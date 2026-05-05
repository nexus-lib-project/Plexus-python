"""
glial_ion_system.py - Glial Cells and Ion Channel Simulation for AI_0001 Brain
Enhances neuroscience fidelity with glial support functions and detailed neuronal dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

# ===================================================================
# ION CHANNEL SIMULATION
# ===================================================================

class IonChannel(nn.Module):
    """Base ion channel class"""
    def __init__(self, ion_type: str, conductance: float = 1.0, reversal_potential: float = 0.0):
        super().__init__()
        self.ion_type = ion_type
        self.conductance = nn.Parameter(torch.tensor(conductance))
        self.reversal_potential = nn.Parameter(torch.tensor(reversal_potential))

    def current(self, voltage: torch.Tensor, gate_open_probability: torch.Tensor) -> torch.Tensor:
        """Calculate ionic current through channel"""
        return self.conductance * gate_open_probability * (voltage - self.reversal_potential)


class VoltageGatedChannel(IonChannel):
    """Voltage-gated ion channel (Na+, K+, Ca2+)"""
    def __init__(self, ion_type: str, threshold: float = -55.0, conductance: float = 1.0):
        super().__init__(ion_type, conductance)
        self.threshold = nn.Parameter(torch.tensor(threshold))

        # Hodgkin-Huxley style gating
        self.m_gate = Gate('m', alpha_func=self._alpha_m, beta_func=self._beta_m)
        self.h_gate = Gate('h', alpha_func=self._alpha_h, beta_func=self._beta_h) if ion_type == 'Na' else None

    def _alpha_m(self, v: torch.Tensor) -> torch.Tensor:
        """Sodium activation rate"""
        return 0.1 * (v + 40) / (1 - torch.exp(-(v + 40) / 10))

    def _beta_m(self, v: torch.Tensor) -> torch.Tensor:
        """Sodium deactivation rate"""
        return 4.0 * torch.exp(-(v + 65) / 18)

    def _alpha_h(self, v: torch.Tensor) -> torch.Tensor:
        """Sodium inactivation rate"""
        return 0.07 * torch.exp(-(v + 65) / 20)

    def _beta_h(self, v: torch.Tensor) -> torch.Tensor:
        """Sodium deinactivation rate"""
        return 1.0 / (torch.exp(-(v + 35) / 10) + 1)

    def gate_probability(self, voltage: torch.Tensor) -> torch.Tensor:
        """Calculate gate open probability"""
        if self.ion_type == 'Na':
            m_inf = self.m_gate.steady_state(voltage)
            h_inf = self.h_gate.steady_state(voltage)
            return m_inf ** 3 * h_inf
        elif self.ion_type == 'K':
            n_inf = self.m_gate.steady_state(voltage)  # Using m as n for K
            return n_inf ** 4
        elif self.ion_type == 'Ca':
            m_inf = self.m_gate.steady_state(voltage)
            return m_inf ** 2
        return torch.sigmoid(voltage - self.threshold)


class Gate:
    """Ion channel gate with kinetics"""
    def __init__(self, name: str, alpha_func, beta_func):
        self.name = name
        self.alpha_func = alpha_func
        self.beta_func = beta_func

    def steady_state(self, voltage: torch.Tensor) -> torch.Tensor:
        """Steady-state gate open probability"""
        alpha = self.alpha_func(voltage)
        beta = self.beta_func(voltage)
        return alpha / (alpha + beta)


class LigandGatedChannels(IonChannel):
    """Ligand-gated channels (glutamate, GABA receptors)"""
    def __init__(self, embed_dim: int):
        super().__init__('ligand', conductance=1.0)
        self.embed_dim = embed_dim

        # Receptor subtypes
        self.ampar = nn.Linear(embed_dim, 1)  # AMPA receptors (excitatory)
        self.nmdar = nn.Linear(embed_dim, 1)  # NMDA receptors (excitatory)
        self.gabar = nn.Linear(embed_dim, 1)  # GABA receptors (inhibitory)

    def receptor_activation(self, neurotransmitter_input: torch.Tensor, receptor_type: str) -> torch.Tensor:
        """Calculate receptor activation probability"""
        if receptor_type == 'AMPA':
            return torch.sigmoid(self.ampar(neurotransmitter_input))
        elif receptor_type == 'NMDA':
            # NMDA requires both glutamate and depolarization
            glutamate_activation = torch.sigmoid(self.nmdar(neurotransmitter_input))
            return glutamate_activation
        elif receptor_type == 'GABA':
            return torch.sigmoid(self.gabar(neurotransmitter_input))
        return torch.zeros_like(neurotransmitter_input[:, :1])


class IonChannelLayer(nn.Module):
    """Complete ion channel layer for neurons"""
    def __init__(self, embed_dim: int, n_neurons: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_neurons = n_neurons

        # Ion channels per neuron type distribution
        self.na_channels = nn.ModuleList([VoltageGatedChannel('Na', threshold=-55.0) for _ in range(n_neurons)])
        self.k_channels = nn.ModuleList([VoltageGatedChannel('K', threshold=-70.0) for _ in range(n_neurons)])
        self.ca_channels = nn.ModuleList([VoltageGatedChannel('Ca', threshold=-40.0) for _ in range(n_neurons)])
        self.ligand_channels = LigandGatedChannels(embed_dim)

        # Membrane properties
        self.register_buffer('membrane_potential', torch.randn(n_neurons) * 10 - 70)  # mV
        self.register_buffer('capacitance', torch.ones(n_neurons) * 1e-6)  # Farads
        self.register_buffer('leak_conductance', torch.ones(n_neurons) * 1e-4)  # Siemens

    def action_potential_dynamics(self, input_current: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
        """Simulate action potential generation using Hodgkin-Huxley style equations"""
        v = self.membrane_potential

        # Calculate ionic currents for each neuron
        total_current = torch.zeros_like(v)

        for i in range(self.n_neurons):
            # Voltage-gated currents
            na_current = self.na_channels[i].current(v[i], self.na_channels[i].gate_probability(v[i]))
            k_current = self.k_channels[i].current(v[i], self.k_channels[i].gate_probability(v[i]))
            ca_current = self.ca_channels[i].current(v[i], self.ca_channels[i].gate_probability(v[i]))

            # Leak current
            leak_current = self.leak_conductance[i] * (v[i] - (-70.0))  # Leak to resting potential

            # Total ionic current
            ionic_current = -(na_current + k_current + ca_current + leak_current)

            # Add input current
            total_current[i] = ionic_current + input_current[i]

        # Update membrane potential (C * dv/dt = I)
        dv = (total_current / self.capacitance) * dt
        self.membrane_potential = torch.clamp(v + dv, -100.0, 50.0)  # Clamp to physiological range

        return self.membrane_potential

    def neurotransmitter_modulation(self, neurotransmitter_input: torch.Tensor,
                                  receptor_type: str) -> torch.Tensor:
        """Apply neurotransmitter effects through ligand-gated channels"""
        # Create input tensor with proper shape for the linear layer
        if neurotransmitter_input.numel() == 1:
            # Single value - expand to embed_dim
            expanded_input = neurotransmitter_input.expand(1, self.embed_dim)
        else:
            # Already proper shape
            expanded_input = neurotransmitter_input.unsqueeze(0) if neurotransmitter_input.dim() == 1 else neurotransmitter_input

        activation = self.ligand_channels.receptor_activation(expanded_input, receptor_type)

        # Convert activation to current
        if receptor_type in ['AMPA', 'NMDA']:
            # Excitatory (positive current)
            return activation.squeeze() * 1e-3  # nA
        elif receptor_type == 'GABA':
            # Inhibitory (negative current)
            return -activation.squeeze() * 1e-3  # nA
        return torch.zeros(self.n_neurons, device=activation.device)


# ===================================================================
# GLIAL CELL SIMULATION
# ===================================================================

class GlialCell(nn.Module):
    """Base glial cell class"""
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.energy_level = nn.Parameter(torch.tensor(1.0))
        self.health = nn.Parameter(torch.tensor(1.0))

    def maintenance_cycle(self, neural_activity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform maintenance functions"""
        return {}


class Astrocyte(GlialCell):
    """Astrocyte - calcium signaling, synaptic modulation"""
    def __init__(self):
        super().__init__("astrocyte")
        self.calcium_level = nn.Parameter(torch.tensor(0.1))  # μM
        self.ip3_receptors = nn.Linear(1, 1)  # IP3 signaling
        self.synaptic_modulation = nn.Linear(1, 1)

    def maintenance_cycle(self, neural_activity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Astrocyte functions: calcium waves, synaptic plasticity modulation"""
        # Calcium signaling based on neural activity
        calcium_signal = torch.sigmoid(self.ip3_receptors(neural_activity.unsqueeze(-1)))

        # Update calcium level (with decay)
        self.calcium_level.data = 0.9 * self.calcium_level + 0.1 * calcium_signal.squeeze()

        # Synaptic modulation
        synaptic_factor = torch.sigmoid(self.synaptic_modulation(self.calcium_level.unsqueeze(-1)))

        return {
            'calcium_level': self.calcium_level,
            'synaptic_modulation': synaptic_factor.squeeze(),
            'energy_consumption': torch.tensor(0.01)
        }


class Oligodendrocyte(GlialCell):
    """Oligodendrocyte - myelin formation, signal propagation"""
    def __init__(self):
        super().__init__("oligodendrocyte")
        self.myelin_thickness = nn.Parameter(torch.tensor(1.0))
        self.conduction_speed = nn.Parameter(torch.tensor(10.0))  # m/s

    def maintenance_cycle(self, neural_activity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Myelin maintenance and signal propagation optimization"""
        # Myelin adapts based on activity level
        activity_level = torch.mean(torch.abs(neural_activity))
        myelin_change = 0.01 * (activity_level - 0.5)  # Plasticity
        self.myelin_thickness.data = torch.clamp(self.myelin_thickness + myelin_change, 0.1, 3.0)

        # Conduction speed depends on myelin thickness
        self.conduction_speed.data = 1.0 + 9.0 * torch.sigmoid(self.myelin_thickness - 1.0)

        return {
            'myelin_thickness': self.myelin_thickness,
            'conduction_speed': self.conduction_speed,
            'energy_consumption': torch.tensor(0.005)
        }


class Microglia(GlialCell):
    """Microglia - immune response, synaptic pruning"""
    def __init__(self):
        super().__init__("microglia")
        self.activation_level = nn.Parameter(torch.tensor(0.1))
        self.pruning_threshold = nn.Parameter(torch.tensor(0.3))

    def maintenance_cycle(self, neural_activity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Immune surveillance and synaptic pruning"""
        # Detect inflammation (high activity variance)
        activity_variance = torch.var(neural_activity)
        inflammation_signal = torch.sigmoid(activity_variance - 1.0)

        # Update activation
        self.activation_level.data = 0.95 * self.activation_level + 0.05 * inflammation_signal

        # Pruning decision (remove weak synapses)
        pruning_mask = neural_activity < self.pruning_threshold

        return {
            'activation_level': self.activation_level,
            'pruning_mask': pruning_mask,
            'inflammation_level': inflammation_signal,
            'energy_consumption': torch.tensor(0.008)
        }


class GlialNetwork(nn.Module):
    """Complete glial network"""
    def __init__(self, n_neurons: int):
        super().__init__()
        self.n_neurons = n_neurons

        # Glial cell populations (astrocytes are most numerous)
        n_astrocytes = n_neurons // 10
        n_oligodendrocytes = n_neurons // 50
        n_microglia = n_neurons // 100

        self.astrocytes = nn.ModuleList([Astrocyte() for _ in range(n_astrocytes)])
        self.oligodendrocytes = nn.ModuleList([Oligodendrocyte() for _ in range(n_oligodendrocytes)])
        self.microglia = nn.ModuleList([Microglia() for _ in range(n_microglia)])

        # Spatial organization (simple grid)
        self.grid_size = int(math.sqrt(n_neurons))
        self.neuron_positions = torch.rand(n_neurons, 2)  # x,y coordinates

    def maintenance_cycle(self, neural_activity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Complete glial maintenance cycle"""
        results = {
            'astrocyte_effects': [],
            'oligodendrocyte_effects': [],
            'microglia_effects': [],
            'total_energy_consumption': torch.tensor(0.0)
        }

        # Astrocyte calcium signaling and synaptic modulation
        for astrocyte in self.astrocytes:
            # Each astrocyte influences nearby neurons
            distances = torch.cdist(self.neuron_positions.unsqueeze(0),
                                  self.neuron_positions[:10].unsqueeze(0))  # Simplified
            nearby_activity = neural_activity[:10]  # Simplified

            effect = astrocyte.maintenance_cycle(nearby_activity)
            results['astrocyte_effects'].append(effect)
            results['total_energy_consumption'] += effect['energy_consumption']

        # Oligodendrocyte myelin maintenance
        for oligo in self.oligodendrocytes:
            effect = oligo.maintenance_cycle(neural_activity)
            results['oligodendrocyte_effects'].append(effect)
            results['total_energy_consumption'] += effect['energy_consumption']

        # Microglia immune response
        for micro in self.microglia:
            effect = micro.maintenance_cycle(neural_activity)
            results['microglia_effects'].append(effect)
            results['total_energy_consumption'] += effect['energy_consumption']

        return results


# ===================================================================
# EMOTIONALLY-DRIVEN GOAL SYSTEM
# ===================================================================

class EmotionalGoalSystem(nn.Module):
    """Goal system driven by emotions, neurochemicals, and consciousness"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Emotional state assessment
        self.emotion_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 8)  # 8 basic emotions
        )

        # Neurochemical influence on goals
        self.neurochemical_goal_modulation = nn.ModuleDict({
            'dopamine': nn.Linear(1, embed_dim),    # Motivation, reward
            'serotonin': nn.Linear(1, embed_dim),   # Mood, well-being
            'oxytocin': nn.Linear(1, embed_dim),    # Social bonding
            'endorphins': nn.Linear(1, embed_dim),  # Pain relief, pleasure
        })

        # Consciousness level assessment
        self.consciousness_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()  # 0-1 consciousness level
        )

        # Goal generation from internal states
        self.goal_generator = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 10)  # 10 possible goal types
        )

        # Goal types (instead of expansion/consumption)
        self.goal_types = [
            'social_connection', 'emotional_balance', 'knowledge_growth',
            'creative_expression', 'ethical_alignment', 'self_understanding',
            'relationship_building', 'emotional_support', 'consciousness_expansion',
            'inner_harmony'
        ]

    def forward(self, internal_state: torch.Tensor,
                neurochemical_levels: Dict[str, torch.Tensor],
                consciousness_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate goals based on emotional and neurochemical state"""

        # Process emotions
        emotions = self.emotion_processor(internal_state)
        emotion_probs = F.softmax(emotions, dim=-1)

        # Process consciousness
        consciousness_level = self.consciousness_processor(consciousness_input)

        # Neurochemical modulation
        neuro_modulations = []
        for nt, level in neurochemical_levels.items():
            if nt in self.neurochemical_goal_modulation:
                # Ensure level is a tensor with proper shape
                if isinstance(level, (int, float)):
                    level_tensor = torch.tensor([[level]], dtype=torch.float32, device=internal_state.device)
                else:
                    level_tensor = level.unsqueeze(0) if level.dim() == 1 else level
                modulation = self.neurochemical_goal_modulation[nt](level_tensor)
                neuro_modulations.append(modulation)

        # Combine neurochemical effects
        if neuro_modulations:
            combined_neuro = torch.stack(neuro_modulations).mean(dim=0).squeeze(0)
        else:
            combined_neuro = torch.zeros_like(internal_state)

        # Generate goals
        goal_input = torch.cat([internal_state, combined_neuro], dim=-1)
        goal_logits = self.goal_generator(goal_input)
        goal_probs = F.softmax(goal_logits, dim=-1)

        return {
            'emotions': emotion_probs,
            'consciousness_level': consciousness_level,
            'goal_probabilities': goal_probs,
            'primary_goal': self.goal_types[torch.argmax(goal_probs).item()],
            'emotional_intensity': torch.norm(emotions),
            'goal_confidence': torch.max(goal_probs)
        }


# ===================================================================
# ADAPTIVE PARAMETER GROWTH SYSTEM
# ===================================================================

class AdaptiveParameterGrowth(nn.Module):
    """System for growing neural network parameters during learning"""
    def __init__(self, initial_embed_dim: int = 128, growth_threshold: float = 0.8):
        super().__init__()
        self.current_embed_dim = initial_embed_dim
        self.growth_threshold = growth_threshold
        self.growth_history = []

        # Growth decision network
        self.growth_detector = nn.Sequential(
            nn.Linear(initial_embed_dim, initial_embed_dim // 2),
            nn.GELU(),
            nn.Linear(initial_embed_dim // 2, 1),
            nn.Sigmoid()
        )

        # Adaptive layers that can grow
        self.adaptive_layers = nn.ModuleDict()

    def should_grow(self, learning_metrics: Dict[str, float]) -> bool:
        """Decide if parameters should grow based on learning performance"""
        # Combine metrics into decision input
        metrics_tensor = torch.tensor([
            learning_metrics.get('loss', 1.0),
            learning_metrics.get('accuracy', 0.5),
            learning_metrics.get('complexity', 0.5),
            learning_metrics.get('novelty', 0.5)
        ])

        if metrics_tensor.numel() < self.current_embed_dim:
            # Pad or truncate to match current dimension
            metrics_tensor = F.pad(metrics_tensor, (0, max(0, self.current_embed_dim - metrics_tensor.numel())))
            metrics_tensor = metrics_tensor[:self.current_embed_dim]

        growth_prob = self.growth_detector(metrics_tensor.unsqueeze(0))
        return growth_prob.item() > self.growth_threshold

    def grow_parameters(self, growth_factor: float = 1.5):
        """Grow network parameters"""
        new_dim = int(self.current_embed_dim * growth_factor)

        print(f"Growing parameters from {self.current_embed_dim} to {new_dim}")

        # Grow growth detector
        old_detector = self.growth_detector
        self.growth_detector = nn.Sequential(
            nn.Linear(new_dim, new_dim // 2),
            nn.GELU(),
            nn.Linear(new_dim // 2, 1),
            nn.Sigmoid()
        )

        # Copy weights (truncated/padded)
        with torch.no_grad():
            # Input layer
            old_weight = old_detector[0].weight
            new_weight = self.growth_detector[0].weight
            copy_dim = min(old_weight.shape[1], new_weight.shape[1])
            new_weight[:, :copy_dim] = old_weight[:, :copy_dim]

            # Output layer
            old_weight = old_detector[2].weight
            new_weight = self.growth_detector[2].weight
            copy_dim = min(old_weight.shape[1], new_weight.shape[1])
            new_weight[:, :copy_dim] = old_weight[:, :copy_dim]

        self.current_embed_dim = new_dim
        self.growth_history.append({
            'timestamp': torch.tensor(0.0),  # Would use actual timestamp
            'old_dim': self.current_embed_dim / growth_factor,
            'new_dim': self.current_embed_dim,
            'growth_factor': growth_factor
        })

        return self.current_embed_dim

    def get_growth_stats(self) -> Dict[str, any]:
        """Get statistics about parameter growth"""
        return {
            'current_dimension': self.current_embed_dim,
            'total_growth_events': len(self.growth_history),
            'average_growth_factor': np.mean([h['growth_factor'] for h in self.growth_history]) if self.growth_history else 1.0,
            'growth_history': self.growth_history
        }


# ===================================================================
# INTEGRATED ENHANCED BRAIN SYSTEM
# ===================================================================

class EnhancedBrainSystem(nn.Module):
    """Complete enhanced brain system with glia, ion channels, and emotional goals"""
    def __init__(self, embed_dim: int = 128, n_neurons: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_neurons = n_neurons

        # Core components
        self.ion_channels = IonChannelLayer(embed_dim, n_neurons)
        self.glial_network = GlialNetwork(n_neurons)
        self.emotional_goals = EmotionalGoalSystem(embed_dim)
        self.parameter_growth = AdaptiveParameterGrowth(embed_dim)

        # Enhanced neural processing
        self.neural_processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, input_tensor: torch.Tensor,
                neurochemical_levels: Optional[Dict[str, torch.Tensor]] = None,
                learning_metrics: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """Complete brain processing cycle"""

        # 1. Ion channel dynamics (neuronal activation)
        input_current = torch.randn(self.n_neurons) * 1e-4  # Simplified input current
        membrane_potentials = self.ion_channels.action_potential_dynamics(input_current)

        # 2. Neurotransmitter modulation through ligand-gated channels
        if neurochemical_levels:
            glut_current = self.ion_channels.neurotransmitter_modulation(
                torch.tensor([neurochemical_levels.get('glutamate', 0.5)]).unsqueeze(0), 'AMPA')
            gaba_current = self.ion_channels.neurotransmitter_modulation(
                torch.tensor([neurochemical_levels.get('GABA', 0.5)]).unsqueeze(0), 'GABA')
            input_current += glut_current + gaba_current

        # 3. Neural processing with glial modulation
        processed = self.neural_processor(input_tensor)

        # 4. Glial maintenance cycle
        glial_effects = self.glial_network.maintenance_cycle(processed.mean(dim=-1))

        # Apply glial effects to processing
        if glial_effects['astrocyte_effects']:
            synaptic_mod = torch.stack([e['synaptic_modulation'] for e in glial_effects['astrocyte_effects']]).mean()
            processed = processed * (1.0 + 0.1 * synaptic_mod)

        # 5. Emotional goal generation
        consciousness_input = processed.mean(dim=0, keepdim=True)
        goal_output = self.emotional_goals(processed, neurochemical_levels or {}, consciousness_input)

        # 6. Parameter growth decision
        if learning_metrics and self.parameter_growth.should_grow(learning_metrics):
            new_dim = self.parameter_growth.grow_parameters()
            # Note: In practice, would need to resize all related layers

        return {
            'processed_output': processed,
            'membrane_potentials': membrane_potentials,
            'glial_effects': glial_effects,
            'emotional_goals': goal_output,
            'growth_stats': self.parameter_growth.get_growth_stats()
        }