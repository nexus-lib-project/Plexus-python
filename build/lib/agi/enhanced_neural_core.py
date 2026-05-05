"""
Enhanced Neural Core with Ion Channel Simulation
===============================================
Extends RichNeuron with detailed ion channel dynamics and integrates
with astrocyte networks for biologically accurate neural processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from astrocyte_network import AstrocyteNetwork, AstrocyteCognitiveProcessor
from .affective_cognition import NEUROTRANSMITTER_EMOTION_MAP


class IonChannel(nn.Module):
    """
    Individual Ion Channel Simulation
    Models voltage-gated and ligand-gated ion channels
    """

    def __init__(self, ion_type: str = 'Na', conductance: float = 1.0):
        super().__init__()
        self.ion_type = ion_type
        self.baseline_conductance = conductance

        # Voltage/ligand gating parameters
        self.gate_params = nn.ParameterDict({
            'activation_v_half': nn.Parameter(torch.tensor(0.0)),  # Half-activation voltage
            'activation_slope': nn.Parameter(torch.tensor(5.0)),   # Slope of activation curve
            'inactivation_v_half': nn.Parameter(torch.tensor(0.0)), # Inactivation voltage
            'inactivation_slope': nn.Parameter(torch.tensor(-5.0)), # Inactivation slope
            'time_constant': nn.Parameter(torch.tensor(1.0))       # Channel kinetics
        })

        # Conductance modulation
        self.conductance_modulator = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Ion-specific reversal potentials (mV)
        self.reversal_potentials = {
            'Na': 50.0, 'K': -90.0, 'Ca': 120.0, 'Cl': -70.0
        }

    def forward(self, membrane_potential: torch.Tensor,
                ligand_concentration: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute ion current through channel
        """
        v = membrane_potential

        # Activation gating (voltage-dependent)
        activation_gate = torch.sigmoid(
            self.gate_params['activation_slope'] *
            (v - self.gate_params['activation_v_half'])
        )

        # Inactivation gating
        inactivation_gate = 1 - torch.sigmoid(
            self.gate_params['inactivation_slope'] *
            (v - self.gate_params['inactivation_v_half'])
        )

        # Ligand gating (if applicable)
        ligand_gate = 1.0
        if ligand_concentration is not None:
            ligand_gate = torch.sigmoid(ligand_concentration)

        # Effective conductance
        base_g = self.baseline_conductance * activation_gate * inactivation_gate * ligand_gate
        modulated_g = self.conductance_modulator(base_g.unsqueeze(-1)).squeeze(-1)

        # Current = conductance * (V - V_rev)
        v_rev = self.reversal_potentials.get(self.ion_type, 0.0)
        current = modulated_g * (v - v_rev)

        return current


class NeuronWithIonChannels(nn.Module):
    """
    Neuron with detailed ion channel simulation
    Extends RichNeuron with biophysical accuracy
    """

    def __init__(self, input_dim: int, output_dim: int,
                 num_concepts: int = 20, concept_dim: int = 64):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Membrane properties
        self.membrane_capacitance = 1e-6  # Farads
        self.membrane_potential = nn.Parameter(torch.tensor(-70.0))  # Resting potential

        # Ion channels
        self.ion_channels = nn.ModuleDict({
            'Na': IonChannel('Na', conductance=120.0),    # Sodium channels
            'K': IonChannel('K', conductance=36.0),      # Potassium channels
            'Ca': IonChannel('Ca', conductance=0.1),     # Calcium channels
            'Cl': IonChannel('Cl', conductance=0.5),     # Chloride channels
            'leak': IonChannel('K', conductance=0.3)     # Leak channels
        })

        # Neurotransmitter receptors (ligand-gated)
        self.neurotransmitter_receptors = nn.ModuleDict({
            'glutamate': IonChannel('Na', conductance=50.0),  # AMPA/NMDA
            'GABA': IonChannel('Cl', conductance=30.0),      # GABA_A
            'acetylcholine': IonChannel('Na', conductance=20.0), # Nicotinic
        })

        # Astrocyte modulation
        self.astrocyte_modulation = nn.Linear(input_dim, len(self.ion_channels))

        # Semantic processing (from RichNeuron)
        self.semantic_processor = nn.Sequential(
            nn.Linear(input_dim, concept_dim),
            nn.GELU(),
            nn.Linear(concept_dim, output_dim)
        )

        # Calcium dynamics for synaptic plasticity
        self.calcium_level = 0.0
        self.calcium_decay = 0.9

    def compute_ion_currents(self, membrane_potential: torch.Tensor,
                           neurotransmitter_levels: Dict[str, float]) -> torch.Tensor:
        """Compute total ionic current through all channels"""
        total_current = torch.zeros_like(membrane_potential)

        # Voltage-gated channels
        for channel_name, channel in self.ion_channels.items():
            if channel_name != 'leak':  # Leak doesn't depend on voltage
                current = channel(membrane_potential)
                total_current += current

        # Ligand-gated channels (neurotransmitter receptors)
        for nt, level in neurotransmitter_levels.items():
            if nt in self.neurotransmitter_receptors:
                nt_conc = torch.tensor(level, device=membrane_potential.device)
                current = self.neurotransmitter_receptors[nt](membrane_potential, nt_conc)
                total_current += current

        return total_current

    def update_membrane_potential(self, input_current: torch.Tensor,
                                neurotransmitter_levels: Dict[str, float],
                                dt: float = 0.001) -> torch.Tensor:
        """Update membrane potential using Hodgkin-Huxley style dynamics"""
        # Compute ionic currents
        ionic_currents = self.compute_ion_currents(self.membrane_potential, neurotransmitter_levels)

        # Membrane potential change (C * dV/dt = -I_ionic + I_input)
        dv_dt = (-ionic_currents + input_current) / self.membrane_capacitance
        new_potential = self.membrane_potential + dv_dt * dt

        # Update calcium for plasticity
        self.calcium_level = self.calcium_decay * self.calcium_level + torch.abs(new_potential - self.membrane_potential).item()

        self.membrane_potential.data = new_potential
        return new_potential

    def forward(self, x: torch.Tensor,
                neurotransmitter_levels: Dict[str, float] = None,
                astrocyte_modulation: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with ion channel dynamics
        """
        if neurotransmitter_levels is None:
            neurotransmitter_levels = {'glutamate': 0.5, 'GABA': 0.3}

        batch_size = x.size(0)

        # Convert input to input current
        input_current = x.mean(dim=-1) * 1e-9  # Convert to amperes

        # Update membrane potential
        membrane_potential = self.update_membrane_potential(input_current, neurotransmitter_levels)

        # Semantic processing (spike-encoded output)
        semantic_output = self.semantic_processor(x)

        # Action potential generation (simplified)
        threshold = -50.0  # mV
        spikes = (membrane_potential > threshold).float()

        # Modulate output by spikes
        final_output = semantic_output * spikes.unsqueeze(-1).expand_as(semantic_output)

        # Astrocyte modulation
        if astrocyte_modulation is not None:
            channel_modulations = self.astrocyte_modulation(astrocyte_modulation)
            # Apply modulation to channel conductances
            for i, (channel_name, channel) in enumerate(self.ion_channels.items()):
                if i < channel_modulations.size(-1):
                    mod_factor = torch.sigmoid(channel_modulations[..., i])
                    # This would modulate the channel conductance

        return {
            'output': final_output,
            'membrane_potential': membrane_potential,
            'spikes': spikes,
            'calcium_level': torch.tensor(self.calcium_level),
            'ion_currents': self.compute_ion_currents(membrane_potential, neurotransmitter_levels)
        }


class EnhancedRichNeuron(nn.Module):
    """
    Rich Neuron with ion channel simulation and astrocyte integration
    """

    def __init__(self, input_dim: int, output_dim: int,
                 num_concepts: int = 20, concept_dim: int = 64,
                 num_astrocytes: int = 4):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Core neuron with ion channels
        self.ion_neuron = NeuronWithIonChannels(input_dim, output_dim, num_concepts, concept_dim)

        # Astrocyte network for modulation
        self.astrocyte_network = AstrocyteNetwork(
            num_astrocytes=num_astrocytes,
            input_dim=input_dim
        )

        # Neurotransmitter state
        self.neurotransmitter_state = {
            name: 0.5 for name in ['dopamine', 'serotonin', 'glutamate', 'GABA',
                                 'acetylcholine', 'norepinephrine', 'oxytocin']
        }

        # Emotion computation from neurotransmitters
        self.emotion_computer = nn.Sequential(
            nn.Linear(len(self.neurotransmitter_state), 32),
            nn.GELU(),
            nn.Linear(32, len(NEUROTRANSMITTER_EMOTION_MAP))
        )

    def update_neurotransmitters(self, emotional_input: torch.Tensor):
        """Update neurotransmitter levels based on emotional context"""
        # Simple update rule - in real implementation would be more complex
        for nt in self.neurotransmitter_state:
            # Modulate based on emotional input
            modulation = emotional_input.mean().item() * 0.1
            self.neurotransmitter_state[nt] = np.clip(
                self.neurotransmitter_state[nt] + modulation, 0.0, 1.0
            )

    def compute_emotion(self) -> Dict[str, float]:
        """Compute emotions from current neurotransmitter state"""
        nt_values = torch.tensor(list(self.neurotransmitter_state.values())).unsqueeze(0)
        emotion_logits = self.emotion_computer(nt_values)
        emotion_probs = F.softmax(emotion_logits, dim=-1)

        emotions = {}
        for i, emotion_name in enumerate(NEUROTRANSMITTER_EMOTION_MAP.keys()):
            emotions[emotion_name] = emotion_probs[0, i].item()

        return emotions

    def forward(self, x: torch.Tensor,
                emotional_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with astrocyte modulation and ion channels
        """
        batch_size = x.size(0)

        # Update neurotransmitters if emotional context provided
        if emotional_context is not None:
            self.update_neurotransmitters(emotional_context)

        # Astrocyte processing
        astro_result = self.astrocyte_network(x)
        astrocyte_modulation = astro_result['network_modulation']

        # Neuron processing with ion channels
        neuron_result = self.ion_neuron(
            x,
            neurotransmitter_levels=self.neurotransmitter_state,
            astrocyte_modulation=astrocyte_modulation
        )

        # Compute emotions
        emotions = self.compute_emotion()

        # Enhanced output
        enhanced_output = neuron_result['output'] * astro_result['consolidation_signal'].unsqueeze(-1)

        return {
            'output': enhanced_output,
            'emotions': emotions,
            'membrane_potential': neuron_result['membrane_potential'],
            'spikes': neuron_result['spikes'],
            'calcium_level': neuron_result['calcium_level'],
            'astrocyte_activity': astro_result['calcium_levels'],
            'neurotransmitter_state': self.neurotransmitter_state.copy()
        }


class NeuroscienceEnhancedBrain(nn.Module):
    """
    Complete neuroscience-enhanced brain with ion channels, astrocytes,
    and emotion-driven processing
    """

    def __init__(self, embed_dim: int = 256, num_neurons: int = 128,
                 num_astrocytes: int = 16):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_neurons = num_neurons

        # Layer of enhanced rich neurons
        self.neural_layer = nn.ModuleList([
            EnhancedRichNeuron(embed_dim, embed_dim, num_astrocytes=num_astrocytes//4)
            for _ in range(num_neurons)
        ])

        # Global astrocyte network
        self.global_astrocyte_network = AstrocyteCognitiveProcessor(
            embed_dim=embed_dim,
            num_astrocytes=num_astrocytes
        )

        # Emotion integration
        self.emotion_integrator = nn.Sequential(
            nn.Linear(len(NEUROTRANSMITTER_EMOTION_MAP), embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Social cognition layer
        self.social_cognition = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Goal generation based on emotions and social factors
        self.goal_generator = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # emotions + social context
            nn.GELU(),
            nn.Linear(embed_dim, 10)  # 10 possible goals
        )

    def compute_social_context(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """Extract social context from input"""
        # Simple social context detection - in real implementation would be more sophisticated
        return self.social_cognition(input_embeds)

    def generate_goals(self, emotions: Dict[str, float],
                      social_context: torch.Tensor) -> torch.Tensor:
        """Generate goals based on emotions and social factors"""
        # Convert emotions to tensor
        emotion_tensor = torch.tensor(list(emotions.values())).to(social_context.device)
        emotion_tensor = emotion_tensor.unsqueeze(0).expand(social_context.size(0), -1)

        # Combine emotions and social context
        combined = torch.cat([emotion_tensor, social_context], dim=-1)
        goal_logits = self.goal_generator(combined)

        return F.softmax(goal_logits, dim=-1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through neuroscience-enhanced brain
        """
        batch_size = x.size(0)

        # Process through individual neurons
        neuron_outputs = []
        all_emotions = []
        all_spikes = []
        all_membrane_potentials = []

        for neuron in self.neural_layer:
            result = neuron(x)
            neuron_outputs.append(result['output'])
            all_emotions.append(result['emotions'])
            all_spikes.append(result['spikes'])
            all_membrane_potentials.append(result['membrane_potential'])

        # Aggregate neuron outputs
        neural_output = torch.stack(neuron_outputs, dim=1).mean(dim=1)

        # Aggregate emotions (average across neurons)
        avg_emotions = {}
        for emotion_name in all_emotions[0].keys():
            avg_emotions[emotion_name] = np.mean([e[emotion_name] for e in all_emotions])

        # Global astrocyte processing
        astro_result = self.global_astrocyte_network(neural_output)

        # Social context processing
        social_context = self.compute_social_context(x)

        # Goal generation
        goals = self.generate_goals(avg_emotions, social_context)

        # Emotion integration
        emotion_tensor = torch.tensor(list(avg_emotions.values())).unsqueeze(0).to(x.device)
        emotion_enhanced = self.emotion_integrator(emotion_tensor)

        # Final output modulated by emotions and goals
        final_output = neural_output * emotion_enhanced.unsqueeze(0) * astro_result['output']

        return {
            'output': final_output,
            'emotions': avg_emotions,
            'goals': goals,
            'social_context': social_context,
            'astrocyte_activity': astro_result['calcium_waves'],
            'spike_rate': torch.stack(all_spikes).mean(),
            'avg_membrane_potential': torch.stack(all_membrane_potentials).mean(),
            'neural_activity': neural_output
        }


# Global instance
_enhanced_brain_instance = None

def get_enhanced_brain(embed_dim: int = 256) -> NeuroscienceEnhancedBrain:
    """Get or create enhanced neuroscience brain"""
    global _enhanced_brain_instance
    if _enhanced_brain_instance is None:
        _enhanced_brain_instance = NeuroscienceEnhancedBrain(embed_dim=embed_dim)
        print(f"[NEUROSCIENCE_BRAIN] Created enhanced brain with ion channels and astrocytes")
    return _enhanced_brain_instance