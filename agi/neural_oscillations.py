#!/usr/bin/env python3
"""
neural_oscillations.py - Neural Oscillations System
Implements brain wave patterns for different cognitive states
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

class NeuralOscillations(nn.Module):
    """Implements brain wave patterns for different cognitive states"""

    def __init__(self, embed_dim=8192, num_regions=11, simulation_freq=1000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_regions = num_regions
        self.simulation_freq = simulation_freq  # Hz

        # Oscillation frequencies (Hz)
        self.wave_frequencies = {
            'delta': 0.5,      # Deep sleep, unconscious processing
            'theta': 6.0,      # Meditation, REM sleep, memory consolidation
            'alpha': 10.0,     # Relaxed wakefulness, creativity
            'beta': 20.0,      # Active thinking, problem solving
            'gamma': 40.0,     # High cognition, insight, peak consciousness
            'epsilon': 80.0    # Ultra-high cognition, transcendent states
        }

        # Phase-locked loops for each region and frequency
        self.phase_generators = nn.ModuleDict()
        self.frequency_modulators = nn.ModuleDict()

        brain_regions = self._get_brain_regions()
        for region in brain_regions:
            self.phase_generators[region] = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.Tanh(),
                nn.Linear(embed_dim // 4, len(self.wave_frequencies))
            )

            self.frequency_modulators[region] = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 8),
                nn.ReLU(),
                nn.Linear(embed_dim // 8, len(self.wave_frequencies)),
                nn.Sigmoid()
            )

        # Cross-regional coupling matrix (learnable)
        self.coupling_matrix = nn.Parameter(torch.randn(num_regions, num_regions) * 0.1)

        # State-dependent frequency modulation
        self.state_modulator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, len(self.wave_frequencies))
        )

        # Cognitive state classifier
        self.cognitive_state_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 8)  # 8 cognitive states
        )

        # Initialize phases
        self.phases = {freq: torch.zeros(num_regions) for freq in self.wave_frequencies}
        self.phase_velocities = {freq: torch.ones(num_regions) * freq * 2 * math.pi
                               for freq in self.wave_frequencies}

        print(f"[NEURAL_OSCILLATIONS] Initialized with {num_regions} regions, {len(self.wave_frequencies)} frequency bands")

    def _get_brain_regions(self) -> List[str]:
        """Get list of brain regions"""
        return ['prefrontal', 'motor', 'sensory', 'visual', 'auditory',
                'limbic', 'hippocampus', 'amygdala', 'thalamus', 'cerebellum', 'brainstem']

    def _classify_cognitive_state(self, cognitive_input: torch.Tensor) -> str:
        """Classify the current cognitive state"""
        state_logits = self.cognitive_state_classifier(cognitive_input)
        state_idx = torch.argmax(state_logits).item()

        cognitive_states = [
            'deep_sleep', 'meditation', 'relaxed', 'active_thinking',
            'problem_solving', 'insight', 'peak_consciousness', 'transcendent'
        ]

        return cognitive_states[state_idx]

    def _get_state_specific_frequencies(self, cognitive_state: str) -> Dict[str, float]:
        """Get frequency adjustments based on cognitive state"""
        state_freq_mods = {
            'deep_sleep': {'delta': 1.2, 'theta': 0.8, 'alpha': 0.5, 'beta': 0.3, 'gamma': 0.1, 'epsilon': 0.0},
            'meditation': {'delta': 0.8, 'theta': 1.5, 'alpha': 1.2, 'beta': 0.8, 'gamma': 0.6, 'epsilon': 0.0},
            'relaxed': {'delta': 0.5, 'theta': 0.8, 'alpha': 1.5, 'beta': 1.0, 'gamma': 0.7, 'epsilon': 0.0},
            'active_thinking': {'delta': 0.3, 'theta': 0.6, 'alpha': 0.8, 'beta': 1.5, 'gamma': 1.0, 'epsilon': 0.0},
            'problem_solving': {'delta': 0.2, 'theta': 0.5, 'alpha': 0.7, 'beta': 1.8, 'gamma': 1.2, 'epsilon': 0.0},
            'insight': {'delta': 0.1, 'theta': 0.4, 'alpha': 0.6, 'beta': 1.2, 'gamma': 2.0, 'epsilon': 0.5},
            'peak_consciousness': {'delta': 0.0, 'theta': 0.3, 'alpha': 0.5, 'beta': 1.0, 'gamma': 1.8, 'epsilon': 1.0},
            'transcendent': {'delta': 0.0, 'theta': 0.2, 'alpha': 0.3, 'beta': 0.8, 'gamma': 1.5, 'epsilon': 2.0}
        }

        return state_freq_mods.get(cognitive_state, {freq: 1.0 for freq in self.wave_frequencies})

    def update_oscillations(self, dt: float, cognitive_state_embedding: torch.Tensor):
        """Update neural oscillations based on cognitive state"""

        # Classify cognitive state
        cognitive_state = self._classify_cognitive_state(cognitive_state_embedding)

        # Get state-specific frequency modulations
        freq_mods = self._get_state_specific_frequencies(cognitive_state)

        # Update phases for each frequency band
        for freq_name, base_freq in self.wave_frequencies.items():
            # Apply state-dependent modulation
            freq_mod = freq_mods[freq_name]
            current_freq = base_freq * freq_mod

            # Update phase velocities
            target_velocity = current_freq * 2 * math.pi
            self.phase_velocities[freq_name] = 0.9 * self.phase_velocities[freq_name] + 0.1 * target_velocity

            # Update phases
            self.phases[freq_name] += self.phase_velocities[freq_name] * dt

            # Keep phases in reasonable range
            self.phases[freq_name] = self.phases[freq_name] % (2 * math.pi)

    def generate_oscillatory_signals(self, cognitive_state_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate oscillatory signals for all frequency bands"""

        oscillations = {}
        region_phases = {}

        brain_regions = self._get_brain_regions()

        for freq_name in self.wave_frequencies:
            # Generate base oscillations
            base_oscillation = torch.sin(self.phases[freq_name])

            # Apply regional phase adjustments
            regional_phases = []
            for i, region in enumerate(brain_regions):
                region_input = cognitive_state_embedding + base_oscillation[i].unsqueeze(0)
                phase_adjustment = self.phase_generators[region](region_input)
                regional_phase = self.phases[freq_name][i] + phase_adjustment.squeeze()
                regional_phases.append(regional_phase)

            regional_phases = torch.stack(regional_phases)
            oscillations[freq_name] = torch.sin(regional_phases)

        # Apply cross-regional coupling
        coupled_oscillations = {}
        for freq_name, osc_signal in oscillations.items():
            coupling_effect = torch.matmul(self.coupling_matrix, osc_signal.unsqueeze(-1)).squeeze(-1)
            coupled_oscillations[freq_name] = osc_signal + 0.1 * coupling_effect

        # Store region phases for output
        for i, region in enumerate(brain_regions):
            region_phases[region] = torch.stack([coupled_oscillations[freq][i] for freq in self.wave_frequencies])

        return {
            'oscillations': coupled_oscillations,
            'region_phases': region_phases,
            'dominant_frequency': self._get_dominant_frequency(coupled_oscillations)
        }

    def _get_dominant_frequency(self, oscillations: Dict[str, torch.Tensor]) -> str:
        """Determine the dominant frequency band"""
        amplitudes = {freq: osc.abs().mean().item() for freq, osc in oscillations.items()}
        return max(amplitudes, key=amplitudes.get)

    def modulate_cognitive_processing(self, input_tensor: torch.Tensor,
                                    oscillations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Modulate neural processing based on oscillation patterns"""

        # Extract dominant oscillation
        dominant_freq = oscillations['dominant_frequency']
        osc_signal = oscillations['oscillations'][dominant_freq]

        # Apply oscillatory modulation
        modulation = 1.0 + 0.1 * osc_signal.mean()

        # Phase-based attention modulation
        phase_coherence = self._compute_phase_coherence(oscillations)
        attention_modulation = 1.0 + 0.05 * phase_coherence

        # Apply modulations
        modulated_output = input_tensor * modulation * attention_modulation

        return modulated_output

    def _compute_phase_coherence(self, oscillations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute phase coherence across regions"""
        # Simplified phase coherence calculation
        dominant_osc = oscillations['oscillations'][oscillations['dominant_frequency']]
        mean_phase = dominant_osc.mean()
        coherence = 1.0 - (dominant_osc - mean_phase).abs().mean()
        return coherence

    def synchronize_regions(self, target_regions: List[str], sync_strength: float = 0.5):
        """Synchronize oscillations between specific regions"""
        if len(target_regions) < 2:
            return

        region_indices = [self._get_brain_regions().index(r) for r in target_regions]

        # Apply synchronization through coupling matrix updates
        for i in region_indices:
            for j in region_indices:
                if i != j:
                    self.coupling_matrix.data[i, j] += sync_strength * 0.01
                    self.coupling_matrix.data[j, i] += sync_strength * 0.01

        # Normalize coupling matrix
        self.coupling_matrix.data.clamp_(-1.0, 1.0)

    def get_oscillation_status(self) -> Dict:
        """Get current oscillation status"""
        dominant_freq = self._get_dominant_frequency(
            {freq: torch.sin(self.phases[freq]) for freq in self.wave_frequencies}
        )

        return {
            'dominant_frequency': dominant_freq,
            'phase_coherence': self._compute_phase_coherence({
                'oscillations': {freq: torch.sin(self.phases[freq]) for freq in self.wave_frequencies},
                'dominant_frequency': dominant_freq
            }).item(),
            'mean_phase_velocity': np.mean([vel.mean().item() for vel in self.phase_velocities.values()]),
            'coupling_strength': self.coupling_matrix.abs().mean().item()
        }

    def forward(self, cognitive_state: torch.Tensor, dt: float = 0.001) -> Dict[str, torch.Tensor]:
        """Main forward pass"""

        # Update oscillations
        self.update_oscillations(dt, cognitive_state)

        # Generate oscillatory signals
        oscillation_output = self.generate_oscillatory_signals(cognitive_state)

        # Apply modulation to input
        modulated_output = self.modulate_cognitive_processing(cognitive_state, oscillation_output)

        return {
            'modulated_output': modulated_output,
            'oscillations': oscillation_output['oscillations'],
            'region_phases': oscillation_output['region_phases'],
            'dominant_frequency': oscillation_output['dominant_frequency'],
            'phase_coherence': self._compute_phase_coherence(oscillation_output)
        }