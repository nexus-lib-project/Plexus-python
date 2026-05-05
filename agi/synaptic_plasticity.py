#!/usr/bin/env python3
"""
synaptic_plasticity.py - Advanced Synaptic Plasticity with Neuromodulation
Implements neuromodulated Hebbian learning with spike-timing dependent plasticity
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

class AdvancedSynapticPlasticity(nn.Module):
    """Advanced Hebbian learning with neuromodulation"""

    def __init__(self, embed_dim=8192, num_neurons=1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons

        # Base synaptic weights (learnable)
        self.weights = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.01)

        # Neuromodulator systems
        self.dopamine_modulation = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_neurons)
        )

        self.serotonin_modulation = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_neurons)
        )

        self.norepinephrine_modulation = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_neurons)
        )

        # Homeostatic scaling
        self.homeostatic_scaling = nn.Linear(num_neurons, num_neurons)

        # Long-term potentiation/depression tracking
        self.ltp_ltd_memory = torch.zeros(num_neurons, num_neurons)

        # Spike-timing dependent plasticity (STDP) kernel
        self.stdp_kernel = self._create_stdp_kernel()

        # Neuromodulator state tracking
        self.neuromodulator_state = {
            'dopamine': torch.zeros(num_neurons),
            'serotonin': torch.zeros(num_neurons),
            'norepinephrine': torch.zeros(num_neurons)
        }

        print(f"[SYNAPTIC_PLASTICITY] Initialized with {num_neurons} neurons")

    def _create_stdp_kernel(self) -> torch.Tensor:
        """Create STDP learning window"""
        # Time differences from -50ms to +50ms
        t_pre = torch.linspace(-50, 50, 101)

        # STDP parameters
        A_plus = 0.1    # LTP amplitude
        A_minus = -0.05 # LTD amplitude
        tau_plus = 20.0  # LTP time constant
        tau_minus = 20.0 # LTD time constant

        # STDP window function
        stdp_window = torch.where(
            t_pre >= 0,
            A_plus * torch.exp(-t_pre / tau_plus),
            A_minus * torch.exp(t_pre / tau_minus)
        )

        return nn.Parameter(stdp_window.unsqueeze(0).unsqueeze(0))

    def _get_spike_times(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Extract spike times from spike train"""
        # Simple threshold-based spike detection
        threshold = spike_train.mean() + spike_train.std()
        spikes = (spike_train > threshold).float()

        # Convert to time indices
        spike_times = []
        for i in range(spike_train.shape[0]):
            spike_indices = torch.where(spikes[i] > 0)[0]
            if len(spike_indices) > 0:
                spike_times.append(spike_indices.float().mean())
            else:
                spike_times.append(torch.tensor(float('inf')))

        return torch.stack(spike_times)

    def _compute_stdp_updates(self, spike_times_pre: torch.Tensor,
                            spike_times_post: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Compute STDP weight updates"""
        # Compute time differences
        time_diff = spike_times_pre.unsqueeze(1) - spike_times_post.unsqueeze(0)

        # Apply STDP kernel
        stdp_updates = torch.zeros_like(self.weights)

        # Vectorized STDP computation
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if time_diff[i, j].isinf() or (-50 <= time_diff[i, j] <= 50):
                    # Interpolate STDP kernel
                    idx = int((time_diff[i, j] + 50) / 100 * 100)  # Map to kernel indices
                    idx = max(0, min(100, idx))
                    stdp_updates[i, j] = self.stdp_kernel[0, 0, idx]

        return stdp_updates

    def update_neuromodulators(self, neuromodulator_inputs: Dict[str, torch.Tensor]):
        """Update neuromodulator states"""
        for modulator, input_tensor in neuromodulator_inputs.items():
            if modulator in self.neuromodulator_state:
                # Apply neuromodulator-specific processing
                if modulator == 'dopamine':
                    update = self.dopamine_modulation(input_tensor)
                elif modulator == 'serotonin':
                    update = self.serotonin_modulation(input_tensor)
                elif modulator == 'norepinephrine':
                    update = self.norepinephrine_modulation(input_tensor)

                # Update state with decay
                decay_rate = 0.9
                self.neuromodulator_state[modulator] = (
                    decay_rate * self.neuromodulator_state[modulator] +
                    (1 - decay_rate) * torch.tanh(update)
                )

    def apply_plasticity_updates(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                               neuromodulators: Dict[str, torch.Tensor], dt: float = 1.0):
        """Apply synaptic plasticity with neuromodulation"""

        # Update neuromodulator states
        self.update_neuromodulators(neuromodulators)

        # Compute STDP updates
        spike_times_pre = self._get_spike_times(pre_spikes)
        spike_times_post = self._get_spike_times(post_spikes)
        delta_w_stdp = self._compute_stdp_updates(spike_times_pre, spike_times_post, dt)

        # Apply neuromodulation
        dopamine_effect = torch.sigmoid(self.neuromodulator_state['dopamine'])
        serotonin_effect = torch.sigmoid(self.neuromodulator_state['serotonin'])
        ne_effect = torch.sigmoid(self.neuromodulator_state['norepinephrine'])

        # Modulate plasticity
        modulation_factor = 1.0 + dopamine_effect * 0.5 + serotonin_effect * 0.3 + ne_effect * 0.2
        modulation_factor = modulation_factor.unsqueeze(1).expand_as(self.weights)

        # Apply Hebbian learning component
        hebbian_update = torch.matmul(pre_spikes.T, post_spikes) / pre_spikes.shape[0]
        hebbian_update = hebbian_update * 0.01  # Scale factor

        # Combine updates
        delta_w = (delta_w_stdp + hebbian_update) * modulation_factor * dt

        # Apply homeostatic scaling
        activity_levels = self.weights.abs().mean(dim=1, keepdim=True)
        homeostatic_factor = self.homeostatic_scaling(activity_levels.squeeze())
        homeostatic_factor = torch.sigmoid(homeostatic_factor).unsqueeze(1).expand_as(self.weights)

        # Update weights
        self.weights.data += delta_w
        self.weights.data *= homeostatic_factor

        # Weight bounds and normalization
        self.weights.data.clamp_(-2.0, 2.0)

        # Update LTP/LTD memory
        self.ltp_ltd_memory = 0.95 * self.ltp_ltd_memory + 0.05 * delta_w

        return delta_w

    def get_synaptic_state(self) -> Dict:
        """Get current synaptic state information"""
        return {
            'mean_weight': self.weights.mean().item(),
            'weight_std': self.weights.std().item(),
            'sparsity': (self.weights.abs() < 0.01).float().mean().item(),
            'neuromodulator_levels': {
                mod: state.mean().item()
                for mod, state in self.neuromodulator_state.items()
            },
            'plasticity_history': self.ltp_ltd_memory.mean().item()
        }

    def reset_plasticity(self):
        """Reset plasticity state (for testing or initialization)"""
        self.weights.data = torch.randn_like(self.weights) * 0.01
        self.ltp_ltd_memory.zero_()
        for modulator in self.neuromodulator_state:
            self.neuromodulator_state[modulator].zero_()

    def forward(self, pre_activity: torch.Tensor, post_activity: torch.Tensor,
                neuromodulators: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass - apply synaptic transmission with plasticity"""

        # Ensure correct dimensions
        if pre_activity.dim() == 1:
            pre_activity = pre_activity.unsqueeze(0)
        if post_activity.dim() == 1:
            post_activity = post_activity.unsqueeze(0)

        # Apply synaptic weights
        output = torch.matmul(pre_activity, self.weights.t())

        # Apply plasticity updates
        self.apply_plasticity_updates(pre_activity, post_activity, neuromodulators)

        return output.squeeze(0) if output.shape[0] == 1 else output