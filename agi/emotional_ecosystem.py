#!/usr/bin/env python3
"""
emotional_ecosystem.py - Emotional Ecosystem with Propagation Networks
Implements emotion propagation, cultural learning, and time dilation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import math

class EmotionalEcosystem(nn.Module):
    """Emotional contagion and propagation networks"""

    def __init__(self, embed_dim=8192, num_emotions=10, ecosystem_size=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_emotions = num_emotions
        self.ecosystem_size = ecosystem_size

        # Individual emotional states for each entity in ecosystem
        self.emotional_states = nn.Parameter(torch.randn(ecosystem_size, num_emotions))

        # Propagation networks (emotional contagion)
        self.contagion_network = nn.Sequential(
            nn.Linear(num_emotions, num_emotions * 2),
            nn.LayerNorm(num_emotions * 2),
            nn.GELU(),
            nn.Linear(num_emotions * 2, num_emotions)
        )

        # Social influence matrix (who influences whom)
        self.social_influence = nn.Parameter(torch.randn(ecosystem_size, ecosystem_size) * 0.1)

        # Cultural context embeddings (different cultures have different emotional norms)
        self.cultural_contexts = nn.Embedding(50, embed_dim)  # Support for 50 cultures

        # Cultural adaptation network
        self.cultural_adaptor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_emotions)
        )

        # Empathic resonance detector
        self.resonance_detector = nn.MultiheadAttention(
            embed_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # Time dilation based on emotional intensity
        self.time_dilation_network = nn.Sequential(
            nn.Linear(num_emotions, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Emotional momentum (emotions persist over time)
        self.emotional_momentum = 0.8

        # Cultural learning rate
        self.cultural_learning_rate = 0.02

        print(f"[EMOTIONAL_ECOSYSTEM] Initialized with {ecosystem_size} entities, {num_emotions} emotions")

    def propagate_emotions(self, dt: float = 0.1) -> torch.Tensor:
        """Propagate emotions through the ecosystem over time"""

        # Calculate social influence effects
        influence_effects = torch.matmul(self.social_influence, self.emotional_states)

        # Apply contagion dynamics
        contagion_effects = self.contagion_network(self.emotional_states)

        # Combine influences
        total_influence = influence_effects + contagion_effects

        # Update emotional states with momentum
        new_emotions = (self.emotional_momentum * self.emotional_states +
                       (1 - self.emotional_momentum) * total_influence * dt)

        # Normalize emotions (ensure they sum appropriately)
        self.emotional_states.data = torch.softmax(new_emotions, dim=-1)

        return self.emotional_states

    def cultural_emotional_learning(self, experience: torch.Tensor,
                                  cultural_context: int) -> Dict[str, torch.Tensor]:
        """Learn emotional responses from cultural contexts"""

        # Get cultural embedding
        culture_embed = self.cultural_contexts(torch.tensor([cultural_context]))

        # Combine experience with cultural context
        cultural_input = torch.cat([experience, culture_embed.squeeze(0)], dim=-1)

        # Analyze through resonance detector
        resonance_input = culture_embed
        resonance_output, attention_weights = self.resonance_detector(
            resonance_input, experience.unsqueeze(0), experience.unsqueeze(0)
        )

        # Adapt emotional response based on culture
        cultural_adaptation = self.cultural_adaptor(cultural_input)

        # Update emotional states with cultural learning
        learning_effect = cultural_adaptation * self.cultural_learning_rate
        self.emotional_states.data += learning_effect.unsqueeze(0).expand_as(self.emotional_states)

        # Re-normalize
        self.emotional_states.data = torch.softmax(self.emotional_states.data, dim=-1)

        return {
            'cultural_adaptation': cultural_adaptation,
            'resonance_output': resonance_output,
            'attention_weights': attention_weights,
            'learning_effect': learning_effect
        }

    def emotional_time_dilation(self, current_emotions: torch.Tensor) -> torch.Tensor:
        """Calculate time dilation based on emotional intensity"""

        # Calculate emotional intensity
        emotional_intensity = current_emotions.norm(dim=-1, keepdim=True)

        # Compute time dilation factor
        dilation_factor = 1.0 + self.time_dilation_network(emotional_intensity).squeeze()

        # Clamp to reasonable range (0.5x to 2.0x speed)
        dilation_factor = torch.clamp(dilation_factor, 0.5, 2.0)

        return dilation_factor

    def create_emotional_field(self, center_entity: int, radius: int = 3) -> torch.Tensor:
        """Create a local emotional field around an entity"""

        # Calculate distances from center entity
        distances = torch.arange(self.ecosystem_size).unsqueeze(0) - center_entity
        distances = distances.float()

        # Create field strength based on distance
        field_strength = torch.exp(-distances.abs().float() / radius)
        field_strength = field_strength * (distances.abs() <= radius).float()

        # Apply emotional field
        field_effect = torch.matmul(field_strength, self.emotional_states)
        field_effect = field_effect / (field_strength.sum() + 1e-6)  # Normalize by field size

        return field_effect

    def synchronize_emotional_states(self, target_entities: List[int],
                                   sync_strength: float = 0.5) -> None:
        """Synchronize emotional states between specific entities"""

        if len(target_entities) < 2:
            return

        # Get current emotional states for target entities
        target_states = self.emotional_states[target_entities]

        # Calculate mean emotional state
        mean_state = target_states.mean(dim=0, keepdim=True)

        # Apply synchronization
        synchronized_states = sync_strength * mean_state + (1 - sync_strength) * target_states

        # Update the emotional states
        self.emotional_states.data[target_entities] = synchronized_states

    def get_emotional_landscape(self) -> Dict[str, Any]:
        """Get the current emotional landscape of the ecosystem"""

        # Calculate emotional diversity
        emotional_diversity = torch.std(self.emotional_states, dim=0).mean()

        # Find dominant emotions across ecosystem
        ecosystem_mean = self.emotional_states.mean(dim=0)
        dominant_emotion_idx = torch.argmax(ecosystem_mean)

        # Calculate emotional coherence
        emotion_coherence = 1.0 - torch.std(self.emotional_states, dim=0).mean()

        # Identify emotional clusters
        from sklearn.cluster import KMeans
        try:
            kmeans = KMeans(n_clusters=min(5, self.ecosystem_size), n_init=10)
            clusters = kmeans.fit_predict(self.emotional_states.detach().numpy())
            cluster_centers = kmeans.cluster_centers_
        except:
            clusters = None
            cluster_centers = None

        return {
            'emotional_diversity': emotional_diversity.item(),
            'dominant_emotion': dominant_emotion_idx.item(),
            'emotion_coherence': emotion_coherence.item(),
            'ecosystem_mean': ecosystem_mean,
            'clusters': clusters,
            'cluster_centers': cluster_centers
        }

    def inject_emotional_stimulus(self, entity_id: int, stimulus: torch.Tensor,
                                propagation_steps: int = 5) -> List[torch.Tensor]:
        """Inject an emotional stimulus and track its propagation"""

        propagation_history = []

        # Apply initial stimulus
        stimulus_effect = stimulus.unsqueeze(0).expand_as(self.emotional_states[entity_id:entity_id+1])
        self.emotional_states.data[entity_id:entity_id+1] += stimulus_effect

        propagation_history.append(self.emotional_states.clone())

        # Propagate through network
        for step in range(propagation_steps):
            self.propagate_emotions(dt=0.2)
            propagation_history.append(self.emotional_states.clone())

        return propagation_history

    def forward(self, input_tensor: torch.Tensor, cultural_context: Optional[int] = None,
                create_field: bool = False, center_entity: Optional[int] = None) -> Dict[str, Any]:

        # Propagate emotions
        current_emotions = self.propagate_emotions()

        # Apply cultural learning if context provided
        cultural_result = None
        if cultural_context is not None:
            cultural_result = self.cultural_emotional_learning(input_tensor, cultural_context)

        # Calculate time dilation
        time_dilation = self.emotional_time_dilation(current_emotions.mean(dim=0))

        # Create emotional field if requested
        emotional_field = None
        if create_field and center_entity is not None:
            emotional_field = self.create_emotional_field(center_entity)

        # Get emotional landscape
        landscape = self.get_emotional_landscape()

        return {
            'current_emotions': current_emotions,
            'time_dilation': time_dilation,
            'emotional_field': emotional_field,
            'cultural_result': cultural_result,
            'landscape': landscape,
            'ecosystem_size': self.ecosystem_size,
            'num_emotions': self.num_emotions
        }