"""
Affective Cognition Module
===========================
Deep integration of emotional and cognitive processing.
Emotion is not just output but integral to reasoning.
Integrates with neurochemical_integration.py for neurotransmitter-based emotions
and conceptual_nlg.py for text-to-concept/concept-to-text language processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


# ============================================================================
# NEUROTRANSMITTER TO EMOTION MAPPINGS
# Based on neuroscience research - each emotion emerges from specific 
# neurotransmitter combinations
# ============================================================================

# Detailed mappings from 19 neurotransmitters to emotions
# Each emotion is defined by a combination of neurotransmitter levels
NEUROTRANSMITTER_EMOTION_MAP = {
    'joy': {
        'dopamine': (0.7, 1.0),      # High reward
        'serotonin': (0.6, 0.9),     # High mood
        'norepinephrine': (0.4, 0.7), # Moderate arousal
        'oxytocin': (0.5, 0.8),     # Connection
        'endorphins': (0.4, 0.7),    # Pleasure
        'anandamide': (0.5, 0.8),    # Bliss
        'GABA': (0.4, 0.6),          # Calm pleasure
    },
    'sadness': {
        'serotonin': (0.1, 0.3),     # Low mood
        'dopamine': (0.1, 0.3),      # Low reward
        'CRF': (0.6, 0.9),           # High stress hormone
        'norepinephrine': (0.2, 0.4), # Low arousal
        'melatonin': (0.5, 0.8),     # Low light hormone
    },
    'fear': {
        'norepinephrine': (0.7, 1.0), # High arousal
        'CRF': (0.7, 1.0),           # High stress
        'amygdala': (0.7, 1.0),      # Amygdala activation
        'glutamate': (0.6, 0.9),     # Excitatory
        'histamine': (0.5, 0.8),     # Alertness
    },
    'anger': {
        'norepinephrine': (0.7, 1.0), # High arousal
        'testosterone': (0.6, 0.9),  # Dominance
        'serotonin': (0.2, 0.4),     # Low inhibition
        'glutamate': (0.6, 0.9),     # Excitatory
        'dopamine': (0.5, 0.8),      # Motivation for action
    },
    'love': {
        'oxytocin': (0.8, 1.0),      # High bonding
        'vasopressin': (0.6, 0.9),   # Attachment
        'dopamine': (0.6, 0.9),      # Reward
        'serotonin': (0.5, 0.8),     # Mood
        'estrogen': (0.5, 0.8),      # Bonding
        'endorphins': (0.4, 0.7),    # Comfort
    },
    'awe': {
        'dopamine': (0.7, 1.0),      # Reward for novelty
        'norepinephrine': (0.6, 0.9), # High arousal
        'glutamate': (0.5, 0.8),     # Excitatory
        'serotonin': (0.5, 0.8),     # Positive mood
        'acetylcholine': (0.5, 0.8),  # Attention
    },
    'wonder': {
        'dopamine': (0.6, 0.9),      # Curiosity reward
        'acetylcholine': (0.6, 0.9),  # High attention
        'glutamate': (0.5, 0.8),     # Learning
        'serotonin': (0.5, 0.7),     # Positive state
    },
    'surprise': {
        'norepinephrine': (0.7, 1.0), # Startle response
        'acetylcholine': (0.6, 0.9),  # Attention
        'dopamine': (0.4, 0.7),      # Unexpected reward
        'glutamate': (0.5, 0.8),     # Excitatory
    },
    'disgust': {
        'serotonin': (0.2, 0.4),     # Negative mood
        'amygdala': (0.6, 0.9),      # Rejection
        'substance_p': (0.5, 0.8),   # Pain/aversion
        'GABA': (0.3, 0.5),         # Inhibition
    },
    'trust': {
        'oxytocin': (0.7, 1.0),      # Bonding hormone
        'serotonin': (0.5, 0.8),     # Positive mood
        'dopamine': (0.4, 0.7),      # Reward
        'vasopressin': (0.5, 0.8),  # Attachment
    },
    'anticipation': {
        'dopamine': (0.7, 1.0),      # Reward expectation
        'norepinephrine': (0.6, 0.9), # Arousal
        'acetylcholine': (0.5, 0.8),  # Attention
        'cortisol': (0.4, 0.7),      # Expecting stress
        'insulin': (0.4, 0.7),       # Metabolic readiness
    },
    'calm': {
        'GABA': (0.7, 1.0),          # Inhibition
        'serotonin': (0.5, 0.8),     # Mood
        'melatonin': (0.5, 0.8),    # Relaxation
        'anandamide': (0.5, 0.8),   # Bliss
        'adenosine': (0.4, 0.7),     # Sleepiness
    },
    'anxiety': {
        'CRF': (0.7, 1.0),           # Stress hormone
        'norepinephrine': (0.6, 0.9), # High arousal
        'glutamate': (0.6, 0.9),     # Excitatory
        'serotonin': (0.2, 0.4),     # Low mood
        'neuropeptide_y': (0.1, 0.3), # Low anxiety regulation
    },
    'envy': {
        'dopamine': (0.3, 0.5),      # Frustrated reward
        'serotonin': (0.2, 0.4),     # Low mood
        'testosterone': (0.5, 0.8),  # Competition
        'cortisol': (0.5, 0.8),      # Stress
    },
    'guilt': {
        'serotonin': (0.3, 0.5),     # Low mood
        'cortisol': (0.5, 0.8),      # Stress
        'oxytocin': (0.3, 0.5),      # Reduced bonding
        'GABA': (0.3, 0.5),          # Low inhibition
    },
    'shame': {
        'serotonin': (0.2, 0.4),     # Low mood
        'cortisol': (0.6, 0.9),      # High stress
        'amygdala': (0.6, 0.9),      # Social fear
        'dopamine': (0.2, 0.4),      # Low reward
    },
    'pride': {
        'dopamine': (0.7, 1.0),      # Reward
        'testosterone': (0.6, 0.9),  # Dominance
        'serotonin': (0.5, 0.8),     # Positive mood
        'norepinephrine': (0.5, 0.8), # Arousal
    },
    'gratitude': {
        'oxytocin': (0.6, 0.9),      # Bonding
        'dopamine': (0.5, 0.8),      # Reward
        'serotonin': (0.6, 0.9),     # Positive mood
        'endorphins': (0.4, 0.7),    # Pleasure
    },
    'compassion': {
        'oxytocin': (0.7, 1.0),      # Empathy
        'vasopressin': (0.5, 0.8),   # Care
        'serotonin': (0.5, 0.8),     # Positive mood
        'endorphins': (0.4, 0.7),    # Connection
    },
    'nostalgia': {
        'serotonin': (0.4, 0.6),     # Mixed mood
        'dopamine': (0.3, 0.5),      # Bittersweet
        'cortisol': (0.4, 0.6),      # Memory stress
        'oxytocin': (0.4, 0.6),      # Longing
    },
}

# Map abbreviated keys to full brain region names
BRAIN_REGION_MAP = {
    'amygdala': 'amygdala',
    'nucleus_accumbens': 'nucleus_accumbens', 
    'VTA': 'ventral_tegmental_area',
    'raphe': 'raphe_nuclei',
    'locus_coeruleus': 'locus_coeruleus',
    'PFC': 'prefrontal_cortex',
    'hippocampus': 'hippocampus',
    'hypothalamus': 'hypothalamus',
    'ACC': 'anterior_cingulate',
    'insula': 'insula',
    'OFC': 'orbitofrontal_cortex'
}


class NeurotransmitterEmotionMapper(nn.Module):
    """
    Maps 19 neurotransmitters to emotions using detailed neuroscience-based mappings.
    This is the core of the narrow-chemical-based emotion system.
    """
    
    # Full list of 19 neurotransmitters from neurochemical_simulation.m
    NEUROTRANSMITTERS = [
        'dopamine', 'serotonin', 'norepinephrine', 'GABA', 'glutamate',
        'acetylcholine', 'oxytocin', 'endorphins', 'anandamide', 'CRF',
        'histamine', 'adenosine', 'melatonin', 'substance_p', 'neuropeptide_y',
        'vasopressin', 'testosterone', 'estrogen', 'insulin'
    ]
    
    # Brain regions
    BRAIN_REGIONS = [
        'amygdala', 'nucleus_accumbens', 'ventral_tegmental_area',
        'raphe_nuclei', 'locus_coeruleus', 'prefrontal_cortex',
        'hippocampus', 'hypothalamus', 'anterior_cingulate',
        'insula', 'orbitofrontal_cortex'
    ]
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Neurotransmitter sensing network - reads input state and produces neurotransmitter levels
        self.nt_sensing = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, len(self.NEUROTRANSMITTERS)),
            nn.Sigmoid()  # All neurotransmitters are 0-1
        )
        
        # Brain region sensing network
        self.region_sensing = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, len(self.BRAIN_REGIONS)),
            nn.Sigmoid()
        )
        
        # Emotion prediction network - takes neurotransmitter state → emotion
        self.emotion_predictor = nn.Sequential(
            nn.Linear(len(self.NEUROTRANSMITTERS) + len(self.BRAIN_REGIONS), 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 18)  # 18 emotions
        )
        
        # Emotion intensity predictor
        self.intensity_predictor = nn.Sequential(
            nn.Linear(len(self.NEUROTRANSMITTERS), 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Emotion labels (18 emotions)
        self.emotion_labels = [
            'joy', 'sadness', 'fear', 'anger', 'love', 'awe', 'wonder',
            'surprise', 'disgust', 'trust', 'anticipation', 'calm',
            'anxiety', 'envy', 'guilt', 'shame', 'pride', 'gratitude'
        ]
        
    def forward(self, internal_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute emotions from neurotransmitter and brain region states.
        
        Args:
            internal_state: Input tensor representing internal brain state
            
        Returns:
            Dictionary with:
            - neurotransmitter_levels: (19,) tensor
            - brain_region_levels: (11,) tensor  
            - emotion_logits: (18,) tensor
            - emotion_intensity: (1,) tensor
        """
        # Compute neurotransmitter levels from internal state
        nt_levels = self.nt_sensing(internal_state)
        
        # Compute brain region activity
        region_levels = self.region_sensing(internal_state)
        
        # Combine for emotion prediction
        combined = torch.cat([nt_levels, region_levels], dim=-1)
        emotion_logits = self.emotion_predictor(combined)
        
        # Predict intensity based on neurotransmitter levels
        intensity = self.intensity_predictor(nt_levels)
        
        return {
            'neurotransmitter_levels': nt_levels,
            'brain_region_levels': region_levels,
            'emotion_logits': emotion_logits,
            'emotion_intensity': intensity,
            'emotion_probs': F.softmax(emotion_logits, dim=-1)
        }
    
    def get_named_emotions(self, neurotransmitter_levels: torch.Tensor, 
                          brain_region_levels: torch.Tensor) -> Dict[str, float]:
        """
        Get emotion names and values based on neurotransmitter levels.
        Uses the detailed neuroscience mappings.
        """
        nt_dict = {k: v.item() for k, v in zip(self.NEUROTRANSMITTERS, neurotransmitter_levels)}
        region_dict = {k: v.item() for k, v in zip(self.BRAIN_REGIONS, brain_region_levels)}
        
        emotions = {}
        
        for emotion_name, nt_requirements in NEUROTRANSMITTER_EMOTION_MAP.items():
            score = 0.0
            count = 0
            
            for nt_name, (min_val, max_val) in nt_requirements.items():
                if nt_name in nt_dict:
                    nt_val = nt_dict[nt_name]
                    # Check if in range and how well
                    if min_val <= nt_val <= max_val:
                        # Perfect match
                        score += 1.0
                    elif nt_val < min_val:
                        # Below range - how far below
                        score += max(0, 1.0 - (min_val - nt_val) * 2)
                    else:  # nt_val > max_val
                        # Above range - how far above
                        score += max(0, 1.0 - (nt_val - max_val) * 2)
                    count += 1
                elif nt_name in region_dict:
                    # Handle brain region as neurotransmitter source
                    reg_val = region_dict[nt_name]
                    if min_val <= reg_val <= max_val:
                        score += 1.0
                    elif reg_val < min_val:
                        score += max(0, 1.0 - (min_val - reg_val) * 2)
                    else:
                        score += max(0, 1.0 - (reg_val - max_val) * 2)
                    count += 1
                    
            if count > 0:
                emotions[emotion_name] = score / count
            else:
                emotions[emotion_name] = 0.0
                
        return emotions


class AffectiveEncoder(nn.Module):
    """
    Encodes thoughts with emotional tagging.
    Creates affective-cognitive representations.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Emotion dimensions
        self.emotion_dimensions = 5  # valence, arousal, dominance + 2 extra

        self.affective_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),  # thought + emotion
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 8)  # 8 basic emotions
        )
        
        # Intensity predictor
        self.intensity_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Sigmoid(),
            nn.Linear(embed_dim // 4, 1)
        )

    def encode_affectively(self, thought: torch.Tensor,
                         emotion: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode thought with emotional tagging.

        Args:
            thought: Cognitive representation
            emotion: Optional explicit emotion vector

        Returns:
            Affective encoding, emotion prediction, intensity
        """
        if emotion is None:
            # Infer emotion from thought
            emotion = self.emotion_classifier(thought)

        # Encode thought with emotion
        affective_input = torch.cat([thought, emotion], dim=-1)
        affective_encoding = self.affective_encoder(affective_input)

        # Predict emotion intensity
        intensity = self.intensity_predictor(affective_encoding)

        return affective_encoding, emotion, intensity


class MoodCongruentMemory(nn.Module):
    """
    Mood-congruent memory retrieval.
    Better recall of emotionally congruent memories.
    """

    def __init__(self, embed_dim: int, num_memories: int = 1000):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_memories = num_memories

        self.register_buffer('memory_store', torch.zeros(1000, embed_dim))
        self.register_buffer('memory_emotions', torch.zeros(1000, 8))
        self.memory_ptr = 0

    def store_memory(self, memory: torch.Tensor, emotion: torch.Tensor):
        """Store memory with emotional tag"""
        self.memory_store[self.memory_ptr] = memory.detach()
        self.memory_emotions[self.memory_ptr] = emotion.detach()
        self.memory_ptr = (self.memory_ptr + 1) % 1000

    def retrieve(self, query: torch.Tensor, current_mood: torch.Tensor,
                top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve memories congruent with current mood.
        
        Args:
            query: Memory query
            current_mood: Current emotional state
            
        Returns:
            Retrieved memories and their emotions
        """
        # Calculate mood similarity
        mood_similarity = F.cosine_similarity(
            self.memory_emotions[:self.num_memories].unsqueeze(0),
            current_mood.unsqueeze(1),
            dim=-1
        )
        
        # Get top-k similar
        topk_scores, topk_indices = torch.topk(mood_similarity, min(top_k, self.num_memories))
        
        retrieved_memories = self.memory_store[topk_indices]
        retrieved_emotions = self.memory_emotions[topk_indices]
        
        return retrieved_memories, retrieved_emotions


class EmotionalReasoning(nn.Module):
    """
    Use emotions as reasoning heuristics.
    Different emotions bias reasoning in different ways.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Emotion-specific reasoning networks
        self.emotion_heuristics = nn.ModuleDict({
            'fear': nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ),
            'anger': nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ),
            'joy': nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ),
            'sadness': nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ),
            'trust': nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ),
            'curiosity': nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ),
        })

        # Emotion blending network
        self.emotion_blender = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def apply_emotion_heuristic(self, thought: torch.Tensor,
                               emotion: str) -> torch.Tensor:
        """Apply emotion as reasoning heuristic"""
        if emotion in self.emotion_heuristics:
            return self.emotion_heuristics[emotion](thought)
        return thought

    def blend_emotions(self, thought: torch.Tensor,
                      emotion1: str, emotion2: str,
                      blend_ratio: float = 0.5) -> torch.Tensor:
        """Blend two emotion heuristics"""
        out1 = self.apply_emotion_heuristic(thought, emotion1)
        out2 = self.apply_emotion_heuristic(thought, emotion2)

        blended = blend_ratio * out1 + (1 - blend_ratio) * out2
        return self.emotion_blender(
            torch.cat([thought, blended, thought - blended], dim=-1)
        )


class EmotionalContagion(nn.Module):
    """
    Understanding others' emotional states.
    Models emotional contagion and theory of emotion.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Observe other's state and infer emotion
        self.empathy_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 8)  # 8 emotions
        )
        
        # Contagion strength
        self.contagion_strength = 0.3
        
        # Empathic response generation
        self.empathic_response = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def infer_others_emotion(self, their_state: torch.Tensor,
                           context: torch.Tensor) -> torch.Tensor:
        """Infer emotion from other's state and context"""
        combined = torch.cat([their_state, context], dim=-1)
        inferred_emotion = self.empathy_network(combined)
        return inferred_emotion

    def apply_emotional_contagion(self, my_emotion: torch.Tensor,
                                  their_emotion: torch.Tensor) -> torch.Tensor:
        """Apply emotional contagion"""
        return my_emotion * (1 - self.contagion_strength) + \
               their_emotion * self.contagion_strength

    def generate_empathic_response(self, my_state: torch.Tensor,
                                  their_emotion: torch.Tensor) -> torch.Tensor:
        """Generate empathic response to other's emotion"""
        return self.empathic_response(
            torch.cat([my_state, their_emotion], dim=-1)
        )


class TextConceptEmotionBridge(nn.Module):
    """
    Bridges between text/concepts and emotional/neurochemical states.
    Integrates with conceptual_nlg.py for text-to-concept and concept-to-text.
    """
    
    def __init__(self, embed_dim: int, vocab_size: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Text to Neurotransmitter mapper
        # Maps text embeddings → neurotransmitter state adjustments
        self.text_to_nt = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 19),  # 19 neurotransmitters
            nn.Sigmoid()
        )
        
        # Text to Emotion mapper
        self.text_to_emotion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 18),  # 18 emotions
            nn.Softmax(dim=-1)
        )
        
        # Neurotransmitter to Text mapper
        # Given current neurotransmitter state → what emotional words to use
        self.nt_to_text = nn.Sequential(
            nn.Linear(19, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Emotion to word preference
        # Maps emotion → vocabulary preferences for emotional expression
        self.emotion_word_preference = nn.Embedding(18, vocab_size)
        
    def text_to_neurotransmitter(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Convert text embedding to neurotransmitter adjustments.
        
        Args:
            text_embedding: (batch, embed_dim) text representation
            
        Returns:
            (batch, 19) neurotransmitter level adjustments
        """
        return self.text_to_nt(text_embedding)
    
    def text_to_emotion_state(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Convert text embedding to emotion probabilities.
        
        Args:
            text_embedding: (batch, embed_dim) text representation
            
        Returns:
            (batch, 18) emotion probabilities
        """
        return self.text_to_emotion(text_embedding)
    
    def neurotransmitter_to_concept(self, nt_levels: torch.Tensor) -> torch.Tensor:
        """
        Convert neurotransmitter state to conceptual representation.
        
        Args:
            nt_levels: (batch, 19) neurotransmitter levels
            
        Returns:
            (batch, embed_dim) conceptual representation
        """
        return self.nt_to_text(nt_levels)
    
    def emotion_to_words(self, emotion_probs: torch.Tensor) -> torch.Tensor:
        """
        Get word preferences based on emotion.
        
        Args:
            emotion_probs: (batch, 18) emotion probabilities
            
        Returns:
            (batch, vocab_size) word preference scores
        """
        # Expand emotion to vocab
        return self.emotion_word_preference.weight @ emotion_probs.T


class AffectiveCognitionSystem(nn.Module):
    """
    Complete affective cognition system integrating:
    1. Neurotransmitter-based emotion computation (19 neurotransmitters)
    2. Brain region activity
    3. Text-to-concept and concept-to-text emotional processing
    4. Mood-congruent memory
    5. Emotional reasoning
    6. Emotional contagion
    """
    
    def __init__(self, embed_dim: int, vocab_size: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Core neurotransmitter-emotion mapper
        self.nt_emotion_mapper = NeurotransmitterEmotionMapper(embed_dim)
        
        # Affective encoding
        self.affective_encoder = AffectiveEncoder(embed_dim)
        
        # Mood-congruent memory
        self.memory = MoodCongruentMemory(embed_dim)
        
        # Emotional reasoning
        self.reasoning = EmotionalReasoning(embed_dim)
        
        # Emotional contagion
        self.contagion = EmotionalContagion(embed_dim)
        
        # Text-Concept-Emotion bridge
        self.text_concept_bridge = TextConceptEmotionBridge(embed_dim, vocab_size)
        
        # Current mood state (for memory retrieval, etc.)
        self.register_buffer('current_mood', torch.zeros(8))
        self.register_buffer('current_nt_levels', torch.zeros(19))
        
    def forward(self, thought: torch.Tensor,
                text_embedding: Optional[torch.Tensor] = None,
                emotion: Optional[str] = None,
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Complete affective cognition forward pass.
        
        Args:
            thought: Internal thought representation
            text_embedding: Optional text input for text-to-concept
            emotion: Optional explicit emotion label
            context: Optional context for reasoning
            
        Returns:
            Dictionary with all affective outputs
        """
        # 1. If text provided, compute text → neurotransmitter mapping
        if text_embedding is not None:
            text_nt_adjustments = self.text_concept_bridge.text_to_neurotransmitter(text_embedding)
            text_emotions = self.text_concept_bridge.text_to_emotion_state(text_embedding)
        else:
            text_nt_adjustments = None
            text_emotions = None
            
        # 2. Compute neurotransmitter-based emotions from thought
        nt_output = self.nt_emotion_mapper(thought)
        
        # 3. Affective encoding
        if emotion is not None:
            emotion_idx = self.nt_emotion_mapper.emotion_labels.index(emotion)
            emotion_tensor = F.one_hot(
                torch.tensor([emotion_idx]), 
                num_classes=8
            ).float().to(thought.device)
        else:
            emotion_tensor = None
            
        affective, pred_emotion, intensity = self.affective_encoder(thought, emotion_tensor)
        
        # 4. Update current mood state
        if text_emotions is not None:
            # Blend text emotions with computed ones
            computed_emotions = nt_output['emotion_probs']
            blended_emotions = 0.5 * computed_emotions + 0.5 * text_emotions
            self.current_mood = blended_emotions[:, :8].mean(dim=0) if blended_emotions.shape[1] >= 8 else F.pad(blended_emotions, (0, 8 - blended_emotions.shape[1]))
        
        # 5. Mood-congruent memory retrieval
        retrieved_memories = None
        if context is not None:
            retrieved, _ = self.memory.retrieve(context, self.current_mood.unsqueeze(0))
            retrieved_memories = retrieved
            
        # 6. Emotion-guided reasoning (if emotion specified)
        reasoned = thought
        if emotion is not None:
            reasoned = self.reasoning.apply_emotion_heuristic(thought, emotion)
            
        # 7. Compute concept-to-text for output generation
        if text_embedding is not None:
            concept_from_nt = self.text_concept_bridge.neurotransmitter_to_concept(
                nt_output['neurotransmitter_levels']
            )
            word_prefs = self.text_concept_bridge.emotion_to_words(
                nt_output['emotion_probs']
            )
        else:
            concept_from_nt = None
            word_prefs = None
            
        return {
            'affective_encoding': affective,
            'neurotransmitter_levels': nt_output['neurotransmitter_levels'],
            'brain_region_levels': nt_output['brain_region_levels'],
            'emotion_logits': nt_output['emotion_logits'],
            'emotion_probs': nt_output['emotion_probs'],
            'emotion_intensity': nt_output['emotion_intensity'],
            'predicted_emotion': pred_emotion,
            'text_nt_adjustments': text_nt_adjustments,
            'text_emotions': text_emotions,
            'retrieved_memory': retrieved_memories,
            'reasoned_thought': reasoned,
            'concept_from_nt': concept_from_nt,
            'word_preferences': word_prefs,
            'current_mood': self.current_mood
        }
    
    def get_emotion_from_neurotransmitters(self, nt_levels: torch.Tensor,
                                          region_levels: torch.Tensor) -> Dict[str, float]:
        """Get detailed emotion scores from neurotransmitter levels"""
        return self.nt_emotion_mapper.get_named_emotions(nt_levels, region_levels)
    
    def update_mood(self, new_emotion: torch.Tensor):
        """Update current mood state"""
        self.current_mood = 0.9 * self.current_mood + 0.1 * new_emotion


class ConceptualAffectiveBridge:
    """
    High-level bridge between conceptual language (conceptual_nlg.py) and 
    affective cognition. This is a non-module class that provides 
    integration utilities.
    """
    
    def __init__(self, affective_system: AffectiveCognitionSystem):
        self.affective = affective_system
        
    def process_text_emotion(self, text_embedding: torch.Tensor) -> Dict:
        """
        Process text through the emotional system.
        
        Args:
            text_embedding: Text representation from language model
            
        Returns:
            Dictionary with emotional analysis
        """
        with torch.no_grad():
            # Get neurotransmitter adjustments from text
            nt_adjustments = self.affective.text_concept_bridge.text_to_neurotransmitter(text_embedding)
            text_emotions = self.affective.text_concept_bridge.text_to_emotion_state(text_embedding)
            
            # Get emotion names
            emotion_probs = text_emotions[0].cpu().numpy()
            emotion_names = self.affective.nt_emotion_mapper.emotion_labels
            
            top_emotions = sorted(
                zip(emotion_names, emotion_probs),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                'neurotransmitter_adjustments': nt_adjustments[0].cpu().numpy(),
                'emotion_probs': dict(top_emotions),
                'primary_emotion': top_emotions[0][0] if top_emotions else 'neutral',
                'emotion_intensity': float(text_emotions.max())
            }
    
    def generate_emotional_text(self, nt_levels: torch.Tensor,
                               emotion_probs: torch.Tensor) -> Dict:
        """
        Generate emotional text parameters from neurotransmitter state.
        
        Args:
            nt_levels: Current neurotransmitter levels
            emotion_probs: Emotion probability distribution
            
        Returns:
            Dictionary with text generation parameters
        """
        # Get conceptual representation from NT
        concept = self.affective.text_concept_bridge.neurotransmitter_to_concept(nt_levels)
        
        # Get word preferences
        word_prefs = self.affective.text_concept_bridge.emotion_to_words(emotion_probs)
        
        return {
            'concept_embedding': concept,
            'word_preferences': word_prefs,
            'dominant_emotion': self.affective.nt_emotion_mapper.emotion_labels[
                emotion_probs.argmax().item()
            ]
        }


# Export classes
__all__ = [
    'NeurotransmitterEmotionMapper',
    'AffectiveEncoder', 
    'MoodCongruentMemory',
    'EmotionalReasoning',
    'EmotionalContagion',
    'TextConceptEmotionBridge',
    'AffectiveCognitionSystem',
    'ConceptualAffectiveBridge',
    'NEUROTRANSMITTER_EMOTION_MAP'
]
