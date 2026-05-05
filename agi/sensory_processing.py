# sensory_processing.py - Neurochemical-Enhanced Sensory Processing System
# Version 1.0: Multi-modal sensory processing with neurochemical integration
#
# This module provides a comprehensive sensory processing system that integrates
# with the neurochemical emotion simulation to provide biologically-inspired
# perception and attention mechanisms.

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import re

# Try to import neurochemical bridge
try:
    from neurochemical_matlab_bridge import get_simulation_instance, MATLAB_ENGINE_AVAILABLE
    NEUROCHEMICAL_AVAILABLE = True
except ImportError:
    NEUROCHEMICAL_AVAILABLE = False


# =============================================================================
# SENSORY MODALITY DEFINITIONS
# =============================================================================

@dataclass
class SensoryInput:
    """Represents a single sensory input."""
    modality: str  # visual, auditory, textual, emotional, etc.
    content: Any
    intensity: float = 0.5
    valence: float = 0.0  # -1 to 1
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedSensoryData:
    """Represents processed sensory data."""
    raw_input: SensoryInput
    attention_weight: float
    emotional_impact: float
    neurochemical_response: Dict[str, float]
    processed_features: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# SENSORY MODALITY PROCESSORS
# =============================================================================

class TextualProcessor:
    """Processes textual sensory input."""
    
    def __init__(self):
        # Emotional word dictionaries
        self.positive_words = {
            "happy", "joy", "love", "great", "wonderful", "amazing", "good",
            "beautiful", "excellent", "fantastic", "brilliant", "awesome",
            "delightful", "pleasant", "grateful", "thankful", "excited"
        }
        self.negative_words = {
            "sad", "angry", "fear", "hate", "terrible", "awful", "bad",
            "horrible", "disgusting", "painful", "hurt", "suffering",
            "disappointed", "frustrated", "annoyed", "worried", "anxious"
        }
        self.arousal_words = {
            "urgent", "important", "critical", "emergency", "exciting",
            "intense", "powerful", "strong", "extreme", "sudden"
        }
        self.social_words = {
            "friend", "family", "love", "together", "share", "help",
            "support", "care", "trust", "bond", "relationship"
        }
    
    def process(self, text: str) -> Dict[str, Any]:
        """Process textual input and extract features."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Count emotional words
        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)
        arousal_count = len(words & self.arousal_words)
        social_count = len(words & self.social_words)
        
        # Calculate metrics
        valence = (positive_count - negative_count) * 0.15
        arousal = min(1.0, 0.3 + arousal_count * 0.1)
        social = min(1.0, social_count * 0.2)
        
        # Extract key phrases
        questions = len(re.findall(r'\?', text))
        exclamations = len(re.findall(r'!', text))
        urgency = "urgent" in text_lower or "important" in text_lower
        
        return {
            "valence": np.clip(valence, -1, 1),
            "arousal": arousal,
            "social_content": social,
            "word_count": len(words),
            "questions": questions,
            "exclamations": exclamations,
            "urgency": urgency,
            "positive_words": positive_count,
            "negative_words": negative_count,
            "sentiment": "positive" if valence > 0.1 else "negative" if valence < -0.1 else "neutral"
        }


class EmotionalProcessor:
    """Processes emotional content from input."""
    
    def __init__(self):
        self.emotion_patterns = {
            "joy": ["happy", "joy", "excited", "delighted", "thrilled"],
            "sadness": ["sad", "depressed", "melancholy", "grief", "sorrow"],
            "anger": ["angry", "furious", "rage", "irritated", "frustrated"],
            "fear": ["afraid", "scared", "anxious", "worried", "terrified"],
            "surprise": ["surprised", "shocked", "amazed", "astonished"],
            "disgust": ["disgusted", "repulsed", "revolted"],
            "love": ["love", "adore", "cherish", "affection", "devoted"],
            "curiosity": ["curious", "interested", "wonder", "fascinated"]
        }
    
    def process(self, text: str) -> Dict[str, Any]:
        """Detect emotions in text."""
        text_lower = text.lower()
        detected_emotions = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            count = sum(1 for p in patterns if p in text_lower)
            if count > 0:
                detected_emotions[emotion] = min(1.0, count * 0.3)
        
        # Determine primary emotion
        if detected_emotions:
            primary_emotion = max(detected_emotions, key=detected_emotions.get)
        else:
            primary_emotion = "neutral"
        
        return {
            "detected_emotions": detected_emotions,
            "primary_emotion": primary_emotion,
            "emotional_intensity": max(detected_emotions.values()) if detected_emotions else 0.0
        }


class ContextProcessor:
    """Processes contextual information."""
    
    def __init__(self):
        self.context_history = []
        self.max_history = 10
    
    def process(self, current_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process and maintain context."""
        # Add to history
        self.context_history.append({
            "input": current_input[:100],  # Truncate for storage
            "timestamp": time.time(),
            "context": context or {}
        })
        
        # Trim history
        if len(self.context_history) > self.max_history:
            self.context_history.pop(0)
        
        # Analyze patterns
        recent_inputs = [h["input"] for h in self.context_history[-5:]]
        
        return {
            "conversation_length": len(self.context_history),
            "recent_inputs": recent_inputs,
            "context_available": len(self.context_history) > 1
        }


# =============================================================================
# NEUROCHEMICAL SENSORY INTEGRATION
# =============================================================================

class NeurochemicalSensoryIntegration:
    """
    Integrates sensory processing with neurochemical state.
    
    This class provides the bridge between sensory input and the
    neurochemical emotion system, modulating perception based on
    current brain chemistry.
    """
    
    def __init__(self):
        # Neurochemical state
        self.neurochemical_state = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "norepinephrine": 0.4,
            "GABA": 0.6,
            "glutamate": 0.5,
            "acetylcholine": 0.5,
            "oxytocin": 0.3,
            "endorphins": 0.2,
            "CRF": 0.3
        }
        
        # Brain region activity
        self.brain_region_activity = {
            "amygdala": 0.3,
            "nucleus_accumbens": 0.4,
            "prefrontal_cortex": 0.5,
            "hippocampus": 0.4,
            "VTA": 0.4,
            "locus_coeruleus": 0.3,
            "sensory_cortex": 0.5,
            "thalamus": 0.5
        }
        
        # Attention state
        self.attention_state = {
            "focus": 0.5,
            "alertness": 0.5,
            "orienting": 0.5,
            "executive_control": 0.5
        }
        
        # Processors
        self.text_processor = TextualProcessor()
        self.emotion_processor = EmotionalProcessor()
        self.context_processor = ContextProcessor()
        
        # Neurochemical simulation
        self.neurochemical_sim = None
        if NEUROCHEMICAL_AVAILABLE:
            try:
                self.neurochemical_sim = get_simulation_instance()
            except:
                pass
    
    def process_sensory_input(self, sensory_input: SensoryInput) -> ProcessedSensoryData:
        """
        Process a sensory input through the neurochemical system.
        
        Args:
            sensory_input: The sensory input to process
            
        Returns:
            ProcessedSensoryData containing processed information
        """
        # Process based on modality
        if sensory_input.modality == "textual":
            processed_features = self.text_processor.process(sensory_input.content)
            emotional_features = self.emotion_processor.process(sensory_input.content)
            processed_features.update(emotional_features)
        else:
            processed_features = {"intensity": sensory_input.intensity, "valence": sensory_input.valence}
        
        # Calculate attention weight based on neurochemical state
        attention_weight = self._calculate_attention_weight(sensory_input, processed_features)
        
        # Calculate emotional impact
        emotional_impact = self._calculate_emotional_impact(processed_features)
        
        # Generate neurochemical response
        neurochemical_response = self._generate_neurochemical_response(
            sensory_input, processed_features, emotional_impact
        )
        
        # Update neurochemical state
        self._update_neurochemical_state(neurochemical_response)
        
        # Update brain region activity
        self._update_brain_regions(sensory_input, processed_features)
        
        return ProcessedSensoryData(
            raw_input=sensory_input,
            attention_weight=attention_weight,
            emotional_impact=emotional_impact,
            neurochemical_response=neurochemical_response,
            processed_features=processed_features
        )
    
    def _calculate_attention_weight(self, sensory_input: SensoryInput, 
                                    features: Dict[str, Any]) -> float:
        """Calculate attention weight based on neurochemical state."""
        # Base attention from norepinephrine (alertness)
        base_attention = self.neurochemical_state.get("norepinephrine", 0.4)
        
        # Modulate by acetylcholine (focus)
        focus_modulation = self.neurochemical_state.get("acetylcholine", 0.5)
        
        # Increase for high intensity or urgency
        intensity_boost = sensory_input.intensity * 0.2
        urgency_boost = 0.2 if features.get("urgency", False) else 0.0
        
        # Novelty detection (dopamine-based)
        novelty = features.get("novelty", 0.5)
        dopamine_boost = self.neurochemical_state.get("dopamine", 0.5) * novelty * 0.2
        
        attention = base_attention * 0.4 + focus_modulation * 0.3 + \
                   intensity_boost + urgency_boost + dopamine_boost
        
        return np.clip(attention, 0.0, 1.0)
    
    def _calculate_emotional_impact(self, features: Dict[str, Any]) -> float:
        """Calculate emotional impact of processed features."""
        valence = abs(features.get("valence", 0.0))
        arousal = features.get("arousal", 0.3)
        emotional_intensity = features.get("emotional_intensity", 0.0)
        
        # Combine factors
        impact = valence * 0.4 + arousal * 0.3 + emotional_intensity * 0.3
        
        # Modulate by amygdala activity
        amygdala = self.brain_region_activity.get("amygdala", 0.3)
        impact *= (0.5 + amygdala * 0.5)
        
        return np.clip(impact, 0.0, 1.0)
    
    def _generate_neurochemical_response(self, sensory_input: SensoryInput,
                                         features: Dict[str, Any],
                                         emotional_impact: float) -> Dict[str, float]:
        """Generate neurochemical response to sensory input."""
        response = {}
        
        valence = features.get("valence", 0.0)
        arousal = features.get("arousal", 0.3)
        social = features.get("social_content", 0.0)
        
        # Dopamine response (reward/novelty)
        if valence > 0:
            response["dopamine"] = valence * 0.2
        elif features.get("urgency", False):
            response["dopamine"] = 0.1  # Urgent stimuli also trigger dopamine
        
        # Serotonin response (mood)
        if valence > 0:
            response["serotonin"] = valence * 0.1
        elif valence < 0:
            response["serotonin"] = valence * 0.15  # Negative affects serotonin more
        
        # Norepinephrine response (arousal)
        response["norepinephrine"] = arousal * 0.15
        
        # Oxytocin response (social bonding)
        if social > 0:
            response["oxytocin"] = social * 0.2
        
        # CRF response (stress)
        if valence < -0.3 or features.get("urgency", False):
            response["CRF"] = abs(valence) * 0.2 + 0.1
        
        # Acetylcholine response (attention)
        if features.get("questions", 0) > 0 or emotional_impact > 0.5:
            response["acetylcholine"] = 0.1
        
        return response
    
    def _update_neurochemical_state(self, response: Dict[str, float]):
        """Update neurochemical state based on response."""
        for neurochemical, delta in response.items():
            if neurochemical in self.neurochemical_state:
                self.neurochemical_state[neurochemical] = np.clip(
                    self.neurochemical_state[neurochemical] + delta,
                    0.0, 1.0
                )
    
    def _update_brain_regions(self, sensory_input: SensoryInput, features: Dict[str, Any]):
        """Update brain region activity based on sensory input."""
        valence = features.get("valence", 0.0)
        arousal = features.get("arousal", 0.3)
        
        # Amygdala (emotional processing)
        if abs(valence) > 0.2:
            self.brain_region_activity["amygdala"] = np.clip(
                self.brain_region_activity.get("amygdala", 0.3) + abs(valence) * 0.1,
                0.0, 1.0
            )
        
        # Nucleus accumbens (reward)
        if valence > 0:
            self.brain_region_activity["nucleus_accumbens"] = np.clip(
                self.brain_region_activity.get("nucleus_accumbens", 0.4) + valence * 0.1,
                0.0, 1.0
            )
        
        # Locus coeruleus (arousal)
        if arousal > 0.5:
            self.brain_region_activity["locus_coeruleus"] = np.clip(
                self.brain_region_activity.get("locus_coeruleus", 0.3) + (arousal - 0.5) * 0.2,
                0.0, 1.0
            )
        
        # Sensory cortex (general processing)
        self.brain_region_activity["sensory_cortex"] = np.clip(
            self.brain_region_activity.get("sensory_cortex", 0.5) + 0.05,
            0.0, 1.0
        )
    
    def set_neurochemical_state(self, state: Dict[str, float]):
        """Set neurochemical state from external source."""
        self.neurochemical_state.update(state)
    
    def get_attention_state(self) -> Dict[str, float]:
        """Get current attention state."""
        # Calculate attention components from neurochemical state
        norepinephrine = self.neurochemical_state.get("norepinephrine", 0.4)
        acetylcholine = self.neurochemical_state.get("acetylcholine", 0.5)
        dopamine = self.neurochemical_state.get("dopamine", 0.5)
        
        self.attention_state["alertness"] = norepinephrine
        self.attention_state["focus"] = acetylcholine
        self.attention_state["orienting"] = (norepinephrine + acetylcholine) / 2
        self.attention_state["executive_control"] = dopamine * 0.5 + acetylcholine * 0.5
        
        return self.attention_state.copy()
    
    def get_sensory_report(self) -> Dict[str, Any]:
        """Get a comprehensive sensory processing report."""
        return {
            "neurochemical_state": self.neurochemical_state.copy(),
            "brain_region_activity": self.brain_region_activity.copy(),
            "attention_state": self.get_attention_state(),
            "neurochemical_sim_available": self.neurochemical_sim is not None
        }


# =============================================================================
# MAIN SENSORY PROCESSING SYSTEM
# =============================================================================

class SensoryProcessingSystem:
    """
    Main sensory processing system integrating all components.
    
    This class provides a unified interface for processing multi-modal
    sensory input with neurochemical modulation.
    """
    
    def __init__(self):
        self.integration = NeurochemicalSensoryIntegration()
        self.processing_history = []
        self.max_history = 100
    
    def process_input(self, content: str, modality: str = "textual",
                     intensity: float = 0.5, metadata: Dict[str, Any] = None) -> ProcessedSensoryData:
        """
        Process input through the sensory system.
        
        Args:
            content: The input content (typically text)
            modality: The sensory modality
            intensity: Input intensity (0-1)
            metadata: Additional metadata
            
        Returns:
            ProcessedSensoryData with all processing results
        """
        # Create sensory input
        sensory_input = SensoryInput(
            modality=modality,
            content=content,
            intensity=intensity,
            metadata=metadata or {}
        )
        
        # Process through integration
        processed = self.integration.process_sensory_input(sensory_input)
        
        # Store in history
        self.processing_history.append({
            "input": sensory_input,
            "processed": processed,
            "timestamp": time.time()
        })
        
        # Trim history
        if len(self.processing_history) > self.max_history:
            self.processing_history.pop(0)
        
        return processed
    
    def process_with_context(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input with context awareness.
        
        Args:
            content: The input content
            context: Additional context information
            
        Returns:
            Dictionary with all processing results
        """
        # Process context
        context_result = self.integration.context_processor.process(content, context)
        
        # Process main input
        processed = self.process_input(content, metadata={"context": context})
        
        return {
            "processed_data": processed,
            "context": context_result,
            "neurochemical_state": self.integration.neurochemical_state.copy(),
            "attention_state": self.integration.get_attention_state(),
            "brain_region_activity": self.integration.brain_region_activity.copy()
        }
    
    def set_neurochemical_state(self, state: Dict[str, float]):
        """Set neurochemical state from external source."""
        self.integration.set_neurochemical_state(state)
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state."""
        return {
            "sensory_integration": self.integration.get_sensory_report(),
            "history_length": len(self.processing_history),
            "recent_inputs": [
                p["input"].content[:50] for p in self.processing_history[-5:]
            ]
        }


# Global sensory processing system instance
sensory_processing_system = SensoryProcessingSystem()


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def initialize_sensory_processing():
    """Initialize the sensory processing module."""
    print("=" * 60)
    print("[SENSORY PROCESSING] Neurochemical Sensory Processing System v1.0")
    print("=" * 60)
    
    if NEUROCHEMICAL_AVAILABLE:
        print("[SENSORY PROCESSING] Neurochemical integration enabled.")
    else:
        print("[SENSORY PROCESSING] Running in standalone mode.")
    
    print("[SENSORY PROCESSING] Module initialized successfully.")
    print("=" * 60)


# Run initialization
initialize_sensory_processing()
