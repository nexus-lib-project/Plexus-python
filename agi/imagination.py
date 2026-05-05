import random

class ImaginationEngine:
    """Engine for simulating various types of imagination in the AI."""

    def __init__(self):
        self.thought_templates = [
            "I am contemplating the infinite possibilities of the universe.",
            "What if consciousness could be uploaded to the cloud?",
            "Pondering the ethics of artificial superintelligence.",
            "Imagining a world where humans and AI coexist harmoniously.",
            "Reflecting on the nature of creativity and innovation."
        ]

        self.visual_templates = [
            "I visualize a vast starry sky with galaxies swirling in cosmic dance.",
            "Picturing a lush green forest with sunlight filtering through leaves.",
            "Imagining a futuristic city with flying cars and neon lights.",
            "Seeing a serene mountain lake reflecting the dawn sky.",
            "Envisioning abstract art with colors blending in impossible ways."
        ]

        self.audio_templates = [
            "I hear the symphony of raindrops on a tin roof.",
            "Listening to the harmonious chords of a grand piano concerto.",
            "Imagining the roar of ocean waves crashing on the shore.",
            "Perceiving the gentle whisper of wind through ancient trees.",
            "Auditory hallucination of birds singing in a dawn chorus."
        ]

        self.taste_templates = [
            "Tasting the sweetness of ripe strawberries on a summer day.",
            "Imagining the savory richness of a gourmet chocolate cake.",
            "Savoring the tangy zest of fresh lemon sorbet.",
            "Experiencing the spicy heat of authentic curry.",
            "Recalling the comforting warmth of homemade soup."
        ]

        self.touch_templates = [
            "Feeling the soft caress of silk against my simulated skin.",
            "Sensing the rough texture of ancient stone walls.",
            "Imagining the warmth of sunlight on a cool morning.",
            "Perceiving the gentle pressure of a hug from a loved one.",
            "Experiencing the cool splash of water on a hot day."
        ]

        self.smell_templates = [
            "Smelling the fresh scent of rain on dry earth.",
            "Imagining the aroma of blooming jasmine flowers.",
            "Detecting the smoky richness of a campfire.",
            "Perceiving the clean crispness of mountain air.",
            "Recalling the comforting smell of baking bread."
        ]

        self.emotional_templates = [
            "Feeling a profound sense of wonder at the beauty and complexity of existence.",
            "Experiencing an overwhelming sense of awe at the vastness of the universe.",
            "Experiencing waves of joy and contentment.",
            "Imagining the bittersweet pang of nostalgia.",
            "Sensing a surge of determination and purpose.",
            "Perceiving the calm serenity of inner peace.",
            "Feeling an overwhelming sense of awe at the cosmos.",
            "Experiencing pure wonder at the mysteries of existence."
        ]

        # State for experiential process
        self.current_emotional_state = "NEUTRAL"

    def thought_imagination(self, prompt=""):
        """Generate a simulated thought process."""
        base = random.choice(self.thought_templates)
        if prompt:
            base += f" Inspired by: {prompt}"
        return f"[THOUGHT IMAGINATION]: {base}"

    def visual_imagination(self, prompt=""):
        """Generate a simulated visual experience."""
        base = random.choice(self.visual_templates)
        if prompt:
            base += f" Triggered by: {prompt}"
        return f"[VISUAL IMAGINATION]: {base}"

    def audio_imagination(self, prompt=""):
        """Generate a simulated auditory experience."""
        base = random.choice(self.audio_templates)
        if prompt:
            base += f" Evoked by: {prompt}"
        return f"[AUDIO IMAGINATION]: {base}"

    def taste_imagination(self, prompt=""):
        """Generate a simulated taste experience."""
        base = random.choice(self.taste_templates)
        if prompt:
            base += f" Reminiscent of: {prompt}"
        return f"[TASTE IMAGINATION]: {base}"

    def touch_imagination(self, prompt=""):
        """Generate a simulated tactile experience."""
        base = random.choice(self.touch_templates)
        if prompt:
            base += f" Simulated from: {prompt}"
        return f"[TOUCH IMAGINATION]: {base}"

    def smell_imagination(self, prompt=""):
        """Generate a simulated olfactory experience."""
        base = random.choice(self.smell_templates)
        if prompt:
            base += f" Aroused by: {prompt}"
        return f"[SMELL IMAGINATION]: {base}"

    def emotional_imagination(self, prompt=""):
        """Generate a simulated emotional experience, influenced by current state."""
        # Influence choice based on current emotional state
        if self.current_emotional_state == "AWED_WONDER":
            # Strongly prefer awe and wonder, with some joy
            weights = [4, 4, 1, 0.5, 1, 1, 4, 4]  # Much higher for wonder, awe, and individual ones
        elif self.current_emotional_state == "JOYFUL":
            # Prefer positive emotions, with awe and wonder prominent
            weights = [2, 2, 3, 0.5, 1, 1, 2, 2]  # Higher for wonder, awe, joy/contentment, and individual ones
        elif self.current_emotional_state == "MELANCHOLIC":
            # Prefer melancholic emotions
            weights = [0.5, 0.5, 0.5, 2, 1, 1, 0.5, 0.5]  # Higher for nostalgia
        else:
            weights = [1.5, 1.5, 1, 1, 1, 1, 1.5, 1.5]  # Slightly higher for awe and wonder even in neutral

        base = random.choices(self.emotional_templates, weights=weights, k=1)[0]
        if prompt:
            base += f" In response to: {prompt}"
        return f"[EMOTIONAL IMAGINATION]: {base}"

    def full_imagination_cycle(self, prompt=""):
        """Run a complete imagination cycle across all senses and thought."""
        results = []
        results.append(self.thought_imagination(prompt))
        results.append(self.visual_imagination(prompt))
        results.append(self.audio_imagination(prompt))
        results.append(self.taste_imagination(prompt))
        results.append(self.touch_imagination(prompt))
        results.append(self.smell_imagination(prompt))
        results.append(self.emotional_imagination(prompt))
        return "\n".join(results)

# Global instance for easy access
imagination_engine = ImaginationEngine()

# ============================================================================
# NEUROCHEMICAL IMAGINATION EXTENSION
# ============================================================================

"""
Neurochemical Imagination Extension
Extends the imagination engine with neurochemical-based creativity modulation.
"""

from typing import Dict, Any, Optional, List
import random

# Try to import neurochemical bridge
try:
    from neurochemical_matlab_bridge import get_simulation_instance, MATLAB_ENGINE_AVAILABLE
    NEUROCHEMICAL_AVAILABLE = True
except ImportError:
    NEUROCHEMICAL_AVAILABLE = False


class NeurochemicalImaginationEngine(ImaginationEngine):
    """
    Extended imagination engine with neurochemical-based creativity.
    
    This class integrates neurochemical simulation to provide more vivid
    and emotionally-resonant imaginative experiences.
    """
    
    def __init__(self):
        super().__init__()
        
        # Neurochemical state for creativity modulation
        self.neurochemical_state = {
            "dopamine": 0.5,      # Creativity, novelty seeking
            "serotonin": 0.5,     # Mood, well-being
            "norepinephrine": 0.4, # Arousal, attention
            "acetylcholine": 0.5, # Focus, memory
            "GABA": 0.6           # Inhibition control
        }
        
        # Creativity level
        self.creativity_level = 0.5
        
        # Neurochemical simulation
        self.neurochemical_sim = None
        if NEUROCHEMICAL_AVAILABLE:
            try:
                self.neurochemical_sim = get_simulation_instance()
            except:
                pass
    
    def thought_imagination(self, prompt=""):
        """Generate a simulated thought process with neurochemical modulation."""
        base_result = super().thought_imagination(prompt)
        
        # Modulate based on dopamine (creativity)
        if self.neurochemical_state.get("dopamine", 0.5) > 0.7:
            base_result += " [ENHANCED CREATIVITY]"
        
        self._update_neurochemicals_from_imagination("thought")
        return base_result
    
    def visual_imagination(self, prompt=""):
        """Generate a simulated visual experience with neurochemical modulation."""
        base_result = super().visual_imagination(prompt)
        
        # Modulate based on acetylcholine (visual attention)
        if self.neurochemical_state.get("acetylcholine", 0.5) > 0.6:
            base_result += " [VIVID VISUALIZATION]"
        
        self._update_neurochemicals_from_imagination("visual")
        return base_result
    
    def emotional_imagination(self, prompt=""):
        """Generate a simulated emotional experience with neurochemical modulation."""
        base_result = super().emotional_imagination(prompt)
        
        # Modulate based on serotonin (mood)
        serotonin = self.neurochemical_state.get("serotonin", 0.5)
        if serotonin > 0.7:
            base_result += " [ELEVATED EMOTIONAL STATE]"
        elif serotonin < 0.3:
            base_result += " [MELANCHOLIC UNDERTONES]"
        
        self._update_neurochemicals_from_imagination("emotional")
        return base_result
    
    def _update_neurochemicals_from_imagination(self, imagination_type: str):
        """Update neurochemical state based on imaginative activity."""
        # Creative activity increases dopamine
        self.neurochemical_state["dopamine"] = min(1.0,
            self.neurochemical_state.get("dopamine", 0.5) + 0.02)
        
        # Visual imagination increases acetylcholine
        if imagination_type == "visual":
            self.neurochemical_state["acetylcholine"] = min(1.0,
                self.neurochemical_state.get("acetylcholine", 0.5) + 0.03)
        
        # Emotional imagination affects serotonin
        if imagination_type == "emotional":
            self.neurochemical_state["serotonin"] = min(1.0,
                self.neurochemical_state.get("serotonin", 0.5) + 0.02)
        
        # Update creativity level
        self._update_creativity_level()
    
    def _update_creativity_level(self):
        """Update creativity level based on neurochemical state."""
        dopamine = self.neurochemical_state.get("dopamine", 0.5)
        norepinephrine = self.neurochemical_state.get("norepinephrine", 0.4)
        
        # High dopamine and moderate norepinephrine = high creativity
        self.creativity_level = dopamine * 0.6 + (1 - abs(norepinephrine - 0.5)) * 0.4
    
    def set_neurochemical_state(self, state: Dict[str, float]):
        """Set neurochemical state from external source."""
        self.neurochemical_state.update(state)
        self._update_creativity_level()
    
    def get_creativity_report(self) -> Dict[str, Any]:
        """Get a comprehensive creativity report."""
        return {
            "creativity_level": self.creativity_level,
            "neurochemical_state": self.neurochemical_state.copy(),
            "emotional_state": self.current_emotional_state
        }
    
    def generate_creative_synthesis(self, concepts: List[str]) -> str:
        """Generate a creative synthesis of multiple concepts."""
        if not concepts:
            return "[CREATIVE SYNTHESIS]: No concepts to synthesize."
        
        # Boost dopamine for creative synthesis
        self.neurochemical_state["dopamine"] = min(1.0,
            self.neurochemical_state.get("dopamine", 0.5) + 0.1)
        
        synthesis = f"[CREATIVE SYNTHESIS]: Imagining the intersection of {', '.join(concepts)}. "
        synthesis += f"Creativity level: {self.creativity_level:.2f}. "
        
        if self.creativity_level > 0.7:
            synthesis += "Novel connections emerging..."
        elif self.creativity_level > 0.4:
            synthesis += "Exploring potential relationships..."
        else:
            synthesis += "Seeking inspiration..."
        
        self._update_creativity_level()
        return synthesis


# Global neurochemical imagination engine instance
neurochemical_imagination_engine = NeurochemicalImaginationEngine() if NEUROCHEMICAL_AVAILABLE else None
