"""
AGI Library for Plexus
=====================

This package provides neurochemical and deep learning systems supporting AGI development.
Note: Core AGI logic is proprietary and not included.
"""

from .neurochemical_matlab_bridge import *
from .glial_ion_system import *
from .astrocyte_network import *
from .synaptic_plasticity import *
from .neural_oscillations import *
from .emotional_ecosystem import *
from .enhanced_neural_core import *
from .sensory_processing import *
from .reasoning_core import *
from .imagination import *
from .longterm_memory import *
from .concept_vector_database import *
from .affective_cognition import *

__all__ = [
    # Neurochemical systems
    'NeurochemicalSimulation', 'get_simulation_instance', 'MATLAB_ENGINE_AVAILABLE',
    'GlialIonSystem', 'IonChannel', 'AstrocyteNetwork',
    'AstrocyteCognitiveProcessor', 'SynapticPlasticity', 'NeuralOscillations',
    'EmotionalEcosystem', 'EmotionProcessor',
    
    # Deep learning systems
    'EnhancedNeuralCore', 'RichNeuron', 'SensoryProcessing',
    'ReasoningCore', 'ImaginationEngine', 'LongTermMemory',
    'ConceptVectorDatabase', 'NEUROTRANSMITTER_EMOTION_MAP'
]
