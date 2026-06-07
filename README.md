# Plexus AGI Framework

*A biologically-inspired Python library for neurochemical and deep learning systems supporting AGI development.*

A Python library providing neurochemical and deep learning systems for artificial general intelligence development.

## Features

### Neurochemical Systems
- **Neurochemical Simulation**: MATLAB-based neurochemical dynamics modeling
- **Glial Ion System**: Glial cell support with ion channel simulation
- **Astrocyte Network**: Astrocyte-mediated neural processing
- **Synaptic Plasticity**: Biologically accurate synaptic learning
- **Neural Oscillations**: Brain wave simulation and cognitive processing
- **Emotional Ecosystem**: Emotion processing integrated with neurochemicals

### Deep Learning Components
- **Enhanced Neural Core**: Advanced neuron models with ion channels
- **Sensory Processing**: Multi-modal sensory integration
- **Reasoning Core**: Neurochemical-modulated reasoning system
- **Imagination Engine**: Creative cognitive processing
- **Long-term Memory**: Persistent knowledge storage
- **Concept Vector Database**: Semantic concept representation

## Installation

```bash
pip install .
```

## Usage

### Basic Import

```python
# Import everything
from plexus import *

# Or import specific modules
from plexus.agi import EnhancedNeuralCore, ReasoningCore
```

### Neurochemical Systems

#### MATLAB Neurochemical Simulation
```python
from plexus.agi import get_simulation_instance

# Initialize MATLAB engine and simulation
neurochem_sim = get_simulation_instance()

# Simulate neurochemical dynamics
initial_state = {
    'dopamine': 0.5,
    'serotonin': 0.6,
    'glutamate': 0.4
}
result = neurochem_sim.simulate_neurochemical_dynamics(initial_state, time_duration=1.0)
print("Final state:", result['neurochemical_state'])
```

#### Glial Ion System
```python
from plexus.agi import GlialIonSystem
import torch

# Create glial system
glial_system = GlialIonSystem(num_neurons=100, num_glia=20)

# Process neural activity
neural_input = torch.randn(100, 64)  # batch_size=100, features=64
glial_output = glial_system(neural_input, neurochemical_levels={'glutamate': 0.7})
```

#### Astrocyte Network
```python
from plexus.agi import AstrocyteNetwork, AstrocyteCognitiveProcessor

# Create astrocyte network
astrocyte_net = AstrocyteNetwork(num_astrocytes=50, embed_dim=256)

# Create cognitive processor
processor = AstrocyteCognitiveProcessor(astrocyte_net, embed_dim=256)

# Process cognitive input
input_tensor = torch.randn(32, 256)  # sequence_length=32, embed_dim=256
cognitive_output = processor(input_tensor)
```

### Deep Learning Components

#### Enhanced Neural Core
```python
from plexus.agi import EnhancedNeuralCore
import torch

# Create enhanced neural network
neural_core = EnhancedNeuralCore(embed_dim=512, num_layers=6)

# Forward pass with neurochemical modulation
input_tensor = torch.randn(10, 512)  # batch_size=10
neurochemicals = {'dopamine': 0.8, 'serotonin': 0.6}
output = neural_core(input_tensor, neurochemical_levels=neurochemicals)
```

#### Reasoning Core
```python
from plexus.agi import ReasoningCore

# Create reasoning system
reasoner = ReasoningCore()

# Set neurochemical state
reasoner.set_neurochemical_state({
    'dopamine': 0.7,
    'acetylcholine': 0.5,
    'norepinephrine': 0.4
})

# Perform reasoning
problem = "What is the capital of France?"
context = ["France is a country in Europe"]
reasoning_result = reasoner.reason(problem, context)
print("Conclusion:", reasoning_result['conclusion'])
```

#### Sensory Processing
```python
from plexus.agi import SensoryProcessing
import numpy as np

# Create sensory processor
sensory = SensoryProcessing()

# Process visual input
image_data = np.random.rand(224, 224, 3)  # RGB image
visual_result = sensory.process_input("visual", image_data)

# Process audio input
audio_data = np.random.rand(16000)  # 1 second at 16kHz
audio_result = sensory.process_input("audio", audio_data)
```

#### Imagination Engine
```python
from plexus.agi import ImaginationEngine

# Create imagination system
imagination = ImaginationEngine(embed_dim=256)

# Generate creative concepts
seed_concept = "flying car"
imagined_concepts = imagination.generate_ideas(seed_concept, num_ideas=5)
for concept in imagined_concepts:
    print("Idea:", concept['description'])
```

#### Long-term Memory
```python
from plexus.agi import LongTermMemory

# Create memory system
memory = LongTermMemory(embed_dim=512, max_memories=10000)

# Store information
memory.store("Paris is the capital of France", importance=0.9)

# Retrieve relevant memories
query = "What is the capital of France?"
relevant_memories = memory.retrieve(query, top_k=3)
for mem in relevant_memories:
    print("Memory:", mem['content'], "Relevance:", mem['score'])
```

#### Concept Vector Database
```python
from plexus.agi import ConceptVectorDatabase

# Create concept database
concept_db = ConceptVectorDatabase(embed_dim=300)

# Add concepts
concept_db.add_concept("artificial_intelligence", description="Machines performing tasks that typically require human intelligence")
concept_db.add_concept("neural_network", description="Computing systems inspired by biological neural networks")

# Find related concepts
related = concept_db.find_related("machine_learning", top_k=5)
for concept, similarity in related:
    print(f"Concept: {concept}, Similarity: {similarity}")
```

### Emotional Ecosystem
```python
from plexus.agi import EmotionalEcosystem

# Create emotional processing system
emotion_system = EmotionalEcosystem()

# Process emotional stimulus
stimulus = {
    'type': 'social_interaction',
    'valence': 0.8,  # positive
    'arousal': 0.6   # moderate
}
emotional_response = emotion_system.process_stimulus(stimulus)
print("Primary emotion:", emotional_response['primary_emotion'])
print("Neurochemical changes:", emotional_response['neurochemical_changes'])
```

## Architecture

The framework integrates neuroscience principles with deep learning:
- Neurotransmitter-based emotional processing
- Ion channel dynamics in neural models
- Astrocyte-neuron interactions
- Biologically inspired learning mechanisms

## Note

This library contains supporting systems for AGI development. Core AGI logic is proprietary and not included.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- MATLAB Engine (optional, for neurochemical simulation)
