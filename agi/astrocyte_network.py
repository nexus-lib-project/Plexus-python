# astrocyte_network.py - Astrocyte Network Neural Architecture
# =========================================================
# Biological Astrocyte Network Implementation for AI_0001
# 
# Astrocytes are star-shaped glial cells that:
# - Form networks via gap junctions
# - Communicate via calcium waves
# - Modulate synaptic transmission
# - Are involved in memory consolidation
# - Regulate neural network activity

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import random


class Astrocyte(nn.Module):
    """
    Single Astrocyte cell - the fundamental unit of the astrocyte network.
    Implements calcium dynamics and synaptic modulation.
    """
    
    def __init__(self, input_dim: int = 256, internal_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.internal_dim = internal_dim
        
        # Calcium dynamics (internal state)
        self.calcium_level = 0.0
        self.calcium_decay = 0.95
        
        # Membrane potential dynamics
        self.membrane_potential = 0.0
        
        # Processes synaptic signals
        self.synapse_processor = nn.Sequential(
            nn.Linear(input_dim, internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
            nn.Tanh()  # Calcium dynamics are often non-linear
        )
        
        # Output modulation (astrocytes modulate neuron activity)
        self.modulation_output = nn.Sequential(
            nn.Linear(internal_dim, input_dim),
            nn.Sigmoid()  # Modulation strength 0-1
        )
        
        # Internal state update network
        self.state_update = nn.GRUCell(input_dim, internal_dim)
        
    def forward(self, neural_input: torch.Tensor, 
               prev_calcium: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process neural input through astrocyte.
        
        Returns:
            modulation: Modulation signal to apply to neurons
            calcium: New calcium level
        """
        batch_size = neural_input.size(0)
        
        # Process synaptic input
        processed = self.synapse_processor(neural_input)
        
        # Update internal calcium state
        if prev_calcium is not None:
            # Calcium wave propagation - decay + new signal
            calcium_signal = self.calcium_decay * prev_calcium + processed.mean(dim=-1, keepdim=True)
        else:
            calcium_signal = processed.mean(dim=-1, keepdim=True)
        
        # Generate modulation output
        modulation = self.modulation_output(processed)
        
        return modulation, calcium_signal


class AstrocyteNetwork(nn.Module):
    """
    Full Astrocyte Network - A network of interconnected astrocytes
    that modulates neural activity and enables memory consolidation.
    """
    
    def __init__(self, num_astrocytes: int = 16, 
                 input_dim: int = 256,
                 internal_dim: int = 128,
                 connectivity_prob: float = 0.3):
        super().__init__()
        
        self.num_astrocytes = num_astrocytes
        self.input_dim = input_dim
        self.connectivity_prob = connectivity_prob
        
        # Create population of astrocytes
        self.astrocytes = nn.ModuleList([
            Astrocyte(input_dim, internal_dim)
            for _ in range(num_astrocytes)
        ])
        
        # Gap junction network (astrocyte-to-astrocyte connections)
        # Learning which astrocytes connect to form networks
        self.gap_junctions = nn.Parameter(
            torch.randn(num_astrocytes, num_astrocytes) * 0.1
        )
        
        # Network-level modulation
        self.network_modulation = nn.Sequential(
            nn.Linear(num_astrocytes * input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        # Calcium wave propagation speed
        self.wave_speed = 0.5
        
        # Memory consolidation pathway
        self.consolidation_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Sigmoid()
        )
        
        # Track calcium levels for wave propagation
        self.calcium_levels = torch.zeros(num_astrocytes)
        
    def forward(self, neural_input: torch.Tensor, 
               learning: bool = True) -> Dict[str, torch.Tensor]:
        """
        Process input through astrocyte network.
        
        The astrocyte network:
        1. Receives neural activity
        2. Each astrocyte processes and generates modulation
        3. Gap junctions allow calcium wave propagation
        4. Network output modulates the original neural activity
        """
        batch_size = neural_input.size(0)
        
        # Process through each astrocyte
        modulations = []
        calcium_states = []
        
        for i, astrocyte in enumerate(self.astrocytes):
            prev_calcium = self.calcium_levels[i].unsqueeze(0).expand(batch_size, -1)
            
            mod, calcium = astrocyte(neural_input, prev_calcium)
            modulations.append(mod)
            calcium_states.append(calcium)
        
        # Stack modulations
        modulations = torch.stack(modulations, dim=1)  # [batch, num_astrocytes, input_dim]
        
        # Apply gap junction influence (calcium wave propagation)
        if learning and self.training:
            # Learn which connections are strong
            gap_junction_strength = torch.sigmoid(self.gap_junctions)
            calcium_wave = torch.matmul(calcium_states[0].squeeze(-1), gap_junction_strength)
            calcium_wave = calcium_wave.unsqueeze(-1).expand(-1, self.input_dim)
        else:
            calcium_wave = torch.zeros(batch_size, self.input_dim, device=neural_input.device)
        
        # Aggregate network modulation
        flat_modulations = modulations.view(batch_size, -1)
        network_modulation = self.network_modulation(flat_modulations)
        
        # Apply modulation to original input
        modulated_output = neural_input * network_modulation
        
        # Memory consolidation signal
        consolidation_signal = self.consolidation_gate(modulated_output)
        
        # Update calcium levels (for wave propagation)
        new_calcium = torch.stack([c.squeeze(-1).mean() for c in calcium_states])
        self.calcium_levels = self.wave_speed * new_calcium.detach() + \
                              (1 - self.wave_speed) * self.calcium_levels
        
        return {
            'output': modulated_output,
            'modulation': network_modulation,
            'consolidation_signal': consolidation_signal,
            'calcium_wave': calcium_wave,
            'individual_modulations': modulations
        }
    
    def reset_calcium(self):
        """Reset calcium levels (like a biological reset)"""
        self.calcium_levels = torch.zeros(self.num_astrocytes)


class AstrocyteCognitiveProcessor(nn.Module):
    """
    High-level cognitive processor using astrocyte networks
    as the primary neural architecture.
    """
    
    def __init__(self, embed_dim: int = 256, num_astrocytes: int = 16):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_astrocytes = num_astrocytes
        
        # Primary astrocyte network (conscious processing)
        self.primary_network = AstrocyteNetwork(
            num_astrocytes=num_astrocytes,
            input_dim=embed_dim,
            internal_dim=embed_dim // 2
        )
        
        # Secondary astrocyte network (subconscious processing)
        self.subconscious_network = AstrocyteNetwork(
            num_astrocytes=num_astrocytes // 2,
            input_dim=embed_dim,
            internal_dim=embed_dim // 2
        )
        
        # Integration network (combining conscious and subconscious)
        self.integration = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Memory gating (astrocytes regulate memory)
        self.memory_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Attention via astrocyte modulation
        self.attention_modulation = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, input_embeds: torch.Tensor,
                memory_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process input using astrocyte network architecture.
        
        The astrocyte network provides:
        - Neural modulation (regulating activity)
        - Memory consolidation (via calcium signaling)
        - Attention (via selective modulation)
        """
        batch_size = input_embeds.size(0)
        
        # Primary conscious processing
        primary_result = self.primary_network(input_embeds)
        
        # Secondary subconscious processing
        subconscious_result = self.subconscious_network(input_embeds)
        
        # Integrate conscious and subconscious
        combined = torch.cat([
            primary_result['output'],
            subconscious_result['output']
        ], dim=-1)
        
        integrated = self.integration(combined)
        
        # Apply memory gating (regulated by astrocyte activity)
        memory_signal = self.memory_gate(primary_result['consolidationation_signal'])
        
        if memory_context is not None:
            # Incorporate memory context
            integrated = integrated + memory_context * memory_signal
        
        # Generate attention via astrocyte modulation
        attention_weights = torch.softmax(
            self.attention_modulation(integrated), dim=-1
        )
        attended = integrated * attention_weights
        
        return {
            'output': attended,
            'primary_modulation': primary_result['modulation'],
            'subconscious_modulation': subconscious_result['modulation'],
            'consolidation_signal': primary_result['consolidation_signal'],
            'calcium_wave': primary_result['calcium_wave'],
            'memory_gate': memory_signal,
            'attention_weights': attention_weights
        }
    
    def process_for_memory(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        Process input specifically for memory consolidation.
        Astrocytes are heavily involved in memory consolidation.
        """
        result = self.primary_network(input_embeds, learning=True)
        return result['consolidation_signal']
    
    def recall_with_calcium(self, query_embeds: torch.Tensor, 
                           memory_bank: torch.Tensor) -> torch.Tensor:
        """
        Recall memories using calcium wave dynamics.
        """
        # Query through astrocyte network
        query_result = self.primary_network(query_embeds)
        
        # Calcium wave influences recall
        calcium_influence = query_result['calcium_wave']
        
        # Calculate similarity with memory bank
        similarities = F.cosine_similarity(
            query_embeds + calcium_influence,
            memory_bank.unsqueeze(0).expand(query_embeds.size(0), -1, -1),
            dim=-1
        )
        
        # Retrieve top memories
        top_memories = torch.topk(similarities, k=3, dim=-1)
        
        return top_memories.values


class AstrocytePrefrontalCortex(nn.Module):
    """
    Astrocyte-based prefrontal cortex for executive control.
    Astrocytes in the PFC regulate decision-making and executive functions.
    """
    
    def __init__(self, input_dim: int = 256, num_astrocytes: int = 8):
        super().__init__()
        
        # Astrocyte network for executive function
        self.executive_network = AstrocyteNetwork(
            num_astrocytes=num_astrocytes,
            input_dim=input_dim,
            internal_dim=input_dim // 2
        )
        
        # Decision making network
        self.decision_network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 4)  # 4 possible actions
        )
        
        # Impulse control (astrocytes regulate impulsivity)
        self.impulse_control = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Sigmoid()
        )
        
        # Working memory (astrocytes maintain calcium levels for short-term memory)
        self.working_memory = torch.zeros(1, input_dim)
        
    def forward(self, context: torch.Tensor, 
                options: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process through astrocyte-based prefrontal cortex.
        """
        # Executive processing
        executive_result = self.executive_network(context)
        
        # Impulse control
        impulse_signal = self.impulse_control(executive_result['output'])
        
        # Update working memory (calcium persistence)
        self.working_memory = 0.9 * self.working_memory + 0.1 * context.detach()
        
        # Decision making
        if options is not None:
            # Evaluate options
            decision_logits = self.decision_network(options * impulse_signal)
        else:
            # Generate default decision
            decision_logits = self.decision_network(executive_result['output'])
        
        return {
            'output': executive_result['output'],
            'decision': decision_logits,
            'impulse_control': impulse_signal,
            'working_memory': self.working_memory,
            'modulation': executive_result['modulation']
        }


# Global instance for chatbot integration
astrocyte_processor: Optional[AstrocyteCognitiveProcessor] = None
astrocyte_pfc: Optional[AstrocytePrefrontalCortex] = None


def initialize_astrocyte_system(embed_dim: int = 256) -> Tuple[AstrocyteCognitiveProcessor, AstrocytePrefrontalCortex]:
    """Initialize the astrocyte network system"""
    global astrocyte_processor, astrocyte_pfc
    
    astrocyte_processor = AstrocyteCognitiveProcessor(
        embed_dim=embed_dim,
        num_astrocytes=16
    )
    
    astrocyte_pfc = AstrocytePrefrontalCortex(
        input_dim=embed_dim,
        num_astrocytes=8
    )
    
    print("[ASTROCYTE_NETWORK] Initialized astrocyte network architecture")
    print("[ASTROCYTE_NETWORK] - Cognitive processor with 16 astrocytes")
    print("[ASTROCYTE_NETWORK] - Prefrontal cortex with 8 astrocytes")
    print("[ASTROCYTE_NETWORK] - Calcium wave propagation enabled")
    print("[ASTROCYTE_NETWORK] - Memory consolidation pathway active")
    
    return astrocyte_processor, astrocyte_pfc


def get_astrocyte_processor() -> Optional[AstrocyteCognitiveProcessor]:
    """Get the astrocyte processor instance"""
    global astrocyte_processor
    return astrocyte_processor


def get_astrocyte_pfc() -> Optional[AstrocytePrefrontalCortex]:
    """Get the astrocyte PFC instance"""
    global astrocyte_pfc
    return astrocyte_pfc


# Convenience functions for chatbot integration
def process_with_astrocytes(input_embeds: torch.Tensor, 
                            memory_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Process input through astrocyte network"""
    processor = get_astrocyte_processor()
    if processor is None:
        initialize_astrocyte_system(input_embeds.size(-1))
        processor = get_astrocyte_processor()
    
    return processor(input_embeds, memory_context)


def astrocyte_prefrontal_process(context: torch.Tensor,
                                  options: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Process through astrocyte-based PFC"""
    pfc = get_astrocyte_pfc()
    if pfc is None:
        initialize_astrocyte_system(context.size(-1))
        pfc = get_astrocyte_pfc()
    
    return pfc(context, options)


def consolidate_to_memory(input_embeds: torch.Tensor) -> torch.Tensor:
    """Consolidate input to memory via astrocyte pathway"""
    processor = get_astrocyte_processor()
    if processor is None:
        return input_embeds
    
    return processor.process_for_memory(input_embeds)


def recall_with_astrocytes(query_embeds: torch.Tensor, 
                           memory_bank: torch.Tensor) -> torch.Tensor:
    """Recall memories using astrocyte calcium dynamics"""
    processor = get_astrocyte_processor()
    if processor is None:
        return query_embeds
    
    return processor.recall_with_calcium(query_embeds, memory_bank)
