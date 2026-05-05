# longterm_memory.py - ChromaDB-based Long-Term Memory for AI_0001 Chatbot
# ========================================================================
# Integrates AGI_0001's ChromaDB persistent memory system for years-long storage
# into the AI_0001 chatbot system.

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[LONGTERM_MEMORY] Warning: ChromaDB not available")

# Try to import spacy for NLP
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except:
        SPACY_AVAILABLE = False
except ImportError:
    nlp = None
    SPACY_AVAILABLE = False

# Try to import torch for neural processing
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# CHROMADB PERSISTENT MEMORY - YEARS-LONG STORAGE
# =============================================================================

class LongTermMemoryDB:
    """
    ChromaDB-based persistent memory that lasts for YEARS on disk.
    Provides semantic vector search for memory retrieval.
    """
    
    def __init__(self, persist_directory: str = "./AI_0001/agi_longterm_memory"):
        self.persist_directory = persist_directory
        self.embed_dim = 384  # Default embedding dimension
        
        if CHROMADB_AVAILABLE:
            # Ensure directory exists
            os.makedirs(persist_directory, exist_ok=True)
            
            # Create persistent client that saves to disk
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Main conversation memory collection
            self.conversation_collection = self.client.get_or_create_collection(
                name="chatbot_conversations",
                metadata={
                    "description": "AI_0001 chatbot conversation memory - persists for years",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            # Knowledge/facts collection
            self.knowledge_collection = self.client.get_or_create_collection(
                name="chatbot_knowledge",
                metadata={"description": "AI_0001 persistent knowledge base"}
            )
            
            # User profiles collection
            self.user_collection = self.client.get_or_create_collection(
                name="chatbot_users",
                metadata={"description": "AI_0001 user profiles and history"}
            )
            
            print(f"[LONGTERM_MEMORY] Initialized at {persist_directory}")
            print(f"[LONGTERM_MEMORY] Conversations: {self.conversation_collection.count()}")
            print(f"[LONGTERM_MEMORY] Knowledge: {self.knowledge_collection.count()}")
            print(f"[LONGTERM_MEMORY] Users: {self.user_collection.count()}")
        else:
            self.client = None
            self.conversation_collection = None
            self.knowledge_collection = None
            self.user_collection = None
            print("[LONGTERM_MEMORY] Using in-memory fallback")
    
    def store_conversation(self, user_id: str, message: str, response: str, 
                          embedding: np.ndarray, metadata: Optional[Dict] = None):
        """Store a conversation exchange that persists for years"""
        if not CHROMADB_AVAILABLE or self.conversation_collection is None:
            return
        
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000000)}"
        
        full_text = f"User: {message}\nAI: {response}"
        
        self.conversation_collection.add(
            embeddings=[embedding.tolist()],
            documents=[full_text],
            metadatas=[metadata or {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "type": "conversation"
            }],
            ids=[conversation_id]
        )
    
    def store_knowledge(self, fact: str, embedding: np.ndarray, 
                        metadata: Optional[Dict] = None):
        """Store persistent knowledge"""
        if not CHROMADB_AVAILABLE or self.knowledge_collection is None:
            return
        
        fact_id = f"fact_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000000)}"
        
        self.knowledge_collection.add(
            embeddings=[embedding.tolist()],
            documents=[fact],
            metadatas=[metadata or {
                "timestamp": datetime.now().isoformat(),
                "type": "knowledge"
            }],
            ids=[fact_id]
        )
    
    def store_user_profile(self, user_id: str, profile_data: str, 
                           embedding: np.ndarray, metadata: Optional[Dict] = None):
        """Store user profile information"""
        if not CHROMADB_AVAILABLE or self.user_collection is None:
            return
        
        profile_id = f"user_{user_id}_{np.random.randint(1000000)}"
        
        self.user_collection.add(
            embeddings=[embedding.tolist()],
            documents=[profile_data],
            metadatas=[metadata or {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "type": "user_profile"
            }],
            ids=[profile_id]
        )
    
    def retrieve_conversations(self, query_embedding: np.ndarray, 
                               n_results: int = 5) -> Tuple[List[str], List[Dict]]:
        """Retrieve similar past conversations"""
        if not CHROMADB_AVAILABLE or self.conversation_collection is None:
            return [], []
        
        if self.conversation_collection.count() == 0:
            return [], []
        
        results = self.conversation_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.conversation_collection.count())
        )
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        return documents, metadatas
    
    def retrieve_knowledge(self, query_embedding: np.ndarray, 
                          n_results: int = 5) -> Tuple[List[str], List[Dict]]:
        """Retrieve relevant knowledge"""
        if not CHROMADB_AVAILABLE or self.knowledge_collection is None:
            return [], []
        
        if self.knowledge_collection.count() == 0:
            return [], []
        
        results = self.knowledge_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.knowledge_collection.count())
        )
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        return documents, metadatas
    
    def retrieve_user_history(self, user_id: str, query_embedding: np.ndarray,
                              n_results: int = 5) -> Tuple[List[str], List[Dict]]:
        """Retrieve user's conversation history"""
        if not CHROMADB_AVAILABLE or self.conversation_collection is None:
            return [], []
        
        if self.conversation_collection.count() == 0:
            return [], []
        
        results = self.conversation_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.conversation_collection.count()),
            where={"user_id": user_id}
        )
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        return documents, metadatas
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        if SPACY_AVAILABLE:
            doc = nlp(text)
            embedding = doc.vector
        else:
            # Fallback: random embedding
            embedding = np.random.randn(self.embed_dim)
        
        # Ensure proper dimension
        if len(embedding) > self.embed_dim:
            embedding = embedding[:self.embed_dim]
        elif len(embedding) < self.embed_dim:
            embedding = np.pad(embedding, (0, self.embed_dim - len(embedding)))
        
        return embedding
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics"""
        return {
            "conversations": self.conversation_collection.count() if self.conversation_collection else 0,
            "knowledge": self.knowledge_collection.count() if self.knowledge_collection else 0,
            "users": self.user_collection.count() if self.user_collection else 0
        }


# =============================================================================
# RNN BRAIN FROM IDEA'S_FOR_AI - INTEGRATED
# =============================================================================

if TORCH_AVAILABLE:
    class ChatbotBrain(nn.Module):
        """
        Neural brain for processing chatbot inputs.
        Based on SkynetBrain from Idea's_for_AI (RNN_0.py)
        """
        
        def __init__(self, input_dim: int = 384, hidden_dim: int = 96, output_dim: int = 5): 
            super().__init__()
            self.hidden_dim = hidden_dim
            self.recurrent_layer = nn.Linear(hidden_dim, hidden_dim)
            self.output_head = nn.Linear(hidden_dim, output_dim)
            self.hidden_state = torch.zeros(1, hidden_dim)

        def forward(self, input_tensor: torch.Tensor, 
                    hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            """Process input with recurrent dynamics"""
            if hidden_state is None:
                hidden_state = self.hidden_state
                
            combined = self.recurrent_layer(hidden_state) + input_tensor[:, :self.hidden_dim]
            new_hidden = F.leaky_relu(combined, negative_slope=0.01)
            logits = self.output_head(new_hidden)
            
            self.hidden_state = new_hidden.detach()
            return logits, new_hidden
        
        def reset_hidden(self):
            """Reset the hidden state"""
            self.hidden_state = torch.zeros(1, self.hidden_dim)
    
    class NeuralChatbotProcessor:
        """Neural processor for chatbot using RNN brain"""
        
        def __init__(self, embed_dim: int = 384):
            self.embed_dim = embed_dim
            self.brain = ChatbotBrain(input_dim=embed_dim, hidden_dim=96, output_dim=5)
            self.brain.eval()
            
        def process_input(self, embedding: np.ndarray) -> Dict:
            """Process input through neural brain"""
            # Ensure proper dimension
            if len(embedding) > self.embed_dim:
                embedding = embedding[:self.embed_dim]
            elif len(embedding) < self.embed_dim:
                embedding = np.pad(embedding, (0, self.embed_dim - len(embedding)))
            
            input_tensor = torch.from_numpy(embedding).unsqueeze(0).float()
            
            with torch.no_grad():
                logits, hidden = self.brain(input_tensor)
            
            return {
                'logits': logits.numpy(),
                'hidden_state': hidden.numpy(),
                'action': torch.argmax(logits, dim=1).item()
            }
        
        def reset(self):
            """Reset brain state"""
            self.brain.reset_hidden()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Global long-term memory instance
longterm_memory: Optional[LongTermMemoryDB] = None
neural_processor: Optional[NeuralChatbotProcessor] = None

def initialize_longterm_memory(persist_directory: str = "./AI_0001/agi_longterm_memory") -> LongTermMemoryDB:
    """Initialize the long-term memory system"""
    global longterm_memory, neural_processor
    
    longterm_memory = LongTermMemoryDB(persist_directory=persist_directory)
    
    if TORCH_AVAILABLE:
        neural_processor = NeuralChatbotProcessor()
    
    return longterm_memory

def get_longterm_memory() -> Optional[LongTermMemoryDB]:
    """Get the long-term memory instance"""
    global longterm_memory
    if longterm_memory is None:
        longterm_memory = initialize_longterm_memory()
    return longterm_memory

def get_neural_processor() -> Optional[NeuralChatbotProcessor]:
    """Get the neural processor instance"""
    global neural_processor
    if neural_processor is None and TORCH_AVAILABLE:
        neural_processor = NeuralChatbotProcessor()
    return neural_processor


# =============================================================================
# CONVENIENCE FUNCTIONS FOR CHATBOT INTEGRATION
# =============================================================================

def store_interaction(user_id: str, message: str, response: str):
    """Store a conversation interaction in long-term memory"""
    memory = get_longterm_memory()
    if memory is None:
        return
    
    # Get embedding for the full interaction
    full_text = f"User: {message}\nAI: {response}"
    embedding = memory.get_text_embedding(full_text)
    
    memory.store_conversation(
        user_id=user_id,
        message=message,
        response=response,
        embedding=embedding
    )

def recall_similar_conversations(query: str, n_results: int = 5) -> List[str]:
    """Recall similar past conversations"""
    memory = get_longterm_memory()
    if memory is None:
        return []
    
    embedding = memory.get_text_embedding(query)
    documents, _ = memory.retrieve_conversations(embedding, n_results)
    return documents

def store_knowledge_fact(fact: str):
    """Store a knowledge fact"""
    memory = get_longterm_memory()
    if memory is None:
        return
    
    embedding = memory.get_text_embedding(fact)
    memory.store_knowledge(fact, embedding)

def recall_knowledge(query: str, n_results: int = 5) -> List[str]:
    """Recall relevant knowledge"""
    memory = get_longterm_memory()
    if memory is None:
        return []
    
    embedding = memory.get_text_embedding(query)
    documents, _ = memory.retrieve_knowledge(embedding, n_results)
    return documents

def get_memory_statistics() -> Dict[str, int]:
    """Get memory statistics"""
    memory = get_longterm_memory()
    if memory is None:
        return {"conversations": 0, "knowledge": 0, "users": 0}
    return memory.get_memory_stats()


# =============================================================================
# AUTO-INITIALIZE IF IMPORTED
# =============================================================================

if __name__ != "__main__":
    # Auto-initialize when imported
    try:
        initialize_longterm_memory()
    except Exception as e:
        print(f"[LONGTERM_MEMORY] Auto-init failed: {e}")
