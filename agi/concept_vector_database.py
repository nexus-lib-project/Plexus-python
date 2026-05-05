#!/usr/bin/env python3
"""
concept_vector_database.py - Dynamic Dimensional Concept Vector Database
========================================================================
A vector database that stores concepts with N dimensions where N = number of concepts.
Similar concepts are stored closer together, dissimilar concepts further apart.
Storage format: Mix of .parquet, .bin, and .h5/.hdf5 files.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[CONCEPT_VDB] pandas not available, will use alternative formats")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("[CONCEPT_VDB] h5py not available, will use alternative formats")

# Default database directory
VECTOR_DB_DIR = "AI_0001/concept_vector_db"


class ConceptVectorDatabase:
    """
    Dynamic dimensional vector database for concepts.
    
    Key Features:
    - Dimensions = number of concepts (e.g., 500 concepts = 500 dimensions)
    - Similar concepts stored closer together using cosine similarity
    - Dissimilar concepts stored further apart
    - Hybrid storage: .parquet, .bin, .h5 files
    - Always accessible to the AI
    """
    
    def __init__(self, db_dir: str = VECTOR_DB_DIR, init_dim: int = 0):
        """
        Initialize with init_dim=0 so dimension equals num_concepts by default.
        """
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)
        
        # Core concept storage
        self.concepts: Dict[str, np.ndarray] = {}
        self.concept_names: List[str] = []
        self.concept_metadata: Dict[str, Dict] = {}
        
        # Dynamic dimension - starts at 0, equals num_concepts
        # Set to at least 1 to avoid empty array issues
        self.dimension = max(init_dim, 1)
        
        # Storage file paths
        self.parquet_file = os.path.join(db_dir, "concepts.parquet")
        self.bin_file = os.path.join(db_dir, "vectors.bin")
        self.h5_file = os.path.join(db_dir, "concepts.h5")
        self.meta_file = os.path.join(db_dir, "metadata.json")
        self.index_file = os.path.join(db_dir, "concept_index.json")
        
        # Load existing database if available
        self._load_database()
        
        print(f"[CONCEPT_VDB] Initialized with {self.dimension} dimensions, {len(self.concepts)} concepts")
    
    def _load_database(self):
        """Load existing database from disk"""
        # Try to load metadata
        if os.path.exists(self.meta_file):
            try:
                with open(self.meta_file, 'r') as f:
                    meta = json.load(f)
                    self.concept_names = meta.get('concept_names', [])
                    self.concept_metadata = meta.get('metadata', {})
            except Exception as e:
                print(f"[CONCEPT_VDB] Error loading metadata: {e}")
        
        # Load vectors from available formats
        loaded = False
        
        # Try HDF5 first (most complete)
        if os.path.exists(self.h5_file) and HAS_H5PY:
            try:
                with h5py.File(self.h5_file, 'r') as f:
                    for name in self.concept_names:
                        if name in f:
                            self.concepts[name] = f[name][:]
                loaded = True
                print(f"[CONCEPT_VDB] Loaded from HDF5: {len(self.concepts)} concepts")
            except Exception as e:
                print(f"[CONCEPT_VDB] Error loading HDF5: {e}")
        
        # Try parquet if HDF5 failed
        if not loaded and os.path.exists(self.parquet_file) and HAS_PANDAS:
            try:
                df = pd.read_parquet(self.parquet_file)
                for idx, row in df.iterrows():
                    name = row['concept']
                    vector = row['vector']
                    self.concepts[name] = np.array(vector)
                loaded = True
                print(f"[CONCEPT_VDB] Loaded from Parquet: {len(self.concepts)} concepts")
            except Exception as e:
                print(f"[CONCEPT_VDB] Error loading Parquet: {e}")
        
        # Try bin as fallback
        if not loaded and os.path.exists(self.bin_file):
            try:
                vectors = np.fromfile(self.bin_file, dtype=np.float32)
                num_vectors = len(self.concept_names)
                # Estimate dimension from file size
                if num_vectors > 0 and len(vectors) % num_vectors == 0:
                    loaded_dim = len(vectors) // num_vectors
                    self.dimension = loaded_dim
                    vectors = vectors.reshape(num_vectors, loaded_dim)
                    for i, name in enumerate(self.concept_names):
                        self.concepts[name] = vectors[i]
                    loaded = True
                    print(f"[CONCEPT_VDB] Loaded from BIN: {len(self.concepts)} concepts, dimension: {loaded_dim}")
            except Exception as e:
                print(f"[CONCEPT_VDB] Error loading BIN: {e}")
        
        # If no concepts loaded, set dimension to max(num_concepts, 1)
        if self.dimension < len(self.concepts):
            self.dimension = max(len(self.concepts), 1)
    
    def _save_database(self):
        """Save database to disk in multiple formats"""
        # Save metadata JSON
        meta = {
            'dimension': self.dimension,
            'concept_names': self.concept_names,
            'metadata': self.concept_metadata,
            'created_at': datetime.now().isoformat(),
            'num_concepts': len(self.concepts)
        }
        with open(self.meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Save as HDF5 (primary format)
        if HAS_H5PY:
            try:
                with h5py.File(self.h5_file, 'w') as f:
                    for name, vector in self.concepts.items():
                        f.create_dataset(name, data=vector)
                print(f"[CONCEPT_VDB] Saved to HDF5: {self.h5_file}")
            except Exception as e:
                print(f"[CONCEPT_VDB] Error saving HDF5: {e}")
        
        # Save as Parquet (if pandas available)
        if HAS_PANDAS:
            try:
                data = []
                for name, vector in self.concepts.items():
                    data.append({
                        'concept': name,
                        'vector': vector.tolist(),
                        'metadata': json.dumps(self.concept_metadata.get(name, {}))
                    })
                df = pd.DataFrame(data)
                df.to_parquet(self.parquet_file, index=False)
                print(f"[CONCEPT_VDB] Saved to Parquet: {self.parquet_file}")
            except Exception as e:
                print(f"[CONCEPT_VDB] Error saving Parquet: {e}")
        
        # Save as binary (fallback)
        try:
            vectors = np.array(list(self.concepts.values()), dtype=np.float32)
            vectors.tofile(self.bin_file)
            print(f"[CONCEPT_VDB] Saved to BIN: {self.bin_file}")
        except Exception as e:
            print(f"[CONCEPT_VDB] Error saving BIN: {e}")
        
        # Save concept index for fast lookup
        index = {name: i for i, name in enumerate(self.concept_names)}
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _expand_dimensions(self, new_dim: int):
        """Expand vector dimensions to accommodate new concepts"""
        if new_dim <= self.dimension:
            return
        
        old_dim = self.dimension
        self.dimension = new_dim
        
        # Expand existing vectors
        for name in self.concepts:
            old_vector = self.concepts[name]
            if len(old_vector) < new_dim:
                # Pad with small random values to maintain structure
                padding = np.random.randn(new_dim - len(old_vector)) * 0.01
                self.concepts[name] = np.concatenate([old_vector, padding])
        
        print(f"[CONCEPT_VDB] Expanded dimensions: {old_dim} -> {new_dim}")
    
    def add_concept(self, name: str, vector: Optional[np.ndarray] = None, 
                   metadata: Optional[Dict] = None) -> bool:
        """
        Add a new concept to the database.
        Expands dimensions if needed to match concept count.
        If no vector is provided, automatically generates one based on concept name
        and similarity to existing concepts.
        
        Args:
            name: Concept name/identifier
            vector: Optional vector - if None, auto-generates one
            metadata: Optional metadata dict
            
        Returns:
            True if added successfully
        """
        if name in self.concepts:
            print(f"[CONCEPT_VDB] Concept '{name}' already exists")
            return False
        
        # Calculate required dimensions based on total concepts
        # Each concept adds a dimension
        new_num_concepts = len(self.concepts) + 1
        required_dim = max(new_num_concepts, self.dimension)
        
        # Expand if needed
        if required_dim > self.dimension:
            self._expand_dimensions(required_dim)
        
        # Generate vector if not provided
        if vector is None:
            vec = self._generate_concept_vector(name)
        else:
            vec = np.array(vector, dtype=np.float32)
        
        # Ensure correct dimension
        if len(vec) < self.dimension:
            vec = np.pad(vec, (0, self.dimension - len(vec)))
        elif len(vec) > self.dimension:
            vec = vec[:self.dimension]
        
        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        self.concepts[name] = vec
        self.concept_names.append(name)
        
        if metadata:
            self.concept_metadata[name] = metadata
        else:
            self.concept_metadata[name] = {}
        
        # Auto-save after adding
        self._save_database()
        
        print(f"[CONCEPT_VDB] Added concept '{name}' (dimension: {self.dimension}, auto-generated: {vector is None})")
        return True
    
    def _generate_concept_vector(self, name: str) -> np.ndarray:
        """
        Generate a vector for a concept based on its name and existing concepts.
        Uses hash-based seeding with similarity adjustment to existing concepts.
        """
        import hashlib
        
        # Create deterministic base vector from concept name
        name_hash = hashlib.md5(name.encode()).digest()
        # Use first bytes as seed
        seed = int.from_bytes(name_hash[:4], 'big') % (2**32)
        np.random.seed(seed)
        
        # Generate base vector
        vec = np.random.randn(self.dimension).astype(np.float32)
        
        # Adjust vector to be more similar to semantically similar existing concepts
        # Use word-level similarity for basic adjustment
        if self.concepts:
            similar_adjustments = []
            for existing_name, existing_vec in self.concepts.items():
                # Ensure existing vector has correct dimension
                if len(existing_vec) != self.dimension:
                    if len(existing_vec) < self.dimension:
                        existing_vec = np.pad(existing_vec, (0, self.dimension - len(existing_vec)))
                    else:
                        existing_vec = existing_vec[:self.dimension]
                    self.concepts[existing_name] = existing_vec
                
                # Simple word-level similarity check
                similarity = self._calculate_name_similarity(name, existing_name)
                if similarity > 0.3:
                    # Adjust toward similar concepts
                    similar_adjustments.append((existing_vec, similarity))
            
            # Apply adjustments
            if similar_adjustments:
                for existing_vec, sim in similar_adjustments:
                    vec = vec + existing_vec * sim * 0.3
        
        return vec
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate simple word-level similarity between two concept names.
        Returns value between 0 (no similarity) and 1 (identical).
        """
        # Simple character overlap coefficient
        set1 = set(name1.lower().replace('_', ' ').replace('-', ' ').split())
        set2 = set(name2.lower().replace('_', ' ').replace('-', ' ').split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_concept(self, name: str) -> Optional[np.ndarray]:
        """Get vector for a specific concept"""
        return self.concepts.get(name)
    
    def get_all_concepts(self) -> Dict[str, np.ndarray]:
        """Get all concepts and their vectors"""
        return self.concepts.copy()
    
    def find_similar(self, query_vector: np.ndarray, 
                     top_k: int = 10, 
                     exclude: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Find similar concepts using cosine similarity.
        Similar concepts have vectors closer together (higher similarity).
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            exclude: List of concept names to exclude
            
        Returns:
            List of (concept_name, similarity_score) tuples
        """
        if not self.concepts:
            return []
        
        # Normalize query
        query = np.array(query_vector, dtype=np.float32)
        if len(query) > self.dimension:
            query = query[:self.dimension]
        elif len(query) < self.dimension:
            query = np.pad(query, (0, self.dimension - len(query)))
        
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        similarities = []
        
        for name, vector in self.concepts.items():
            if exclude and name in exclude:
                continue
            
            # Cosine similarity
            sim = np.dot(query, vector)
            similarities.append((name, float(sim)))
        
        # Sort by similarity (highest = most similar)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def find_dissimilar(self, query_vector: np.ndarray,
                        top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find dissimilar concepts.
        Dissimilar concepts have vectors further apart (lower similarity).
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            
        Returns:
            List of (concept_name, dissimilarity_score) tuples
        """
        similar = self.find_similar(query_vector, top_k=len(self.concepts))
        
        # Return least similar (lowest cosine similarity)
        dissimilar = similar[-top_k:] if len(similar) >= top_k else similar
        dissimilar = [(name, 1.0 - sim) for name, sim in dissimilar]
        
        return dissimilar
    
    def get_dimension(self) -> int:
        """Get current dimension (equals number of concepts)"""
        return self.dimension
    
    def get_num_concepts(self) -> int:
        """Get number of concepts in database"""
        return len(self.concepts)
    
    def update_concept(self, name: str, new_vector: np.ndarray) -> bool:
        """Update an existing concept's vector"""
        if name not in self.concepts:
            return False
        
        vec = np.array(new_vector, dtype=np.float32)
        
        # Ensure correct dimension
        if len(vec) < self.dimension:
            vec = np.pad(vec, (0, self.dimension - len(vec)))
        elif len(vec) > self.dimension:
            vec = vec[:self.dimension]
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        self.concepts[name] = vec
        self._save_database()
        
        print(f"[CONCEPT_VDB] Updated concept '{name}'")
        return True
    
    def delete_concept(self, name: str) -> bool:
        """Delete a concept from the database"""
        if name not in self.concepts:
            return False
        
        del self.concepts[name]
        self.concept_names.remove(name)
        
        if name in self.concept_metadata:
            del self.concept_metadata[name]
        
        self._save_database()
        
        print(f"[CONCEPT_VDB] Deleted concept '{name}'")
        return True
    
    def get_database_matrix(self) -> np.ndarray:
        """
        Get the full database as a matrix.
        Shape: (num_concepts, dimension)
        """
        if not self.concepts:
            return np.array([])
        
        return np.array(list(self.concepts.values()))
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise cosine similarity between all concepts.
        Similar concepts have higher values, dissimilar have lower.
        """
        if not self.concepts:
            return np.array([])
        
        vectors = self.get_database_matrix()
        
        # Cosine similarity matrix
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = vectors / norms
        
        similarity = np.dot(normalized, normalized.T)
        
        return similarity
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'num_concepts': len(self.concepts),
            'dimension': self.dimension,
            'storage_files': {
                'h5': self.h5_file if HAS_H5PY else None,
                'parquet': self.parquet_file if HAS_PANDAS else None,
                'bin': self.bin_file
            },
            'total_size_bytes': sum(
                os.path.getsize(os.path.join(self.db_dir, f)) 
                for f in os.listdir(self.db_dir) 
                if os.path.isfile(os.path.join(self.db_dir, f))
            ) if os.path.exists(self.db_dir) else 0
        }


# Global instance
_concept_vdb_instance = None


def get_concept_vector_database(db_dir: str = VECTOR_DB_DIR) -> ConceptVectorDatabase:
    """Get or create the global concept vector database instance"""
    global _concept_vdb_instance
    if _concept_vdb_instance is None:
        _concept_vdb_instance = ConceptVectorDatabase(db_dir)
    return _concept_vdb_instance


def add_concept(name: str, vector: Optional[np.ndarray] = None, metadata: Optional[Dict] = None) -> bool:
    """Add a concept to the database (auto-generates vector if not provided)"""
    return get_concept_vector_database().add_concept(name, vector, metadata)


def find_similar_concepts(query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
    """Find similar concepts to the query vector"""
    return get_concept_vector_database().find_similar(query_vector, top_k)


def get_all_concepts() -> Dict[str, np.ndarray]:
    """Get all concepts from the database"""
    return get_concept_vector_database().get_all_concepts()


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Concept Vector Database Demo")
    print("=" * 60)
    
    # Create database
    vdb = get_concept_vector_database()
    
    # Add some sample concepts with vectors
    # Each concept adds a dimension
    sample_concepts = {
        'cat': np.array([0.9, 0.1, 0.2, 0.3]),
        'dog': np.array([0.85, 0.15, 0.25, 0.2]),
        'computer': np.array([0.1, 0.9, 0.3, 0.1]),
        'keyboard': np.array([0.15, 0.85, 0.2, 0.15]),
        'apple_fruit': np.array([0.7, 0.3, 0.8, 0.2]),
        'apple_tech': np.array([0.2, 0.8, 0.3, 0.7])
    }
    
    print("\n--- Adding Concepts ---")
    for name, vector in sample_concepts.items():
        vdb.add_concept(name, vector, {'category': name.split('_')[0]})
    
    print(f"\nDatabase stats: {vdb.get_stats()}")
    print(f"Dimension (should equal num_concepts): {vdb.get_dimension()}")
    print(f"Number of concepts: {vdb.get_num_concepts()}")
    
    # Test similarity search
    print("\n--- Finding Similar to 'cat' ---")
    cat_vector = vdb.get_concept('cat')
    similar = vdb.find_similar(cat_vector, top_k=3)
    for name, sim in similar:
        print(f"  {name}: {sim:.4f}")
    
    print("\n--- Finding Dissimilar to 'computer' ---")
    computer_vector = vdb.get_concept('computer')
    dissimilar = vdb.find_dissimilar(computer_vector, top_k=3)
    for name, diss in dissimilar:
        print(f"  {name}: {diss:.4f}")
    
    print("\n--- Similarity Matrix (first 4 concepts) ---")
    sim_matrix = vdb.compute_similarity_matrix()
    print(f"Shape: {sim_matrix.shape}")
    print(f"First 4x4:\n{sim_matrix[:4, :4]}")
    
    print("\n--- Database Complete ---")
    print(f"Files in {VECTOR_DB_DIR}:")
    for f in os.listdir(VECTOR_DB_DIR):
        print(f"  - {f}")