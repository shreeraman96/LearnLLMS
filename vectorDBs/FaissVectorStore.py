import faiss
import numpy as np
import json
import os
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer
from .VectorStore import VectorStore


class FaissVectorStore(VectorStore):
    """
    FAISS implementation of the VectorStore abstract class.
    
    This class provides vector storage capabilities using FAISS library,
    specifically designed for storing and retrieving text documents and strings.
    Supports multiple embedding models and dimensions.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", index_type: str = "flat", 
                 dimension: Optional[int] = None, normalize: bool = True, 
                 allow_different_dimensions: bool = False):
        """
        Initialize the FAISS vector store.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            dimension: Vector dimension (auto-detected if None)
            normalize: Whether to normalize vectors before storage
            allow_different_dimensions: Whether to allow vectors of different dimensions
        """
        self.embedding_model = embedding_model
        self.index_type = index_type
        self.dimension = dimension
        self.normalize = normalize
        self.allow_different_dimensions = allow_different_dimensions
        
        # Initialize the embedding model
        self.model = SentenceTransformer(embedding_model)
        
        # Initialize FAISS index
        self.index = None
        self.vector_dimension = dimension or self.model.get_sentence_embedding_dimension()
        
        # Storage for metadata and mappings
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        
        # Track different dimensions if allowed
        self.dimension_tracker: Dict[int, int] = {}  # dimension -> count
        
        # Initialize the index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the FAISS index based on the specified type."""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.vector_dimension)
        elif self.index_type == "ivf":
            # IVF index with 100 clusters
            quantizer = faiss.IndexFlatL2(self.vector_dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.vector_dimension, 100)
        elif self.index_type == "hnsw":
            # HNSW index with 32 neighbors
            self.index = faiss.IndexHNSWFlat(self.vector_dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Add normalization if requested
        if self.normalize:
            self.index = faiss.IndexIDMap2(self.index)
    
    def _validate_vector_dimension(self, vectors: np.ndarray) -> bool:
        """
        Validate that vectors have the expected dimension.
        
        Args:
            vectors: numpy array of vectors to validate
        
        Returns:
            bool: True if dimensions are valid
        """
        if vectors.shape[1] != self.vector_dimension:
            if not self.allow_different_dimensions:
                raise ValueError(
                    f"Vector dimension {vectors.shape[1]} does not match expected dimension {self.vector_dimension}. "
                    f"Set allow_different_dimensions=True to allow different dimensions."
                )
            else:
                # Track this dimension
                self.dimension_tracker[vectors.shape[1]] = self.dimension_tracker.get(vectors.shape[1], 0) + vectors.shape[0]
                return False
        return True
    
    def _encode_texts(self, texts: Union[str, List[str]], 
                     model_name: Optional[str] = None) -> np.ndarray:
        """
        Encode texts to vectors using the sentence transformer model.
        
        Args:
            texts: Single text string or list of text strings
            model_name: Optional model name to use (if different from default)
        
        Returns:
            np.ndarray: Encoded vectors
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Use specified model or default model
        if model_name and model_name != self.embedding_model:
            # Create temporary model for different embedding
            temp_model = SentenceTransformer(model_name)
            embeddings = temp_model.encode(texts, show_progress_bar=False)
        else:
            embeddings = self.model.encode(texts, show_progress_bar=False)
        
        if self.normalize:
            faiss.normalize_L2(embeddings)
        
        return embeddings.astype(np.float32)
    
    def add_vectors(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None, 
                   ids: Optional[List[str]] = None) -> List[str]:
        """
        Add vectors to the FAISS index.
        
        Args:
            vectors: numpy array of vectors to store
            metadata: Optional list of metadata dictionaries
            ids: Optional list of custom IDs
        
        Returns:
            List[str]: List of IDs for the added vectors
        """
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")
        
        n_vectors = vectors.shape[0]
        
        # Validate vector dimensions
        dimension_matches = self._validate_vector_dimension(vectors)
        
        if not dimension_matches and not self.allow_different_dimensions:
            raise ValueError(
                f"Cannot add vectors with dimension {vectors.shape[1]} to index with dimension {self.vector_dimension}"
            )
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n_vectors)]
        
        # Initialize metadata if not provided
        if metadata is None:
            metadata = [{} for _ in range(n_vectors)]
        
        # Ensure all lists have the same length
        if len(ids) != n_vectors or len(metadata) != n_vectors:
            raise ValueError("Length mismatch between vectors, ids, and metadata")
        
        # If dimensions don't match and we allow different dimensions,
        # we need to handle this differently (e.g., store separately or reject)
        if not dimension_matches:
            # For now, we'll reject vectors with different dimensions
            # In a more advanced implementation, you could:
            # 1. Store them in separate indices
            # 2. Use a multi-index approach
            # 3. Project them to the same dimension
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} differs from index dimension {self.vector_dimension}. "
                f"This is not yet supported in the current implementation."
            )
        
        # Get current index size for mapping
        start_index = self.index.ntotal
        
        # Add vectors to FAISS index
        self.index.add(vectors)
        
        # Store metadata and create mappings
        for i, (vector_id, meta) in enumerate(zip(ids, metadata)):
            index_pos = start_index + i
            self.metadata_store[vector_id] = meta
            self.id_to_index[vector_id] = index_pos
            self.index_to_id[index_pos] = vector_id
        
        return ids
    
    def add_texts(self, texts: Union[str, List[str]], metadata: Optional[List[Dict[str, Any]]] = None,
                  ids: Optional[List[str]] = None, model_name: Optional[str] = None) -> List[str]:
        """
        Add text documents to the vector store.
        
        Args:
            texts: Single text string or list of text strings
            metadata: Optional list of metadata dictionaries
            ids: Optional list of custom IDs
            model_name: Optional model name to use for encoding
        
        Returns:
            List[str]: List of IDs for the added texts
        """
        # Encode texts to vectors
        vectors = self._encode_texts(texts, model_name)
        
        # Add vectors to the store
        return self.add_vectors(vectors, metadata, ids)
    
    def search(self, query_vector: np.ndarray, top_k: int = 5, 
               filter_dict: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search for similar vectors in the FAISS index.
        
        Args:
            query_vector: Query vector to search for
            top_k: Number of top results to return
            filter_dict: Optional dictionary of metadata filters (not implemented in basic version)
        
        Returns:
            Tuple[List[Dict[str, Any]], List[float]]: Results and scores
        """
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")
        
        # Validate query vector dimension
        if query_vector.shape[0] != self.vector_dimension:
            raise ValueError(
                f"Query vector dimension {query_vector.shape[0]} does not match index dimension {self.vector_dimension}"
            )
        
        # Normalize query vector if needed
        if self.normalize:
            query_vector = query_vector.astype(np.float32)
            faiss.normalize_L2(query_vector.reshape(1, -1))
        
        # Search in FAISS index
        distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        
        results = []
        scores = []
        
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx in self.index_to_id:  # Valid index
                vector_id = self.index_to_id[idx]
                metadata = self.metadata_store.get(vector_id, {}).copy()
                metadata['id'] = vector_id
                
                results.append(metadata)
                scores.append(float(distance))
        
        return results, scores
    
    def search_texts(self, query_text: str, top_k: int = 5, 
                    filter_dict: Optional[Dict[str, Any]] = None,
                    model_name: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search for similar texts using a text query.
        
        Args:
            query_text: Text query to search for
            top_k: Number of top results to return
            filter_dict: Optional dictionary of metadata filters
            model_name: Optional model name to use for encoding the query
        
        Returns:
            Tuple[List[Dict[str, Any]], List[float]]: Results and scores
        """
        # Encode query text to vector
        query_vector = self._encode_texts(query_text, model_name)
        
        # Search using the vector
        return self.search(query_vector, top_k, filter_dict)
    
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a specific vector by ID.
        
        Args:
            vector_id: ID of the vector to retrieve
        
        Returns:
            Optional[np.ndarray]: The vector if found, None otherwise
        """
        if vector_id not in self.id_to_index:
            return None
        
        # FAISS doesn't support direct vector retrieval by ID
        # This is a limitation - we would need to store vectors separately
        # For now, return None to indicate this feature is not available
        return None
    
    def update_vector(self, vector_id: str, new_vector: np.ndarray, 
                     new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing vector and its metadata.
        
        Args:
            vector_id: ID of the vector to update
            new_vector: New vector to replace the existing one
            new_metadata: Optional new metadata
        
        Returns:
            bool: True if update was successful, False otherwise
        """
        if vector_id not in self.id_to_index:
            return False
        
        # Validate new vector dimension
        if new_vector.shape[0] != self.vector_dimension:
            raise ValueError(
                f"New vector dimension {new_vector.shape[0]} does not match index dimension {self.vector_dimension}"
            )
        
        # FAISS doesn't support direct vector updates
        # We need to remove and re-add the vector
        if not self.delete_vector(vector_id):
            return False
        
        # Add the new vector
        new_ids = self.add_vectors(new_vector.reshape(1, -1), 
                                 [new_metadata] if new_metadata else None,
                                 [vector_id])
        
        return len(new_ids) > 0
    
    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector from the store.
        
        Args:
            vector_id: ID of the vector to delete
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if vector_id not in self.id_to_index:
            return False
        
        # FAISS doesn't support direct deletion
        # This is a limitation - we would need to rebuild the index
        # For now, just remove from metadata and mappings
        index_pos = self.id_to_index[vector_id]
        
        del self.metadata_store[vector_id]
        del self.id_to_index[vector_id]
        del self.index_to_id[index_pos]
        
        return True
    
    def clear(self) -> bool:
        """
        Clear all vectors from the store.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        try:
            self._initialize_index()
            self.metadata_store.clear()
            self.id_to_index.clear()
            self.index_to_id.clear()
            self.dimension_tracker.clear()
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict[str, Any]: Dictionary containing statistics
        """
        stats = {
            'total_vectors': self.index.ntotal if self.index else 0,
            'vector_dimension': self.vector_dimension,
            'index_type': self.index_type,
            'embedding_model': self.embedding_model,
            'normalize': self.normalize,
            'metadata_count': len(self.metadata_store),
            'allow_different_dimensions': self.allow_different_dimensions
        }
        
        if self.index:
            stats['index_size'] = self.index.ntotal
            stats['index_dimension'] = self.index.d
        
        if self.dimension_tracker:
            stats['dimension_distribution'] = self.dimension_tracker
        
        return stats
    
    def save(self, path: str) -> bool:
        """
        Save the vector store to disk.
        
        Args:
            path: Path where to save the vector store
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            index_path = f"{path}.idx"
            faiss.write_index(self.index, index_path)
            
            # Save metadata and mappings
            metadata_path = f"{path}.json"
            save_data = {
                'metadata_store': self.metadata_store,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'embedding_model': self.embedding_model,
                'index_type': self.index_type,
                'dimension': self.vector_dimension,
                'normalize': self.normalize,
                'allow_different_dimensions': self.allow_different_dimensions,
                'dimension_tracker': self.dimension_tracker
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving vector store: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            path: Path from where to load the vector store
        
        Returns:
            bool: True if load was successful, False otherwise
        """
        try:
            # Load FAISS index
            index_path = f"{path}.idx"
            self.index = faiss.read_index(index_path)
            
            # Load metadata and mappings
            metadata_path = f"{path}.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                load_data = json.load(f)
            
            self.metadata_store = load_data['metadata_store']
            self.id_to_index = load_data['id_to_index']
            self.index_to_id = load_data['index_to_id']
            self.embedding_model = load_data['embedding_model']
            self.index_type = load_data['index_type']
            self.vector_dimension = load_data['dimension']
            self.normalize = load_data['normalize']
            self.allow_different_dimensions = load_data.get('allow_different_dimensions', False)
            self.dimension_tracker = load_data.get('dimension_tracker', {})
            
            # Reinitialize the model
            self.model = SentenceTransformer(self.embedding_model)
            
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
