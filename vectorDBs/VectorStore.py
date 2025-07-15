from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np


class VectorStore(ABC):
    """
    Abstract base class for vector storage systems.
    
    This class defines the interface that all vector storage implementations must follow.
    It provides abstract methods for storing, retrieving, and managing vector embeddings.
    """
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None, 
                   ids: Optional[List[str]] = None) -> List[str]:
        """
        Add vectors to the vector store.
        
        Args:
            vectors: numpy array of vectors to store (shape: n_vectors x vector_dim)
            metadata: Optional list of metadata dictionaries for each vector
            ids: Optional list of custom IDs for each vector
        
        Returns:
            List[str]: List of IDs for the added vectors
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5, 
               filter_dict: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search for similar vectors in the store.
        
        Args:
            query_vector: Query vector to search for (shape: vector_dim)
            top_k: Number of top results to return
            filter_dict: Optional dictionary of metadata filters
        
        Returns:
            Tuple[List[Dict[str, Any]], List[float]]: 
                - List of result dictionaries containing metadata and IDs
                - List of similarity scores
        """
        pass
    
    @abstractmethod
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a specific vector by ID.
        
        Args:
            vector_id: ID of the vector to retrieve
        
        Returns:
            Optional[np.ndarray]: The vector if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update_vector(self, vector_id: str, new_vector: np.ndarray, 
                     new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing vector and its metadata.
        
        Args:
            vector_id: ID of the vector to update
            new_vector: New vector to replace the existing one
            new_metadata: Optional new metadata to replace existing metadata
        
        Returns:
            bool: True if update was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector from the store.
        
        Args:
            vector_id: ID of the vector to delete
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all vectors from the store.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict[str, Any]: Dictionary containing statistics like:
                - 'total_vectors': Number of vectors in store
                - 'vector_dimension': Dimension of vectors
                - 'index_type': Type of index used
                - Additional implementation-specific stats
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> bool:
        """
        Save the vector store to disk.
        
        Args:
            path: Path where to save the vector store
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            path: Path from where to load the vector store
        
        Returns:
            bool: True if load was successful, False otherwise
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear()
