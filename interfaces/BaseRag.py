from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union


class BaseRag(ABC):
    """
    Abstract base class for RAG (Retrieval-Augmented Generation) systems.
    
    This class defines the interface that all RAG implementations must follow.
    It provides abstract methods for inserting documents and querying the knowledge base.
    """
    
    @abstractmethod
    def insert(self, documents: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """
        Insert documents into the RAG system's knowledge base.
        
        Args:
            documents: Documents to insert. Can be:
                - A single document as string
                - A list of document strings
                - A single document as dictionary with metadata
                - A list of document dictionaries with metadata
        
        Returns:
            bool: True if insertion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def query(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Query the RAG system to retrieve relevant documents.
        
        Args:
            query: The search query string
            top_k: Number of top results to return (default: 5)
            **kwargs: Additional parameters specific to the implementation
        
        Returns:
            List[Dict[str, Any]]: List of retrieved documents with their metadata.
                Each dictionary should contain at least:
                - 'content': The document content
                - 'score': Relevance score (if applicable)
                - Additional metadata as needed
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all documents from the RAG system.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear()
