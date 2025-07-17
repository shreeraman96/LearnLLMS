import os
import json
import uuid
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np

# Handle imports for both direct execution and package import
import os
import sys

# Add parent directory to path if running directly
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from interfaces.BaseRag import BaseRag
    from vectorDBs.FaissVectorStore import FaissVectorStore
except ImportError:
    # Fallback for relative imports when used as package
    from ..interfaces.BaseRag import BaseRag
    from ..vectorDBs.FaissVectorStore import FaissVectorStore


class FaissRag(BaseRag):
    """
    FAISS-based RAG (Retrieval-Augmented Generation) implementation.
    
    This class implements the BaseRag interface using FAISS for vector storage
    and similarity search. It provides document insertion, querying, and
    management capabilities with support for metadata and custom configurations.
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 index_type: str = "flat",
                 dimension: Optional[int] = None,
                 normalize: bool = True,
                 allow_different_dimensions: bool = False,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 metadata_fields: Optional[List[str]] = None):
        """
        Initialize the FAISS RAG system.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            dimension: Vector dimension (auto-detected if None)
            normalize: Whether to normalize vectors before storage
            allow_different_dimensions: Whether to allow vectors of different dimensions
            chunk_size: Size of text chunks when splitting documents
            chunk_overlap: Overlap between consecutive chunks
            metadata_fields: List of metadata fields to extract and store
        """
        self.embedding_model = embedding_model
        self.index_type = index_type
        self.dimension = dimension
        self.normalize = normalize
        self.allow_different_dimensions = allow_different_dimensions
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata_fields = metadata_fields or []
        
        # Initialize the vector store
        self.vector_store = FaissVectorStore(
            embedding_model=embedding_model,
            index_type=index_type,
            dimension=dimension,
            normalize=normalize,
            allow_different_dimensions=allow_different_dimensions
        )
        
        # Initialize the embedding model for direct use
        self.model = SentenceTransformer(embedding_model)
        
        # Document tracking
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunk_to_doc: Dict[str, str] = {}
        
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks for better retrieval.
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to find a good break point (sentence boundary)
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + self.chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
                # If no sentence ending found, look for word boundary
                else:
                    for i in range(end, max(start + self.chunk_size - 50, start), -1):
                        if text[i] == ' ':
                            end = i
                            break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _extract_metadata(self, document: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract metadata from document input.
        
        Args:
            document: Document as string or dictionary
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        if isinstance(document, str):
            return {
                'content': document,
                'type': 'text',
                'length': len(document),
                'chunks': len(self._split_text(document))
            }
        else:
            metadata = document.copy()
            if 'content' not in metadata:
                raise ValueError("Document dictionary must contain 'content' field")
            
            # Add computed metadata
            metadata['type'] = metadata.get('type', 'document')
            metadata['length'] = len(metadata['content'])
            metadata['chunks'] = len(self._split_text(metadata['content']))
            
            return metadata
    
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
        try:
            # Normalize input to list of dictionaries
            if isinstance(documents, str):
                documents = [{'content': documents}]
            elif isinstance(documents, list) and all(isinstance(doc, str) for doc in documents):
                documents = [{'content': doc} for doc in documents]
            elif isinstance(documents, dict):
                documents = [documents]
            
            # Process each document
            for doc_data in documents:
                metadata = self._extract_metadata(doc_data)
                doc_id = metadata.get('id', str(uuid.uuid4()))
                
                # Store document metadata
                self.documents[doc_id] = metadata
                
                # Split document into chunks
                content = metadata['content']
                chunks = self._split_text(content)
                
                # Prepare chunk metadata
                chunk_metadata_list = []
                chunk_ids = []
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    chunk_metadata = {
                        'chunk_id': chunk_id,
                        'document_id': doc_id,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'content': chunk,
                        'chunk_length': len(chunk),
                        'type': 'chunk'
                    }
                    
                    # Add original document metadata
                    for key, value in metadata.items():
                        if key not in ['content', 'chunks']:  # Skip content and computed fields
                            chunk_metadata[f"doc_{key}"] = value
                    
                    chunk_metadata_list.append(chunk_metadata)
                    chunk_ids.append(chunk_id)
                    self.chunk_to_doc[chunk_id] = doc_id
                
                # Add chunks to vector store
                self.vector_store.add_texts(
                    texts=chunks,
                    metadata=chunk_metadata_list,
                    ids=chunk_ids
                )
            
            return True
            
        except Exception as e:
            print(f"Error inserting documents: {e}")
            return False
    
    def query(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Query the RAG system to retrieve relevant documents.
        
        Args:
            query: The search query string
            top_k: Number of top results to return (default: 5)
            **kwargs: Additional parameters:
                - filter_dict: Optional metadata filters
                - include_metadata: Whether to include full metadata (default: True)
                - group_by_document: Whether to group results by document (default: False)
        
        Returns:
            List[Dict[str, Any]]: List of retrieved documents with their metadata.
                Each dictionary contains:
                - 'content': The document content
                - 'score': Relevance score
                - 'metadata': Additional metadata
        """
        try:
            # Extract additional parameters
            filter_dict = kwargs.get('filter_dict')
            include_metadata = kwargs.get('include_metadata', True)
            group_by_document = kwargs.get('group_by_document', False)
            
            # Search in vector store
            results, scores = self.vector_store.search_texts(
                query_text=query,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            # Format results
            formatted_results = []
            for result, score in zip(results, scores):
                formatted_result = {
                    'content': result.get('content', ''),
                    'score': float(score),
                    'chunk_id': result.get('chunk_id'),
                    'document_id': result.get('document_id'),
                    'chunk_index': result.get('chunk_index'),
                    'total_chunks': result.get('total_chunks')
                }
                
                if include_metadata:
                    formatted_result['metadata'] = result
                
                formatted_results.append(formatted_result)
            
            # Group by document if requested
            if group_by_document:
                return self._group_results_by_document(formatted_results)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error querying RAG system: {e}")
            return []
    
    def _group_results_by_document(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group query results by document.
        
        Args:
            results: List of query results
            
        Returns:
            List[Dict[str, Any]]: Grouped results by document
        """
        doc_groups = {}
        
        for result in results:
            doc_id = result.get('document_id')
            if doc_id not in doc_groups:
                doc_groups[doc_id] = {
                    'document_id': doc_id,
                    'document_metadata': self.documents.get(doc_id, {}),
                    'chunks': [],
                    'best_score': 0.0,
                    'total_chunks': 0
                }
            
            doc_groups[doc_id]['chunks'].append(result)
            doc_groups[doc_id]['best_score'] = max(doc_groups[doc_id]['best_score'], result['score'])
            doc_groups[doc_id]['total_chunks'] += 1
        
        # Convert to list and sort by best score
        grouped_results = list(doc_groups.values())
        grouped_results.sort(key=lambda x: x['best_score'], reverse=True)
        
        return grouped_results
    
    def clear(self) -> bool:
        """
        Clear all documents from the RAG system.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        try:
            self.vector_store.clear()
            self.documents.clear()
            self.chunk_to_doc.clear()
            return True
        except Exception as e:
            print(f"Error clearing RAG system: {e}")
            return False
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Document metadata if found, None otherwise
        """
        return self.documents.get(document_id)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a specific document and all its chunks.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            if document_id not in self.documents:
                return False
            
            # Find all chunks for this document
            chunks_to_delete = []
            for chunk_id, doc_id in self.chunk_to_doc.items():
                if doc_id == document_id:
                    chunks_to_delete.append(chunk_id)
            
            # Delete chunks from vector store
            for chunk_id in chunks_to_delete:
                self.vector_store.delete_vector(chunk_id)
                del self.chunk_to_doc[chunk_id]
            
            # Delete document metadata
            del self.documents[document_id]
            
            return True
            
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dict[str, Any]: Statistics including document count, chunk count, etc.
        """
        vector_stats = self.vector_store.get_stats()
        
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunk_to_doc),
            'embedding_model': self.embedding_model,
            'index_type': self.index_type,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'vector_store_stats': vector_stats
        }
    
    def save(self, path: str) -> bool:
        """
        Save the RAG system to disk.
        
        Args:
            path: Path where to save the RAG system
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save vector store
            vector_store_path = os.path.join(path, "vector_store")
            if not self.vector_store.save(vector_store_path):
                return False
            
            # Save RAG-specific data
            rag_data = {
                'embedding_model': self.embedding_model,
                'index_type': self.index_type,
                'dimension': self.dimension,
                'normalize': self.normalize,
                'allow_different_dimensions': self.allow_different_dimensions,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'metadata_fields': self.metadata_fields,
                'documents': self.documents,
                'chunk_to_doc': self.chunk_to_doc
            }
            
            rag_file = os.path.join(path, "rag_data.json")
            with open(rag_file, 'w', encoding='utf-8') as f:
                json.dump(rag_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error saving RAG system: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load the RAG system from disk.
        
        Args:
            path: Path from where to load the RAG system
            
        Returns:
            bool: True if load was successful, False otherwise
        """
        try:
            # Load vector store
            vector_store_path = os.path.join(path, "vector_store")
            if not self.vector_store.load(vector_store_path):
                return False
            
            # Load RAG-specific data
            rag_file = os.path.join(path, "rag_data.json")
            with open(rag_file, 'r', encoding='utf-8') as f:
                rag_data = json.load(f)
            
            # Restore configuration
            self.embedding_model = rag_data['embedding_model']
            self.index_type = rag_data['index_type']
            self.dimension = rag_data['dimension']
            self.normalize = rag_data['normalize']
            self.allow_different_dimensions = rag_data['allow_different_dimensions']
            self.chunk_size = rag_data['chunk_size']
            self.chunk_overlap = rag_data['chunk_overlap']
            self.metadata_fields = rag_data['metadata_fields']
            
            # Restore data
            self.documents = rag_data['documents']
            self.chunk_to_doc = rag_data['chunk_to_doc']
            
            
            return True
            
        except Exception as e:
            print(f"Error loading RAG system: {e}")
            return False
    
    def search_similar_documents(self, document_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document.
        
        Args:
            document_id: ID of the document to find similar documents for
            top_k: Number of similar documents to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents
        """
        try:
            if document_id not in self.documents:
                return []
            
            # Get the first chunk of the document as query
            document_chunks = [chunk_id for chunk_id, doc_id in self.chunk_to_doc.items() 
                             if doc_id == document_id]
            
            if not document_chunks:
                return []
            
            # Use the first chunk as query
            first_chunk_id = document_chunks[0]
            first_chunk_vector = self.vector_store.get_vector(first_chunk_id)
            
            if first_chunk_vector is None:
                return []
            
            # Search for similar vectors
            results, scores = self.vector_store.search(
                query_vector=first_chunk_vector,
                top_k=top_k * 2  # Get more results to filter out the same document
            )
            
            # Filter out chunks from the same document and group by document
            similar_docs = {}
            for result, score in zip(results, scores):
                chunk_doc_id = result.get('document_id')
                if chunk_doc_id != document_id:
                    if chunk_doc_id not in similar_docs:
                        similar_docs[chunk_doc_id] = {
                            'document_id': chunk_doc_id,
                            'document_metadata': self.documents.get(chunk_doc_id, {}),
                            'best_score': score,
                            'chunks_found': 0
                        }
                    similar_docs[chunk_doc_id]['chunks_found'] += 1
                    similar_docs[chunk_doc_id]['best_score'] = max(
                        similar_docs[chunk_doc_id]['best_score'], score
                    )
            
            # Convert to list and sort by score
            similar_docs_list = list(similar_docs.values())
            similar_docs_list.sort(key=lambda x: x['best_score'], reverse=True)
            
            return similar_docs_list[:top_k]
            
        except Exception as e:
            print(f"Error finding similar documents: {e}")
            return []
