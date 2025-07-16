# RAG Store Implementation

This directory contains a FAISS-based RAG (Retrieval-Augmented Generation) implementation that extends the `BaseRag` abstract class.

## Overview

The `FaissRag` class provides a complete RAG system implementation using FAISS for vector storage and similarity search. It supports document insertion, querying, and management with advanced features like document chunking, metadata handling, and persistence.

## Features

### Core RAG Functionality
- **Document Insertion**: Support for single documents, lists of documents, and documents with metadata
- **Intelligent Chunking**: Automatic text splitting with configurable chunk size and overlap
- **Semantic Search**: Vector-based similarity search using FAISS
- **Metadata Support**: Rich metadata handling for documents and chunks
- **Result Grouping**: Option to group search results by document

### Advanced Features
- **Multiple FAISS Index Types**: Support for flat, IVF, and HNSW indices
- **Configurable Embedding Models**: Any sentence transformer model
- **Document Management**: Individual document retrieval and deletion
- **Similar Document Search**: Find documents similar to a given document
- **Persistence**: Save and load RAG systems to/from disk
- **Statistics**: Comprehensive system statistics and monitoring

### Configuration Options
- **Chunk Size**: Configurable text chunk size (default: 1000 characters)
- **Chunk Overlap**: Overlap between consecutive chunks (default: 200 characters)
- **Embedding Model**: Any sentence transformer model (default: "all-MiniLM-L6-v2")
- **Index Type**: FAISS index type (flat, ivf, hnsw)
- **Vector Normalization**: Optional L2 normalization
- **Metadata Fields**: Custom metadata field extraction

## Installation

The implementation requires the following dependencies:

```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install sentence-transformers
pip install numpy
```

## Usage

### Basic Usage

```python
from ragstore.FaissRag import FaissRag

# Initialize the RAG system
rag = FaissRag(
    embedding_model="all-MiniLM-L6-v2",
    index_type="flat",
    chunk_size=500,
    chunk_overlap=100
)

# Insert documents
documents = [
    "This is the first document about artificial intelligence.",
    "This is the second document about machine learning.",
    {
        "content": "This is a document with metadata.",
        "title": "AI Document",
        "author": "John Doe",
        "category": "technology"
    }
]

success = rag.insert(documents)

# Query the system
results = rag.query("What is artificial intelligence?", top_k=3)

for result in results:
    print(f"Content: {result['content']}")
    print(f"Score: {result['score']}")
    print(f"Document ID: {result['document_id']}")
```

### Advanced Usage

```python
# Initialize with custom configuration
rag = FaissRag(
    embedding_model="all-mpnet-base-v2",  # Different embedding model
    index_type="hnsw",                    # HNSW index for better performance
    chunk_size=1000,
    chunk_overlap=200,
    metadata_fields=["title", "author", "category"]
)

# Insert documents with metadata
documents = [
    {
        "content": "Long document about AI...",
        "title": "AI Overview",
        "author": "Expert",
        "category": "technology",
        "date": "2024-01-01"
    }
]

rag.insert(documents)

# Query with grouping
grouped_results = rag.query(
    "machine learning applications",
    top_k=5,
    group_by_document=True,
    include_metadata=True
)

# Find similar documents
similar_docs = rag.search_similar_documents("doc_id_123", top_k=3)

# Save and load the system
rag.save("my_rag_system")
new_rag = FaissRag()
new_rag.load("my_rag_system")
```

## API Reference

### FaissRag Class

#### Constructor Parameters

- `embedding_model` (str): Sentence transformer model name (default: "all-MiniLM-L6-v2")
- `index_type` (str): FAISS index type - "flat", "ivf", or "hnsw" (default: "flat")
- `dimension` (int, optional): Vector dimension (auto-detected if None)
- `normalize` (bool): Whether to normalize vectors (default: True)
- `allow_different_dimensions` (bool): Allow vectors of different dimensions (default: False)
- `chunk_size` (int): Size of text chunks (default: 1000)
- `chunk_overlap` (int): Overlap between chunks (default: 200)
- `metadata_fields` (List[str], optional): Metadata fields to extract

#### Methods

##### Core Methods (from BaseRag)

- `insert(documents)`: Insert documents into the knowledge base
- `query(query, top_k=5, **kwargs)`: Query the system for relevant documents
- `clear()`: Clear all documents from the system

##### Additional Methods

- `get_document(document_id)`: Retrieve a specific document by ID
- `delete_document(document_id)`: Delete a specific document and its chunks
- `search_similar_documents(document_id, top_k=5)`: Find documents similar to a given document
- `get_stats()`: Get system statistics
- `save(path)`: Save the RAG system to disk
- `load(path)`: Load the RAG system from disk

### Query Parameters

The `query` method accepts additional parameters:

- `filter_dict` (Dict): Metadata filters for search
- `include_metadata` (bool): Include full metadata in results (default: True)
- `group_by_document` (bool): Group results by document (default: False)

### Result Format

Query results contain:

- `content`: The document/chunk content
- `score`: Relevance score (higher is better)
- `chunk_id`: ID of the specific chunk
- `document_id`: ID of the parent document
- `chunk_index`: Index of the chunk within the document
- `total_chunks`: Total number of chunks in the document
- `metadata`: Full metadata (if include_metadata=True)

## Performance Considerations

### Index Types

- **Flat**: Best accuracy, slower for large datasets
- **IVF**: Good balance of speed and accuracy
- **HNSW**: Fastest, slightly lower accuracy

### Chunking Strategy

- **Small chunks**: Better precision, more results
- **Large chunks**: Better context, fewer results
- **Overlap**: Helps maintain context across chunks

### Embedding Models

- **all-MiniLM-L6-v2**: Fast, good quality (384 dimensions)
- **all-mpnet-base-v2**: Slower, better quality (768 dimensions)
- **all-MiniLM-L12-v2**: Balanced option (384 dimensions)

## Example Script

Run the example script to see the implementation in action:

```bash
cd ragstore
python example_usage.py
```

This will demonstrate:
- Document insertion with metadata
- Various query types
- Result grouping
- Similar document search
- Save/load functionality

## File Structure

```
ragstore/
├── FaissRag.py          # Main RAG implementation
├── example_usage.py     # Usage examples
└── README.md           # This file
```

## Dependencies

- `faiss-cpu` or `faiss-gpu`: Vector similarity search
- `sentence-transformers`: Text embedding models
- `numpy`: Numerical operations
- `interfaces.BaseRag`: Abstract base class
- `vectorDBs.FaissVectorStore`: Vector storage backend

## Extending the Implementation

The `FaissRag` class can be extended to add:

- **Hybrid Search**: Combine vector search with keyword search
- **Reranking**: Post-process results with more sophisticated ranking
- **Multi-modal Support**: Handle images, audio, etc.
- **Real-time Updates**: Incremental document updates
- **Distributed Storage**: Scale across multiple nodes
- **Custom Embeddings**: Support for custom embedding functions

## Troubleshooting

### Common Issues

1. **Memory Usage**: Large documents or many chunks can consume significant memory
2. **Index Performance**: Flat indices become slow with large datasets
3. **Embedding Model**: Some models may not be available or require additional setup
4. **Chunking**: Very short documents may not be chunked effectively

### Performance Tips

1. Use HNSW index for large datasets
2. Adjust chunk size based on your use case
3. Consider using smaller embedding models for speed
4. Monitor memory usage with large document collections 