#!/usr/bin/env python3
"""
Simple example runner for FaissRag.

Run this directly from the ragstore directory.
"""

import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import and use FaissRag
from interfaces.BaseRag import BaseRag
from vectorDBs.FaissVectorStore import FaissVectorStore

# Import our implementation
from FaissRag import FaissRag


def main():
    """Simple test of the FaissRag implementation."""
    print("Testing FaissRag implementation...")
    
    # Initialize the RAG system
    rag = FaissRag(
        embedding_model="all-MiniLM-L6-v2",
        index_type="flat",
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Test documents
    documents = [
        "Artificial Intelligence is a branch of computer science.",
        "Machine Learning is a subset of artificial intelligence.",
        {
            "content": "Natural Language Processing deals with human language.",
            "title": "NLP Overview",
            "category": "technology"
        }
    ]
    
    # Insert documents
    print("Inserting test documents...")
    success = rag.insert(documents)
    if success:
        print("✅ Documents inserted successfully")
    else:
        print("❌ Failed to insert documents")
        return
    
    # Test query
    print("Testing query functionality...")
    results = rag.query("What is artificial intelligence?", top_k=2)
    
    if results:
        print(f"✅ Query successful, found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"  Result {i}: {result['content'][:50]}... (score: {result['score']:.4f})")
    else:
        print("❌ No results found")
    
    # Test statistics
    print("Testing statistics...")
    stats = rag.get_stats()
    print(f"✅ Statistics: {stats['total_documents']} documents, {stats['total_chunks']} chunks")
    
    print("\n🎉 Test completed successfully!")


if __name__ == "__main__":
    main() 