#!/usr/bin/env python3
"""
Example usage of the FaissRag implementation.

This script demonstrates how to:
1. Initialize a FaissRag system
2. Insert documents
3. Query the system
4. Save and load the system
"""

# Handle imports for both direct execution and package import
import os
import sys

# Add parent directory to path if running directly
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ragstore.FaissRag import FaissRag
except ImportError:
    # Fallback for direct import when running from ragstore directory
    from FaissRag import FaissRag


def main():
    """Main example function."""
    
    # Initialize the RAG system
    print("Initializing FaissRag system...")
    rag = FaissRag(
        embedding_model="all-MiniLM-L6-v2",
        index_type="flat",
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Sample documents
    documents = [
        {
            "content": """
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines that work and react like humans. Some of the activities 
            computers with artificial intelligence are designed for include speech recognition, 
            learning, planning, and problem solving. AI has applications in various fields 
            including healthcare, finance, transportation, and entertainment.
            """,
            "title": "Introduction to AI",
            "category": "technology",
            "author": "Tech Expert"
        },
        {
            "content": """
            Machine Learning is a subset of artificial intelligence that provides systems 
            the ability to automatically learn and improve from experience without being 
            explicitly programmed. Machine learning focuses on the development of computer 
            programs that can access data and use it to learn for themselves. The process 
            of learning begins with observations or data, such as examples, direct experience, 
            or instruction, in order to look for patterns in data and make better decisions 
            in the future based on the examples that we provide.
            """,
            "title": "Machine Learning Basics",
            "category": "technology",
            "author": "ML Researcher"
        },
        {
            "content": """
            Natural Language Processing (NLP) is a subfield of linguistics, computer science, 
            and artificial intelligence concerned with the interactions between computers 
            and human language, in particular how to program computers to process and 
            analyze large amounts of natural language data. Challenges in natural language 
            processing frequently involve speech recognition, natural language understanding, 
            and natural language generation.
            """,
            "title": "Natural Language Processing",
            "category": "technology",
            "author": "NLP Specialist"
        },
        {
            "content": """
            Deep Learning is part of a broader family of machine learning methods based on 
            artificial neural networks with representation learning. Learning can be 
            supervised, semi-supervised or unsupervised. Deep learning architectures such 
            as deep neural networks, deep belief networks, recurrent neural networks and 
            convolutional neural networks have been applied to fields including computer 
            vision, speech recognition, natural language processing, audio recognition, 
            social network filtering, machine translation, bioinformatics, drug design, 
            medical image analysis, material inspection and board game programs.
            """,
            "title": "Deep Learning Overview",
            "category": "technology",
            "author": "DL Expert"
        }
    ]
    
    # Insert documents
    print("Inserting documents...")
    success = rag.insert(documents)
    if success:
        print(f"Successfully inserted {len(documents)} documents")
    else:
        print("Failed to insert documents")
        return
    
    # Get system statistics
    print("\nSystem Statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        if key != 'vector_store_stats':
            print(f"  {key}: {value}")
    
    # Example queries
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain natural language processing",
        "What are neural networks?",
        "Tell me about deep learning applications"
    ]
    
    print("\n" + "="*50)
    print("QUERYING THE RAG SYSTEM")
    print("="*50)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        # Perform query
        results = rag.query(query, top_k=3, include_metadata=True)
        print(len(results))
        
        if results:
            for j, result in enumerate(results, 1):
                print(f"\nResult {j}:")
                print(f"  Content: {result['content'][:150]}...")
                print(f"  Score: {result['score']:.4f}")
                print(f"  Document ID: {result['document_id']}")
                if 'metadata' in result:
                    title = result['metadata'].get('doc_title', 'N/A')
                    author = result['metadata'].get('doc_author', 'N/A')
                    print(f"  Title: {title}")
                    print(f"  Author: {author}")
        else:
            print("  No results found")
    
    # Example of grouping results by document
    print("\n" + "="*50)
    print("GROUPED RESULTS EXAMPLE")
    print("="*50)
    
    query = "machine learning and artificial intelligence"
    print(f"\nQuery: {query}")
    print("-" * 40)
    
    grouped_results = rag.query(
        query, 
        top_k=5, 
        group_by_document=True,
        include_metadata=True
    )
    
    for i, group in enumerate(grouped_results, 1):
        print(f"\nDocument Group {i}:")
        print(f"  Document ID: {group['document_id']}")
        print(f"  Best Score: {group['best_score']:.4f}")
        print(f"  Chunks Found: {group['total_chunks']}")
        
        if 'document_metadata' in group:
            title = group['document_metadata'].get('title', 'N/A')
            author = group['document_metadata'].get('author', 'N/A')
            print(f"  Title: {title}")
            print(f"  Author: {author}")
    
    # Example of finding similar documents
    print("\n" + "="*50)
    print("SIMILAR DOCUMENTS EXAMPLE")
    print("="*50)
    
    # Get the first document ID
    first_doc_id = list(rag.documents.keys())[0]
    print(f"\nFinding documents similar to: {first_doc_id}")
    
    similar_docs = rag.search_similar_documents(first_doc_id, top_k=3)
    
    for i, similar_doc in enumerate(similar_docs, 1):
        print(f"\nSimilar Document {i}:")
        print(f"  Document ID: {similar_doc['document_id']}")
        print(f"  Similarity Score: {similar_doc['best_score']:.4f}")
        print(f"  Chunks Found: {similar_doc['chunks_found']}")
        
        if 'document_metadata' in similar_doc:
            title = similar_doc['document_metadata'].get('title', 'N/A')
            print(f"  Title: {title}")
    
    # Save and load example
    print("\n" + "="*50)
    print("SAVE AND LOAD EXAMPLE")
    print("="*50)
    
    # Save the system
    save_path = "./example_rag_system"
    print(f"\nSaving RAG system to: {save_path}")
    if rag.save(save_path):
        print("Successfully saved RAG system")
    else:
        print("Failed to save RAG system")
        return
    
    # Create a new RAG system and load the saved one
    print("\nCreating new RAG system and loading saved data...")
    new_rag = FaissRag()
    print(save_path)
    if new_rag.load(save_path):
        print("Successfully loaded RAG system")
        
        # Test query on loaded system
        test_query = "What is AI?"
        results = new_rag.query(test_query, top_k=2)
        
        print(f"\nTest query on loaded system: '{test_query}'")
        if results:
            print(f"Found {len(results)} results")
            for result in results:
                print(f"  - {result['content'][:100]}... (score: {result['score']:.4f})")
        else:
            print("No results found")
    else:
        print("Failed to load RAG system")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 