#!/usr/bin/env python3
"""
Test script for the FaissRag implementation.

Run this from the project root directory to test the RAG system.
"""

from ragstore.FaissRag import FaissRag


def test_basic_functionality():
    """Test basic RAG functionality."""
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
    if not success:
        print("❌ Failed to insert documents")
        return False
    
    print("✅ Documents inserted successfully")
    
    # Test query
    print("Testing query functionality...")
    results = rag.query("What is artificial intelligence?", top_k=2)
    
    if results:
        print(f"✅ Query successful, found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"  Result {i}: {result['content'][:50]}... (score: {result['score']:.4f})")
    else:
        print("❌ No results found")
        return False
    
    # Test statistics
    print("Testing statistics...")
    stats = rag.get_stats()
    print(f"✅ Statistics: {stats['total_documents']} documents, {stats['total_chunks']} chunks")
    
    # Test clear
    print("Testing clear functionality...")
    rag.clear()
    stats_after_clear = rag.get_stats()
    if stats_after_clear['total_documents'] == 0:
        print("✅ Clear functionality works")
    else:
        print("❌ Clear functionality failed")
        return False
    
    print("\n🎉 All tests passed!")
    return True


if __name__ == "__main__":
    try:
        success = test_basic_functionality()
        if success:
            print("\n✅ FaissRag implementation is working correctly!")
        else:
            print("\n❌ Some tests failed!")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc() 