import faiss
import pandas as pd
import numpy as np
import os
import tiktoken
import time
from sentence_transformers import SentenceTransformer

def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):  
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files

def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens

def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content

def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results

def encode_texts_with_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2"):
    """
    Encode a list of texts using a sentence transformer model
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

if __name__ == "__main__":
    # Start timing for the entire indexing process
    indexing_start_time = time.time()
    
    # 1. Read the text files in the path provided
    print("Step 1: Reading text files...")
    file_reading_start = time.time()
    WEEK_LIST = find_txt_files("dataset/LiHua-World/data/LiHuaWorld/LiHua-World")

    texts = []
    for WEEK in WEEK_LIST:
        id = WEEK_LIST.index(WEEK)
        with open(WEEK) as f:
            try:
                texts.append(f.read())
            except:
                pass
    
    file_reading_time = time.time() - file_reading_start
    print(f"Loaded {len(texts)} text files in {file_reading_time*1000:.1f} milliseconds")
    
    # 2. Use the chunking function to chunk the text data in the files
    print("Step 2: Chunking text data...")
    chunking_start = time.time()
    all_chunks = []
    chunk_metadata = []
    
    for file_idx, text in enumerate(texts):
        chunks = chunking_by_token_size(text)
        for chunk in chunks:
            all_chunks.append(chunk["content"])
            chunk_metadata.append({
                "file_index": file_idx,
                "chunk_order_index": chunk["chunk_order_index"],
                "tokens": chunk["tokens"]
            })
    
    chunking_time = time.time() - chunking_start
    print(f"Created {len(all_chunks)} chunks from {len(texts)} files in {chunking_time*1000:.1f} milliseconds")
    
    # 3. Encode the chunked texts using the encoding function and an embedding model
    print("Step 3: Encoding chunks with embedding model...")
    embedding_start = time.time()
    embeddings = encode_texts_with_embeddings(all_chunks)
    embedding_time = time.time() - embedding_start
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Encoding completed in {embedding_time*1000:.1f} milliseconds")
    
    # 4. Add the chunks into numpy array using faiss library
    print("Step 4: Creating FAISS index...")
    faiss_start = time.time()
    
    # Convert embeddings to float32 for FAISS
    embeddings = embeddings.astype(np.float32)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    
    # Add vectors to the index
    index.add(embeddings)
    
    
    faiss_time = time.time() - faiss_start
    print(f"FAISS index created with {index.ntotal} vectors")
    print(f"Index dimension: {index.d}")
    print(f"FAISS indexing completed in {faiss_time*1000:.1f} milliseconds")
    
    # Optional: Save the index and metadata
    print("Step 5: Saving index and metadata...")
    saving_start = time.time()
    
    faiss.write_index(index, "faiss_index.idx")
    
    # Save metadata as JSON for later reference
    import json
    with open("chunk_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)
    
    saving_time = time.time() - saving_start
    print("Index and metadata saved successfully!")
    print(f"Saving completed in {saving_time*1000:.1f} milliseconds")
    
    # Calculate total indexing time
    total_indexing_time = time.time() - indexing_start_time
    print(f"\n📊 INDEXING PERFORMANCE SUMMARY:")
    print(f"   File reading: {file_reading_time*1000:.1f} milliseconds")
    print(f"   Text chunking: {chunking_time*1000:.1f} milliseconds")
    print(f"   Embedding generation: {embedding_time*1000:.1f} milliseconds")
    print(f"   FAISS indexing: {faiss_time*1000:.1f} milliseconds")
    print(f"   File saving: {saving_time*1000:.1f} milliseconds")
    print(f"   ⏱️  TOTAL INDEXING TIME: {total_indexing_time*1000:.1f} milliseconds")
    
    # 5. Retrieve the closest chunk to a specific query
    print(f"\n🔍 RETRIEVAL PERFORMANCE TEST")
    query_text = 'What does LiHua predict will happen in "The Rings of Power"?'
    print(f"Searching for closest chunks to: '{query_text}'")
    
    # Time the retrieval process
    retrieval_start = time.time()
    
    # Encode the query text
    query_encoding_start = time.time()
    query_embedding = encode_texts_with_embeddings([query_text])
    query_embedding = query_embedding.astype(np.float32)
    query_encoding_time = time.time() - query_encoding_start
    
    # Search for the closest chunks
    search_start = time.time()
    k = 5  # Number of closest chunks to retrieve
    distances, indices = index.search(query_embedding, k)
    search_time = time.time() - search_start
    
    total_retrieval_time = time.time() - retrieval_start
    
    print(f"\nTop {k} closest chunks:")
    print("-" * 50)
    
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(all_chunks):
            chunk_content = all_chunks[idx]
            metadata = chunk_metadata[idx]
            
            print(f"\n{i+1}. Distance: {distance:.4f}")
            print(f"   File Index: {metadata['file_index']}")
            print(f"   Chunk Index: {metadata['chunk_order_index']}")
            print(f"   Tokens: {metadata['tokens']}")
            print(f"   Content Preview: {chunk_content[:200]}...")
            print("-" * 50)
        else:
            print(f"\n{i+1}. Invalid index: {idx}")
    
    # Show the most similar chunk in full
    if len(indices[0]) > 0 and indices[0][0] < len(all_chunks):
        most_similar_idx = indices[0][0]
        most_similar_chunk = all_chunks[most_similar_idx]
        most_similar_metadata = chunk_metadata[most_similar_idx]
        
        print(f"\n🎯 MOST SIMILAR CHUNK:")
        print(f"Distance: {distances[0][0]:.4f}")
        print(f"File Index: {most_similar_metadata['file_index']}")
        print(f"Chunk Index: {most_similar_metadata['chunk_order_index']}")
        print(f"Tokens: {most_similar_metadata['tokens']}")
        print(f"Full Content:\n{most_similar_chunk}")
        print("-" * 50)
        # Display retrieval performance summary
    print(f"\n📊 RETRIEVAL PERFORMANCE SUMMARY:")
    print(f"   Query encoding: {query_encoding_time*1000:.1f} milliseconds")
    print(f"   FAISS search: {search_time*1000:.1f} milliseconds")
    print(f"   ⏱️  TOTAL RETRIEVAL TIME: {total_retrieval_time*1000:.1f} milliseconds")
    print(f"   📈 Search speed: {k} results in {search_time*1000:.1f} milliseconds ({k/(search_time*1000):.1f} results/millisecond)")
    
