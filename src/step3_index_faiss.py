"""
Stage 1 - Step 3: Vector Indexing with FAISS and Gemini Embeddings
1. Loads chunked abstracts from data/chunks.json.
2. Embeds all chunk texts using Gemini's text-embedding-004 model.
3. L2-normalizes vectors so Inner Product equals Cosine Similarity.
4. Builds a local faiss.IndexFlatIP index and writes to data/faiss_index.bin.
5. Saves data/chunk_metadata.json in the identical row order (row i = chunk i).
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss

from src.config import (
    CHUNKS_FILE,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    GEMINI_EMBEDDING_MODEL
)
from src.gemini_client import embed_texts

def build_faiss_index(
    chunks_path: Path = CHUNKS_FILE,
    index_path: Path = FAISS_INDEX_FILE,
    metadata_path: Path = METADATA_FILE
) -> faiss.IndexFlatIP:
    """
    Reads chunks.json, generates normalized embeddings via Gemini,
    creates and saves a FAISS Inner Product (Cosine) index, and saves ordered metadata.
    """
    print(f"\n==================================================")
    print(f"  Step 3: Embedding Chunks & Building FAISS Index")
    print(f"==================================================")
    print(f"Loading chunks from: {chunks_path}")

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file '{chunks_path}' not found. Please run Step 2 (step2_chunk_papers.py) first!"
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    if not chunks:
        raise ValueError(f"Chunks file '{chunks_path}' is empty!")

    print(f"Total chunks to embed: {len(chunks)}")
    print(f"Using Gemini Embedding Model: '{GEMINI_EMBEDDING_MODEL}'")
    print("Generating embeddings via Google Gemini API (with rate-limit backoff)...")

    # Extract all texts to embed
    texts_to_embed = [chunk["text"] for chunk in chunks]

    # Task type RETRIEVAL_DOCUMENT for indexing documents in Gemini embedding models
    raw_embeddings = embed_texts(
        texts=texts_to_embed,
        model=GEMINI_EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
        batch_size=16
    )

    # Convert to float32 numpy array
    vectors = np.array(raw_embeddings, dtype=np.float32)
    num_vectors, dimension = vectors.shape
    print(f"Successfully generated {num_vectors} embeddings of dimension {dimension}.")

    # L2-normalize vectors so that Inner Product (IndexFlatIP) is exact Cosine Similarity
    print("Normalizing vectors (L2-norm) for cosine similarity calculation...")
    faiss.normalize_L2(vectors)

    # Create Inner Product Index
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    print(f"FAISS index built. Total indexed vectors: {index.ntotal}")

    # Save FAISS index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to: {index_path}")

    # Save metadata mapping in exact row order
    # row i in FAISS index -> metadata[i]
    metadata_list = []
    for i, chunk in enumerate(chunks):
        metadata_entry = {
            "faiss_row_id": i,
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "paper_title": chunk["paper_title"],
            "paper_url": chunk["paper_url"],
            "chunk_index": chunk["chunk_index"],
            "word_count": chunk["word_count"],
            "text": chunk["text"]
        }
        metadata_list.append(metadata_entry)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"Saved ordered chunk metadata to: {metadata_path}")
    print("Indexing complete!\n")

    return index

if __name__ == "__main__":
    build_faiss_index()
