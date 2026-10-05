"""
Stage 1 - Step 2: Paper Abstract Chunker
Splits paper abstracts into overlapping word-level chunks (100-150 words with ~20 words overlap).
Attaches full provenance metadata (paper id, title, URL, unique chunk_id) to every chunk.
Saves to data/chunks.json.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from src.config import (
    PAPERS_FILE,
    CHUNKS_FILE,
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS
)

def chunk_text_sliding_window(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS
) -> List[str]:
    """
    Splits text into overlapping chunks using a word-based sliding window.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        
        # Avoid creating a tiny trailing orphan chunk (< 15 words) if we already have chunks
        if len(chunk_words) < 15 and chunks:
            # Merge with previous chunk
            chunks[-1] = chunks[-1] + " " + " ".join(chunk_words)
            break

        chunks.append(" ".join(chunk_words))
        
        if end >= len(words):
            break
        start += step

    return chunks


def process_papers_into_chunks(
    input_file: Path = PAPERS_FILE,
    output_file: Path = CHUNKS_FILE,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS
) -> List[Dict[str, Any]]:
    """
    Loads papers from papers.json, chunks each abstract, attaches citation metadata,
    and writes chunks.json.
    """
    print(f"\n==================================================")
    print(f"  Step 2: Splitting Abstracts into Overlapping Chunks")
    print(f"==================================================")
    print(f"Input file: {input_file}")
    print(f"Chunk size: ~{chunk_size} words | Overlap: ~{overlap} words")

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input papers file '{input_file}' not found. Please run Step 1 (step1_fetch_arxiv.py) first!"
        )

    with open(input_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    all_chunks = []
    chunk_counter = 0

    for paper in papers:
        abstract = paper.get("abstract", "")
        text_chunks = chunk_text_sliding_window(abstract, chunk_size=chunk_size, overlap=overlap)

        for idx, text in enumerate(text_chunks):
            chunk_counter += 1
            chunk_record = {
                "chunk_id": f"{paper['id']}_c{idx + 1}",
                "global_chunk_num": chunk_counter,
                "paper_id": paper["id"],
                "paper_title": paper["title"],
                "paper_url": paper["url"],
                "chunk_index": idx + 1,
                "total_chunks_in_paper": len(text_chunks),
                "word_count": len(text.split()),
                "text": text
            }
            all_chunks.append(chunk_record)

    # Save to data/chunks.json
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Successfully processed {len(papers)} papers into {len(all_chunks)} chunks.")
    print(f"Saved chunks to: {output_file}\n")

    # Preview sample chunk
    if all_chunks:
        sample = all_chunks[0]
        print("Sample Chunk Preview:")
        print("-" * 75)
        print(f"Chunk ID    : {sample['chunk_id']}")
        print(f"Paper Title : {sample['paper_title']}")
        print(f"Paper URL   : {sample['paper_url']}")
        print(f"Word Count  : {sample['word_count']} words")
        print(f"Text Snippet: {sample['text'][:150]}...\n")

    return all_chunks

if __name__ == "__main__":
    process_papers_into_chunks()
