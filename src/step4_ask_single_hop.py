"""
Stage 1 - Step 4: Single-Hop RAG Query Pipeline
1. Takes a user research question.
2. Embeds the question using Gemini embedding model (task_type=RETRIEVAL_QUERY).
3. Searches the local FAISS index for top 5 closest chunks (Cosine Similarity).
4. Formulates a strict citation-prompt for Gemini LLM.
5. Prints the cited answer and an explicit citation key mapping [1], [2], ...
   to real arXiv paper titles and clickable URLs.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

from src.config import (
    FAISS_INDEX_FILE,
    METADATA_FILE,
    STAGE1_TOP_K,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL
)
from src.gemini_client import embed_texts, generate_text

def load_index_and_metadata(
    index_path: Path = FAISS_INDEX_FILE,
    metadata_path: Path = METADATA_FILE
) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """Loads the FAISS vector index and ordered metadata from disk."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index file '{index_path}' not found. Please run Step 3 (step3_index_faiss.py) first!"
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file '{metadata_path}' not found. Please run Step 3 (step3_index_faiss.py) first!"
        )

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def retrieve_top_k(
    query: str,
    index: faiss.IndexFlatIP,
    metadata: List[Dict[str, Any]],
    top_k: int = STAGE1_TOP_K
) -> List[Dict[str, Any]]:
    """
    Embeds the search query, normalizes it, and queries FAISS for top-k chunks.
    Returns list of chunk dicts annotated with similarity scores and source rankings [1..k].
    """
    # Embed query using Gemini
    query_embeddings = embed_texts(
        texts=[query],
        model=GEMINI_EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY"
    )
    query_vec = np.array(query_embeddings, dtype=np.float32)
    faiss.normalize_L2(query_vec)

    # Perform Inner Product search (equivalent to Cosine Similarity on normalized vectors)
    actual_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, actual_k)

    retrieved = []
    for rank, (score, row_idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if row_idx < 0 or row_idx >= len(metadata):
            continue
        chunk_data = dict(metadata[row_idx])
        chunk_data["source_num"] = rank
        chunk_data["similarity_score"] = float(score)
        retrieved.append(chunk_data)

    return retrieved


def build_rag_prompt(question: str, sources: List[Dict[str, Any]]) -> str:
    """
    Constructs a strict grounding prompt requiring inline citations and refusal to hallucinate.
    """
    context_blocks = []
    for s in sources:
        num = s["source_num"]
        title = s["paper_title"]
        url = s["paper_url"]
        text = s["text"]
        context_blocks.append(
            f"Source [{num}]:\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {text}\n"
        )
    context_str = "\n".join(context_blocks)

    prompt = f"""You are an academic research assistant answering a research question based strictly on provided academic paper excerpts.

CRITICAL INSTRUCTIONS:
1. Answer the research question relying ONLY on the facts explicitly stated in the provided Sources below.
2. Every claim or factual statement you write MUST have an inline citation tag like [1], [2], etc., identifying exactly which Source provides the evidence.
3. If multiple sources support a sentence, cite them together, for example [1][3].
4. If the provided sources do NOT fully answer the question, or if certain details are missing, you MUST state explicitly: "The retrieved papers do not provide sufficient information to answer [specific aspect]." DO NOT speculate or fill gaps using outside knowledge.
5. Do NOT make claims without citing one of the numbered sources.

=== RETRIEVED SOURCES ===
{context_str}

=== RESEARCH QUESTION ===
{question}

=== YOUR GROUNDED ANSWER WITH CITATIONS ==="""

    return prompt


def ask_question(question: str, top_k: int = STAGE1_TOP_K) -> Dict[str, Any]:
    """
    Executes the full single-hop RAG pipeline for a given question.
    """
    print(f"\n==================================================")
    print(f"  Step 4: Single-Hop RAG Question Answering")
    print(f"==================================================")
    print(f"Question: \"{question}\"")
    print(f"Top-K Retrieval: {top_k} chunks")

    index, metadata = load_index_and_metadata()
    print(f"Loaded FAISS index ({index.ntotal} vectors) and metadata ({len(metadata)} chunks).")

    print("Embedding query and retrieving closest chunks via FAISS...")
    sources = retrieve_top_k(question, index, metadata, top_k=top_k)

    print(f"\nRetrieved {len(sources)} Most Relevant Chunks:")
    print("-" * 75)
    for s in sources:
        print(f"[{s['source_num']}] Similarity: {s['similarity_score']:.4f} | {s['paper_title']}")
        print(f"    URL: {s['paper_url']} (Chunk ID: {s['chunk_id']})")
        print(f"    Snippet: {s['text'][:120]}...\n")

    print(f"Sending prompt to Gemini ({GEMINI_MODEL}) with strict citation grounding...")
    prompt = build_rag_prompt(question, sources)
    answer = generate_text(prompt=prompt, model=GEMINI_MODEL, temperature=0.1)

    print("\n" + "=" * 75)
    print("  GROUNDED ANSWER")
    print("=" * 75)
    print(answer)

    print("\n" + "=" * 75)
    print("  CITATION KEY (Source Papers & Working URLs)")
    print("=" * 75)
    for s in sources:
        print(f"[{s['source_num']}] {s['paper_title']}")
        print(f"    URL     : {s['paper_url']}")
        print(f"    Chunk ID: {s['chunk_id']}")
        print(f"    Cos-Sim : {s['similarity_score']:.4f}\n")

    return {
        "question": question,
        "sources": sources,
        "answer": answer
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Hop RAG Question Answering.")
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="The research question to answer."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=STAGE1_TOP_K,
        help=f"Number of chunks to retrieve (default: {STAGE1_TOP_K})"
    )
    args = parser.parse_args()

    if args.question:
        q = args.question
    else:
        # Interactive prompt
        print("\nEnter your research question (e.g. 'How does retrieval augmented generation automate literature reviews?'):")
        q = input("> ").strip()
        if not q:
            q = "How does retrieval augmented generation automate literature reviews?"
            print(f"Using default question: \"{q}\"")

    ask_question(question=q, top_k=args.top_k)
