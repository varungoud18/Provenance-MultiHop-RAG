"""
Stage 2 - Node 2: RETRIEVE
Embeds the current sub-question, queries the local FAISS index,
and pulls a broad candidate set (approx. 20 chunks) to feed the Cross-Encoder.
"""

from typing import Dict, Any, List
import numpy as np
import faiss

from src.state import ResearchAgentState
from src.config import (
    FAISS_INDEX_FILE,
    METADATA_FILE,
    STAGE2_CANDIDATE_K,
    GEMINI_EMBEDDING_MODEL
)
from src.gemini_client import embed_texts
from src.step4_ask_single_hop import load_index_and_metadata

# Cached in-memory index & metadata with automatic mtime freshness validation
_index: faiss.IndexFlatIP = None
_metadata: List[Dict[str, Any]] = None
_last_mtime: float = 0.0

def get_faiss_resources(force_reload: bool = False):
    global _index, _metadata, _last_mtime
    current_mtime = FAISS_INDEX_FILE.stat().st_mtime if FAISS_INDEX_FILE.exists() else 0.0
    if force_reload or _index is None or _metadata is None or current_mtime != _last_mtime:
        _index, _metadata = load_index_and_metadata(FAISS_INDEX_FILE, METADATA_FILE)
        _last_mtime = current_mtime
    return _index, _metadata


def retrieve_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Broad vector retrieval of up to 20 candidate chunks.
    """
    sub_question = state["current_sub_question"]
    print(f"\n==================================================")
    print(f"  [NODE 2: RETRIEVE] Broad Vector Search")
    print(f"==================================================")
    print(f"Retrieving candidate passages for: \"{sub_question}\"")

    index, metadata = get_faiss_resources()
    target_k = min(STAGE2_CANDIDATE_K, index.ntotal)

    # Embed query using Gemini
    query_emb = embed_texts(
        texts=[sub_question],
        model=GEMINI_EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY"
    )
    query_vec = np.array(query_emb, dtype=np.float32)
    faiss.normalize_L2(query_vec)

    # Search FAISS
    scores, indices = index.search(query_vec, target_k)

    candidates = []
    for rank, (score, row_idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if row_idx < 0 or row_idx >= len(metadata):
            continue
        chunk = dict(metadata[row_idx])
        chunk["faiss_rank"] = rank
        chunk["faiss_score"] = float(score)
        candidates.append(chunk)

    print(f"Retrieved {len(candidates)} wide candidate chunks from FAISS.")
    if candidates:
        top_cand = candidates[0]
        print(f"  Top Candidate Score: {top_cand['faiss_score']:.4f} | \"{top_cand['paper_title']}\"")

    return {
        "retrieved_candidates": candidates
    }
