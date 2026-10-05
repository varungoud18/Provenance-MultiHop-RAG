"""
Stage 2 - Node 3: RERANK
Runs the local Cross-Encoder (ms-marco-MiniLM-L-6-v2) on the sub-question
and candidate chunks. Re-ranks candidates by joint semantic cross-attention
and selects the top 5 highest-scoring passages.
"""

from typing import Dict, Any, List
from sentence_transformers import CrossEncoder

from src.state import ResearchAgentState
from src.config import RERANKER_MODEL_NAME, STAGE2_RERANK_K

# Singleton Cross-Encoder model
_cross_encoder: CrossEncoder = None

def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"Loading local Cross-Encoder: '{RERANKER_MODEL_NAME}'...")
        _cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
    return _cross_encoder


def rerank_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Local cross-encoder reranking of candidate chunks.
    """
    sub_question = state["current_sub_question"]
    candidates = state.get("retrieved_candidates", [])

    print(f"\n==================================================")
    print(f"  [NODE 3: RERANK] Local Cross-Encoder Reranking")
    print(f"==================================================")
    print(f"Scoring {len(candidates)} candidates against: \"{sub_question}\"")

    if not candidates:
        print("[WARNING] No candidate chunks available to rerank!")
        return {"reranked_chunks": []}

    model = get_cross_encoder()

    # Form (query, text) pairs
    pairs = [(sub_question, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    # Attach scores
    scored_candidates = []
    for cand, score in zip(candidates, scores):
        item = dict(cand)
        item["cross_encoder_score"] = float(score)
        scored_candidates.append(item)

    # Sort descending by cross-encoder score
    scored_candidates.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    # Select top-k
    top_k = scored_candidates[:STAGE2_RERANK_K]

    # Re-assign canonical 1-based source numbers for strict citing [1..k]
    for idx, chunk in enumerate(top_k, start=1):
        chunk["source_num"] = idx

    print(f"Reranked Top {len(top_k)} Chunks:")
    print("-" * 75)
    for c in top_k:
        print(f"[{c['source_num']}] CE Score: {c['cross_encoder_score']:+.4f} (FAISS Cosine: {c['faiss_score']:.4f})")
        print(f"    Paper : {c['paper_title']}")
        print(f"    URL   : {c['paper_url']} (Chunk: {c['chunk_id']})")
        print(f"    Text  : {c['text'][:110]}...\n")

    return {
        "reranked_chunks": top_k
    }
