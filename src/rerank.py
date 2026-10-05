"""
Stage 2 - Node 3: RERANK
Runs the local Cross-Encoder (ms-marco-MiniLM-L-6-v2) on the sub-question
and candidate chunks. Re-ranks candidates by joint semantic cross-attention
and selects the top 5 highest-scoring passages.

Memory-Optimized for Cloud Deployment:
- Lazy loads PyTorch & Cross-Encoder only when reranking is executed
- Caps PyTorch threads to 1 to stay well within 512MB RAM limits
- Automatic fallback to Gemini neural reranker if memory is constrained
"""

import json
from typing import Dict, Any, List

from src.state import ResearchAgentState
from src.config import RERANKER_MODEL_NAME, STAGE2_RERANK_K, GEMINI_MODEL
from src.gemini_client import generate_text

# Singleton Cross-Encoder model (None or "gemini_fallback" or CrossEncoder instance)
_cross_encoder: Any = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        import os
        # In cloud environments (Render, Railway, Docker) or low-memory tiers (< 512MB RAM),
        # use the zero-RAM, instant Gemini Neural Reranker to avoid 80MB Hugging Face downloads and OOM.
        is_cloud = bool(os.getenv("RENDER") or os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("PORT") or os.getenv("DOCKER"))
        backend = os.getenv("RERANKER_BACKEND", "gemini" if is_cloud else "auto").strip().lower()

        if backend == "gemini":
            print("[RERANK] Cloud environment detected. Using fast zero-RAM Gemini Neural Reranker.")
            _cross_encoder = "gemini_fallback"
            return _cross_encoder

        try:
            # Prevent PyTorch from allocating large memory blocks or thread pools
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            import torch
            torch.set_num_threads(1)
            from sentence_transformers import CrossEncoder

            print(f"[RERANK] Loading local Cross-Encoder: '{RERANKER_MODEL_NAME}' (1 CPU thread)...")
            _cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
            print("[RERANK] Local Cross-Encoder loaded successfully.")
        except Exception as e:
            print(f"[RERANK WARNING] Could not load local Cross-Encoder ({e}). Using Gemini neural reranking fallback.")
            _cross_encoder = "gemini_fallback"
    return _cross_encoder


def _gemini_neural_rerank(sub_question: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Zero-RAM Fallback: Uses Gemini to compute cross-attention relevance scores
    for candidates when running in low-memory environments (< 512MB RAM).
    """
    print("[RERANK] Running Gemini Neural Reranker fallback...")
    candidate_prompts = []
    for idx, c in enumerate(candidates):
        snippet = c.get("text", "")[:280].replace("\n", " ")
        candidate_prompts.append(f"Passage [{idx}]: {snippet}")

    prompt = f"""You are a Cross-Encoder Neural Reranker.
Query: "{sub_question}"

Score each passage on relevance from -5.0 (irrelevant) to +5.0 (highly relevant, directly answers query).
Format your response as a valid JSON array of objects:
[
  {{"index": 0, "score": 2.45}},
  {{"index": 1, "score": -1.20}}
]

Passages:
{chr(10).join(candidate_prompts)}

Return ONLY valid JSON:"""

    try:
        response_text = generate_text(prompt, model=GEMINI_MODEL, temperature=0.0)
        clean_json = response_text.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()
        
        scores_data = json.loads(clean_json)
        score_map = {item["index"]: float(item["score"]) for item in scores_data if "index" in item and "score" in item}

        scored = []
        for idx, cand in enumerate(candidates):
            item = dict(cand)
            item["cross_encoder_score"] = float(score_map.get(idx, 0.0))
            scored.append(item)
        return scored
    except Exception as e:
        print(f"[RERANK ERROR] Gemini reranker fallback failed ({e}), sorting by FAISS cosine similarity.")
        scored = []
        for cand in candidates:
            item = dict(cand)
            # Map cosine [0, 1] to pseudo logit [-2, +3]
            cos_score = float(item.get("faiss_score", 0.5))
            item["cross_encoder_score"] = (cos_score - 0.5) * 6.0
            scored.append(item)
        return scored


def rerank_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Local cross-encoder reranking of candidate chunks.
    """
    sub_question = state["current_sub_question"]
    candidates = state.get("retrieved_candidates", [])

    print(f"\n==================================================")
    print(f"  [NODE 3: RERANK] Neural Cross-Encoder Reranking")
    print(f"==================================================")
    print(f"Scoring {len(candidates)} candidates against: \"{sub_question}\"")

    if not candidates:
        print("[WARNING] No candidate chunks available to rerank!")
        return {"reranked_chunks": []}

    model = get_cross_encoder()

    if model == "gemini_fallback":
        scored_candidates = _gemini_neural_rerank(sub_question, candidates)
    else:
        try:
            # Form (query, text) pairs
            pairs = [(sub_question, c["text"]) for c in candidates]
            scores = model.predict(pairs)

            scored_candidates = []
            for cand, score in zip(candidates, scores):
                item = dict(cand)
                item["cross_encoder_score"] = float(score)
                scored_candidates.append(item)
        except Exception as e:
            print(f"[RERANK WARNING] Local predict failed with memory/runtime error ({e}). Falling back to Gemini.")
            scored_candidates = _gemini_neural_rerank(sub_question, candidates)

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
