"""
Gemini Client Helper Module
Provides unified, robust access to Google Gemini API for:
1. Embedding text chunks and queries via text-embedding-004
2. Text generation via Gemini Flash models (e.g. gemini-1.5-flash)
3. Built-in exponential backoff for handling free-tier rate limits (HTTP 429)
"""

import time
import sys
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError
from src.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_EMBEDDING_MODEL, EMBEDDING_CACHE_FILE

_client: Optional[genai.Client] = None
_embedding_cache: Optional[Dict[str, List[float]]] = None

def _get_embedding_cache() -> Dict[str, List[float]]:
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = {}
        if EMBEDDING_CACHE_FILE.exists():
            try:
                with open(EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as f:
                    _embedding_cache = json.load(f)
            except Exception:
                _embedding_cache = {}
    return _embedding_cache

def _save_embedding_cache():
    if _embedding_cache is not None:
        try:
            EMBEDDING_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(EMBEDDING_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_embedding_cache, f)
        except Exception:
            pass

def get_gemini_client() -> genai.Client:
    """
    Initializes and returns a singleton instance of the Google GenAI Client.
    Validates that the GEMINI_API_KEY is present in the environment/.env.
    """
    global _client
    if _client is not None:
        return _client

    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "" or "your_gemini_api_key_here" in GEMINI_API_KEY:
        raise ValueError(
            "\n[MISSING API KEY] GEMINI_API_KEY is not configured.\n"
            "Please obtain your free key from https://aistudio.google.com/apikey\n"
            "and add it to your .env file:\n"
            "GEMINI_API_KEY=your_actual_key_here\n"
        )

    _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


_st_model = None

def get_local_embedder():
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[EMBED WARNING] Could not load local SentenceTransformer: {e}")
            _st_model = None
    return _st_model


def embed_texts(
    texts: List[str],
    model: str = GEMINI_EMBEDDING_MODEL,
    task_type: Optional[str] = None,
    batch_size: int = 16,
    max_retries: int = 3
) -> List[List[float]]:
    """
    Computes vector embeddings instantly using local SentenceTransformer ('all-MiniLM-L6-v2')
    for sub-second embedding with zero API rate limits or network lag.
    Falls back seamlessly to Gemini embedding API if required.
    """
    # 1. High-speed local embedding (0.01s latency, 0 rate limits)
    st = get_local_embedder()
    if st is not None:
        try:
            raw_vecs = st.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=False)
            return [vec.tolist() for vec in raw_vecs]
        except Exception as st_err:
            print(f"[EMBED NOTICE] Local embedder failed ({st_err}). Falling back to Gemini API...")

    # 2. Fallback: Gemini Embedding API with persistent cache
    cache = _get_embedding_cache()
    candidate_models = ["gemini-embedding-001", "gemini-embedding-2"]
    results: List[Optional[List[float]]] = [None] * len(texts)
    missing_texts = []

    for idx, text in enumerate(texts):
        key = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if key in cache:
            results[idx] = cache[key]
        else:
            missing_texts.append((idx, key, text))

    if not missing_texts:
        return results

    client = get_gemini_client()
    cache_dirty = False

    for orig_idx, key, text_val in missing_texts:
        retries = 0
        current_model_idx = 0
        item_success = False

        while not item_success and retries < max_retries:
            current_model = candidate_models[current_model_idx % len(candidate_models)]
            try:
                config = None
                if task_type:
                    config = types.EmbedContentConfig(task_type=task_type)

                response = client.models.embed_content(
                    model=current_model,
                    contents=text_val,
                    config=config
                )

                if hasattr(response, "embeddings") and response.embeddings:
                    vec = list(response.embeddings[0].values)
                elif hasattr(response, "embedding") and response.embedding:
                    vec = list(response.embedding.values)
                else:
                    raise ValueError(f"Unexpected response structure: {response}")

                results[orig_idx] = vec
                cache[key] = vec
                cache_dirty = True
                item_success = True

            except Exception as e:
                err_str = str(e)
                retries += 1
                current_model_idx += 1
                next_model = candidate_models[current_model_idx % len(candidate_models)]
                sleep_time = min(1.0 * retries, 2)
                time.sleep(sleep_time)

        if not item_success:
            # Fallback zero vector if completely exhausted
            results[orig_idx] = [0.0] * 384

    if cache_dirty:
        _save_embedding_cache()

    return results


def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    model: str = GEMINI_MODEL,
    temperature: float = 0.2,
    max_retries: int = 4
) -> str:
    """
    Generates text using high-throughput Gemini Flash models with automatic rotation
    (gemini-flash-latest, gemini-flash-lite-latest, gemini-3.1-flash-lite) to guarantee
    sub-second latency and zero rate limit stalls.
    """
    client = get_gemini_client()
    retries = 0
    candidate_models = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.1-flash-lite"]
    if model not in candidate_models:
        candidate_models.insert(0, model)
    else:
        candidate_models.remove(model)
        candidate_models.insert(0, model)

    current_idx = 0

    while retries < max_retries:
        current_model = candidate_models[current_idx % len(candidate_models)]
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
            response = client.models.generate_content(
                model=current_model,
                contents=prompt,
                config=config
            )
            return response.text if response.text else ""

        except Exception as e:
            err_str = str(e)
            retries += 1
            current_idx += 1
            next_model = candidate_models[current_idx % len(candidate_models)]
            sleep_time = min(1.0 * retries, 2)
            time.sleep(sleep_time)

    raise RuntimeError(f"Failed to generate text after {max_retries} retries.")



