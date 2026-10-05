import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Data artifact filepaths
PAPERS_FILE = DATA_DIR / "papers.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.bin"
METADATA_FILE = DATA_DIR / "chunk_metadata.json"
EMBEDDING_CACHE_FILE = DATA_DIR / "embedding_cache.json"

# Gemini API settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2").strip()


# Text Chunking Settings (Word-based sliding window)
CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 20

# Search & Retrieval Settings
STAGE1_TOP_K = 5
STAGE2_CANDIDATE_K = 20
STAGE2_RERANK_K = 5

# Local Cross-Encoder Reranker model
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
