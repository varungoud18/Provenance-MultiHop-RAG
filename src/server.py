"""
Web Backend Server for AI Academic Research Agent (Stage 3)
FastAPI application providing:
- Real-time SSE streaming for LangGraph agent execution
- arXiv paper ingestion API (Fetch -> Chunk -> FAISS Index)
- Corpus metadata and status endpoints
- Static file serving for modern dark-mode frontend
"""

import asyncio
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure standard output and error never crash with charmap encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import (
    DATA_DIR,
    PAPERS_FILE,
    CHUNKS_FILE,
    FAISS_INDEX_FILE,
    GEMINI_MODEL,
    GEMINI_EMBEDDING_MODEL,
    RERANKER_MODEL_NAME
)
from src.step1_fetch_arxiv import fetch_arxiv_papers
from src.step2_chunk_papers import process_papers_into_chunks
from src.step3_index_faiss import build_faiss_index
from src.agent import build_research_agent_graph
from src.state import ResearchAgentState

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"

app = FastAPI(
    title="PROVENANCE Academic Research Agent",
    description="Autonomous Academic Agent & Citation Auditor with ArXiv, FAISS, Cross-Encoder, and Gemini",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lock for ingestion to prevent race conditions on vector index
ingest_lock = threading.Lock()


# =====================================================================
# Request / Response Models
# =====================================================================

class IngestRequest(BaseModel):
    query: str
    max_results: int = 10


# =====================================================================
# Status & Corpus Endpoints
# =====================================================================

@app.get("/api/status")
def get_system_status() -> Dict[str, Any]:
    """Returns the current state of papers, chunks, FAISS vector index, and models."""
    papers_count = 0
    chunks_count = 0
    index_ready = False
    last_indexed = None

    if PAPERS_FILE.exists():
        try:
            with open(PAPERS_FILE, "r", encoding="utf-8") as f:
                papers_data = json.load(f)
                papers_count = len(papers_data)
        except Exception:
            pass

    if CHUNKS_FILE.exists():
        try:
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                chunks_count = len(chunks_data)
        except Exception:
            pass

    if FAISS_INDEX_FILE.exists():
        index_ready = True
        mtime = FAISS_INDEX_FILE.stat().st_mtime
        last_indexed = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))

    return {
        "status": "ready" if index_ready else "needs_indexing",
        "papers_count": papers_count,
        "chunks_count": chunks_count,
        "index_ready": index_ready,
        "last_indexed": last_indexed,
        "gemini_model": GEMINI_MODEL,
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL_NAME
    }


@app.get("/api/papers")
def get_ingested_papers() -> Dict[str, Any]:
    """Returns list of papers currently indexed in data/papers.json."""
    if not PAPERS_FILE.exists():
        return {"papers": []}
    try:
        with open(PAPERS_FILE, "r", encoding="utf-8") as f:
            papers = json.load(f)
        return {"papers": papers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest")
def trigger_ingestion(req: IngestRequest) -> Dict[str, Any]:
    """
    Executes the full harvest & indexing pipeline:
    1. Fetch papers from arXiv Atom API
    2. Sliding-window chunking
    3. Gemini text embeddings & FAISS index creation
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    if not ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="An ingestion job is already running")

    try:
        t0 = time.time()
        # 1. Fetch papers
        papers = fetch_arxiv_papers(query=req.query.strip(), max_results=req.max_results)
        if not papers:
            return {
                "status": "warning",
                "message": f"arXiv returned 0 papers for query '{req.query}'. Index was not modified.",
                "papers_count": 0,
                "chunks_count": 0
            }

        # 2. Chunk
        chunks = process_papers_into_chunks()

        # 3. Index FAISS
        build_faiss_index()

        elapsed = time.time() - t0
        return {
            "status": "success",
            "message": f"Successfully fetched {len(papers)} papers, generated {len(chunks)} chunks, and built FAISS vector index.",
            "papers_count": len(papers),
            "chunks_count": len(chunks),
            "elapsed_seconds": round(elapsed, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        ingest_lock.release()


# =====================================================================
# Real-Time SSE Agent Execution Stream
# =====================================================================

def run_agent_thread(initial_state: ResearchAgentState, event_queue: queue.Queue, auto_harvest: bool = False):
    """
    Runs the compiled LangGraph workflow in a dedicated worker thread,
    emitting step events into the thread-safe queue.
    If auto_harvest is True, harvests fresh papers from arXiv and re-indexes FAISS before executing.
    """
    try:
        if auto_harvest:
            event_queue.put({
                "type": "harvest_start",
                "message": f"Auto-harvesting live arXiv papers matching: '{initial_state['original_query']}'..."
            })
            with ingest_lock:
                try:
                    papers = fetch_arxiv_papers(query=initial_state["original_query"], max_results=10)
                    if papers:
                        chunks = process_papers_into_chunks()
                        build_faiss_index()
                        event_queue.put({
                            "type": "harvest_complete",
                            "papers_count": len(papers),
                            "chunks_count": len(chunks),
                            "message": f"Successfully indexed {len(chunks)} fresh chunks from {len(papers)} papers on arXiv."
                        })
                    else:
                        event_queue.put({
                            "type": "harvest_complete",
                            "papers_count": 0,
                            "chunks_count": 0,
                            "message": "arXiv returned no matching preprints; proceeding with indexed corpus."
                        })
                except Exception as harvest_err:
                    print(f"[HARVEST NOTICE] {harvest_err}")
                    event_queue.put({
                        "type": "harvest_complete",
                        "papers_count": 0,
                        "chunks_count": 0,
                        "message": f"Harvest bypassed; proceeding with existing indexed literature."
                    })

        app_graph = build_research_agent_graph()


        event_queue.put({
            "type": "init",
            "query": initial_state["original_query"],
            "max_hops": initial_state["max_hops"]
        })

        # Track cumulative state as graph streams
        accumulated_state = dict(initial_state)

        for step_output in app_graph.stream(initial_state):
            for node_name, node_update in step_output.items():
                accumulated_state.update(node_update)

                payload = {
                    "type": "node_update",
                    "node": node_name,
                    "hop": accumulated_state.get("hop_count", 0),
                    "sub_question": accumulated_state.get("current_sub_question"),
                    "evidence_sufficient": accumulated_state.get("evidence_sufficient", False),
                }

                if node_name == "plan":
                    payload["sub_question"] = node_update.get("current_sub_question")
                    payload["query_history"] = node_update.get("query_history", [])

                elif node_name == "retrieve":
                    cands = node_update.get("retrieved_candidates", [])
                    payload["candidate_count"] = len(cands)
                    payload["top_candidates"] = [
                        {
                            "chunk_id": c.get("chunk_id"),
                            "paper_title": c.get("paper_title"),
                            "cosine_similarity": round(float(c.get("similarity", 0.0)), 4),
                            "snippet": c.get("text", "")[:140] + "..."
                        }
                        for c in cands[:5]
                    ]

                elif node_name == "rerank":
                    reranked = node_update.get("reranked_chunks", [])
                    payload["reranked_count"] = len(reranked)
                    payload["reranked_chunks"] = [
                        {
                            "source_num": r.get("source_num"),
                            "paper_title": r.get("paper_title"),
                            "paper_url": r.get("paper_url"),
                            "cross_encoder_score": round(float(r.get("cross_encoder_score", 0.0)), 4),
                            "text": r.get("text")
                        }
                        for r in reranked
                    ]

                elif node_name == "reason":
                    payload["draft_answer"] = node_update.get("draft_answer", "")

                elif node_name == "verify":
                    verifs = node_update.get("verification_results", [])
                    payload["verifications"] = verifs
                    payload["verified_count"] = sum(1 for v in verifs if v.get("is_entailed"))
                    payload["total_claims"] = len(verifs)
                    payload["evidence_sufficient"] = node_update.get("evidence_sufficient", False)

                elif node_name == "controller":
                    payload["hop_count"] = node_update.get("hop_count", 0)
                    payload["hop_history"] = node_update.get("hop_history", [])

                elif node_name == "report":
                    payload["final_report"] = node_update.get("final_report", "")
                    payload["draft_answer"] = accumulated_state.get("draft_answer", "")
                    payload["original_query"] = accumulated_state.get("original_query", "")
                    payload["evidence_sufficient"] = accumulated_state.get("evidence_sufficient", False)
                    payload["verifications"] = accumulated_state.get("verification_results", [])
                    payload["reranked_chunks"] = accumulated_state.get("reranked_chunks", [])
                    payload["hop_history"] = accumulated_state.get("hop_history", [])
                    payload["query_history"] = accumulated_state.get("query_history", [])

                event_queue.put(payload)

        event_queue.put({"type": "complete", "status": "success"})

    except Exception as e:
        event_queue.put({"type": "error", "error": str(e)})


@app.get("/api/research/stream")
async def stream_research(
    query: str = Query(..., min_length=3),
    max_hops: int = Query(3, ge=1, le=5),
    auto_harvest: bool = Query(True)
):
    """
    Streams LangGraph agent execution step-by-step using Server-Sent Events (SSE).
    Supports on-demand arXiv auto-harvesting for zero-friction dynamic research.
    """
    if not auto_harvest and not FAISS_INDEX_FILE.exists():
        raise HTTPException(status_code=400, detail="FAISS index not found. Please enable Live arXiv Auto-Harvest or ingest papers first.")

    initial_state: ResearchAgentState = {
        "original_query": query,
        "current_sub_question": query,
        "hop_count": 0,
        "max_hops": max_hops,
        "retrieved_candidates": [],
        "reranked_chunks": [],
        "draft_answer": "",
        "verification_results": [],
        "evidence_sufficient": False,
        "final_report": "",
        "query_history": [],
        "hop_history": []
    }

    event_queue: queue.Queue = queue.Queue()
    worker_thread = threading.Thread(
        target=run_agent_thread,
        args=(initial_state, event_queue, auto_harvest),
        daemon=True
    )
    worker_thread.start()

    async def sse_event_publisher():
        while True:
            try:
                # Use asyncio.to_thread to wait on queue without blocking async event loop
                event = await asyncio.to_thread(event_queue.get, timeout=45.0)
                event_data = json.dumps(event)
                yield f"data: {event_data}\n\n"

                if event.get("type") in ("complete", "error"):
                    break
            except queue.Empty:
                # Keep-alive heartbeat comment
                yield ": heartbeat\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                break

    return StreamingResponse(
        sse_event_publisher(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =====================================================================
# Serve Frontend Static Assets
# =====================================================================

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = WEB_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Web UI directory exists, but index.html is not created yet."}
