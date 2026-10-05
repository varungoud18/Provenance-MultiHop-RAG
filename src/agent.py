"""
Stage 2: Full Agentic Loop Orchestrator with LangGraph
Wires the autonomous research agent loop:
PLAN -> RETRIEVE -> RERANK -> REASON -> VERIFY -> CONTROLLER -> (PLAN or REPORT)
"""

import argparse
import sys
import time
from typing import Dict, Any

# Ensure immediate unbuffered terminal output
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from langgraph.graph import StateGraph, START, END

from src.state import ResearchAgentState
from src.plan import plan_node
from src.retrieve import retrieve_node
from src.rerank import rerank_node
from src.reason import reason_node
from src.verify import verify_node
from src.controller import controller_node, should_continue
from src.report import report_node

def build_research_agent_graph():
    """
    Constructs and compiles the cyclic LangGraph workflow.
    """
    workflow = StateGraph(ResearchAgentState)

    # Register all modular nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("reason", reason_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("controller", controller_node)
    workflow.add_node("report", report_node)

    # Wire edges
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "reason")
    workflow.add_edge("reason", "verify")
    workflow.add_edge("verify", "controller")

    # Conditional branching at CONTROLLER
    workflow.add_conditional_edges(
        "controller",
        should_continue,
        {
            "plan": "plan",
            "report": "report"
        }
    )

    workflow.add_edge("report", END)

    # Compile the graph
    app = workflow.compile()
    return app


def run_research_agent(query: str, max_hops: int = 3, auto_harvest: bool = False) -> Dict[str, Any]:
    """
    Executes the full agentic loop for a given research query.
    If auto_harvest is True, pulls fresh papers from arXiv and builds FAISS index before executing.
    """
    start_time = time.time()
    print("=" * 80)
    print("      INITIALIZING AUTONOMOUS AGENTIC RAG SYSTEM (STAGE 2)")
    print("=" * 80)
    print(f"Research Question : \"{query}\"")
    print(f"Max Hop Limit     : {max_hops}")
    print(f"Auto-Harvest Mode : {'ENABLED (Live arXiv Search)' if auto_harvest else 'DISABLED (Using Local Corpus)'}")
    print("=" * 80)

    if auto_harvest:
        from src.step1_fetch_arxiv import fetch_arxiv_papers
        from src.step2_chunk_papers import process_papers_into_chunks
        from src.step3_index_faiss import build_faiss_index

        print(f"\n[AUTO-HARVEST] Fetching live arXiv papers matching: '{query}'...")
        fetch_arxiv_papers(query=query, max_results=10)
        process_papers_into_chunks()
        build_faiss_index()

    app = build_research_agent_graph()

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

    final_state = app.invoke(initial_state)
    elapsed = time.time() - start_time
    print(f"\n[EXECUTION COMPLETE] Total execution time: {elapsed:.2f} seconds.")
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Academic Research Agent (LangGraph Loop).")
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="The academic research question to investigate."
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=3,
        help="Maximum number of iterative retrieval/reasoning hops (default: 3)."
    )
    parser.add_argument(
        "--auto-harvest",
        action="store_true",
        default=False,
        help="Automatically harvest and index fresh arXiv papers matching the question before running."
    )
    args = parser.parse_args()

    if args.question:
        q = args.question
    else:
        print("\nEnter your research question (or press Enter for default):")
        q = input("> ").strip()
        if not q:
            q = "What specific models and datasets were evaluated for automated literature review generation?"
            print(f"Using default question: \"{q}\"")

    run_research_agent(query=q, max_hops=args.max_hops, auto_harvest=args.auto_harvest)
