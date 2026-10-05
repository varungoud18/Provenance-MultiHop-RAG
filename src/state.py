"""
Stage 2: LangGraph State Definition
Defines the shared state dictionary passed across all nodes in the agent loop.
"""

from typing import TypedDict, List, Dict, Any, Optional

class VerificationItem(TypedDict):
    claim_id: int
    raw_sentence: str
    claim_text: str
    citations: List[int]
    is_entailed: bool
    verdict: str  # "YES" or "NO"
    reason: str
    cited_passages: List[Dict[str, Any]]

class ResearchAgentState(TypedDict):
    # Core query inputs
    original_query: str
    current_sub_question: str
    
    # Loop counters & control
    hop_count: int
    max_hops: int
    
    # Retrieved & reranked evidence
    retrieved_candidates: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    
    # Generation & verification
    draft_answer: str
    verification_results: List[VerificationItem]
    evidence_sufficient: bool
    
    # Final output & logs
    final_report: str
    query_history: List[str]
    hop_history: List[Dict[str, Any]]
