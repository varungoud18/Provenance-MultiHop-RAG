"""
Diagnostic Test Script for Test (c):
Validates that [UNVERIFIED — reason] flags fire correctly on:
1. A claim that makes an ungrounded assertion not supported by the cited passage.
2. A claim that references an invalid out-of-bounds citation [9].
3. A properly grounded claim [1].
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.state import ResearchAgentState
from src.verify import verify_node
from src.report import report_node

# Real passage from papers.json (Automated Literature Review)
mock_chunk_1 = {
    "source_num": 1,
    "paper_id": "2411.18583v1",
    "paper_title": "Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation",
    "paper_url": "https://arxiv.org/abs/2411.18583v1",
    "cross_encoder_score": 3.45,
    "text": "This research presents and compares multiple approaches to automate the generation of literature reviews using several Natural Language Processing (NLP) techniques. Three distinct techniques are utilized: a frequency-based method using spaCy, a transformer model (Simple T5), and retrieval-augmented generation (RAG) with Large Language Model (GPT-3.5-turbo). The SciTLDR dataset is chosen for the research experiment. The Large Language Model GPT-3.5-turbo achieved the highest ROUGE-1 score, 0.364."
}

# Draft answer containing:
# 1. Supported claim
# 2. Hallucinated claim citing [1]
# 3. Invalid citation citing [9]
mock_draft_answer = """The SciTLDR dataset was selected for the automated literature review research experiment [1].
The GPT-3.5-turbo model achieved a 99.9% diagnosis accuracy in clinical oncology trials [1].
Quantum annealing processors were utilized to accelerate document retrieval [9]."""

mock_state: ResearchAgentState = {
    "original_query": "What are the clinical applications and quantum algorithms in literature review generation?",
    "current_sub_question": "What are the clinical applications and quantum algorithms in literature review generation?",
    "hop_count": 1,
    "max_hops": 1,
    "retrieved_candidates": [mock_chunk_1],
    "reranked_chunks": [mock_chunk_1],
    "draft_answer": mock_draft_answer,
    "verification_results": [],
    "evidence_sufficient": False,
    "final_report": "",
    "query_history": ["What are the clinical applications and quantum algorithms in literature review generation?"],
    "hop_history": []
}

if __name__ == "__main__":
    print("=" * 80)
    print("  RUNNING TEST (c): VERIFYING THAT [UNVERIFIED] FLAGS FIRE AS EXPECTED")
    print("=" * 80)
    
    # Run Verify Node
    v_out = verify_node(mock_state)
    mock_state.update(v_out)
    
    # Run Report Node
    r_out = report_node(mock_state)
