"""
Stage 2 - Node 1: PLAN
Formulates the research sub-question for the current hop:
- Hop 0: Uses the original user query verbatim.
- Retry hops (hop > 0): Analyzes previous draft answers and unverified claims,
  then queries Gemini to rewrite the query targeting the missing evidence.
"""

from typing import Dict, Any, List
from src.state import ResearchAgentState
from src.gemini_client import generate_text
from src.config import GEMINI_MODEL

def plan_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Determines the sub-question to retrieve evidence for.
    """
    hop = state.get("hop_count", 0)
    original_query = state["original_query"]
    query_history = list(state.get("query_history", []))

    print(f"\n==================================================")
    print(f"  [NODE 1: PLAN] Executing Hop {hop}")
    print(f"==================================================")

    if hop == 0:
        # First hop: directly use the original query
        sub_question = original_query
        print(f"Initial Hop: Sub-question is original query -> \"{sub_question}\"")
    else:
        # Retry hop: identify gaps from previous attempt
        draft_answer = state.get("draft_answer", "")
        verifications = state.get("verification_results", [])
        
        # Summarize unverified or missing claims
        unverified_claims = [
            f"- Claim: \"{v['claim_text']}\" (Failed reason: {v['reason']})"
            for v in verifications if not v.get("is_entailed", False)
        ]
        
        gap_summary = "\n".join(unverified_claims) if unverified_claims else "Previous answer was incomplete or unverified."

        plan_prompt = f"""You are an academic research query planner in an Agentic RAG system.
The original user research question is:
"{original_query}"

Previous search queries already tried:
{query_history}

Previous draft answer:
\"\"\"{draft_answer}\"\"\"

Evidence gaps and unverified claims from previous hop:
{gap_summary}

TASK:
Formulate a new, specific search query / sub-question targeting the missing evidence or specific terminology needed from academic papers.
Focus on keywords and concepts that were missing or unverified.

CRITICAL: Return ONLY the rewritten query text. Do not wrap in quotes or add explanatory notes."""

        print(f"Reformulating query targeting missing evidence...")
        raw_sub = generate_text(
            prompt=plan_prompt,
            model=GEMINI_MODEL,
            temperature=0.2
        ).strip().strip('"').strip("'")
        sub_question = raw_sub if len(raw_sub) >= 3 else original_query
        print(f"Reformulated Sub-Question: \"{sub_question}\"")

    query_history.append(sub_question)

    return {
        "current_sub_question": sub_question,
        "query_history": query_history
    }
