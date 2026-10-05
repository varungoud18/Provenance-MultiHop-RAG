"""
Stage 2 - Node 6: CONTROLLER
Increments the hop counter, logs hop history, and evaluates loop termination conditions:
- If evidence_sufficient == True: stop and proceed to REPORT.
- If hop_count >= max_hops: stop and proceed to REPORT (boundary check: 3 full attempts 0, 1, 2).
- Otherwise: loop back to PLAN with updated state.
"""

from typing import Dict, Any, Literal
from src.state import ResearchAgentState

def controller_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Updates loop bookkeeping after a verification pass.
    """
    current_hop = state.get("hop_count", 0)
    next_hop = current_hop + 1
    max_hops = state.get("max_hops", 3)
    sufficient = state.get("evidence_sufficient", False)
    verifications = state.get("verification_results", [])

    print(f"\n==================================================")
    print(f"  [NODE 6: CONTROLLER] Decision & Loop Control")
    print(f"==================================================")
    print(f"Completed Hop  : {current_hop} (Attempt {next_hop} of {max_hops})")
    print(f"Evidence Status: {'SUFFICIENT' if sufficient else 'INSUFFICIENT'}")

    # Record snapshot in hop_history
    hop_history = list(state.get("hop_history", []))
    hop_history.append({
        "hop_index": current_hop,
        "sub_question": state.get("current_sub_question"),
        "evidence_sufficient": sufficient,
        "verified_count": sum(1 for v in verifications if v.get("is_entailed", False)),
        "total_claims": len(verifications),
        "draft_answer": state.get("draft_answer"),
        "verification_results": verifications,
        "reranked_chunks": state.get("reranked_chunks", [])
    })

    return {
        "hop_count": next_hop,
        "hop_history": hop_history
    }


def should_continue(state: ResearchAgentState) -> Literal["plan", "report"]:
    """
    LangGraph Conditional Edge Router.
    Decides whether to continue looping to 'plan' or exit to 'report'.
    """
    hop_count = state.get("hop_count", 0)
    max_hops = state.get("max_hops", 3)
    sufficient = state.get("evidence_sufficient", False)

    if sufficient:
        print(f"-> Decision: Evidence is SUFFICIENT. Terminating loop and proceeding to REPORT.\n")
        return "report"

    if hop_count >= max_hops:
        print(f"-> Decision: Reached maximum hop limit ({hop_count}/{max_hops}). Terminating loop and proceeding to REPORT.\n")
        return "report"

    print(f"-> Decision: Evidence insufficient and hops remaining ({hop_count}/{max_hops}). Looping back to PLAN.\n")
    return "plan"
