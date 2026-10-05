"""
Stage 2 - Node 4: REASON
Generates a grounded academic answer using the top 5 reranked chunks.
Explicitly instructs the LLM to write ONE claim per sentence with immediate inline citations
so each statement can be independently audited in the subsequent verification step.
"""

from typing import Dict, Any, List
from src.state import ResearchAgentState
from src.gemini_client import generate_text
from src.config import GEMINI_MODEL

def build_reasoning_prompt(query: str, sources: List[Dict[str, Any]]) -> str:
    """Constructs prompt enforcing one-claim-per-sentence structure for verification."""
    source_blocks = []
    for s in sources:
        num = s["source_num"]
        title = s["paper_title"]
        url = s["paper_url"]
        text = s["text"]
        source_blocks.append(
            f"Source [{num}]:\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Passage: {text}\n"
        )
    sources_text = "\n".join(source_blocks)

    max_src_num = len(sources)

    prompt = f"""You are an academic research assistant answering a research question based strictly on provided academic paper excerpts.

CRITICAL INSTRUCTIONS:
1. Answer the research question relying ONLY on the facts explicitly stated in the Sources below.
2. Structure your answer with clear, concise statements.
3. Write strictly ONE factual claim per sentence, followed immediately by its citation tag like [1] or [1][2].
4. Avoid long compound sentences with multiple assertions so that each sentence can be independently verified.
5. Only cite source numbers between [1] and [{max_src_num}].
6. If the sources do not provide enough evidence to answer the question or missing aspects, state explicitly:
   "The retrieved papers do not provide sufficient information to answer [specific aspect]."
   DO NOT guess or extrapolate beyond what is written in the passages.

=== SOURCES ===
{sources_text}

=== RESEARCH QUESTION ===
{query}

=== GROUNDED ANSWER (One factual claim per sentence with citation) ==="""
    return prompt


def reason_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Generates a cited draft answer from reranked chunks.
    """
    query = state["current_sub_question"]
    sources = state.get("reranked_chunks", [])

    print(f"\n==================================================")
    print(f"  [NODE 4: REASON] Generating Cited Answer")
    print(f"==================================================")
    print(f"Reasoning over {len(sources)} reranked passages for: \"{query}\"")

    if not sources:
        print("[WARNING] No reranked sources provided to reason node.")
        draft_answer = "The retrieved papers do not provide sufficient information to answer this question."
        return {"draft_answer": draft_answer}

    prompt = build_reasoning_prompt(query, sources)
    draft_answer = generate_text(
        prompt=prompt,
        model=GEMINI_MODEL,
        temperature=0.1
    )

    print("\nDraft Answer Generated:")
    print("-" * 75)
    print(draft_answer)
    print("-" * 75)

    return {
        "draft_answer": draft_answer
    }
