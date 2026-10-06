"""
Stage 2 - Node 7: REPORT
Generates the final comprehensive research report:
- Itemizes every claim with strict verification flags: [VERIFIED] or [UNVERIFIED — reason].
- Lists hop count, evidence sufficiency verdict, and query planning history.
- Provides a clickable bibliography of real academic papers and arXiv URLs.
"""

from typing import Dict, Any, List
from src.state import ResearchAgentState

def report_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Formats final audited report with per-claim flags and source bibliography.
    """
    query = state["original_query"]
    hop_count = state.get("hop_count", 0)
    max_hops = state.get("max_hops", 3)
    sufficient = state.get("evidence_sufficient", False)
    verifications = state.get("verification_results", [])
    reranked_chunks = state.get("reranked_chunks", [])
    query_history = state.get("query_history", [])
    draft_answer = state.get("draft_answer", "")

    lines = []
    lines.append("# PROVENANCE // Academic Research Audit Dossier\n")
    lines.append(f"**Research Question:** {query}  ")
    lines.append(f"**Total Hops Executed:** {hop_count} / {max_hops}  ")
    verdict_label = "[SUFFICIENT]" if sufficient else "[INSUFFICIENT LITERATURE EVIDENCE]"
    lines.append(f"**Evidence Sufficiency:** {verdict_label}  ")
    if query_history:
        lines.append(f"**Query Trajectory:** " + " -> ".join([f"`{q}`" for q in query_history]) + "  ")
    lines.append("\n---\n")

    lines.append("## Audited Findings & Verified Claims\n")
    if verifications:
        for idx, v in enumerate(verifications, 1):
            cit_str = "".join([f"[{c}]" for c in v.get("citations", [])])
            if v.get("is_entailed"):
                lines.append(f"- [VERIFIED] {v.get('claim_text', '')} {cit_str}")
            else:
                lines.append(f"- [UNVERIFIED] {v.get('claim_text', '')} {cit_str}")
                lines.append(f"  - Audit Reason: {v.get('reason', 'Evidence does not entail claim')}")
    else:
        lines.append(f"> **Academic Scope Advisory:**\n> {draft_answer}\n")
        lines.append("No factual claims could be strictly entailed from the retrieved academic literature.")

    # Document any refuted / unverified assertions from earlier hops (e.g. Hop 0)
    hop_history = state.get("hop_history", [])
    earlier_unverified = []
    for h in hop_history:
        h_idx = h.get("hop_index", 0)
        # Collect unverified claims from previous hops before the final convergence
        if h_idx < hop_count - 1 or (h_idx == 0 and hop_count > 1):
            for v in h.get("verification_results", []):
                if not v.get("is_entailed"):
                    earlier_unverified.append({
                        "hop": h_idx,
                        "claim_text": v.get("claim_text", ""),
                        "citations": v.get("citations", []),
                        "reason": v.get("reason", "Evidence does not entail claim")
                    })

    if earlier_unverified:
        lines.append("\n---\n")
        lines.append("## Multi-Hop Self-Correction & Refuted Hypotheses Log\n")
        lines.append("*The following candidate assertions were investigated in earlier reasoning loops, flagged as unverified by the NLI auditor, and triggered loopback query reformulation:*\n")
        for u in earlier_unverified:
            cit_str = "".join([f"[{c}]" for c in u.get("citations", [])])
            lines.append(f"- [REFUTED IN HOP {u['hop']}] {u['claim_text']} {cit_str}")
            lines.append(f"  - Audit Reason: {u['reason']}")

    lines.append("\n---\n")
    lines.append("## Working Academic Evidence Sources & Bibliography\n")

    # Collect unique papers from reranked chunks
    seen_urls = set()
    source_idx = 1
    for chunk in reranked_chunks:
        url = chunk.get("paper_url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        s_num = chunk.get("source_num", source_idx)
        score_info = f" (Cross-Encoder: {chunk['cross_encoder_score']:+.4f})" if "cross_encoder_score" in chunk else ""
        lines.append(f"[{s_num}] **{chunk.get('paper_title')}**{score_info}")
        is_arxiv = "arxiv.org" in (url or "")
        url_label = "arXiv URL" if is_arxiv else "Paper URL / DOI"
        lines.append(f"- **{url_label}:** {url}")
        lines.append(f"- **Paper ID:** `{chunk.get('paper_id')}`\n")
        source_idx += 1

    final_report = "\n".join(lines)

    try:
        print("\n" + final_report + "\n")
    except Exception:
        try:
            print("\n" + final_report.encode("ascii", errors="replace").decode("ascii") + "\n")
        except Exception:
            pass

    return {
        "final_report": final_report
    }


