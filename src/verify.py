"""
Stage 2 - Node 5: VERIFY
Splits draft answer into individual cited sentences/claims.
For every claim, fetches the cited chunk(s) and audits entailment with Gemini.
Supports unified batch verification for maximum speed and zero free-tier rate-limit throttling,
while evaluating every single claim with an individual YES/NO verdict and specific reason.
"""

import re
import json
from typing import Dict, Any, List, Tuple
from src.state import ResearchAgentState, VerificationItem
from src.gemini_client import generate_text
from src.config import GEMINI_MODEL


def parse_claims_from_answer(text: str) -> List[Dict[str, Any]]:
    """
    Robust claim extractor for markdown bullet points, numbered lists, and paragraphs.
    Preserves numbers with decimals (e.g. 0.364, GPT-3.5) and binds trailing citations.
    """
    extracted = []
    lines = text.split("\n")

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Ignore obvious meta/disclaimer lines that contain no citations
        if not re.search(r"\[\d+\]", line_clean):
            continue

        # Step 1: Normalize citations occurring right after period: "text. [1]" -> "text [1]."
        normalized_line = re.sub(r"\.\s*(\[\d+\]+)", r" \1.", line_clean)

        # Step 2: Strip leading markdown bullets like '*', '-', '1.'
        content = re.sub(r"^[*\-•]\s+", "", normalized_line)
        content = re.sub(r"^\d+\.\s+", "", content)

        # Step 3: Split into sentences, protecting decimals and abbreviations
        sentence_end_pattern = r"(?<!\d)(?<=[.!?])\s+(?=[A-Z*#\-])"
        raw_sentences = re.split(sentence_end_pattern, content)

        for s in raw_sentences:
            s_str = s.strip()
            if not s_str:
                continue

            citations = [int(m) for m in re.findall(r"\[(\d+)\]", s_str)]
            if not citations:
                continue

            # Strip citation brackets from claim text for entailment check
            claim_text = re.sub(r"\[\d+\]", "", s_str).strip()

            # Strip leading bold label if present, e.g. '**Method:** The model...' -> 'The model...'
            clean_statement = re.sub(r"^\*\*[^*]+:\*\*\s*", "", claim_text).strip()
            if not clean_statement:
                clean_statement = claim_text

            extracted.append({
                "raw_sentence": s_str,
                "claim_text": clean_statement,
                "citations": sorted(list(set(citations)))
            })

    return extracted


def batch_verify_claims(
    claims_to_check: List[Dict[str, Any]],
    source_map: Dict[int, Dict[str, Any]]
) -> List[Tuple[bool, str, str]]:
    """
    Audits all claims in a single API call to prevent hitting the 15 RPM free-tier rate limit,
    while evaluating each claim with an independent YES/NO verdict and explanation.
    """
    blocks = []
    for idx, c in enumerate(claims_to_check, start=1):
        claim_text = c["claim_text"]
        citations = c["citations"]
        passages = [f"Source [{cit}]: {source_map[cit]['text']}" for cit in citations if cit in source_map]
        combined = "\n".join(passages)
        blocks.append(
            f"[CLAIM {idx}]\n"
            f"Claim Statement: \"{claim_text}\"\n"
            f"Cited Evidence:\n{combined}\n"
        )

    claims_text = "\n".join(blocks)

    prompt = f"""You are an objective academic verification auditor.
Your job is to decide whether the provided Cited Evidence strictly ENTAILS (logically supports with direct textual facts) each Claim.

=== EVALUATION RULES ===
- Answer YES only if ALL statements in the Claim are directly supported by the passage.
- Answer NO if the Claim adds outside facts, exaggerates metrics, makes unmentioned claims, or contradicts the passage.

=== CLAIMS TO AUDIT ===
{claims_text}

=== AUDIT INSTRUCTIONS ===
For EACH claim from 1 to {len(claims_to_check)}, output your assessment strictly in the following format:
CLAIM <number>:
VERDICT: YES (or NO)
REASON: <concise explanation of why it is entailed or what specific fact is unsupported>"""

    response = generate_text(prompt=prompt, model=GEMINI_MODEL, temperature=0.0).strip()

    # Parse per-claim blocks
    results = []
    # Pattern to match CLAIM <n>: ... VERDICT: ... REASON: ...
    claim_sections = re.split(r"(?i)CLAIM\s+(\d+)\s*:", response)
    
    parsed_map = {}
    if len(claim_sections) > 1:
        for i in range(1, len(claim_sections), 2):
            c_num = int(claim_sections[i])
            c_body = claim_sections[i + 1].strip()
            
            verdict = "NO"
            reason = "No reason provided"
            is_entailed = False
            
            for line in c_body.split("\n"):
                line_s = line.strip()
                if line_s.upper().startswith("VERDICT:"):
                    v_val = line_s.split(":", 1)[1].strip().upper()
                    if "YES" in v_val:
                        verdict = "YES"
                        is_entailed = True
                    else:
                        verdict = "NO"
                        is_entailed = False
                elif line_s.upper().startswith("REASON:"):
                    reason = line_s.split(":", 1)[1].strip()
            
            parsed_map[c_num] = (is_entailed, verdict, reason)

    # Reconstruct in order
    for idx in range(1, len(claims_to_check) + 1):
        if idx in parsed_map:
            results.append(parsed_map[idx])
        else:
            # Fallback if specific claim block was missed
            results.append((False, "NO", "Could not parse audit block"))

    return results


def verify_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Audits every claim in the draft answer against its cited source chunks.
    """
    draft_answer = state.get("draft_answer", "")
    sources = state.get("reranked_chunks", [])

    print(f"\n==================================================")
    print(f"  [NODE 5: VERIFY] Per-Claim Entailment Verification")
    print(f"==================================================")

    # Map source_num -> source dict
    source_map = {s["source_num"]: s for s in sources}
    claims = parse_claims_from_answer(draft_answer)

    if not claims:
        print("[NOTICE] No cited claims found in draft answer.")
        return {
            "verification_results": [],
            "evidence_sufficient": False
        }


    print(f"Auditing {len(claims)} individual factual claims against evidence chunks...\n")

    # Filter out claims with out-of-bounds citations before LLM check
    valid_claims = []
    valid_indices = []
    pre_results: Dict[int, VerificationItem] = {}

    for idx, c in enumerate(claims, start=1):
        claim_text = c["claim_text"]
        citations = c["citations"]
        raw_sentence = c["raw_sentence"]

        invalid = [cit for cit in citations if cit not in source_map]
        if invalid:
            v_item: VerificationItem = {
                "claim_id": idx,
                "raw_sentence": raw_sentence,
                "claim_text": claim_text,
                "citations": citations,
                "is_entailed": False,
                "verdict": "NO",
                "reason": f"Invalid citation index {invalid} does not exist in retrieved sources.",
                "cited_passages": []
            }
            pre_results[idx] = v_item
            print(f"Claim {idx}: \"{claim_text[:65]}...\"")
            print(f"  Result: [FAIL - UNVERIFIED] Reason: {v_item['reason']}\n")
        else:
            valid_claims.append(c)
            valid_indices.append(idx)

    # Run unified audit on valid claims
    if valid_claims:
        audit_verdicts = batch_verify_claims(valid_claims, source_map)
        for original_idx, c, (is_entailed, verdict, reason) in zip(valid_indices, valid_claims, audit_verdicts):
            cited_sources = [source_map[cit] for cit in c["citations"]]
            v_item: VerificationItem = {
                "claim_id": original_idx,
                "raw_sentence": c["raw_sentence"],
                "claim_text": c["claim_text"],
                "citations": c["citations"],
                "is_entailed": is_entailed,
                "verdict": verdict,
                "reason": reason,
                "cited_passages": cited_sources
            }
            pre_results[original_idx] = v_item
            status_tag = "[PASS - ENTAILED]" if is_entailed else "[FAIL - UNVERIFIED]"
            print(f"Claim {original_idx}: \"{c['claim_text'][:65]}...\"")
            print(f"  Target Citations: {c['citations']}")
            print(f"  Result: {status_tag}")
            print(f"  Auditor Reason: {reason}\n")

    # Assemble complete ordered list
    verification_results = [pre_results[i] for i in range(1, len(claims) + 1)]

    # Evidence is sufficient only if at least one claim was verified and EVERY claim passed entailment
    if verification_results:
        all_passed = all(item["is_entailed"] for item in verification_results)
        evidence_sufficient = all_passed
    else:
        evidence_sufficient = False

    passed_count = sum(1 for v in verification_results if v["is_entailed"])
    print(f"Hop Verification Summary: Sufficient = {evidence_sufficient} "
          f"({passed_count}/{len(verification_results)} claims verified)")

    return {
        "verification_results": verification_results,
        "evidence_sufficient": evidence_sufficient
    }
