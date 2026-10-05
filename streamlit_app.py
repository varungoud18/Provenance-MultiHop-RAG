"""
Stage 3 Alternative: Streamlit Interactive Research Dashboard
Run with: streamlit run streamlit_app.py
"""

import sys
import json
import time
from pathlib import Path

# Project path configuration
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.config import (
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

# Streamlit Page Setup
st.set_page_config(
    page_title="PROVENANCE // Academic Research Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark aesthetic
st.markdown("""
<style>
    .metric-box {
        background-color: #111827;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .verified-claim {
        background-color: rgba(16, 185, 129, 0.08);
        border-left: 4px solid #10b981;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .unverified-claim {
        background-color: rgba(239, 68, 68, 0.08);
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .citation-tag {
        background-color: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Sidebar: System Status & arXiv Harvester
# =====================================================================
st.sidebar.title("⚡ PROVENANCE")
st.sidebar.caption("Autonomous Academic Agent & Citation Auditor")

# Corpus Statistics
papers_count = 0
chunks_count = 0
if PAPERS_FILE.exists():
    try:
        with open(PAPERS_FILE, "r", encoding="utf-8") as f:
            papers_count = len(json.load(f))
    except Exception:
        pass

if CHUNKS_FILE.exists():
    try:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks_count = len(json.load(f))
    except Exception:
        pass

st.sidebar.subheader("System Telemetry")
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Indexed Chunks", chunks_count)
col_s2.metric("arXiv Papers", papers_count)

st.sidebar.text(f"LLM: {GEMINI_MODEL}")
st.sidebar.text(f"Embed: {GEMINI_EMBEDDING_MODEL}")
st.sidebar.text("Rerank: MiniLM-L6 (Local)")

st.sidebar.markdown("---")

# Ingestion Form
st.sidebar.subheader("📥 Ingest New arXiv Papers")
with st.sidebar.form("ingest_form"):
    ingest_topic = st.text_input("Search Query", placeholder="e.g. quantum error correction")
    ingest_max = st.slider("Number of Papers", min_value=3, max_value=15, value=10)
    ingest_btn = st.form_submit_button("Harvest & Re-Index")

    if ingest_btn:
        if not ingest_topic.strip():
            st.sidebar.error("Please enter a query.")
        else:
            with st.sidebar.status("Running Ingestion Pipeline...", expanded=True) as status:
                st.write("1. Fetching papers from arXiv...")
                papers = fetch_arxiv_papers(ingest_topic.strip(), max_results=ingest_max)
                st.write(f"Harvested {len(papers)} papers.")

                st.write("2. Chunking abstracts into overlapping segments...")
                chunks = process_papers_into_chunks()
                st.write(f"Generated {len(chunks)} chunks.")

                st.write("3. Generating embeddings & building FAISS index...")
                build_faiss_index()
                status.update(label="Index updated successfully!", state="complete", expanded=False)
            st.sidebar.success(f"Indexed {len(chunks)} chunks into FAISS!")
            st.rerun()

# =====================================================================
# Main Application Area
# =====================================================================
st.title("🔬 Autonomous Academic Research Agent")
st.markdown("Searches arXiv literature, verifies every claim sentence against evidence, and refines queries across evidence gaps.")

# Preset buttons
st.markdown("**Sample Presets:**")
cols_p = st.columns(4)
preset_query = ""
if cols_p[0].button("SQL Optimization"):
    preset_query = "What optimization strategies are used for complex SQL queries?"
if cols_p[1].button("RAG Literature Reviews"):
    preset_query = "How does retrieval augmented generation automate literature reviews?"
if cols_p[2].button("Simple T5 Baselines"):
    preset_query = "What exact baseline models were compared against Simple T5 in the literature review paper?"
if cols_p[3].button("Out-of-Corpus Query"):
    preset_query = "What is the capital of France and what are its main rivers?"

# Query input
default_q = preset_query if preset_query else "What optimization strategies are used for complex SQL queries?"
research_query = st.text_area("Research Question:", value=default_q, height=75)

auto_harvest = st.checkbox("🌐 Live arXiv Auto-Harvest (Pulls and indexes fresh literature matching this question)", value=True)

col_ctrl1, col_ctrl2 = st.columns([1, 3])
with col_ctrl1:
    max_hops = st.slider("Maximum Agent Hops", min_value=1, max_value=5, value=3)
with col_ctrl2:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Execute Autonomous Research Mission", type="primary", use_container_width=True)

# Run Agent
if run_btn:
    if not research_query.strip():
        st.error("Please enter a research question.")
    elif not auto_harvest and not FAISS_INDEX_FILE.exists():
        st.error("FAISS index not found. Please enable Live arXiv Auto-Harvest or ingest papers first.")
    else:
        st.markdown("---")
        st.subheader("⚙️ Agent Execution Trajectory")

        status_box = st.status("Initializing Autonomous Research Agent...", expanded=True)

        if auto_harvest:
            status_box.write("🌐 [Auto-Harvest] Pulling newest papers from arXiv Atom API...")
            papers = fetch_arxiv_papers(research_query.strip(), max_results=10)
            status_box.write(f"✓ Harvested {len(papers)} papers. Chunking and building FAISS vector index...")
            chunks = process_papers_into_chunks()
            build_faiss_index()
            status_box.write(f"✓ Indexed {len(chunks)} fresh chunks into local FAISS.")

        app_graph = build_research_agent_graph()

        initial_state: ResearchAgentState = {
            "original_query": research_query.strip(),
            "current_sub_question": research_query.strip(),
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

        # Step by step execution
        final_state = dict(initial_state)
        hop_records = []

        start_time = time.time()
        for step in app_graph.stream(initial_state):
            for node_name, node_update in step.items():
                final_state.update(node_update)
                curr_hop = final_state.get("hop_count", 0)

                if node_name == "plan":
                    status_box.write(f"📍 **[Node 1: PLAN]** Hop {curr_hop}: Sub-Question -> *\"{node_update.get('current_sub_question')}\"*")
                elif node_name == "retrieve":
                    cands = node_update.get("retrieved_candidates", [])
                    status_box.write(f"🔍 **[Node 2: RETRIEVE]** Retrieved {len(cands)} candidates from FAISS.")
                elif node_name == "rerank":
                    reranked = node_update.get("reranked_chunks", [])
                    status_box.write(f"⚡ **[Node 3: RERANK]** Local Cross-Encoder selected Top {len(reranked)} chunks.")
                elif node_name == "reason":
                    status_box.write(f"✍️ **[Node 4: REASON]** Synthesized cited draft answer.")
                elif node_name == "verify":
                    verifs = node_update.get("verification_results", [])
                    ent_cnt = sum(1 for v in verifs if v.get("is_entailed"))
                    status_box.write(f"🛡️ **[Node 5: VERIFY]** Audited {len(verifs)} claims: {ent_cnt} Verified, {len(verifs) - ent_cnt} Unverified.")
                elif node_name == "controller":
                    suff = final_state.get("evidence_sufficient", False)
                    if suff:
                        status_box.write(f"✅ **[Node 6: CONTROLLER]** Evidence is sufficient. Proceeding to report.")
                    else:
                        status_box.write(f"🔄 **[Node 6: CONTROLLER]** Evidence incomplete. Looping to next hop.")
                elif node_name == "report":
                    status_box.write(f"📊 **[Node 7: REPORT]** Audited dossier compiled.")

        elapsed = time.time() - start_time
        status_box.update(label=f"Completed in {elapsed:.2f}s!", state="complete", expanded=False)

        # Display Final Audited Dossier
        st.markdown("---")
        st.subheader("📋 Final Audited Research Findings")

        verifications = final_state.get("verification_results", [])
        reranked_chunks = final_state.get("reranked_chunks", [])
        hop_history = final_state.get("hop_history", [])

        # Stats bar
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Hops", len(hop_history))
        col_m2.metric("Evidence Status", "SUFFICIENT" if final_state.get("evidence_sufficient") else "PARTIAL GAPS")
        verified_count = sum(1 for v in verifications if v.get("is_entailed"))
        total_claims = len(verifications)
        rate = int((verified_count / total_claims * 100)) if total_claims > 0 else 100
        col_m3.metric("Verification Rate", f"{rate}%")
        col_m4.metric("Audited Claims", f"{verified_count} / {total_claims}")

        # Render Claims
        for v in verifications:
            cits = " ".join([f"<span class='citation-tag'>[{c}]</span>" for c in v.get("citations", [])])
            if v.get("is_entailed"):
                st.markdown(f"""
                <div class="verified-claim">
                    <strong>✓ VERIFIED CLAIM</strong> {cits}<br>
                    {v.get('claim_text')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="unverified-claim">
                    <strong>✗ UNVERIFIED CLAIM</strong> {cits}<br>
                    {v.get('claim_text')}<br>
                    <small style="color: #f87171;">⚠️ Reason: {v.get('reason')}</small>
                </div>
                """, unsafe_allow_html=True)

        # Bibliography
        st.markdown("---")
        st.subheader("📑 Bibliography & Working arXiv Sources")
        seen_urls = set()
        for chunk in reranked_chunks:
            url = chunk.get("paper_url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            with st.expander(f"[{chunk.get('source_num', 1)}] {chunk.get('paper_title')}"):
                st.markdown(f"**Direct arXiv Link:** [{url}]({url})")
                st.markdown(f"**Cross-Encoder Score:** `{chunk.get('cross_encoder_score', 'N/A')}`")
                st.markdown(f"**Excerpt:** {chunk.get('text')}")

        # Download Report
        st.download_button(
            label="💾 Download Dossier (.md)",
            data=final_state.get("final_report", ""),
            file_name="verified_academic_dossier.md",
            mime="text/markdown"
        )
