# PROVENANCE // Autonomous Multi-Hop Academic Research Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Orchestration-LangGraph-purple?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Vector_DB-FAISS-00ADD8?style=for-the-badge&logo=meta&logoColor=white" alt="FAISS" />
  <img src="https://img.shields.io/badge/Reranker-MiniLM--L6_Cross--Encoder-orange?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch Cross-Encoder" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/API-arXiv_Discovery-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv API" />
</p>

---

## 📌 Executive Summary

> **Core Capability:** Detects and flags ungrounded claims by auditing every generated sentence against cited source text via Natural Language Inference (NLI), iteratively retrieving missing evidence through a self-correcting multi-hop state machine.

Standard Retrieval-Augmented Generation (RAG) suffers from **generative overconfidence**: once passages are retrieved, the LLM freely synthesizes them, frequently introducing subtle hallucinations or misattributing citations. 

**PROVENANCE** decouples generation from factual verification. Instead of blindly trusting LLM generation, our agent:
1. Deconstructs responses into **atomic, sentence-level claims**.
2. Audits each individual claim against its cited evidence using an independent **Natural Language Inference (NLI)** auditor.
3. Automatically identifies knowledge gaps and **formulates targeted queries across multiple hops (1–4 hops)** to find missing proof.
4. Upholds **epistemic humility**: any claim that cannot be substantiated after maximum hops is permanently tagged as `[UNVERIFIED - REFUTED BY SOURCE LITERATURE]`, highlighted in red on an interactive citation knowledge graph, and logged in an agent self-correction audit trail.

---

## 👥 Project Team & Institutional Mentorship

* **Institution:** Department of Computer Science & Engineering
* **Project Guide:** **Dr. Voddelli Sri Lakshmi**
* **Project Investigators:**
  * **K. Varun Goud** — `23BCE7216` ([@varungoud18](https://github.com/varungoud18))
  * **J.T. S Praneeth** — `23BCE7220`
  * **P. Yuvraj** — `23BCE7225`
  * **G. Naga Siddhi Sai** — `23BCE7179`

---

## 🏛️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           1. INGESTION & DISCOVERY                          │
│   arXiv Public Atom XML API ──► XML Parser ──► papers.json (Metadata & DOIs)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2. CHUNKING & DENSE VECTOR INDEXING                      │
│   Sentence Sliding Window (120w / 20w overlap) ──► Gemini Embedding-004     │
│   ──► L2-Normalized 768-dim Vectors ──► FAISS IndexFlatIP (Cosine Cache)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       3. TWO-STAGE NEURAL RETRIEVAL                         │
│   Query ──► FAISS Candidate Pool (k=20)                                      │
│         ──► PyTorch Cross-Encoder (ms-marco-MiniLM-L-6-v2) Reranker (k=5)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     4. MULTI-HOP AGENTIC LANGGRAPH LOOP                     │
│   PLAN ──► RETRIEVE ──► RERANK ──► REASON ──► VERIFY ──► CONTROLLER         │
│     ▲                                                          │            │
│     └──────── Gap-Targeted Reformulation (Hop < MaxHops) ──────┘            │
│                                                                ▼ (Done)     │
│                                                              REPORT         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  5. EXPLAINABLE RESEARCH DASHBOARD (WEB UI)                 │
│   - Force-Directed Citation Graph      - Real-Time Telemetry Bar            │
│   - Factual Hallucination Risk Gauge   - BibTeX LaTeX Exporter              │
│   - Direct arXiv PDF Deep-Links        - Self-Correction Audit Log          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 The 7-Node LangGraph State Machine

The multi-hop loop is managed as an asynchronous state graph:

1. **`PLAN` Node:** Analyzes the research goal or gap diagnosis to formulate focused search queries.
2. **`RETRIEVE` Node:** Extracts the top $k = 20$ dense vector candidates from the local FAISS index.
3. **`RERANK` Node:** Evaluates query-passage cross-attention using local `ms-marco-MiniLM-L-6-v2`, isolating the top $k = 5$ authoritative passages.
4. **`REASON` Node:** Generates an evidence-grounded draft with strict sentence-level citation syntax (`[1]`, `[2]`).
5. **`VERIFY` Node:** Deconstructs generated text into atomic assertions and audits premise-hypothesis entailment against cited source chunks.
6. **`CONTROLLER` Node:** Evaluates loop termination:
   * *If all claims are verified* $\to$ Routes to `REPORT`.
   * *If unverified claims exist & $\text{hop} < \text{max\_hops}$* $\to$ Loops back to `PLAN` with a targeted gap query.
   * *If hop limit reached* $\to$ Routes to `REPORT` preserving audit flags.
7. **`REPORT` Node:** Formats the final audited report, generates BibTeX entries, calculates the factual risk score, and surfaces the self-correction history.

---

## 📐 Mathematical Foundations

### 1. FAISS Cosine Vector Similarity (Bi-Encoder Retrieval)
$$\text{Sim}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\| \|\mathbf{e}_d\|} = \mathbf{e}_q^{\top} \mathbf{e}_d \quad (\text{for L2-normalized embeddings})$$

### 2. Cross-Encoder Joint-Attention Logit (Reranking)
$$z = W \cdot \text{Transformer}\Big( \text{[CLS]} \circ q \circ \text{[SEP]} \circ d \circ \text{[SEP]} \Big) + b \in (-\infty, +\infty)$$
The MiniLM cross-encoder outputs an unnormalized logit $z$. Negative values (e.g. $-3.07$) are standard for background candidate passages.

### 3. Temperature-Scaled Sigmoid Conversion (UI Neural Match)
To convert the unbounded logit $z$ into an intuitive human percentage $[0\%, 100\%]$ without saturation:
$$\text{Neural Match \%} = \text{round}\left( 100 \cdot \sigma(0.4 \cdot z) \right) = \text{round}\left( \frac{100}{1 + e^{-0.4 \cdot z}} \right)$$

| Raw Logit $z$ | Scaled Exponent $(-0.4 \cdot z)$ | Sigmoid $\sigma(0.4 \cdot z)$ | Displayed Match | Semantic Meaning |
|:---:|:---:|:---:|:---:|:---|
| **$+2.60$** | $-1.04$ | $\approx 0.739$ | **$74\%$** | Primary authoritative reference |
| **$+1.00$** | $-0.40$ | $\approx 0.599$ | **$60\%$** | Moderate topical support |
| **$0.00$** | $0.00$ | $= 0.500$ | **$50\%$** | Neutral decision boundary |
| **$-1.50$** | $+0.60$ | $\approx 0.354$ | **$35\%$** | Partial background overlap |
| **$-3.07$** | $+1.228$ | $\approx 0.226$ | **$23\%$** | Distant candidate |

### 4. Factual Hallucination Risk Metric
$$\text{Risk Score} = \left( \frac{\text{Count of Refuted/Unverified Claims}}{\text{Total Asserted Claims}} \right) \times 100\%$$

---

## ⚡ Key Features & Research Add-Ons

* 🌐 **Interactive 2D Citation Knowledge Graph:** Force-directed canvas visualizes query seeds, paper nodes, verified green claims, and red self-corrected assertions.
* 🛡️ **Factual Hallucination Risk Gauge:** Animated SVG telemetry displaying real-time factual confidence.
* 📄 **Direct arXiv Deep-Links & DOIs:** One-click instant navigation to original preprint PDF sources.
* 📚 **One-Click BibTeX Exporter:** Instant `.bib` generation ready for LaTeX / Overleaf compilation.
* ⏱️ **Research Mission History Drawer:** LocalStorage-persisted session recall with instant reload.
* 🏷️ **Epistemic Humility Badges:** Clean visual indicators for entailed facts vs. refuted hypotheses.

---

## 📂 Repository Structure

```text
Provenance-MultiHop-RAG/
│
├── .env.example               <-- Template showing API keys & model options
├── .gitignore                 <-- Protects keys and vector index files from git
├── requirements.txt           <-- Python dependencies
├── README.md                  <-- Project documentation
│
├── data/                      <-- Local Pipeline Data Storage
│   └── .gitkeep               <-- Tracks data directory (cache/indices generated locally)
│
├── src/                       <-- Core Modular Implementation
│   ├── config.py              <-- Central hyperparameters (k=20, k=5, hops=3)
│   ├── gemini_client.py       <-- Resilient Gemini API wrapper with rotation
│   │
│   │   # Stage 1: Ingestion & Baseline
│   ├── step1_fetch_arxiv.py   <-- arXiv public XML API harvester
│   ├── step2_chunk_papers.py  <-- Sliding-window chunker (120w / 20w overlap)
│   ├── step3_index_faiss.py   <-- Local FAISS cosine index builder
│   ├── step4_ask_single_hop.py<-- Single-hop baseline retriever
│   │
│   │   # Stage 2: Multi-Hop LangGraph Agentic Loop
│   ├── state.py               <-- AgentState TypedDict schema
│   ├── plan.py                <-- Node 1: PLAN (initial & gap queries)
│   ├── retrieve.py            <-- Node 2: RETRIEVE (FAISS pool, k=20)
│   ├── rerank.py              <-- Node 3: RERANK (PyTorch Cross-Encoder, k=5)
│   ├── reason.py              <-- Node 4: REASON (synthesizer with citations)
│   ├── verify.py              <-- Node 5: VERIFY (atomic sentence NLI auditor)
│   ├── controller.py          <-- Node 6: CONTROLLER (hop routing & decision gates)
│   ├── report.py              <-- Node 7: REPORT (dossier with self-corrections)
│   ├── agent.py               <-- LangGraph compiler & CLI entrypoint
│   └── server.py              <-- FastAPI asynchronous REST backend
│
├── web/                       <-- High-Performance Web Dashboard
│   ├── index.html             <-- Modern semantic HTML5 interface
│   ├── style.css              <-- Premium responsive dark-mode styling
│   └── app.js                 <-- Interactive 2D canvas, BibTeX export, SSE polling
│
└── tests/                     <-- Unit & Edge-Case Test Suite
    ├── test_claim_splitter.py <-- Regex atomic sentence boundary verification
    ├── test_edge_cases.py     <-- FAISS edge-case handling tests
    └── test_unverified_flag.py<-- Hallucination detection verification
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/varungoud18/Provenance-MultiHop-RAG.git
cd Provenance-MultiHop-RAG
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy `.env.example` to `.env` and insert your [Google Gemini API Key](https://aistudio.google.com/):
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004
```

### 5. Launch the Web Application
```bash
uvicorn src.server:app --host 127.0.0.1 --port 8000 --reload
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 🧪 Running the Baseline CLI & Test Suite

* **Fetch arXiv papers & build index:**
  ```bash
  python -m src.step1_fetch_arxiv
  python -m src.step2_chunk_papers
  python -m src.step3_index_faiss
  ```
* **Run terminal multi-hop agent:**
  ```bash
  python -m src.agent
  ```
* **Execute test suite:**
  ```bash
  pytest tests/
  ```

---

## 📜 License & Citation

This project is released under the **MIT License**.

If you use or build upon PROVENANCE in your academic research, please cite:
```bibtex
@software{provenance2026,
  author = {K. Varun Goud and J.T. S Praneeth and P. Yuvraj and G. Naga Siddhi Sai},
  title = {PROVENANCE: Autonomous Multi-Hop Academic Research Agent with Local Cross-Encoder Reranking and Sentence-Level Citation Entailment Auditing},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/varungoud18/Provenance-MultiHop-RAG}}
}
```
