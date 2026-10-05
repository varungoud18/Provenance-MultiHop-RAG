# PROVENANCE // Final Viva Defense & Technical Reference Guide

---

## 1. Defensible Academic Problem Statement (No Overclaims)

### ❌ The Overclaim to Avoid
> *"Eliminates LLM hallucinations and guarantees 100% factual accuracy."*
*(Any faculty or reviewer familiar with RAG literature will push back: NLI models are probabilistic classifiers and can misjudge nuanced entailment; LLMs can misattribute subtle contextual premises.)*

### ✅ The Defensible, Peer-Review-Safe Framing
> **"Detects and flags ungrounded claims by auditing every generated sentence against cited source text via Natural Language Inference (NLI)."**

#### The 3-Sentence Elevator Pitch for Your Panel:
1. **The Core Defect:** *"Standard Retrieval-Augmented Generation (RAG) blindly trusts the generator; when an LLM synthesizes retrieved passages, it often inserts ungrounded assertions or misattributes citations."*
2. **Our Methodological Shift:** *"PROVENANCE decouples generation from factual verification by treating generation as an unverified hypothesis and subjecting each individual sentence to an automated Natural Language Inference (NLI) auditor."*
3. **The Concrete Guarantee:** *"Rather than claiming to 'eliminate' hallucinations, the system guarantees **transparency**: any claim that fails strict entailment against its cited text is explicitly tagged as `[UNVERIFIED - REFUTED BY SOURCE LITERATURE]`, logged in an agent self-correction audit trail, and visualized on an interactive citation knowledge graph."*

---

## 2. Cross-Encoder Logit vs. Neural Match Percentage

### The Faculty Question:
> *"Why does the hover tooltip show `Cross-Encoder: -3.07` next to `Neural Match: 23%` (or `+1.85` next to `81%`)? Why is the cross-encoder score negative? Does a negative number mean bad match or an error?"*

### Your Exact Technical Explanation:

#### A. Why the Cross-Encoder Score Can Be Negative
* The reranker is **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (a 22M parameter transformer running locally via PyTorch / `sentence-transformers`).
* It performs full joint cross-attention over `[CLS] Query [SEP] Document [SEP]`.
* The final linear classification head outputs an **unbounded raw logit $z \in (-\infty, +\infty)$**, trained using binary cross-entropy loss against MS-MARCO passage relevance labels.
* In information retrieval:
  * A score of $z = 0$ represents the decision boundary (neutral relevance).
  * Scores $z > 0$ indicate higher relevance than the training baseline.
  * Scores $z < 0$ (e.g., $-2.5$, $-3.07$) are **completely standard** for candidate passages returned by top-$k$ dense vector search. Negative numbers simply mean the model assigns lower relevance relative to ideal matches, not that an error occurred.

#### B. The Exact Conversion Formula (In `web/app.js:1794`)
To display an intuitive, human-interpretable match quality on the UI, the frontend maps the unbounded logit $z$ to a bounded percentage $[0\%, 100\%]$ using a **temperature-scaled logistic sigmoid function**:

$$\text{Neural Match \%} = \text{round}\left( 100 \cdot \sigma(0.4 \cdot z) \right) = \text{round}\left( \frac{100}{1 + e^{-0.4 \cdot z}} \right)$$

where:
* $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the standard logistic sigmoid.
* $0.4$ is a **temperature/scaling factor** chosen empirically to prevent the sigmoid from saturating too quickly (flattening out at 0% or 100%) across the typical cross-encoder logit range of $[-8, +4]$.

#### C. Concrete Worked Examples for Viva Defense:

| Raw Logit $z$ | Scaled Exponent $(-0.4 \cdot z)$ | Sigmoid $\sigma(0.4 \cdot z)$ | UI "Neural Match" | Qualitative Meaning |
|---|---|---|---|---|
| **$+2.60$** | $-1.04$ | $\frac{1}{1 + 0.353} \approx 0.739$ | **$74\%$** | Highly relevant primary reference |
| **$+1.00$** | $-0.40$ | $\frac{1}{1 + 0.670} \approx 0.599$ | **$60\%$** | Solid contextual support |
| **$0.00$** | $0.00$ | $\frac{1}{1 + 1.000} = 0.500$ | **$50\%$** | Neutral / boundary candidate |
| **$-1.50$** | $+0.60$ | $\frac{1}{1 + 1.822} \approx 0.354$ | **$35\%$** | Partial topical overlap |
| **$-3.07$** | $+1.228$ | $\frac{1}{1 + 3.414} \approx 0.226$ | **$23\%$** | Distant background candidate |

> **Viva One-Liner:** *"The raw score is the unnormalized logit directly from the MiniLM transformer head, while the Neural Match percentage is that same logit passed through a temperature-scaled sigmoid function $\sigma(0.4 \cdot z)$ for normalized human interpretation."*

---

## 3. Verified Deployed Hyperparameters

Ensure you cite these exact numbers during your demo:

| Component | Variable in Code | Exact Value Deployed | Role in Pipeline |
|---|---|---|---|
| **Stage 2 Candidate Pool** | `STAGE2_CANDIDATE_K` (`src/config.py:46`) | **$k = 20$** | Broad FAISS cosine candidate retrieval |
| **Cross-Encoder Top-$k$** | `STAGE2_RERANK_K` (`src/config.py:47`) | **$k = 5$** | Deep cross-attention reranking passed to LLM |
| **Default Hop Budget** | `max_hops` (`src/config.py`, `src/agent.py`) | **$3$ hops** | Hops 0, 1, and 2 before final synthesis |
| **UI Hop Budget Selector** | `#hopBudgetGroup` (`web/index.html`) | **1, 2, 3, or 4** | User-controllable budget slider/segmented control |
| **Chunk Window Size** | `CHUNK_WORD_TARGET` (`src/config.py:34`) | **$120$ words** | Sentence-boundary-aware chunk target |
| **Chunk Overlap** | `CHUNK_OVERLAP_WORDS` (`src/config.py:35`) | **$20$ words** | Sliding-window overlap preventing boundary losses |
| **Embedding Model** | `GEMINI_EMBED_MODEL` (`src/config.py:27`) | `text-embedding-004` | 768-dim (or 3072-dim fallback) vectors |
| **Vector Index Type** | `IndexFlatIP` (`src/step3_index_faiss.py`) | Inner Product (Cosine) | Normalized vector similarity in FAISS |
| **Cross-Encoder Model** | `CROSS_ENCODER_MODEL` (`src/config.py:30`) | `ms-marco-MiniLM-L-6-v2` | 22M parameter sentence transformer |
| **Loop Routing Policy** | `controller.py:decide_next_node` | Conditional Edge | If unverified claims exist & `hop < max_hops`, loops back to `plan`; else proceeds to `report`. |

---

## 4. Key Questions & Defenses for Panel Review

### Q1: "Why use a separate Cross-Encoder if FAISS already gave you top vectors?"
> *"FAISS uses Bi-Encoder embeddings (vector dot product). Bi-encoders encode queries and documents independently into vectors, meaning they miss word-level token cross-attention interactions. The Cross-Encoder jointly passes both texts through transformer layers simultaneously, allowing query terms to attend directly to document context. This eliminates false-positive cosine matches."*

### Q2: "Why use an agentic multi-hop loop instead of standard single-hop RAG?"
> *"In single-hop RAG, if the first vector retrieval misses a vital premise or retrieves ambiguous text, the LLM hallucinates the missing bridge. In PROVENANCE's multi-hop LangGraph loop, our VERIFY node extracts every atomic claim and verifies it against the text. If any claim is ungrounded, the CONTROLLER node diagnoses the knowledge gap and reformulates a targeted sub-query in Hop 1 and Hop 2 to retrieve the missing evidence."*

### Q3: "What happens if a claim still cannot be verified after maximum hops?"
> *"The system guarantees epistemic humility. Instead of outputting the claim as fact, the REPORT node permanently flags it as `[UNVERIFIED - REFUTED BY SOURCE LITERATURE]`, calculates the factual risk score on the risk gauge, records it in the Self-Correction Audit Log, and highlights the refuted claim node in red on the Citation Knowledge Graph."*
