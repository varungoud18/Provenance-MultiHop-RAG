/**
 * VERITAS-RAG // Interactive Client-Side Agent Controller
 * Connects to FastAPI SSE backend and powers interactive DAG visualizations,
 * hop accordion streaming, citation inspection drawer, and arXiv ingestion.
 */

// Application State Store
const state = {
  currentHop: 0,
  maxHops: 3,
  activeNode: null,
  executionStartTime: null,
  timerInterval: null,
  latestRerankedChunks: {},  // map source_num -> chunk object
  latestVerifications: [],
  hopHistory: [],
  finalReportText: "",
  isStreaming: false,
};

// DOM Element References
const dom = {
  // Status Telemetry
  valFaissChunks: document.getElementById("val-faiss-chunks"),
  valLlmModel: document.getElementById("val-llm-model"),
  
  // Controls
  researchForm: document.getElementById("research-form"),
  researchQuery: document.getElementById("research-query"),
  btnVoiceInput: document.getElementById("btn-voice-input"),
  btnFloatingMic: document.getElementById("btn-floating-mic"),
  micStatusPill: document.getElementById("mic-status-pill"),
  micBtnLabel: document.getElementById("mic-btn-label"),
  maxHopsInput: document.getElementById("max-hops-input"),
  maxHopsVal: document.getElementById("max-hops-val"),
  btnSubmit: document.getElementById("btn-submit-research"),
  actionSpinner: document.getElementById("action-spinner"),
  actionBtnText: document.getElementById("action-btn-text"),
  execTimer: document.getElementById("exec-timer"),
  presetChips: document.querySelectorAll(".preset-chips .chip"),
  toggleAutoHarvest: document.getElementById("toggle-auto-harvest"),

  // DAG Flowchart
  dagPipeline: document.getElementById("dag-pipeline"),
  btnViewStepper: document.getElementById("btn-view-stepper"),
  btnViewGrid: document.getElementById("btn-view-grid"),
  dagHopTag: document.getElementById("dag-hop-tag"),
  dagNodes: {
    plan: document.getElementById("node-plan"),
    retrieve: document.getElementById("node-retrieve"),
    rerank: document.getElementById("node-rerank"),
    reason: document.getElementById("node-reason"),
    verify: document.getElementById("node-verify"),
    controller: document.getElementById("node-controller"),
    report: document.getElementById("node-report")
  },
  loopBackIndicator: document.getElementById("loop-back-indicator"),

  // Tabs & Panes
  tabBtns: document.querySelectorAll(".tab-btn"),
  tabPanes: document.querySelectorAll(".tab-pane"),
  hopBadgeCount: document.getElementById("hop-badge-count"),
  sourceBadgeCount: document.getElementById("source-badge-count"),

  // Findings Content
  summaryStats: document.getElementById("summary-stats"),
  statGaugeCircle: document.getElementById("stat-gauge-circle"),
  statTrustScore: document.getElementById("stat-trust-score"),
  statHops: document.getElementById("stat-hops"),
  statEvidence: document.getElementById("stat-evidence"),
  statVerificationRate: document.getElementById("stat-verification-rate"),
  statBreakdown: document.getElementById("stat-breakdown"),
  dossierActions: document.getElementById("dossier-actions"),
  dossierContent: document.getElementById("dossier-content"),
  btnCopyDossier: document.getElementById("btn-copy-dossier"),
  btnExportBibtex: document.getElementById("btn-export-bibtex"),
  btnDownloadMarkdown: document.getElementById("btn-download-markdown"),

  // Hops & Sources Stream
  hopsStreamContainer: document.getElementById("hops-stream-container"),
  sourcesList: document.getElementById("sources-list"),

  // Knowledge Graph Canvas
  tabBtnGraph: document.getElementById("tab-btn-graph"),
  citationGraphCanvas: document.getElementById("citation-graph-canvas"),
  graphEmptyOverlay: document.getElementById("graph-empty-overlay"),
  btnRecenterGraph: document.getElementById("btn-recenter-graph"),
  graphCanvasWrap: document.getElementById("graph-canvas-wrap"),

  // Citation Drawer
  citationDrawer: document.getElementById("citation-drawer"),
  btnCloseDrawer: document.getElementById("btn-close-drawer"),
  drawerSourceBadge: document.getElementById("drawer-source-badge"),
  drawerPaperTitle: document.getElementById("drawer-paper-title"),
  drawerArxivLink: document.getElementById("drawer-arxiv-link"),
  drawerArxivPdfLink: document.getElementById("drawer-arxiv-pdf-link"),
  drawerRerankScore: document.getElementById("drawer-rerank-score"),
  drawerChunkId: document.getElementById("drawer-chunk-id"),
  drawerChunkText: document.getElementById("drawer-chunk-text"),

  // History Modal
  btnOpenHistory: document.getElementById("btn-open-history"),
  modalHistory: document.getElementById("modal-history"),
  btnCloseHistory: document.getElementById("btn-close-history"),
  btnClearHistory: document.getElementById("btn-clear-history"),
  historyList: document.getElementById("history-list"),
  historyCount: document.getElementById("history-count"),

  // Harvest Modal
  btnOpenHarvest: document.getElementById("btn-open-harvest"),
  modalHarvest: document.getElementById("modal-harvest"),
  btnCloseHarvest: document.getElementById("btn-close-harvest"),
  btnCancelHarvest: document.getElementById("btn-cancel-harvest"),
  harvestForm: document.getElementById("harvest-form"),
  harvestQuery: document.getElementById("harvest-query"),
  harvestMaxResults: document.getElementById("harvest-max-results"),
  harvestLogArea: document.getElementById("harvest-log-area"),
  harvestLogOutput: document.getElementById("harvest-log-output"),
  harvestSpinner: document.getElementById("harvest-spinner"),
  btnStartHarvest: document.getElementById("btn-start-harvest"),

  // Corpus Modal
  btnOpenCorpus: document.getElementById("btn-open-corpus"),
  modalCorpus: document.getElementById("modal-corpus"),
  btnCloseCorpus: document.getElementById("btn-close-corpus"),
  corpusTbody: document.getElementById("corpus-tbody"),
  corpusCount: document.getElementById("corpus-count"),

  // Toasts
  toastContainer: document.getElementById("toast-container")
};

// =====================================================================
// System Initialization & Status Polling
// =====================================================================

async function fetchSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    dom.valFaissChunks.textContent = `${data.papers_count} Papers (${data.chunks_count} Chunks)`;
    if (data.gemini_model === "gemini-3.1-flash-lite") {
      dom.valLlmModel.textContent = "Gemini 3.1 Flash-Lite";
    } else {
      dom.valLlmModel.textContent = data.gemini_model;
    }
  } catch (err) {
    console.warn("Failed to fetch system status:", err);
    dom.valFaissChunks.textContent = "Offline";
  }
}

// =====================================================================
// UI Event Handlers
// =====================================================================

// =====================================================================
// Speech-to-Text / Voice Dictation Controller
// =====================================================================

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isDictating = false;
let preSpeechText = "";

function initSpeechRecognition() {
  if (!SpeechRecognition) {
    console.warn("Web Speech API not supported in this browser environment.");
    return null;
  }

  try {
    const rec = new SpeechRecognition();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-US";

    rec.onstart = () => {
      isDictating = true;
      updateMicUI(true);
      preSpeechText = dom.researchQuery ? dom.researchQuery.value : "";
      showToast("🎙️ Listening... Speak your research question");
    };

    rec.onresult = (event) => {
      let speechTranscript = "";
      for (let i = 0; i < event.results.length; i++) {
        speechTranscript += event.results[i][0].transcript;
      }
      
      const cleanPrefix = preSpeechText.trimEnd();
      const separator = cleanPrefix.length > 0 ? " " : "";
      if (dom.researchQuery) {
        dom.researchQuery.value = cleanPrefix + separator + speechTranscript.trimStart();
        dom.researchQuery.scrollTop = dom.researchQuery.scrollHeight;
      }
    };

    rec.onerror = (event) => {
      console.warn("Speech recognition error:", event.error);
      isDictating = false;
      updateMicUI(false);
      if (event.error === "not-allowed") {
        showToast("⚠️ Microphone access denied. Allow mic permissions in your browser.");
      } else if (event.error === "network") {
        showToast("⚠️ Speech recognition network error.");
      } else if (event.error !== "no-speech") {
        showToast(`Voice input: ${event.error}`);
      }
    };

    rec.onend = () => {
      isDictating = false;
      updateMicUI(false);
    };

    return rec;
  } catch (err) {
    console.error("Failed to initialize SpeechRecognition:", err);
    return null;
  }
}

function updateMicUI(active) {
  if (dom.btnVoiceInput) {
    dom.btnVoiceInput.classList.toggle("listening", active);
    if (dom.micBtnLabel) {
      dom.micBtnLabel.textContent = active ? "Stop Listening" : "Voice Dictate";
    }
  }
  if (dom.btnFloatingMic) {
    dom.btnFloatingMic.classList.toggle("listening", active);
  }
  if (dom.micStatusPill) {
    dom.micStatusPill.classList.toggle("hidden", !active);
  }
  if (active && dom.researchQuery) {
    dom.researchQuery.focus();
  }
}

function toggleVoiceDictation() {
  if (!SpeechRecognition) {
    showToast("⚠️ Speech Recognition is not supported by this browser. Please use Chrome, Edge, or Brave.");
    return;
  }

  if (!recognition) {
    recognition = initSpeechRecognition();
    if (!recognition) {
      showToast("⚠️ Unable to access speech recognition in this browser.");
      return;
    }
  }

  if (isDictating) {
    try {
      recognition.stop();
    } catch (e) {
      console.warn("Error stopping speech recognition:", e);
    }
    isDictating = false;
    updateMicUI(false);
    showToast("Voice dictation stopped.");
  } else {
    try {
      recognition.start();
    } catch (e) {
      console.warn("Error starting speech recognition:", e);
      try {
        recognition.stop();
        setTimeout(() => {
          try { recognition.start(); } catch (err) { console.error(err); }
        }, 150);
      } catch (err) {
        console.error(err);
      }
    }
  }
}

if (dom.btnVoiceInput) {
  dom.btnVoiceInput.addEventListener("click", toggleVoiceDictation);
}
if (dom.btnFloatingMic) {
  dom.btnFloatingMic.addEventListener("click", toggleVoiceDictation);
}

// Max Hops slider (kept for backward compatibility)
if (dom.maxHopsInput) {
  dom.maxHopsInput.addEventListener("input", (e) => {
    if (dom.maxHopsVal) dom.maxHopsVal.textContent = e.target.value;
    state.maxHops = parseInt(e.target.value, 10);
  });
}

// Modern Segmented Hop Selector
document.querySelectorAll(".btn-hop-pill").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".btn-hop-pill").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const hops = parseInt(btn.getAttribute("data-hops"), 10);
    state.maxHops = hops;
    if (dom.maxHopsInput) dom.maxHopsInput.value = hops;
    if (dom.maxHopsVal) dom.maxHopsVal.textContent = hops;
  });
});

// Preset Query Chips
dom.presetChips.forEach(chip => {
  chip.addEventListener("click", () => {
    const query = chip.getAttribute("data-query");
    dom.researchQuery.value = query;
    dom.researchQuery.focus();
  });
});

// Starter Query Cards in Empty State
document.addEventListener("click", (e) => {
  const card = e.target.closest(".starter-card");
  if (card) {
    const query = card.getAttribute("data-query");
    if (query && dom.researchQuery) {
      dom.researchQuery.value = query;
      dom.researchQuery.focus();
    }
  }
});

// Tabs Switching
dom.tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    const targetTab = btn.getAttribute("data-tab");
    dom.tabBtns.forEach(b => b.classList.remove("active"));
    dom.tabPanes.forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const pane = document.getElementById(targetTab);
    if (pane) pane.classList.add("active");
    if (targetTab === "tab-graph") {
      setTimeout(() => {
        if (typeof drawCanvasGraph === "function") drawCanvasGraph();
      }, 50);
    }
  });
});

// Citation Inspector Drawer Controls
dom.btnCloseDrawer.addEventListener("click", () => {
  dom.citationDrawer.classList.add("hidden");
});

function openCitationDrawer(sourceNum) {
  const chunk = state.latestRerankedChunks[sourceNum];
  if (!chunk) {
    showToast(`Citation [${sourceNum}] evidence metadata not found.`);
    return;
  }

  dom.drawerSourceBadge.textContent = `[${sourceNum}]`;
  dom.drawerPaperTitle.textContent = chunk.paper_title || "Unknown Paper";
  dom.drawerArxivLink.textContent = chunk.paper_url || "N/A";
  dom.drawerArxivLink.href = chunk.paper_url || "#";

  if (dom.drawerArxivPdfLink) {
    let pdfUrl = "#";
    if (chunk.paper_url) {
      pdfUrl = chunk.paper_url.replace("/abs/", "/pdf/") + ".pdf";
    }
    dom.drawerArxivPdfLink.href = pdfUrl;
  }

  dom.drawerRerankScore.textContent = chunk.cross_encoder_score !== undefined ? chunk.cross_encoder_score : "N/A";
  dom.drawerChunkId.textContent = chunk.chunk_id || `source_${sourceNum}`;
  dom.drawerChunkText.textContent = chunk.text || "No text available.";

  dom.citationDrawer.classList.remove("hidden");
}

// Global delegated click for citation pills
document.addEventListener("click", (e) => {
  const pill = e.target.closest(".citation-pill");
  if (pill) {
    const sourceNum = parseInt(pill.getAttribute("data-source"), 10);
    if (sourceNum) openCitationDrawer(sourceNum);
  }
});

// =====================================================================
// DAG Flowchart Visualizer & View Mode Switcher
// =====================================================================

if (dom.btnViewStepper && dom.btnViewGrid && dom.dagPipeline) {
  dom.btnViewStepper.addEventListener("click", () => {
    dom.btnViewStepper.classList.add("active");
    dom.btnViewGrid.classList.remove("active");
    dom.dagPipeline.classList.remove("layout-grid");
    dom.dagPipeline.classList.add("layout-stepper");
  });

  dom.btnViewGrid.addEventListener("click", () => {
    dom.btnViewGrid.classList.add("active");
    dom.btnViewStepper.classList.remove("active");
    dom.dagPipeline.classList.remove("layout-stepper");
    dom.dagPipeline.classList.add("layout-grid");
  });
}

function setActiveDagNode(nodeName) {
  state.activeNode = nodeName;
  const nodes = ["plan", "retrieve", "rerank", "reason", "verify", "controller", "report"];
  const targetIdx = nodes.indexOf(nodeName);

  nodes.forEach((n, idx) => {
    const el = dom.dagNodes[n];
    if (!el) return;
    el.classList.remove("active");
    const badge = el.querySelector(".node-status-badge");
    if (idx <= targetIdx) {
      el.classList.add("completed");
      if (badge) badge.textContent = "Done ✓";
    } else if (idx === targetIdx + 1 && targetIdx < nodes.length - 1) {
      el.classList.add("active");
      el.classList.remove("completed");
      if (badge) badge.textContent = "Running...";
    } else {
      el.classList.remove("completed");
      if (badge) badge.textContent = "Waiting";
    }
  });
}

function resetDagNodes() {
  Object.values(dom.dagNodes).forEach(el => {
    el.classList.remove("active", "completed");
    const badge = el.querySelector(".node-status-badge");
    if (badge) badge.textContent = "Ready";
  });
  if (dom.loopBackIndicator) {
    dom.loopBackIndicator.classList.add("hidden");
  }
  if (dom.dagHopTag) {
    dom.dagHopTag.classList.add("hidden");
  }
}

function markDagAllCompleted() {
  state.activeNode = null;
  const nodes = ["plan", "retrieve", "rerank", "reason", "verify", "controller", "report"];
  nodes.forEach(n => {
    const el = dom.dagNodes[n];
    if (!el) return;
    el.classList.remove("active");
    el.classList.add("completed");
    const badge = el.querySelector(".node-status-badge");
    if (badge) badge.textContent = "Done ✓";
  });
}

function startTimer() {
  state.executionStartTime = Date.now();
  dom.execTimer.classList.remove("hidden");
  dom.execTimer.textContent = "00.0s";
  clearInterval(state.timerInterval);
  state.timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - state.executionStartTime) / 1000).toFixed(1);
    dom.execTimer.textContent = `${elapsed}s`;
  }, 100);
}

function stopTimer() {
  clearInterval(state.timerInterval);
}

// =====================================================================
// Stream Reader & Agent Execution
// =====================================================================

dom.researchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = dom.researchQuery.value.trim();
  if (!query || state.isStreaming) return;

  const maxHops = parseInt(dom.maxHopsInput.value, 10);

  // Set running state
  state.isStreaming = true;
  dom.btnSubmit.disabled = true;
  dom.actionSpinner.classList.remove("hidden");
  dom.actionBtnText.textContent = "Agent Executing...";
  resetDagNodes();
  startTimer();

  // Reset UI views
  dom.hopsStreamContainer.innerHTML = "";
  dom.sourcesList.innerHTML = "";
  dom.dossierContent.innerHTML = `
    <div class="empty-state">
      <div class="btn-spinner" style="width: 32px; height: 32px; margin-bottom: 12px;"></div>
      <h3 class="empty-title">Agentic Loop Running</h3>
      <p class="empty-desc">Formulating sub-questions, retrieving candidates, reranking with Cross-Encoder, and auditing claim entailment...</p>
    </div>
  `;
  dom.summaryStats.classList.add("hidden");
  dom.dossierActions.classList.add("hidden");
  dom.hopBadgeCount.textContent = "0";
  dom.sourceBadgeCount.textContent = "0";
  state.latestRerankedChunks = {};
  state.hopHistory = [];

  try {
    const autoHarvest = dom.toggleAutoHarvest ? dom.toggleAutoHarvest.checked : true;
    const sseUrl = `/api/research/stream?query=${encodeURIComponent(query)}&max_hops=${maxHops}&auto_harvest=${autoHarvest}`;
    const response = await fetch(sseUrl);

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({ detail: "Unknown server error" }));
      throw new Error(errJson.detail || `Server returned ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // keep trailing incomplete chunk

      for (const line of lines) {
        if (!line.trim() || line.startsWith(":")) continue; // skip heartbeats
        if (line.startsWith("data: ")) {
          const jsonStr = line.replace(/^data: /, "").trim();
          try {
            const eventData = JSON.parse(jsonStr);
            handleAgentEvent(eventData);
          } catch (pErr) {
            console.warn("SSE JSON Parse error:", pErr, jsonStr);
          }
        }
      }
    }
  } catch (err) {
    showToast(`Execution Error: ${err.message}`);
    dom.dossierContent.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">⚠️</div>
        <h3 class="empty-title">Agent Execution Stopped</h3>
        <p class="empty-desc" style="color: #f87171;">${err.message}</p>
      </div>
    `;
  } finally {
    state.isStreaming = false;
    dom.btnSubmit.disabled = false;
    dom.actionSpinner.classList.add("hidden");
    dom.actionBtnText.textContent = "Execute Agent Loop";
    stopTimer();
    if (state.finalReportText) {
      markDagAllCompleted();
    }
  }
});

// Process individual SSE events from server
function handleAgentEvent(event) {
  if (event.type === "harvest_start") {
    dom.actionBtnText.textContent = "Harvesting arXiv...";
    dom.dossierContent.innerHTML = `
      <div class="empty-state">
        <div class="btn-spinner" style="width: 36px; height: 36px; margin-bottom: 14px;"></div>
        <h3 class="empty-title">🌐 Auto-Harvesting arXiv Literature</h3>
        <p class="empty-desc">${escapeHtml(event.message || "Downloading newest papers matching your query...")}</p>
      </div>
    `;
  } else if (event.type === "harvest_complete") {
    dom.actionBtnText.textContent = "Agent Executing...";
    fetchSystemStatus();
  } else if (event.type === "node_update") {
    const node = event.node;
    setActiveDagNode(node);

    if (node === "plan") {
      dom.loopBackIndicator.classList.add("hidden");
      renderHopPlan(event);
    } else if (node === "retrieve") {
      updateHopRetrieve(event);
    } else if (node === "rerank") {
      updateHopRerank(event);
    } else if (node === "reason") {
      updateHopReason(event);
    } else if (node === "verify") {
      updateHopVerify(event);
    } else if (node === "controller") {
      handleControllerUpdate(event);
    } else if (node === "report") {
      renderFinalReport(event);
      markDagAllCompleted();
    }
  } else if (event.type === "complete") {
    // All nodes done
    markDagAllCompleted();
    showToast("Research Mission Completed!");
  } else if (event.type === "error") {
    showToast(`Error: ${event.error}`);
  }
}

// =====================================================================
// Hop Evolution Rendering
// =====================================================================

function getOrCreateHopCard(hopIndex) {
  let card = document.getElementById(`hop-card-${hopIndex}`);
  if (!card) {
    card = document.createElement("div");
    card.className = "hop-card";
    card.id = `hop-card-${hopIndex}`;
    card.innerHTML = `
      <div class="hop-card-header">
        <div class="hop-card-title">
          <span>Hop ${hopIndex}</span>
          <span class="badge badge-info" id="hop-status-${hopIndex}">Executing...</span>
        </div>
        <span class="font-mono" style="font-size: 11px; color: var(--text-faint);">Click to expand/collapse</span>
      </div>
      <div class="hop-card-body" id="hop-body-${hopIndex}">
        <div id="hop-plan-section-${hopIndex}"></div>
        <div id="hop-retrieve-section-${hopIndex}"></div>
        <div id="hop-rerank-section-${hopIndex}"></div>
        <div id="hop-reason-section-${hopIndex}"></div>
        <div id="hop-verify-section-${hopIndex}"></div>
      </div>
    `;

    // Collapsible header toggle
    card.querySelector(".hop-card-header").addEventListener("click", () => {
      const body = card.querySelector(".hop-card-body");
      body.classList.toggle("hidden");
    });

    dom.hopsStreamContainer.appendChild(card);
    dom.hopBadgeCount.textContent = document.querySelectorAll(".hop-card").length;
  }
  return card;
}

function renderHopPlan(event) {
  const hop = event.hop || 0;
  if (dom.dagHopTag) {
    dom.dagHopTag.textContent = `Hop ${hop + 1}/${state.maxHops}`;
    dom.dagHopTag.classList.remove("hidden");
  }
  getOrCreateHopCard(hop);
  const planSection = document.getElementById(`hop-plan-section-${hop}`);
  if (planSection) {
    planSection.innerHTML = `
      <div class="hop-section-title">Node 1: Plan (Sub-Question)</div>
      <div class="hop-query-box">${escapeHtml(event.sub_question || "Using query")}</div>
    `;
  }
}

function updateHopRetrieve(event) {
  const hop = event.hop || 0;
  const retSection = document.getElementById(`hop-retrieve-section-${hop}`);
  if (retSection) {
    retSection.innerHTML = `
      <div class="hop-section-title">Node 2: Retrieve</div>
      <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 6px;">
        Retrieved <strong>${event.candidate_count || 0} candidate chunks</strong> from local FAISS vector database.
      </p>
    `;
  }
}

function updateHopRerank(event) {
  const hop = event.hop || 0;
  const chunks = event.reranked_chunks || [];
  
  // Track in hop history
  if (!state.hopHistory) state.hopHistory = [];
  if (!state.hopHistory[hop]) state.hopHistory[hop] = { hop_index: hop };
  state.hopHistory[hop].reranked_chunks = chunks;

  // Cache chunks for citation drawer lookup
  chunks.forEach(c => {
    state.latestRerankedChunks[c.source_num] = c;
  });

  const rerankSection = document.getElementById(`hop-rerank-section-${hop}`);
  if (rerankSection) {
    let rowsHtml = chunks.map(c => `
      <tr>
        <td><span class="source-num-pill">[${c.source_num}]</span></td>
        <td style="max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
          ${escapeHtml(c.paper_title)}
        </td>
        <td class="font-mono" style="color: #a5b4fc;">${c.cross_encoder_score}</td>
        <td>
          <a href="${c.paper_url}" target="_blank" style="color: var(--cyan); text-decoration: none;">${c.paper_url && c.paper_url.includes("arxiv.org") ? "arXiv ↗" : "DOI ↗"}</a>
        </td>
      </tr>
    `).join("");

    rerankSection.innerHTML = `
      <div class="hop-section-title">Node 3: Local Cross-Encoder Rerank (Top ${chunks.length})</div>
      <table class="reranked-mini-table">
        <thead>
          <tr>
            <th>Ref</th>
            <th>Paper Title</th>
            <th>Score</th>
            <th>Link</th>
          </tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    `;
  }
}

function updateHopReason(event) {
  const hop = event.hop || 0;
  const reasonSection = document.getElementById(`hop-reason-section-${hop}`);
  if (reasonSection) {
    reasonSection.innerHTML = `
      <div class="hop-section-title">Node 4: Synthesized Claims (Draft Answer)</div>
      <div style="background: var(--bg-surface); padding: 10px 12px; border-radius: var(--radius-sm); font-size: 13px; line-height: 1.6;">
        ${formatClaimsWithInteractiveCitations(event.draft_answer || "")}
      </div>
    `;
  }
}

function updateHopVerify(event) {
  const hop = event.hop || 0;
  const verifs = event.verifications || [];

  // Track in hop history
  if (!state.hopHistory) state.hopHistory = [];
  if (!state.hopHistory[hop]) state.hopHistory[hop] = { hop_index: hop };
  state.hopHistory[hop].verification_results = verifs;
  state.hopHistory[hop].evidence_sufficient = event.evidence_sufficient;
  state.hopHistory[hop].verified_count = event.verified_count;
  state.hopHistory[hop].total_claims = verifs.length;

  const verifySection = document.getElementById(`hop-verify-section-${hop}`);
  const statusBadge = document.getElementById(`hop-status-${hop}`);

  if (statusBadge) {
    statusBadge.textContent = event.evidence_sufficient ? "Evidence Sufficient" : "Incomplete Evidence";
    statusBadge.className = event.evidence_sufficient ? "badge" : "badge badge-info";
    statusBadge.style.background = event.evidence_sufficient ? "rgba(16, 185, 129, 0.2)" : "rgba(245, 158, 11, 0.2)";
    statusBadge.style.color = event.evidence_sufficient ? "#34d399" : "#fbbf24";
  }

  if (verifySection) {
    const rows = verifs.map(v => {
      const isEnt = v.is_entailed;
      const citPills = (v.citations || []).map(c => `<span class="citation-pill" data-source="${c}">[${c}]</span>`).join("");
      return `
        <div style="background: var(--bg-surface); border-left: 3px solid ${isEnt ? 'var(--verified)' : 'var(--unverified)'}; padding: 8px 10px; border-radius: var(--radius-sm); margin-bottom: 6px; font-size: 12px;">
          <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-weight: 700; color: ${isEnt ? '#34d399' : '#f87171'};">
              ${isEnt ? '✓ VERIFIED' : '✗ UNVERIFIED'}
            </span>
            <span>${citPills}</span>
          </div>
          <div>${escapeHtml(v.claim_text)}</div>
          ${!isEnt ? `<div style="color: #fca5a5; font-size: 11px; margin-top: 4px;">Reason: ${escapeHtml(v.reason)}</div>` : ''}
        </div>
      `;
    }).join("");

    verifySection.innerHTML = `
      <div class="hop-section-title">Node 5: Claim Verification Pass (${event.verified_count || 0} / ${verifs.length} Entailed)</div>
      ${rows}
    `;
  }
}

function handleControllerUpdate(event) {
  const hop = event.hop_count;
  const isSufficient = event.evidence_sufficient;

  if (!isSufficient) {
    dom.loopBackIndicator.classList.remove("hidden");
    dom.loopBackIndicator.innerHTML = `
      <span class="loop-back-icon">🔄</span>
      <span class="loop-back-text">Hop ${hop - 1} evidence incomplete. Looping back to PLAN for Reformulated Hop ${hop}...</span>
    `;
    if (dom.dagHopTag) {
      dom.dagHopTag.textContent = `Hop ${hop}/${state.maxHops}`;
      dom.dagHopTag.classList.remove("hidden");
    }
  } else {
    dom.loopBackIndicator.classList.add("hidden");
  }
}

// =====================================================================
// Final Report Rendering
// =====================================================================

// Helper to extract clean rationale from report text if needed
function cleanReportFindingText(text) {
  if (!text) return "";
  return text
    .replace(/^#.*$/gm, "")
    .replace(/^\*\*.*?\*\*.*$/gm, "")
    .replace(/[=\-]{3,}/g, "")
    .replace(/>\s*⚠️\s*\*\*Academic Scope Advisory:\*\*/gi, "")
    .replace(/No factual claims could be strictly entailed.*/gi, "")
    .replace(/## Working arXiv Evidence Sources[\s\S]*/gi, "")
    .replace(/AUDITED FINDINGS & VERIFIED CLAIMS:?/gi, "")
    .replace(/BIBLIOGRAPHY & WORKING ARXIV EVIDENCE SOURCES:?[\s\S]*/gi, "")
    .replace(/AI RESEARCH AGENT: FINAL AUDITED REPORT/gi, "")
    .replace(/Research Question\s*:.*?\n/gi, "")
    .replace(/Total Hops Executed\s*:.*?\n/gi, "")
    .replace(/Evidence Sufficiency\s*:.*?\n/gi, "")
    .replace(/Query Path Taken\s*:.*?\n/gi, "")
    .trim();
}

function renderFinalReport(event) {
  state.finalReportText = event.final_report || "";
  const verifs = event.verifications || [];
  const reranked = event.reranked_chunks || [];
  const queryHistory = event.query_history || [];
  const hops = (event.hop_history || []).length || queryHistory.length || 1;
  const originalQuery = event.original_query || (dom.researchQuery ? dom.researchQuery.value.trim() : "") || "Research Query";
  const draftAnswer = event.draft_answer || "";

  // Cache reranked chunks
  reranked.forEach(c => {
    if (c.source_num) {
      state.latestRerankedChunks[c.source_num] = c;
    }
  });

  // Extract all hops history (from server event or client stream)
  const allHops = (event.hop_history && event.hop_history.length > 0)
    ? event.hop_history
    : (state.hopHistory && state.hopHistory.length > 0 ? state.hopHistory : []);

  // Collect earlier unverified claims across hops (e.g. Hop 0)
  const earlierUnverified = [];
  allHops.forEach((h, idx) => {
    const hIdx = h.hop_index !== undefined ? h.hop_index : idx;
    if (hIdx < hops - 1 || (hIdx === 0 && hops > 1)) {
      const hopVerifs = h.verification_results || h.verifications || [];
      hopVerifs.forEach(v => {
        if (!v.is_entailed) {
          earlierUnverified.push({
            ...v,
            hopIndex: hIdx
          });
        }
      });
    }
  });

  // Calculate statistics with strict factual accuracy
  const totalClaims = verifs.length;
  const verifiedCount = verifs.filter(v => v.is_entailed).length;
  const ratePct = totalClaims > 0 ? Math.round((verifiedCount / totalClaims) * 100) : 0;
  
  // Strict evidence sufficiency: backend flag must be true AND at least one claim must be entailed
  const isSufficient = Boolean(event.evidence_sufficient) && totalClaims > 0 && verifiedCount === totalClaims;
  const hasPartialGaps = totalClaims > 0 && verifiedCount > 0 && !isSufficient;

  // Render Stats Banner
  dom.statHops.textContent = `${hops} Hop${hops > 1 ? 's' : ''}`;
  
  if (isSufficient) {
    dom.statEvidence.textContent = "SUFFICIENT";
    dom.statEvidence.style.color = "#34d399";
  } else if (hasPartialGaps) {
    dom.statEvidence.textContent = "PARTIAL GAPS";
    dom.statEvidence.style.color = "#fbbf24";
  } else {
    dom.statEvidence.textContent = "INSUFFICIENT";
    dom.statEvidence.style.color = "#f87171";
  }

  if (totalClaims === 0) {
    dom.statVerificationRate.textContent = "0%";
    dom.statVerificationRate.style.color = "#f87171";
    dom.statBreakdown.textContent = "0 Claims / Insufficient Evidence";
  } else {
    dom.statVerificationRate.textContent = `${ratePct}%`;
    dom.statVerificationRate.style.color = ratePct >= 80 ? "#34d399" : (ratePct >= 50 ? "#fbbf24" : "#f87171");
    if (earlierUnverified.length > 0) {
      dom.statBreakdown.textContent = `${verifiedCount} Verified (${earlierUnverified.length} Hop 0 Hallucinations Prevented)`;
    } else {
      dom.statBreakdown.textContent = `${verifiedCount} Verified / ${totalClaims - verifiedCount} Unverified`;
    }
  }

  // Animate Circular Trust Index Gauge
  if (dom.statTrustScore && dom.statGaugeCircle) {
    const trustScore = totalClaims > 0 ? ratePct : 0;
    dom.statTrustScore.textContent = `${trustScore}%`;
    dom.statGaugeCircle.setAttribute("stroke-dasharray", `${trustScore}, 100`);
    if (trustScore >= 80) {
      dom.statGaugeCircle.style.stroke = "#10b981";
      dom.statTrustScore.style.color = "#34d399";
    } else if (trustScore >= 50) {
      dom.statGaugeCircle.style.stroke = "#f59e0b";
      dom.statTrustScore.style.color = "#fbbf24";
    } else {
      dom.statGaugeCircle.style.stroke = "#f43f5e";
      dom.statTrustScore.style.color = "#f87171";
    }
  }

  dom.summaryStats.classList.remove("hidden");
  dom.dossierActions.classList.remove("hidden");

  // Render Dossier Content
  if (verifs && verifs.length > 0) {
    // SCENARIO 1: Structured Claims Extracted & Audited
    let dossierHtml = "";

    // 1. Executive Grounded Synthesis Card
    if (draftAnswer) {
      dossierHtml += `
        <div class="executive-synthesis-card">
          <div class="synthesis-header">
            <span class="synthesis-badge">
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" style="display:inline-block; vertical-align: -2px; margin-right: 5px;">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
              </svg>
              Audited Research Synthesis
            </span>
            <span class="synthesis-metric">${verifiedCount} / ${totalClaims} Claims Verified</span>
          </div>
          <div class="synthesis-body">
            ${formatClaimsWithInteractiveCitations(draftAnswer)}
          </div>
        </div>
      `;
    }

    // 2. Itemized Entailment Audit Header
    dossierHtml += `
      <div class="claims-audit-header">
        <h4>Audited Factual Claims (${verifiedCount} of ${totalClaims} Entailed)</h4>
        <span class="claims-subtitle">Click citation pills to view mathematical entailment passage</span>
      </div>
    `;

    // 3. Itemized Claim Cards
    dossierHtml += verifs.map(v => {
      const isEnt = v.is_entailed;
      const citPills = (v.citations || []).map(c => `<span class="citation-pill" data-source="${c}">[${c}]</span>`).join("");
      
      return `
        <article class="claim-card ${isEnt ? 'verified' : 'unverified'}">
          <div class="claim-header">
            <span class="verdict-pill ${isEnt ? 'verified' : 'unverified'}">
              ${isEnt ? '✓ VERIFIED CLAIM' : '✕ UNVERIFIED ASSERTION'}
            </span>
            <div class="claim-citations">${citPills}</div>
          </div>
          <p class="claim-text">${escapeHtml(v.claim_text)}</p>
          ${!isEnt ? `
            <div class="claim-reason">
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink: 0; margin-top: 2px;">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
              </svg>
              <span><strong>Audit Reason:</strong> ${escapeHtml(v.reason || "Source evidence does not entail this assertion.")}</span>
            </div>
          ` : ''}
        </article>
      `;
    }).join("");

    // 4. Multi-Hop Self-Correction & Refuted Claims Log (Hop 0 / Earlier Hops)
    if (earlierUnverified.length > 0) {
      dossierHtml += `
        <div class="hop-refutations-audit-card">
          <div class="refutation-header">
            <span class="refutation-badge">
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" style="display:inline-block; vertical-align: -2px; margin-right: 5px;">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
              Autonomous Agent Self-Correction Log
            </span>
            <span class="refutation-metric">${earlierUnverified.length} Hop 0 Assertion${earlierUnverified.length > 1 ? 's' : ''} Refuted & Resolved</span>
          </div>
          <p class="refutation-explainer">
            During preliminary reasoning in <strong>Hop 0</strong>, the agent extracted candidate claims that were evaluated against arXiv preprints. The NLI entailment auditor detected factual discrepancies or lack of peer-reviewed premise support, marked them as <strong>UNVERIFIED</strong>, and triggered loopback query reformulation:
          </p>
          <div class="refutation-items-list">
            ${earlierUnverified.map(u => `
              <div class="refutation-item">
                <div class="refutation-top">
                  <span class="refutation-pill">✕ REFUTED IN HOP ${u.hopIndex}</span>
                  <span class="refutation-action">Agent Decision: Reformulated Query in Next Hop ➔</span>
                </div>
                <div class="refutation-claim-text">${escapeHtml(u.claim_text)}</div>
                <div class="refutation-reason-box">
                  <strong>Auditor Rejection Proof:</strong> ${escapeHtml(u.reason || "Evidence does not strictly entail claim.")}
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }

    dom.dossierContent.innerHTML = dossierHtml;

  } else {
    // SCENARIO 2: Insufficient Evidence / Out of Academic Scope
    let cleanFinding = draftAnswer;
    if (!cleanFinding || cleanFinding.includes("=====")) {
      cleanFinding = cleanReportFindingText(event.final_report);
    }
    if (!cleanFinding) {
      cleanFinding = "The retrieved papers do not provide sufficient peer-reviewed empirical evidence to formulate and verify factual assertions for this inquiry.";
    }

    dom.dossierContent.innerHTML = `
      <div class="advisory-card insufficient-evidence-card">
        <div class="advisory-header">
          <div class="advisory-badge-wrap">
            <span class="advisory-badge-icon">
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
              </svg>
            </span>
            <span class="advisory-badge-title">INSUFFICIENT ACADEMIC EVIDENCE ON arXiv</span>
          </div>
          <span class="status-chip-danger">0 Claims Entailed</span>
        </div>

        <div class="advisory-body">
          <h3 class="advisory-headline">No Entailed Literature Found on arXiv</h3>
          
          <div class="advisory-query-box">
            <div class="advisory-query-label">RESEARCH QUESTION</div>
            <div class="advisory-query-text">"${escapeHtml(originalQuery)}"</div>
          </div>

          <div class="advisory-rationale-box">
            <div class="rationale-header">
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" class="rationale-icon">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
              </svg>
              <strong>Autonomous Auditor Finding:</strong>
            </div>
            <p class="rationale-text">${escapeHtml(cleanFinding)}</p>
            <div class="guardrail-note">
              <strong>Zero-Hallucination Guardrail:</strong> PROVENANCE strictly audits peer-reviewed preprints. Because retrieved passages failed to establish textual entailment, the auditor refused to formulate unverified assertions.
            </div>
          </div>

          ${queryHistory && queryHistory.length > 0 ? `
            <div class="advisory-trajectory-box">
              <div class="trajectory-label">AGENT REASONING TRAJECTORY (${queryHistory.length} Hops Executed)</div>
              <div class="trajectory-chips">
                ${queryHistory.map((q, idx) => `
                  <div class="trajectory-chip">
                    <span class="chip-hop-num">Hop ${idx + 1}</span>
                    <span class="chip-query">"${escapeHtml(q)}"</span>
                  </div>
                `).join('<span class="trajectory-arrow">➔</span>')}
              </div>
            </div>
          ` : ''}

          ${reranked && reranked.length > 0 ? `
            <div class="advisory-rejected-sources">
              <div class="rejected-label">EVALUATED CANDIDATE PAPERS (Cross-Encoder Rejection)</div>
              <div class="rejected-papers-list">
                ${reranked.slice(0, 4).map(c => `
                  <div class="rejected-paper-item">
                    <div class="rejected-paper-info">
                      <span class="paper-score-tag ${c.cross_encoder_score < -3 ? 'score-negative' : ''}">
                        CE Score: ${typeof c.cross_encoder_score === 'number' ? c.cross_encoder_score.toFixed(2) : '-'}
                      </span>
                      <span class="rejected-title" title="${escapeHtml(c.paper_title)}">${escapeHtml(c.paper_title)}</span>
                    </div>
                    <a href="${c.paper_url}" target="_blank" rel="noopener noreferrer" class="rejected-link">
                      ${c.paper_url && c.paper_url.includes("arxiv.org") ? "arXiv ↗" : "DOI ↗"}
                    </a>
                  </div>
                `).join('')}
              </div>
            </div>
          ` : ''}

          <div class="advisory-guidance-box">
            <span class="guidance-icon">💡</span>
            <div class="guidance-text">
              <strong>Domain Scope Recommendation:</strong> PROVENANCE retrieves preprints in Computer Science, Artificial Intelligence, Physics, Mathematics, and Quantitative Biology. For general non-academic topics (such as geographic trivia or general world facts), scientific preprints on arXiv do not contain relevant literature.
            </div>
          </div>
        </div>
      </div>
    `;
  }

  // Populate Sources & Working ArXiv Links Tab
  renderSourcesTab(reranked);
  markDagAllCompleted();

  // Add-on: Render Interactive Citation Knowledge Graph with full multi-hop history
  renderCitationGraph(originalQuery, reranked, verifs, allHops);

  // Add-on: Persist research mission to Session History
  saveMissionToHistory(event, originalQuery, verifs, reranked);
}

function renderSourcesTab(chunks) {
  const seenUrls = new Set();
  const uniquePapers = [];

  chunks.forEach(c => {
    if (!seenUrls.has(c.paper_url)) {
      seenUrls.add(c.paper_url);
      uniquePapers.push(c);
    }
  });

  dom.sourceBadgeCount.textContent = uniquePapers.length;

  if (uniquePapers.length === 0) {
    dom.sourcesList.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📖</div>
        <h3 class="empty-title">No Citations Recorded</h3>
      </div>
    `;
    return;
  }

  const itemsHtml = uniquePapers.map(p => `
    <div class="source-item-card">
      <div class="source-item-header">
        <span class="source-num-pill">[${p.source_num}]</span>
        <h4 class="source-title">${escapeHtml(p.paper_title)}</h4>
      </div>
      <div style="display: flex; gap: 8px; align-items: center; margin: 4px 0 8px 0; flex-wrap: wrap;">
        <a href="${p.paper_url}" target="_blank" rel="noopener noreferrer" class="source-url">
          <span>🔗 Abstract</span> <span>↗</span>
        </a>
        <a href="${p.paper_url ? p.paper_url.replace('/abs/', '/pdf/') + '.pdf' : '#'}" target="_blank" rel="noopener noreferrer" class="source-url" style="color: var(--primary); border-color: rgba(99, 102, 241, 0.4);">
          <span>📄 Direct PDF</span> <span>↗</span>
        </a>
      </div>
      <p style="font-size: 12px; color: var(--text-muted); margin-top: 4px; line-height: 1.5;">
        ${escapeHtml(p.text.substring(0, 180))}...
      </p>
    </div>
  `).join("");

  dom.sourcesList.innerHTML = itemsHtml;
}

// =====================================================================
// ArXiv Ingestion Modal Handlers
// =====================================================================

if (dom.btnOpenHarvest) {
  dom.btnOpenHarvest.addEventListener("click", () => {
    if (dom.modalCorpus) dom.modalCorpus.classList.add("hidden");
    dom.modalHarvest.classList.remove("hidden");
    dom.harvestQuery.focus();
  });
}

dom.btnCloseHarvest.addEventListener("click", () => {
  dom.modalHarvest.classList.add("hidden");
});

dom.btnCancelHarvest.addEventListener("click", () => {
  dom.modalHarvest.classList.add("hidden");
});

dom.harvestForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = dom.harvestQuery.value.trim();
  const maxResults = parseInt(dom.harvestMaxResults.value, 10);
  if (!query) return;

  dom.btnStartHarvest.disabled = true;
  dom.harvestLogArea.classList.remove("hidden");
  dom.harvestLogOutput.textContent = `1. Fetching ${maxResults} papers matching '${query}' from arXiv Atom API...\n`;

  try {
    const res = await fetch("/api/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, max_results: maxResults })
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Ingestion failed");
    }

    dom.harvestLogOutput.textContent += `2. Chunking abstracts into overlapping segments...\n`;
    dom.harvestLogOutput.textContent += `3. Generating Gemini text embeddings & writing to local FAISS index...\n`;
    dom.harvestLogOutput.textContent += `\n[COMPLETE] Successfully indexed ${data.chunks_count} chunks from ${data.papers_count} papers in ${data.elapsed_seconds}s!`;

    showToast(`Indexed ${data.chunks_count} chunks into FAISS!`);
    await fetchSystemStatus();

    setTimeout(() => {
      dom.modalHarvest.classList.add("hidden");
      dom.btnStartHarvest.disabled = false;
      dom.harvestLogArea.classList.add("hidden");
    }, 1500);

  } catch (err) {
    dom.harvestLogOutput.textContent += `\n[ERROR] ${err.message}`;
    dom.btnStartHarvest.disabled = false;
    showToast(`Ingestion error: ${err.message}`);
  }
});

// =====================================================================
// Corpus Library Modal Handlers
// =====================================================================

dom.btnOpenCorpus.addEventListener("click", async () => {
  dom.modalCorpus.classList.remove("hidden");
  dom.corpusTbody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 20px;">Loading library...</td></tr>`;

  try {
    const res = await fetch("/api/papers");
    const data = await res.json();
    const papers = data.papers || [];
    dom.corpusCount.textContent = papers.length;

    if (papers.length === 0) {
      dom.corpusTbody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 20px; color: var(--text-muted);">No papers indexed yet. Click "Ingest arXiv" to harvest literature.</td></tr>`;
      return;
    }

    dom.corpusTbody.innerHTML = papers.map((p, idx) => {
      const isArxiv = (p.url && p.url.includes("arxiv.org")) || (p.id && /^\d{4}\.\d{4,5}/.test(p.id));
      const sourceLabel = isArxiv ? "arXiv ↗" : "DOI ↗";
      const pdfUrl = isArxiv
        ? (p.url ? p.url.replace('/abs/', '/pdf/') + '.pdf' : '#')
        : (p.url || '#');

      return `
      <tr>
        <td><strong>${idx + 1}</strong></td>
        <td><strong>${escapeHtml(p.title)}</strong></td>
        <td>${p.published ? p.published.substring(0, 10) : 'N/A'}</td>
        <td class="font-mono" style="font-size: 0.82rem;">${p.id || 'N/A'}</td>
        <td>
          <a href="${p.url}" target="_blank" rel="noopener noreferrer">${sourceLabel}</a>
          <span style="color: var(--text-faint); margin: 0 4px;">•</span>
          <a href="${pdfUrl}" target="_blank" rel="noopener noreferrer" style="color: var(--primary);">${isArxiv ? 'PDF ↗' : 'Source ↗'}</a>
        </td>
      </tr>
      `;
    }).join("");
  } catch (err) {
    dom.corpusTbody.innerHTML = `<tr><td colspan="5" style="color: #f87171; text-align: center; padding: 20px;">Failed to load corpus: ${err.message}</td></tr>`;
  }
});

dom.btnCloseCorpus.addEventListener("click", () => {
  dom.modalCorpus.classList.add("hidden");
});

// =====================================================================
// Export & Copy Handlers
// =====================================================================

dom.btnCopyDossier.addEventListener("click", async () => {
  if (!state.finalReportText) return;
  try {
    await navigator.clipboard.writeText(state.finalReportText);
    showToast("Audited research dossier copied to clipboard!");
  } catch (err) {
    showToast("Failed to copy text.");
  }
});

dom.btnDownloadMarkdown.addEventListener("click", () => {
  if (!state.finalReportText) return;
  const blob = new Blob([state.finalReportText], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `academic_research_dossier_${Date.now()}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast("Downloaded Markdown Dossier!");
});

// =====================================================================
// Utility Helpers
// =====================================================================

function formatClaimsWithInteractiveCitations(text) {
  if (!text) return "";
  // Turn [1], [2] into clickable citation pills
  return escapeHtml(text).replace(/\[(\d+)\]/g, (match, p1) => {
    return `<span class="citation-pill" data-source="${p1}">[${p1}]</span>`;
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  dom.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// =====================================================================
// Addon 1: BibTeX Export (.bib) Generator
// =====================================================================

function generateBibtexContent(chunks) {
  const seenUrls = new Set();
  const papers = [];
  chunks.forEach(c => {
    if (c.paper_url && !seenUrls.has(c.paper_url)) {
      seenUrls.add(c.paper_url);
      papers.push(c);
    }
  });

  if (papers.length === 0) return "";

  return papers.map((p, idx) => {
    let arxivId = "";
    if (p.paper_url) {
      const match = p.paper_url.match(/arxiv\.org\/(?:abs|pdf)\/([0-9]+\.[0-9]+|[a-z\-]+(?:\.[A-Z]{2})?\/\d+)/i);
      arxivId = match ? match[1] : `preprint_${idx + 1}`;
    } else {
      arxivId = `paper_${idx + 1}`;
    }

    const citeKey = `arxiv_${arxivId.replace(/[^a-zA-Z0-9]/g, "_")}`;
    const cleanTitle = (p.paper_title || "Untitled Preprint").replace(/[{}]/g, "");
    const yearMatch = arxivId.match(/^(\d{2})/);
    const pubYear = yearMatch ? `20${yearMatch[1]}` : "2024";

    return `@misc{${citeKey},
  title         = {{${cleanTitle}}},
  author        = {{arXiv Preprint Authors}},
  year          = {${pubYear}},
  eprint        = {${arxivId}},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {${p.paper_url || ""}}
}`;
  }).join("\n\n");
}

if (dom.btnExportBibtex) {
  dom.btnExportBibtex.addEventListener("click", async () => {
    const chunks = Object.values(state.latestRerankedChunks);
    if (!chunks || chunks.length === 0) {
      showToast("No cited papers available to export as BibTeX.");
      return;
    }

    const bibtexStr = generateBibtexContent(chunks);
    if (!bibtexStr) {
      showToast("No valid paper citations found.");
      return;
    }

    // Download .bib file
    const blob = new Blob([bibtexStr], { type: "application/x-bibtex;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `provenance_citations_${Date.now()}.bib`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // Also copy to clipboard
    try {
      await navigator.clipboard.writeText(bibtexStr);
      showToast("BibTeX citations downloaded (.bib) and copied to clipboard!");
    } catch (e) {
      showToast("BibTeX file downloaded!");
    }
  });
}

// =====================================================================
// Addon 2: Interactive Citation & Knowledge Network Graph Engine
// =====================================================================

const graphState = {
  nodes: [],
  edges: [],
  hoveredNode: null,
  transform: { x: 0, y: 0, scale: 1 },
  initialized: false
};

function renderCitationGraph(query, chunks, verifications, hopHistory) {
  const canvas = dom.citationGraphCanvas;
  if (!canvas) return;

  // 1. Collect unique papers across both current chunks AND all earlier hops
  const seenUrls = new Set();
  const papers = [];
  (chunks || []).forEach(c => {
    if (c.paper_url && !seenUrls.has(c.paper_url)) {
      seenUrls.add(c.paper_url);
      papers.push(c);
    }
  });

  (hopHistory || []).forEach(h => {
    (h.reranked_chunks || []).forEach(c => {
      if (c.paper_url && !seenUrls.has(c.paper_url)) {
        seenUrls.add(c.paper_url);
        papers.push(c);
      }
    });
  });

  const verifs = verifications || [];

  // 2. Collect any unverified / refuted claims from earlier hops (e.g. Hop 0)
  const earlierUnverified = [];
  (hopHistory || []).forEach((h, idx) => {
    const hIdx = h.hop_index !== undefined ? h.hop_index : idx;
    // Collect from previous hops if more than 1 hop occurred
    if (hIdx < (hopHistory.length - 1) || (hIdx === 0 && hopHistory.length > 1)) {
      const hopVerifs = h.verification_results || h.verifications || [];
      hopVerifs.forEach((v, cIdx) => {
        if (!v.is_entailed) {
          earlierUnverified.push({
            ...v,
            hopIndex: hIdx,
            originalIndex: cIdx + 1
          });
        }
      });
    }
  });

  if (papers.length === 0 && verifs.length === 0 && earlierUnverified.length === 0) {
    if (dom.graphEmptyOverlay) dom.graphEmptyOverlay.classList.remove("hidden");
    return;
  }

  if (dom.graphEmptyOverlay) dom.graphEmptyOverlay.classList.add("hidden");

  // Build Graph Nodes & Edges
  const nodes = [];
  const edges = [];

  // 1. Root Query Node
  const rootNode = {
    id: "root-query",
    type: "query",
    label: query ? (query.length > 26 ? query.substring(0, 24) + "..." : query) : "Research Query",
    fullText: query || "Academic Query",
    x: 0,
    y: 0,
    radius: 26,
    color: "#6366f1",
    glowColor: "rgba(99, 102, 241, 0.35)",
    stroke: "#818cf8"
  };
  nodes.push(rootNode);

  // 2. Paper Nodes (Middle orbit)
  const paperNodes = [];
  const paperCount = papers.length;
  const paperRadius = 150;

  papers.forEach((p, idx) => {
    const angle = (2 * Math.PI / Math.max(paperCount, 1)) * idx - Math.PI / 2;
    const px = Math.cos(angle) * paperRadius;
    const py = Math.sin(angle) * paperRadius;

    const paperNode = {
      id: `paper-${p.source_num}`,
      sourceNum: p.source_num,
      type: "paper",
      label: `[${p.source_num}] arXiv:${(p.paper_url || '').split('/').pop() || 'Preprint'}`,
      title: p.paper_title || "Preprint",
      url: p.paper_url,
      score: p.cross_encoder_score,
      x: px,
      y: py,
      radius: 19,
      color: "#06b6d4",
      glowColor: "rgba(6, 182, 212, 0.3)",
      stroke: "#38bdf8"
    };
    nodes.push(paperNode);
    paperNodes.push(paperNode);

    // Edge from Root -> Paper (Clean line without raw floats!)
    edges.push({
      source: rootNode,
      target: paperNode,
      color: "rgba(99, 102, 241, 0.4)",
      width: 1.5,
      weight: "" // Keep the connecting wire clean and uncluttered
    });
  });

  // 3. Claim / Verification Nodes (Outer orbit: both Final claims + Hop 0 Refuted claims)
  const allClaims = [
    ...verifs.map((v, idx) => ({ ...v, isEarlier: false, displayIdx: idx + 1 })),
    ...earlierUnverified.map(v => ({ ...v, isEarlier: true, displayIdx: v.originalIndex }))
  ];

  const totalClaimCount = allClaims.length;
  const claimRadius = 260;

  allClaims.forEach((cItem, idx) => {
    const angle = (2 * Math.PI / Math.max(totalClaimCount, 1)) * idx - Math.PI / 2 + 0.15;
    const cx = Math.cos(angle) * claimRadius;
    const cy = Math.sin(angle) * claimRadius;

    const isEnt = Boolean(cItem.is_entailed);
    const isEarlier = Boolean(cItem.isEarlier);

    let labelText = `Claim #${cItem.displayIdx} (${isEnt ? '✓' : '✕'})`;
    if (isEarlier) {
      labelText = `[Hop ${cItem.hopIndex}] Claim #${cItem.displayIdx} (✕)`;
    }

    const claimNode = {
      id: isEarlier ? `earlier-hop-${cItem.hopIndex}-claim-${cItem.displayIdx}` : `claim-${cItem.displayIdx}`,
      type: "claim",
      label: labelText,
      text: cItem.claim_text || "Asserted claim",
      isEntailed: isEnt,
      isEarlierHop: isEarlier,
      hopIndex: cItem.hopIndex,
      reason: cItem.reason,
      citations: cItem.citations || [],
      x: cx,
      y: cy,
      radius: 16,
      color: isEnt ? "#10b981" : "#f43f5e",
      glowColor: isEnt ? "rgba(16, 185, 129, 0.35)" : "rgba(244, 63, 94, 0.45)",
      stroke: isEnt ? "#34d399" : "#fb7185"
    };
    nodes.push(claimNode);

    // Connect Claim to its citing Papers
    let connected = false;
    (cItem.citations || []).forEach(citNum => {
      const pNode = paperNodes.find(pn => pn.sourceNum === citNum);
      if (pNode) {
        connected = true;
        edges.push({
          source: pNode,
          target: claimNode,
          color: isEnt ? "rgba(16, 185, 129, 0.45)" : "rgba(244, 63, 94, 0.55)",
          width: 2,
          weight: isEarlier ? `Hop ${cItem.hopIndex} Refuted` : (isEnt ? "Entails" : "Refutes"),
          isEntailed: isEnt
        });
      }
    });

    if (!connected) {
      edges.push({
        source: rootNode,
        target: claimNode,
        color: isEarlier ? "rgba(244, 63, 94, 0.35)" : "rgba(148, 163, 184, 0.2)",
        width: 1.5,
        weight: isEarlier ? `Hop ${cItem.hopIndex} Refuted` : (isEnt ? "Entails" : "Unverified"),
        isEntailed: isEnt
      });
    }
  });

  graphState.nodes = nodes;
  graphState.edges = edges;
  graphState.transform = { x: 0, y: 0, scale: 1 };

  initCanvasEvents();
  drawCanvasGraph();
}

function initCanvasEvents() {
  const canvas = dom.citationGraphCanvas;
  if (!canvas || graphState.initialized) return;
  graphState.initialized = true;

  window.addEventListener("resize", () => {
    drawCanvasGraph();
  });

  if (dom.btnRecenterGraph) {
    dom.btnRecenterGraph.addEventListener("click", () => {
      graphState.transform = { x: 0, y: 0, scale: 1 };
      drawCanvasGraph();
    });
  }

  let isDown = false;
  let startX = 0, startY = 0;

  canvas.addEventListener("mousedown", (e) => {
    isDown = true;
    startX = e.clientX - graphState.transform.x;
    startY = e.clientY - graphState.transform.y;
    canvas.style.cursor = "grabbing";
  });

  window.addEventListener("mouseup", () => {
    if (isDown) {
      isDown = false;
      canvas.style.cursor = graphState.hoveredNode ? "pointer" : "grab";
    }
  });

  canvas.addEventListener("mousemove", (e) => {
    if (isDown) {
      graphState.transform.x = e.clientX - startX;
      graphState.transform.y = e.clientY - startY;
      drawCanvasGraph();
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const mx = (e.clientX - rect.left) - (canvas.width / (2 * dpr) + graphState.transform.x);
    const my = (e.clientY - rect.top) - (canvas.height / (2 * dpr) + graphState.transform.y);

    let hit = null;
    for (let i = graphState.nodes.length - 1; i >= 0; i--) {
      const n = graphState.nodes[i];
      const dist = Math.hypot((mx / graphState.transform.scale) - n.x, (my / graphState.transform.scale) - n.y);
      if (dist <= n.radius + 6) {
        hit = n;
        break;
      }
    }

    if (hit !== graphState.hoveredNode) {
      graphState.hoveredNode = hit;
      canvas.style.cursor = hit ? "pointer" : "grab";
      drawCanvasGraph();
    }
  });

  canvas.addEventListener("click", () => {
    if (graphState.hoveredNode) {
      const node = graphState.hoveredNode;
      if (node.type === "paper" && node.sourceNum) {
        openCitationDrawer(node.sourceNum);
      } else if (node.type === "claim") {
        showToast(`${node.label}: ${node.text.substring(0, 90)}...`);
      }
    }
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.08 : 0.92;
    graphState.transform.scale = Math.min(Math.max(graphState.transform.scale * zoomFactor, 0.4), 2.5);
    drawCanvasGraph();
  }, { passive: false });
}

function drawCanvasGraph() {
  const canvas = dom.citationGraphCanvas;
  if (!canvas) return;

  const wrap = dom.graphCanvasWrap || canvas.parentElement;
  const rect = wrap ? wrap.getBoundingClientRect() : { width: 800, height: 500 };
  const dpr = window.devicePixelRatio || 1;

  const w = rect.width || 800;
  const h = rect.height || 500;

  if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
  }

  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.scale(dpr, dpr);

  const cx = w / 2 + graphState.transform.x;
  const cy = h / 2 + graphState.transform.y;

  ctx.translate(cx, cy);
  ctx.scale(graphState.transform.scale, graphState.transform.scale);

  // Draw Edges
  graphState.edges.forEach(edge => {
    ctx.beginPath();
    ctx.moveTo(edge.source.x, edge.source.y);
    ctx.lineTo(edge.target.x, edge.target.y);
    ctx.strokeStyle = edge.color;
    ctx.lineWidth = edge.width;
    ctx.stroke();

    // Render edge weight in a clean translucent pill badge
    if (edge.weight) {
      const mx = (edge.source.x + edge.target.x) / 2;
      const my = (edge.source.y + edge.target.y) / 2;
      ctx.save();
      ctx.font = "bold 9px Inter, sans-serif";
      const tw = ctx.measureText(edge.weight).width;
      const pw = tw + 10;
      const ph = 15;
      const isEnt = Boolean(edge.isEntailed);
      ctx.fillStyle = isEnt ? "rgba(6, 78, 59, 0.9)" : "rgba(136, 19, 55, 0.9)";
      ctx.strokeStyle = isEnt ? "#10b981" : "#f43f5e";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(mx - pw / 2, my - ph / 2, pw, ph, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = isEnt ? "#a7f3d0" : "#fecdd3";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(edge.weight, mx, my);
      ctx.restore();
    }
  });

  // Draw Nodes
  graphState.nodes.forEach(node => {
    const isHovered = (graphState.hoveredNode === node);
    const r = isHovered ? node.radius + 3 : node.radius;

    // Outer Glow Ring
    ctx.beginPath();
    ctx.arc(node.x, node.y, r + (isHovered ? 6 : 3), 0, 2 * Math.PI);
    ctx.fillStyle = node.glowColor;
    ctx.fill();

    // Node Circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = node.color;
    ctx.fill();
    ctx.strokeStyle = node.stroke;
    ctx.lineWidth = isHovered ? 2.5 : 1.5;
    ctx.stroke();

    // Node Label
    ctx.fillStyle = "#ffffff";
    ctx.font = isHovered ? "bold 11px Inter, sans-serif" : "10px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(node.label, node.x, node.y + r + 6);
  });

  // Draw Tooltip for Hovered Node
  if (graphState.hoveredNode) {
    const n = graphState.hoveredNode;
    ctx.save();
    let title = n.label;
    let detail = "";
    if (n.type === "query") {
      title = "Research Mission";
      detail = n.fullText || "";
    } else if (n.type === "paper") {
      title = `[${n.sourceNum}] ${n.title}`;
      const scoreVal = Number(n.score);
      const scoreFormatted = isNaN(scoreVal) ? "N/A" : (scoreVal > 0 ? `+${scoreVal.toFixed(2)}` : scoreVal.toFixed(2));
      const relPct = isNaN(scoreVal) ? 75 : Math.round(100 / (1 + Math.exp(-scoreVal * 0.4)));
      detail = `Cross-Encoder: ${scoreFormatted} • Neural Match: ${relPct}%`;
    } else if (n.type === "claim") {
      title = n.isEarlierHop
        ? `✕ Refuted in Hop ${n.hopIndex} (Self-Corrected)`
        : (n.isEntailed ? "✓ Entailed Factual Claim" : "✕ Unverified Assertion");
      detail = `${n.text}${n.reason ? ` | Audit: ${n.reason}` : ''}`;
    }

    ctx.font = "bold 11px Inter, sans-serif";
    const titleWidth = ctx.measureText(title).width;
    ctx.font = "10px Inter, sans-serif";
    const detailSnippet = detail.length > 60 ? detail.substring(0, 57) + "..." : detail;
    const detailWidth = ctx.measureText(detailSnippet).width;
    const boxW = Math.max(titleWidth, detailWidth) + 26;
    const boxH = 46;
    const boxX = n.x - boxW / 2;
    const boxY = n.y - n.radius - boxH - 12;

    ctx.fillStyle = "rgba(15, 23, 42, 0.96)";
    ctx.strokeStyle = n.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, boxW, boxH, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(title, n.x, boxY + 11);

    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px Inter, sans-serif";
    ctx.fillText(detailSnippet, n.x, boxY + 28);
    ctx.restore();
  }

  ctx.restore();
}

// =====================================================================
// Addon 3: Research Mission History & Session Persistence Manager
// =====================================================================

const HISTORY_STORAGE_KEY = "provenance_research_history_v1";

function getMissionHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    return [];
  }
}

function saveMissionToHistory(event, query, verifs, chunks) {
  try {
    const history = getMissionHistory();
    const totalClaims = verifs ? verifs.length : 0;
    const verifiedCount = verifs ? verifs.filter(v => v.is_entailed).length : 0;
    const trustScore = totalClaims > 0 ? Math.round((verifiedCount / totalClaims) * 100) : 0;

    const record = {
      id: `session_${Date.now()}`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' }),
      query: query || "Academic Research Query",
      verifiedCount,
      totalClaims,
      trustScore,
      evidenceVerdict: event.evidence_sufficient ? "SUFFICIENT" : "PARTIAL / INSUFFICIENT",
      papersCount: chunks ? chunks.length : 0,
      eventData: event
    };

    history.unshift(record);
    if (history.length > 20) history.pop();

    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
    updateHistoryBadge();
  } catch (e) {
    console.warn("Unable to save research mission to localStorage:", e);
  }
}

function updateHistoryBadge() {
  const history = getMissionHistory();
  if (dom.historyCount) {
    dom.historyCount.textContent = history.length;
  }
}

function renderHistoryModal() {
  const history = getMissionHistory();
  updateHistoryBadge();

  if (!dom.historyList) return;

  if (history.length === 0) {
    dom.historyList.innerHTML = `
      <div class="empty-state" style="padding: 40px 20px;">
        <div class="empty-icon">📜</div>
        <h4 style="color: var(--text); margin-top: 10px;">No Missions in History</h4>
        <p style="color: var(--text-muted); font-size: 13px; max-width: 320px; margin: 6px auto;">
          Run an audited research query to automatically record sessions and restore them here.
        </p>
      </div>
    `;
    return;
  }

  dom.historyList.innerHTML = history.map(item => `
    <div class="history-item" data-session-id="${item.id}">
      <div class="history-meta">
        <span class="history-query" title="${escapeHtml(item.query)}">${escapeHtml(item.query)}</span>
        <div class="history-tags">
          <span>🕒 ${item.timestamp}</span>
          <span>•</span>
          <span>${item.verifiedCount}/${item.totalClaims} Claims Entailed</span>
          <span>•</span>
          <span>${item.papersCount} Cited Papers</span>
          <span class="history-badge ${item.trustScore >= 80 ? 'verified' : (item.trustScore >= 50 ? 'partial' : 'unverified')}">
            ${item.trustScore}% Trust
          </span>
        </div>
      </div>
      <div class="history-actions">
        <button type="button" class="btn-restore-history" data-session-id="${item.id}">
          Restore Dossier ↺
        </button>
      </div>
    </div>
  `).join("");
}

// History Modal Events
if (dom.btnOpenHistory) {
  dom.btnOpenHistory.addEventListener("click", () => {
    renderHistoryModal();
    if (dom.modalHistory) dom.modalHistory.classList.remove("hidden");
  });
}

if (dom.btnCloseHistory) {
  dom.btnCloseHistory.addEventListener("click", () => {
    if (dom.modalHistory) dom.modalHistory.classList.add("hidden");
  });
}

if (dom.btnClearHistory) {
  dom.btnClearHistory.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_STORAGE_KEY);
    renderHistoryModal();
    showToast("Audit research mission history cleared.");
  });
}

// Restore click event delegation
document.addEventListener("click", (e) => {
  const restoreBtn = e.target.closest(".btn-restore-history");
  if (!restoreBtn) return;

  const sessionId = restoreBtn.getAttribute("data-session-id");
  const history = getMissionHistory();
  const found = history.find(h => h.id === sessionId);

  if (found && found.eventData) {
    if (dom.researchQuery) dom.researchQuery.value = found.query;
    renderFinalReport(found.eventData);

    // Switch to Findings tab
    const findingsBtn = document.querySelector('.tab-btn[data-tab="tab-findings"]');
    if (findingsBtn) findingsBtn.click();

    if (dom.modalHistory) dom.modalHistory.classList.add("hidden");
    showToast(`Restored audited research dossier: "${found.query}"`);
  }
});

// Initial bootstrap
updateHistoryBadge();
fetchSystemStatus();
