// Logic tương tác toàn diện cho Web Interactive Báo Cáo Giữa Kỳ NLP
// Tham chiếu tri thức từ NLP_DATA (data.js)

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initProgressTracker();
  initVectorPlayground();
  initWord2VecSimulator();
  initAvgW2vFlawDemo();
  initLstmSimulator();
  initAttentionExplorer();
  initBenchmarkAnalytics();
  initDefenseSimulator();
  initAiAssistant();
  initQuiz();
});

/* ==========================================================================
   1. NAVIGATION & TAB SWITCHING
   ========================================================================== */
function initNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  const modules = document.querySelectorAll(".learning-module");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-target");
      
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      modules.forEach(mod => {
        if (mod.id === targetId) {
          mod.classList.remove("hidden");
          mod.classList.add("block");
        } else {
          mod.classList.remove("block");
          mod.classList.add("hidden");
        }
      });

      // Cuộn lên đầu vùng học
      const mainContainer = document.getElementById("main-content");
      if (mainContainer) {
        mainContainer.scrollIntoView({ behavior: "smooth" });
      }

      // Kích hoạt render lại canvas nếu chuyển sang tab Vector
      if (targetId === "module-vector") {
        drawVectorCanvas();
      }
      // Kích hoạt render chart nếu chuyển sang tab Benchmark
      if (targetId === "module-benchmark" && window.benchmarkChart) {
        window.benchmarkChart.resize();
      }
    });
  });
}

/* ==========================================================================
   2. TIẾN ĐỘ HỌC TẬP (PROGRESS TRACKER)
   ========================================================================== */
function initProgressTracker() {
  const checkboxes = document.querySelectorAll(".progress-check");
  const progressBar = document.getElementById("total-progress-bar");
  const progressText = document.getElementById("total-progress-text");

  function updateProgress() {
    const total = checkboxes.length;
    let checkedCount = 0;
    checkboxes.forEach(cb => {
      if (cb.checked) checkedCount++;
    });
    const percent = Math.round((checkedCount / total) * 100);
    if (progressBar) progressBar.style.width = `${percent}%`;
    if (progressText) progressText.textContent = `${percent}% Hoàn thành`;
    localStorage.setItem("nlp_learning_progress", JSON.stringify(
      Array.from(checkboxes).map(cb => cb.checked)
    ));
  }

  // Khôi phục từ localStorage
  const saved = localStorage.getItem("nlp_learning_progress");
  if (saved) {
    try {
      const arr = JSON.parse(saved);
      checkboxes.forEach((cb, idx) => {
        if (arr[idx] !== undefined) cb.checked = arr[idx];
      });
    } catch (e) {
      console.warn("Không thể khôi phục tiến độ", e);
    }
  }

  checkboxes.forEach(cb => {
    cb.addEventListener("change", updateProgress);
  });
  updateProgress();
}

/* ==========================================================================
   3. WIDGET 1: VECTOR PLAYGROUND (ONE-HOT VS DENSE WORD EMBEDDING)
   ========================================================================== */
let vectorCanvas, vectorCtx;
let selectedWord1 = "thầy_cô";
let selectedWord2 = "giảng_viên";

function initVectorPlayground() {
  vectorCanvas = document.getElementById("vector-canvas");
  if (!vectorCanvas) return;
  vectorCtx = vectorCanvas.getContext("2d");

  const select1 = document.getElementById("word-select-1");
  const select2 = document.getElementById("word-select-2");

  // Đổ danh sách từ vựng vào 2 select
  NLP_DATA.vectorPlayground.words.forEach(w => {
    const opt1 = document.createElement("option");
    opt1.value = w.word;
    opt1.textContent = `${w.word} (${w.category})`;
    if (w.word === selectedWord1) opt1.selected = true;
    select1.appendChild(opt1);

    const opt2 = document.createElement("option");
    opt2.value = w.word;
    opt2.textContent = `${w.word} (${w.category})`;
    if (w.word === selectedWord2) opt2.selected = true;
    select2.appendChild(opt2);
  });

  select1.addEventListener("change", (e) => {
    selectedWord1 = e.target.value;
    updateVectorComparison();
    drawVectorCanvas();
  });

  select2.addEventListener("change", (e) => {
    selectedWord2 = e.target.value;
    updateVectorComparison();
    drawVectorCanvas();
  });

  // Hỗ trợ click trực tiếp trên canvas để chọn từ
  vectorCanvas.addEventListener("click", (evt) => {
    const rect = vectorCanvas.getBoundingClientRect();
    const clickX = evt.clientX - rect.left;
    const clickY = evt.clientY - rect.top;
    const width = vectorCanvas.width;
    const height = vectorCanvas.height;

    // Chuyển pixel thành toạ độ [-1, 1]
    const normX = (clickX - width / 2) / (width * 0.42);
    const normY = (height / 2 - clickY) / (height * 0.42);

    // Tìm từ gần nhất
    let nearest = null;
    let minDist = 999;
    NLP_DATA.vectorPlayground.words.forEach(w => {
      const dist = Math.hypot(w.vec2d[0] - normX, w.vec2d[1] - normY);
      if (dist < minDist) {
        minDist = dist;
        nearest = w;
      }
    });

    if (nearest && minDist < 0.25) {
      selectedWord2 = nearest.word;
      select2.value = nearest.word;
      updateVectorComparison();
      drawVectorCanvas();
    }
  });

  updateVectorComparison();
  drawVectorCanvas();
}

function calcCosineSimilarity(v1, v2) {
  let dot = 0, norm1 = 0, norm2 = 0;
  for (let i = 0; i < v1.length; i++) {
    dot += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  if (norm1 === 0 || norm2 === 0) return 0;
  return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

function updateVectorComparison() {
  const wObj1 = NLP_DATA.vectorPlayground.words.find(w => w.word === selectedWord1);
  const wObj2 = NLP_DATA.vectorPlayground.words.find(w => w.word === selectedWord2);
  if (!wObj1 || !wObj2) return;

  // Tính Cosine Dense
  const cosSimDense = calcCosineSimilarity(wObj1.dense, wObj2.dense);
  const angleDeg = Math.acos(Math.max(-1, Math.min(1, cosSimDense))) * (180 / Math.PI);

  // One-Hot: Nếu 2 từ khác nhau -> cos = 0; nếu trùng nhau -> cos = 1
  const cosSimOneHot = (wObj1.word === wObj2.word) ? 1.0 : 0.0;

  // Cập nhật DOM
  const cosDenseEl = document.getElementById("dense-cos-val");
  const cosDenseBar = document.getElementById("dense-cos-bar");
  const angleEl = document.getElementById("dense-angle-val");
  const onehotCosEl = document.getElementById("onehot-cos-val");
  const word1Label = document.getElementById("comp-word1-name");
  const word2Label = document.getElementById("comp-word2-name");

  if (word1Label) word1Label.textContent = `"${wObj1.word}"`;
  if (word2Label) word2Label.textContent = `"${wObj2.word}"`;

  if (cosDenseEl) cosDenseEl.textContent = cosSimDense.toFixed(4);
  if (angleEl) angleEl.textContent = `${angleDeg.toFixed(1)}°`;
  if (cosDenseBar) {
    const percent = Math.max(0, Math.min(100, ((cosSimDense + 1) / 2) * 100));
    cosDenseBar.style.width = `${percent}%`;
  }
  if (onehotCosEl) onehotCosEl.textContent = cosSimOneHot.toFixed(4);

  // Hiển thị vector mẫu
  const denseBox1 = document.getElementById("dense-vec-preview-1");
  const denseBox2 = document.getElementById("dense-vec-preview-2");
  if (denseBox1) denseBox1.textContent = `[${wObj1.dense.map(v => v.toFixed(2)).join(", ")}, ...] (d=100)`;
  if (denseBox2) denseBox2.textContent = `[${wObj2.dense.map(v => v.toFixed(2)).join(", ")}, ...] (d=100)`;

  const onehotBox1 = document.getElementById("onehot-vec-preview-1");
  const onehotBox2 = document.getElementById("onehot-vec-preview-2");
  if (onehotBox1) onehotBox1.textContent = `[0, 0, ..., 1 (idx:${wObj1.onehotIdx}), ..., 0] (|V|=10,000)`;
  if (onehotBox2) onehotBox2.textContent = `[0, 0, ..., 1 (idx:${wObj2.onehotIdx}), ..., 0] (|V|=10,000)`;
}

function drawVectorCanvas() {
  if (!vectorCanvas || !vectorCtx) return;
  const ctx = vectorCtx;
  const width = vectorCanvas.width;
  const height = vectorCanvas.height;
  const originX = width / 2;
  const originY = height / 2;
  const scale = width * 0.42;

  ctx.clearRect(0, 0, width, height);

  // Vẽ lưới tọa độ & trục
  ctx.strokeStyle = "rgba(75, 85, 99, 0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  // Vòng tròn đơn vị
  ctx.arc(originX, originY, scale * 0.9, 0, 2 * Math.PI);
  ctx.stroke();

  // Trục X & Y
  ctx.strokeStyle = "rgba(107, 114, 128, 0.6)";
  ctx.beginPath();
  ctx.moveTo(0, originY);
  ctx.lineTo(width, originY);
  ctx.moveTo(originX, 0);
  ctx.lineTo(originX, height);
  ctx.stroke();

  // Vẽ tất cả các từ trong không gian
  NLP_DATA.vectorPlayground.words.forEach(item => {
    const px = originX + item.vec2d[0] * scale;
    const py = originY - item.vec2d[1] * scale;
    const isW1 = item.word === selectedWord1;
    const isW2 = item.word === selectedWord2;

    ctx.beginPath();
    ctx.arc(px, py, isW1 || isW2 ? 7 : 4, 0, 2 * Math.PI);
    if (isW1) {
      ctx.fillStyle = "#6366f1"; // Indigo
      ctx.shadowColor = "#6366f1";
      ctx.shadowBlur = 10;
    } else if (isW2) {
      ctx.fillStyle = "#ec4899"; // Pink
      ctx.shadowColor = "#ec4899";
      ctx.shadowBlur = 10;
    } else {
      ctx.fillStyle = "#9ca3af";
      ctx.shadowBlur = 0;
    }
    ctx.fill();
    ctx.shadowBlur = 0;

    // Nhãn từ
    ctx.font = isW1 || isW2 ? "bold 13px sans-serif" : "11px sans-serif";
    ctx.fillStyle = isW1 ? "#a5b4fc" : (isW2 ? "#f472b6" : "#6b7280");
    ctx.fillText(item.word, px + 9, py + 4);
  });

  // Vẽ 2 vector nổi bật từ gốc (0,0) đến 2 từ đã chọn
  const w1 = NLP_DATA.vectorPlayground.words.find(w => w.word === selectedWord1);
  const w2 = NLP_DATA.vectorPlayground.words.find(w => w.word === selectedWord2);
  if (w1 && w2) {
    const p1x = originX + w1.vec2d[0] * scale;
    const p1y = originY - w1.vec2d[1] * scale;
    const p2x = originX + w2.vec2d[0] * scale;
    const p2y = originY - w2.vec2d[1] * scale;

    // Vector 1 (Indigo)
    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(p1x, p1y);
    ctx.stroke();

    // Vector 2 (Pink)
    ctx.strokeStyle = "#ec4899";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(p2x, p2y);
    ctx.stroke();

    // Vẽ góc arc giữa 2 vector
    const ang1 = Math.atan2(-w1.vec2d[1], w1.vec2d[0]);
    const ang2 = Math.atan2(-w2.vec2d[1], w2.vec2d[0]);
    ctx.strokeStyle = "rgba(234, 179, 8, 0.8)";
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(originX, originY, 40, Math.min(ang1, ang2), Math.max(ang1, ang2));
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

/* ==========================================================================
   4. WIDGET 2: WORD2VEC SLIDING WINDOW (CBOW VS SKIP-GRAM)
   ========================================================================== */
let currentW2vSentenceIdx = 0;
let currentCenterIdx = 2;
let currentWindowSize = 2;
let currentW2vMode = "cbow"; // "cbow" hoặc "skipgram"

function initWord2VecSimulator() {
  const sentenceSelect = document.getElementById("w2v-sent-select");
  const windowSlider = document.getElementById("w2v-window-slider");
  const windowValDisplay = document.getElementById("w2v-window-val");
  const cbowBtn = document.getElementById("mode-cbow-btn");
  const skipgramBtn = document.getElementById("mode-skipgram-btn");

  if (!sentenceSelect) return;

  // Đổ danh sách câu
  NLP_DATA.word2vecSentences.forEach((s, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `[${s.domain}] ${s.tokens.join(" ")}`;
    sentenceSelect.appendChild(opt);
  });

  sentenceSelect.addEventListener("change", (e) => {
    currentW2vSentenceIdx = parseInt(e.target.value);
    currentCenterIdx = NLP_DATA.word2vecSentences[currentW2vSentenceIdx].defaultCenter;
    renderWord2VecPills();
    updateWord2VecDetails();
  });

  windowSlider.addEventListener("input", (e) => {
    currentWindowSize = parseInt(e.target.value);
    if (windowValDisplay) windowValDisplay.textContent = currentWindowSize;
    renderWord2VecPills();
    updateWord2VecDetails();
  });

  cbowBtn.addEventListener("click", () => {
    currentW2vMode = "cbow";
    cbowBtn.classList.add("bg-indigo-600", "text-white");
    cbowBtn.classList.remove("bg-gray-800", "text-gray-300");
    skipgramBtn.classList.remove("bg-indigo-600", "text-white");
    skipgramBtn.classList.add("bg-gray-800", "text-gray-300");
    updateWord2VecDetails();
  });

  skipgramBtn.addEventListener("click", () => {
    currentW2vMode = "skipgram";
    skipgramBtn.classList.add("bg-indigo-600", "text-white");
    skipgramBtn.classList.remove("bg-gray-800", "text-gray-300");
    cbowBtn.classList.remove("bg-indigo-600", "text-white");
    cbowBtn.classList.add("bg-gray-800", "text-gray-300");
    updateWord2VecDetails();
  });

  renderWord2VecPills();
  updateWord2VecDetails();
}

function renderWord2VecPills() {
  const container = document.getElementById("w2v-tokens-container");
  if (!container) return;
  container.innerHTML = "";

  const sent = NLP_DATA.word2vecSentences[currentW2vSentenceIdx];
  const tokens = sent.tokens;

  tokens.forEach((t, idx) => {
    const pill = document.createElement("div");
    pill.classList.add("token-pill", "px-3", "py-2", "rounded-lg", "border", "text-sm", "font-mono", "inline-flex", "items-center", "gap-1");

    const isCenter = idx === currentCenterIdx;
    const isContext = Math.abs(idx - currentCenterIdx) <= currentWindowSize && !isCenter;

    if (isCenter) {
      pill.classList.add("token-center");
      pill.innerHTML = `<span>🎯</span> <strong>${t}</strong> <span class="text-xs bg-red-800/80 px-1 rounded">w(t)</span>`;
    } else if (isContext) {
      pill.classList.add("token-context");
      const offset = idx - currentCenterIdx;
      const sign = offset > 0 ? `+${offset}` : `${offset}`;
      pill.innerHTML = `<span>👁️</span> ${t} <span class="text-xs bg-sky-900/80 px-1 rounded">${sign}</span>`;
    } else {
      pill.classList.add("token-inactive", "border-gray-800");
      pill.textContent = t;
    }

    pill.addEventListener("click", () => {
      currentCenterIdx = idx;
      renderWord2VecPills();
      updateWord2VecDetails();
    });

    container.appendChild(pill);
  });
}

function updateWord2VecDetails() {
  const sent = NLP_DATA.word2vecSentences[currentW2vSentenceIdx];
  const centerWord = sent.tokens[currentCenterIdx];
  const contextWords = [];

  for (let i = 0; i < sent.tokens.length; i++) {
    if (i !== currentCenterIdx && Math.abs(i - currentCenterIdx) <= currentWindowSize) {
      contextWords.push(sent.tokens[i]);
    }
  }

  const modeTitle = document.getElementById("w2v-mode-title");
  const inputList = document.getElementById("w2v-input-list");
  const outputList = document.getElementById("w2v-output-list");
  const flowDesc = document.getElementById("w2v-flow-desc");

  if (currentW2vMode === "cbow") {
    if (modeTitle) modeTitle.textContent = "Kiến trúc CBOW: Cửa sổ trượt dự đoán Từ Trung Tâm";
    if (inputList) inputList.innerHTML = contextWords.map(w => `<span class="bg-sky-900/60 text-sky-200 px-2 py-1 rounded text-xs border border-sky-700">${w}</span>`).join(" ");
    if (outputList) outputList.innerHTML = `<span class="bg-red-900/60 text-red-200 px-2.5 py-1 rounded text-sm font-bold border border-red-700">${centerWord}</span>`;
    if (flowDesc) flowDesc.innerHTML = `Mô hình lấy các vector của <strong>[${contextWords.join(", ")}]</strong>, thực hiện phép <strong>cộng trung bình</strong> để dự đoán xác suất từ đích <strong>"${centerWord}"</strong>. Phù hợp kho dữ liệu lớn, tính toán nhanh.`;
  } else {
    if (modeTitle) modeTitle.textContent = "Kiến trúc Skip-Gram: Từ Trung Tâm phóng tia dự đoán Ngữ Cảnh";
    if (inputList) inputList.innerHTML = `<span class="bg-red-900/60 text-red-200 px-2.5 py-1 rounded text-sm font-bold border border-red-700">${centerWord}</span>`;
    if (outputList) outputList.innerHTML = contextWords.map(w => `<span class="bg-sky-900/60 text-sky-200 px-2 py-1 rounded text-xs border border-sky-700">${w}</span>`).join(" ");
    if (flowDesc) flowDesc.innerHTML = `Mô hình lấy từ trung tâm <strong>"${centerWord}"</strong> làm đầu vào, cực đại hóa xác suất đồng xuất hiện với từng từ ngữ cảnh <strong>[${contextWords.join(", ")}]</strong>. Bắt rất tốt các từ hiếm và ngữ cảnh sắc thái.`;
  }
}

/* ==========================================================================
   5. WIDGET 3: AVERAGE WORD2VEC FLAW DEMO
   ========================================================================== */
function initAvgW2vFlawDemo() {
  const select = document.getElementById("flaw-sample-select");
  if (!select) return;

  NLP_DATA.avgW2vFlawSamples.forEach((item, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = item.title;
    select.appendChild(opt);
  });

  select.addEventListener("change", (e) => {
    renderAvgW2vFlaw(parseInt(e.target.value));
  });

  renderAvgW2vFlaw(0);
}

function renderAvgW2vFlaw(idx) {
  const data = NLP_DATA.avgW2vFlawSamples[idx];
  if (!data) return;

  const sentAEl = document.getElementById("flaw-sent-a");
  const sentBEl = document.getElementById("flaw-sent-b");
  const meanAEl = document.getElementById("flaw-mean-a");
  const meanBEl = document.getElementById("flaw-mean-b");
  const explainEl = document.getElementById("flaw-math-explain");
  const tokenPillsA = document.getElementById("flaw-tokens-a");
  const tokenPillsB = document.getElementById("flaw-tokens-b");

  if (sentAEl) sentAEl.textContent = `"${data.sentA}"`;
  if (sentBEl) sentBEl.textContent = `"${data.sentB}"`;
  if (meanAEl) meanAEl.textContent = data.meaningA;
  if (meanBEl) meanBEl.textContent = data.meaningB;
  if (explainEl) explainEl.textContent = data.mathExplanation;

  if (tokenPillsA) {
    tokenPillsA.innerHTML = data.tokensA.map(t => `<span class="px-2 py-1 bg-emerald-950/60 text-emerald-300 border border-emerald-800 rounded text-xs">${t}</span>`).join(" ");
  }
  if (tokenPillsB) {
    tokenPillsB.innerHTML = data.tokensB.map(t => `<span class="px-2 py-1 bg-rose-950/60 text-rose-300 border border-rose-800 rounded text-xs">${t}</span>`).join(" ");
  }
}

/* ==========================================================================
   6. WIDGET 4: LSTM 3-GATES INTERACTIVE STEPPER
   ========================================================================== */
let lstmCurrentStep = 0;
let lstmPlaying = false;
let lstmInterval = null;

function initLstmSimulator() {
  const prevBtn = document.getElementById("lstm-prev-btn");
  const nextBtn = document.getElementById("lstm-next-btn");
  const playBtn = document.getElementById("lstm-play-btn");
  const resetBtn = document.getElementById("lstm-reset-btn");

  if (!prevBtn) return;

  prevBtn.addEventListener("click", () => {
    if (lstmCurrentStep > 0) {
      lstmCurrentStep--;
      renderLstmStep();
    }
  });

  nextBtn.addEventListener("click", () => {
    if (lstmCurrentStep < NLP_DATA.lstmSimulation.steps.length - 1) {
      lstmCurrentStep++;
      renderLstmStep();
    }
  });

  resetBtn.addEventListener("click", () => {
    lstmCurrentStep = 0;
    if (lstmPlaying) toggleLstmPlay();
    renderLstmStep();
  });

  playBtn.addEventListener("click", toggleLstmPlay);

  renderLstmSentencePills();
  renderLstmStep();
}

function toggleLstmPlay() {
  const playBtn = document.getElementById("lstm-play-btn");
  lstmPlaying = !lstmPlaying;
  if (lstmPlaying) {
    playBtn.innerHTML = "<span>⏸️ Tạm Dừng</span>";
    playBtn.classList.replace("bg-indigo-600", "bg-amber-600");
    lstmInterval = setInterval(() => {
      if (lstmCurrentStep < NLP_DATA.lstmSimulation.steps.length - 1) {
        lstmCurrentStep++;
        renderLstmStep();
      } else {
        toggleLstmPlay();
      }
    }, 1800);
  } else {
    playBtn.innerHTML = "<span>▶️ Tự Động Chạy</span>";
    playBtn.classList.replace("bg-amber-600", "bg-indigo-600");
    clearInterval(lstmInterval);
  }
}

function renderLstmSentencePills() {
  const container = document.getElementById("lstm-sentence-display");
  if (!container) return;
  container.innerHTML = "";

  NLP_DATA.lstmSimulation.sentence.forEach((tok, idx) => {
    const pill = document.createElement("button");
    pill.classList.add("px-2.5", "py-1.5", "rounded-md", "border", "text-sm", "font-mono", "transition-all");
    pill.id = `lstm-token-${idx}`;
    pill.textContent = tok;
    pill.addEventListener("click", () => {
      lstmCurrentStep = idx;
      renderLstmStep();
    });
    container.appendChild(pill);
  });
}

function renderLstmStep() {
  const step = NLP_DATA.lstmSimulation.steps[lstmCurrentStep];
  if (!step) return;

  // Cập nhật pills
  NLP_DATA.lstmSimulation.sentence.forEach((_, idx) => {
    const pill = document.getElementById(`lstm-token-${idx}`);
    if (!pill) return;
    if (idx === lstmCurrentStep) {
      pill.className = "px-3 py-1.5 rounded-md border-2 border-indigo-500 bg-indigo-900/60 text-white font-bold shadow-lg shadow-indigo-500/30 scale-105";
    } else if (idx < lstmCurrentStep) {
      pill.className = "px-2.5 py-1.5 rounded-md border border-gray-700 bg-gray-900/60 text-gray-400";
    } else {
      pill.className = "px-2.5 py-1.5 rounded-md border border-gray-800 bg-gray-950/40 text-gray-600";
    }
  });

  // Cập nhật giá trị Gates
  const forgetFill = document.getElementById("lstm-forget-fill");
  const forgetVal = document.getElementById("lstm-forget-val");
  const inputFill = document.getElementById("lstm-input-fill");
  const inputVal = document.getElementById("lstm-input-val");
  const cellFill = document.getElementById("lstm-cell-fill");
  const cellVal = document.getElementById("lstm-cell-val");
  const outputFill = document.getElementById("lstm-output-fill");
  const outputVal = document.getElementById("lstm-output-val");
  const hiddenVal = document.getElementById("lstm-hidden-val");
  const noteEl = document.getElementById("lstm-step-note");
  const stepCountEl = document.getElementById("lstm-step-count");
  const alertBox = document.getElementById("lstm-special-alert");

  if (stepCountEl) stepCountEl.textContent = `Bước ${lstmCurrentStep + 1}/${NLP_DATA.lstmSimulation.steps.length}: Từ "${step.token}"`;
  
  if (forgetFill) forgetFill.style.width = `${step.forgetGate * 100}%`;
  if (forgetVal) forgetVal.textContent = step.forgetGate.toFixed(2);

  if (inputFill) inputFill.style.width = `${step.inputGate * 100}%`;
  if (inputVal) inputVal.textContent = step.inputGate.toFixed(2);

  // Cell state có thể từ -1 đến 1 -> chuẩn hóa hiển thị %
  if (cellFill) {
    const cellPercent = Math.max(0, Math.min(100, ((step.cellStateVal + 1) / 2) * 100));
    cellFill.style.width = `${cellPercent}%`;
    if (step.cellStateVal < 0) {
      cellFill.style.backgroundColor = "#ef4444"; // Đỏ tiêu cực
    } else {
      cellFill.style.backgroundColor = "#10b981"; // Xanh tích cực
    }
  }
  if (cellVal) cellVal.textContent = (step.cellStateVal > 0 ? "+" : "") + step.cellStateVal.toFixed(2);

  if (outputFill) outputFill.style.width = `${step.outputGate * 100}%`;
  if (outputVal) outputVal.textContent = step.outputGate.toFixed(2);

  if (hiddenVal) hiddenVal.textContent = (step.hiddenStateVal > 0 ? "+" : "") + step.hiddenStateVal.toFixed(2);

  if (noteEl) noteEl.textContent = step.note;

  // Hiển thị alert nếu gặp từ "nhưng"
  if (alertBox) {
    if (step.token === "nhưng") {
      alertBox.classList.remove("hidden");
    } else {
      alertBox.classList.add("hidden");
    }
  }
}

/* ==========================================================================
   7. WIDGET 5: SELF-ATTENTION HEATMAP EXPLORER
   ========================================================================== */
let currentAttentionIdx = 0;

function initAttentionExplorer() {
  const select = document.getElementById("attention-sample-select");
  if (!select) return;

  NLP_DATA.attentionSamples.forEach((item, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `[${item.domain}] ${item.type} - "${item.tokens.map(t => t.word).join(" ")}"`;
    select.appendChild(opt);
  });

  select.addEventListener("change", (e) => {
    currentAttentionIdx = parseInt(e.target.value);
    renderAttentionSample();
  });

  renderAttentionSample();
}

function renderAttentionSample() {
  const sample = NLP_DATA.attentionSamples[currentAttentionIdx];
  if (!sample) return;

  const wordsContainer = document.getElementById("attention-words-display");
  const commentEl = document.getElementById("attention-comment");
  const predBadge = document.getElementById("attention-pred-badge");
  const imgEl = document.getElementById("attention-real-img");

  if (predBadge) predBadge.textContent = `Mô hình dự đoán: ${sample.modelPred}`;
  if (commentEl) commentEl.textContent = sample.comment;
  if (imgEl) imgEl.src = sample.imagePath;

  if (wordsContainer) {
    wordsContainer.innerHTML = "";
    sample.tokens.forEach((t) => {
      const span = document.createElement("span");
      span.classList.add("attention-word", "border", "font-mono");
      
      // Độ đậm tỷ lệ với weight (weight cao -> màu đỏ/cam rực)
      const alpha = Math.min(1, Math.max(0.1, t.weight * 2.2));
      span.style.backgroundColor = `rgba(244, 63, 94, ${alpha})`;
      span.style.borderColor = `rgba(244, 63, 94, ${alpha + 0.2})`;
      span.style.color = alpha > 0.45 ? "#ffffff" : "#fecdd3";

      span.innerHTML = `${t.word} <span class="text-xs opacity-90 font-bold block text-center">${(t.weight * 100).toFixed(1)}%</span>`;
      
      span.title = `Token: ${t.word} | Trọng số Attention: ${(t.weight * 100).toFixed(2)}%`;
      wordsContainer.appendChild(span);
    });
  }
}

/* ==========================================================================
   8. WIDGET 6: BENCHMARK ANALYTICS & CHARTS
   ========================================================================== */
function initBenchmarkAnalytics() {
  renderBenchmarkTable();
  initBenchmarkChart();
  initImageModal();
}

function renderBenchmarkTable() {
  const tbody = document.getElementById("benchmark-table-body");
  if (!tbody) return;

  tbody.innerHTML = "";
  NLP_DATA.benchmarkResults.forEach(item => {
    const tr = document.createElement("tr");
    tr.classList.add("border-b", "border-gray-800", "hover:bg-gray-800/40", "transition-colors");

    const isSota = item.id === 6;
    const isProposed = item.id === 5;
    const badge = isSota 
      ? '<span class="ml-2 px-1.5 py-0.5 bg-amber-500/20 text-amber-400 border border-amber-500/40 rounded text-xs font-bold">🏆 SOTA</span>'
      : (isProposed ? '<span class="ml-2 px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 rounded text-xs font-bold">✨ Đề xuất</span>' : '');

    tr.innerHTML = `
      <td class="p-3 text-sm font-semibold">${item.id}. ${item.model} ${badge}</td>
      <td class="p-3 text-center text-sm font-bold ${isSota || isProposed ? 'text-indigo-400' : 'text-gray-300'}">${item.uitAcc.toFixed(2)}%</td>
      <td class="p-3 text-center text-sm font-mono text-gray-400">${item.uitF1.toFixed(2)}%</td>
      <td class="p-3 text-center text-sm font-bold ${isSota || isProposed ? 'text-amber-400' : 'text-gray-300'}">${item.ecomAcc.toFixed(2)}%</td>
      <td class="p-3 text-center text-sm font-mono text-gray-400">${item.ecomF1.toFixed(2)}%</td>
      <td class="p-3 text-center">
        <button class="px-2 py-1 bg-gray-800 hover:bg-indigo-700 text-xs rounded border border-gray-700 transition" onclick="openCmModal('${item.model}', '${item.cmUit}', '${item.cmEcom}')">
          🔍 Xem CM
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function initBenchmarkChart() {
  const ctx = document.getElementById("benchmark-chart");
  if (!ctx || typeof Chart === "undefined") return;

  const labels = NLP_DATA.benchmarkResults.map(r => r.model.split(" (")[0]);
  const uitAccData = NLP_DATA.benchmarkResults.map(r => r.uitAcc);
  const ecomAccData = NLP_DATA.benchmarkResults.map(r => r.ecomAcc);

  window.benchmarkChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "UIT-VSFC (Giáo dục)",
          data: uitAccData,
          backgroundColor: "rgba(99, 102, 241, 0.8)",
          borderColor: "#6366f1",
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: "E-Commerce (Thương mại)",
          data: ecomAccData,
          backgroundColor: "rgba(245, 158, 11, 0.8)",
          borderColor: "#f59e0b",
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 60,
          max: 100,
          grid: { color: "rgba(75, 85, 99, 0.2)" },
          ticks: { color: "#9ca3af", callback: val => `${val}%` }
        },
        x: {
          grid: { display: false },
          ticks: { color: "#d1d5db", font: { size: 10 } }
        }
      },
      plugins: {
        legend: {
          labels: { color: "#f3f4f6" }
        },
        tooltip: {
          callbacks: {
            label: context => ` ${context.dataset.label}: ${context.raw}%`
          }
        }
      }
    }
  });
}

function initImageModal() {
  const modal = document.getElementById("image-modal");
  const closeBtn = document.getElementById("close-modal-btn");
  if (closeBtn && modal) {
    closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.add("hidden");
    });
  }
}

window.openCmModal = function(modelName, uitImg, ecomImg) {
  const modal = document.getElementById("image-modal");
  const title = document.getElementById("modal-title");
  const img1 = document.getElementById("modal-img-1");
  const img2 = document.getElementById("modal-img-2");
  if (!modal) return;

  if (title) title.textContent = `Ma trận nhầm lẫn (Confusion Matrix): ${modelName}`;
  if (img1) img1.src = uitImg;
  if (img2) img2.src = ecomImg;
  modal.classList.remove("hidden");
};

/* ==========================================================================
   9. WIDGET 7: MOCK DEFENSE SIMULATOR (VẤN ĐÁP VỚI TS. LÊ ANH CƯỜNG)
   ========================================================================== */
function initDefenseSimulator() {
  const container = document.getElementById("defense-questions-list");
  const filterSelect = document.getElementById("defense-topic-filter");
  const searchInput = document.getElementById("defense-search-input");
  if (!container) return;

  // Lấy danh sách topic độc nhất
  const topics = Array.from(new Set(NLP_DATA.defenseQuestions.map(q => q.topic)));
  if (filterSelect) {
    topics.forEach(t => {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      filterSelect.appendChild(opt);
    });

    filterSelect.addEventListener("change", renderDefenseQuestions);
  }

  if (searchInput) {
    searchInput.addEventListener("input", renderDefenseQuestions);
  }

  renderDefenseQuestions();
}

function renderDefenseQuestions() {
  const container = document.getElementById("defense-questions-list");
  const filterSelect = document.getElementById("defense-topic-filter");
  const searchInput = document.getElementById("defense-search-input");
  if (!container) return;

  const selectedTopic = filterSelect ? filterSelect.value : "ALL";
  const query = searchInput ? searchInput.value.toLowerCase().trim() : "";

  container.innerHTML = "";

  const filtered = NLP_DATA.defenseQuestions.filter(q => {
    const matchTopic = (selectedTopic === "ALL" || q.topic === selectedTopic);
    const matchQuery = !query || q.question.toLowerCase().includes(query) || q.academic.toLowerCase().includes(query);
    return matchTopic && matchQuery;
  });

  if (filtered.length === 0) {
    container.innerHTML = `<div class="p-8 text-center text-gray-500">Không tìm thấy câu hỏi phù hợp. Hãy thử từ khóa khác!</div>`;
    return;
  }

  filtered.forEach(q => {
    const card = document.createElement("div");
    card.classList.add("glass-panel", "p-5", "space-y-4", "border-gray-800");

    card.innerHTML = `
      <div class="flex items-start justify-between gap-4">
        <div>
          <span class="px-2 py-0.5 bg-indigo-900/60 text-indigo-300 border border-indigo-700/50 rounded text-xs font-semibold">${q.topic}</span>
          <h4 class="text-base font-bold text-gray-100 mt-2">${q.question}</h4>
        </div>
        <button class="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-medium transition shrink-0" onclick="toggleDefenseAnswer('${q.id}')">
          💡 Xem Lời Giải Chuẩn
        </button>
      </div>

      <div id="defense-answer-${q.id}" class="hidden space-y-3 pt-3 border-t border-gray-800">
        <!-- Bình dân -->
        <div class="bg-gray-900/80 p-3.5 rounded-lg border-l-4 border-amber-500">
          <div class="text-xs font-bold text-amber-400 mb-1 flex items-center gap-1.5">
            <span>🗣️</span> CÁCH NÓI BÌNH DÂN (Hiểu bản chất trong 30 giây):
          </div>
          <p class="text-sm text-gray-300 leading-relaxed">${q.intuitive}</p>
        </div>

        <!-- Học thuật -->
        <div class="bg-gray-900/80 p-3.5 rounded-lg border-l-4 border-indigo-500">
          <div class="text-xs font-bold text-indigo-400 mb-1 flex items-center gap-1.5">
            <span>🎓</span> TRẢ LỜI CHUẨN HỌC THUẬT (Thuyết phục thầy cô phản biện):
          </div>
          <p class="text-sm text-gray-300 leading-relaxed">${q.academic}</p>
        </div>

        <!-- Bẫy câu hỏi -->
        <div class="bg-rose-950/40 p-3 rounded-lg border border-rose-800/40">
          <div class="text-xs font-bold text-rose-400 mb-1 flex items-center gap-1.5">
            <span>⚠️</span> LƯU Ý CÂU HỎI GÀI CỦA GIẢNG VIÊN:
          </div>
          <p class="text-xs text-rose-200/90">${q.gotcha}</p>
        </div>
      </div>
    `;

    container.appendChild(card);
  });
}

window.toggleDefenseAnswer = function(qid) {
  const el = document.getElementById(`defense-answer-${qid}`);
  if (el) {
    el.classList.toggle("hidden");
  }
};

/* ==========================================================================
   10. WIDGET 8: AI DEFENSE TUTOR & SMART Q&A
   ========================================================================== */
function initAiAssistant() {
  const sendBtn = document.getElementById("ai-ask-btn");
  const inputEl = document.getElementById("ai-question-input");
  const quickChips = document.querySelectorAll(".ai-quick-chip");

  if (sendBtn && inputEl) {
    sendBtn.addEventListener("click", () => handleAiQuestion(inputEl.value));
    inputEl.addEventListener("keypress", (e) => {
      if (e.key === "Enter") handleAiQuestion(inputEl.value);
    });
  }

  quickChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-q") || chip.textContent;
      if (inputEl) inputEl.value = q;
      handleAiQuestion(q);
    });
  });
}

function handleAiQuestion(userQuery) {
  if (!userQuery || !userQuery.trim()) return;
  const chatBox = document.getElementById("ai-chat-history");
  const inputEl = document.getElementById("ai-question-input");
  if (!chatBox) return;

  const query = userQuery.trim().toLowerCase();

  // Thêm tin nhắn người dùng
  const userMsg = document.createElement("div");
  userMsg.className = "flex justify-end";
  userMsg.innerHTML = `
    <div class="bg-indigo-600 text-white px-4 py-2.5 rounded-2xl rounded-tr-none text-sm max-w-xl shadow">
      ${userQuery}
    </div>
  `;
  chatBox.appendChild(userMsg);
  if (inputEl) inputEl.value = "";

  // Tìm câu trả lời trong kho tri thức
  const botReply = generateTutorAnswer(query);

  const botMsg = document.createElement("div");
  botMsg.className = "flex justify-start";
  botMsg.innerHTML = `
    <div class="glass-panel p-4 rounded-2xl rounded-tl-none text-sm max-w-2xl border-gray-800 space-y-2">
      <div class="flex items-center gap-2 text-indigo-400 font-bold text-xs">
        <span>🤖</span> NLP DEFENSE TUTOR:
      </div>
      <div class="text-gray-300 leading-relaxed text-sm whitespace-pre-line">
        ${botReply}
      </div>
    </div>
  `;
  chatBox.appendChild(botMsg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function generateTutorAnswer(query) {
  // Tìm khớp từ khóa
  if (query.includes("one-hot") || query.includes("trực giao") || query.includes("số chiều")) {
    return `💡 **Vấn đề cốt lõi của One-Hot Encoding:**\n` +
      `1. **Bùng nổ số chiều**: Kích thước vector bằng kích thước từ điển |V| (thường 10.000 đến 50.000 chiều), khiến ma trận cực thưa, tốn bộ nhớ.\n` +
      `2. **Tính trực giao**: Tích vô hướng giữa 2 vector từ khác nhau luôn bằng 0 (e_i^T * e_j = 0). Máy tính không thể biết 'thầy cô' có liên quan mật thiết với 'giảng viên'.\n\n` +
      `👉 **Khi báo cáo**: Bạn nhấn mạnh Giả thuyết phân bố (Firth 1957) và cách Word2Vec gom các từ cùng ngữ cảnh vào gần nhau trong không gian vector dày đặc d=100.`;
  }

  if (query.includes("cbow") || query.includes("skip-gram") || query.includes("skipgram")) {
    return `💡 **So sánh CBOW vs Skip-Gram:**\n` +
      `• **CBOW**: Input là các từ ngữ cảnh (Context), Output dự đoán từ đích ở giữa (Target). Huấn luyện nhanh hơn, trơn tru trên các từ xuất hiện nhiều.\n` +
      `• **Skip-Gram**: Input là từ đích ở giữa, phóng tia dự đoán các từ ngữ cảnh xung quanh. Học rất tốt các từ hiếm (rare words) vì không bị cào bằng.\n\n` +
      `👉 **Mẹo bảo vệ**: Nếu thầy hỏi 'Đồ án dùng loại nào?', trả lời rằng nhóm đã huấn luyện cả hai bằng thư viện Gensim (vector_size=100, window=5) và sử dụng vector này để khởi tạo ma trận Embedding cho BiLSTM.`;
  }

  if (query.includes("negative sampling") || query.includes("mẫu âm") || query.includes("3/4")) {
    return `💡 **Bản chất của Negative Sampling (SGNS):**\n` +
      `• **Mục đích**: Thay vì tính mẫu số Softmax trên toàn bộ |V| từ (tốn O(|V|)), ta biến bài toán thành phân loại nhị phân Logistic Regression chỉ với 1 mẫu dương (cặp thật) và K mẫu âm (K=5 đến 20).\n` +
      `• **Số mũ 3/4**: Phân phối Unigram mũ 0.75 làm giảm ưu thế tuyệt đối của các từ stop words (như 'và', 'thì', 'là') và tăng cơ hội cho các từ hiếm gặp được chọn làm mẫu âm, giúp vector cập nhật đồng đều hơn.`;
  }

  if (query.includes("average word2vec") || query.includes("đảo ngữ") || query.includes("phim")) {
    return `💡 **Điểm yếu chí mạng của Average Word2Vec:**\n` +
      `• Phép lấy trung bình cộng v_D = 1/N * sum(v_w) có tính chất giao hoán.\n` +
      `• Ví dụ kinh điển: 'phim này hay chứ không dở' (Khen) và 'phim này dở chứ không hay' (Chê) đều cho ra vector giống hệt nhau 100%!\n` +
      `• Do đó, mô hình tuyến tính hoàn toàn bất lực. Đó là lý do bắt buộc phải chuyển sang kiến trúc tuần tự như LSTM.`;
  }

  if (query.includes("lstm") || query.includes("cổng") || query.includes("cell state")) {
    return `💡 **Bản chất 3 Cổng và Cell State của LSTM:**\n` +
      `• **Cell State (c_t)**: 'Băng chuyền ký ức dài hạn', thông tin chỉ được cộng thêm hoặc nhân với cổng quên, giúp gradient truyền ngược mà không bị triệt tiêu (chống Vanishing Gradient).\n` +
      `• **Forget Gate (f_t)**: Quyết định xóa bao nhiêu phần thông tin cũ (dùng hàm Sigmoid trong [0, 1]).\n` +
      `• **Input Gate (i_t)**: Quyết định nạp bao nhiêu thông tin mới từ token hiện tại.\n` +
      `• **Output Gate (o_t)**: Lọc Cell State để tạo Trạng thái ẩn (Hidden State h_t).\n\n` +
      `👉 **Trong đồ án**: BiLSTM ghép nối 2 chiều [h_T_right ; h_1_left] tạo vector văn bản 256 chiều.`;
  }

  if (query.includes("e-commerce") || query.includes("sụt giảm") || query.includes("giảm") || query.includes("thương mại")) {
    return `💡 **Tại sao điểm trên tập E-Commerce lại sụt giảm mạnh?**\n` +
      `1. **Nhiễu từ vựng & OOV**: Người mua hàng dùng rất nhiều teencode ('k', 'ko', 'đc', 'ship'), icon, viết hoa bừa bãi.\n` +
      `2. **Cấu trúc tự do & Đảo ngữ**: Câu không theo quy chuẩn ngữ pháp như tập học sinh UIT-VSFC.\n` +
      `3. **Sắc thái châm biếm (Sarcasm)**: 'Cho shop 5 sao nhưng giao hàng như hạch' làm các mô hình truyền thống bắt nhầm nhãn.\n\n` +
      `👉 **Điểm sáng đồ án**: PhoBERT nhờ tiền huấn luyện 20GB tiếng Việt và 12 tầng Attention đã duy trì độ chính xác ấn tượng 86.17% (vượt TF-IDF tới +15.84%).`;
  }

  if (query.includes("phobert") || query.includes("transformer") || query.includes("sota")) {
    return `💡 **Vì sao PhoBERT lại vượt trội áp đảo?**\n` +
      `• **Pre-trained khổng lồ**: PhoBERT được học trước trên 20GB văn bản báo chí tiếng Việt qua nhiệm vụ Masked Language Modeling.\n` +
      `• **Dynamic Contextual Embedding**: Không giống Word2Vec có vector tĩnh, PhoBERT tạo vector động theo ngữ cảnh nhờ 12 tầng Transformer.\n` +
      `• **Đạt kỷ lục trong đồ án**: 95.50% trên UIT-VSFC và 86.17% trên E-Commerce. Vector [CLS] gom trọn ý nghĩa sâu sắc của câu.`;
  }

  // Mặc định: Trả lời tổng quát
  return `💡 **Gợi ý từ trợ lý học tập NLP:**\n` +
    `Câu hỏi của bạn liên quan đến nội dung trọng tâm của đề tài. Đồ án giữa kỳ này gồm 4 trụ cột chính:\n` +
    `1. Biểu diễn từ: One-Hot -> BoW/TF-IDF -> Word2Vec (CBOW, Skip-Gram, SGNS).\n` +
    `2. Biểu diễn câu: Average Word2Vec (mất trật tự) -> BiLSTM (ghép 2 chiều xuôi/ngược d=256).\n` +
    `3. Cơ chế thích nghi: Additive Self-Attention (bản đồ nhiệt giải thích từ then chốt) và PhoBERT Transformer SOTA.\n` +
    `4. Thực nghiệm 2 miền: UIT-VSFC (chuẩn mực) vs E-Commerce (nhiễu teencode, dịch chuyển miền).\n\n` +
    `Bạn có thể bấm vào các thẻ câu hỏi gợi ý bên trên hoặc chọn tab 'Luyện Vấn Đáp' để xem chi tiết từng công thức!`;
}

/* ==========================================================================
   11. WIDGET 9: INTERACTIVE SELF-ASSESSMENT QUIZ
   ========================================================================== */
function initQuiz() {
  const container = document.getElementById("quiz-container");
  if (!container) return;

  container.innerHTML = "";

  NLP_DATA.quizQuestions.forEach((q, idx) => {
    const card = document.createElement("div");
    card.classList.add("glass-panel", "p-5", "space-y-3", "border-gray-800");
    card.id = `quiz-card-${q.id}`;

    const optionsHtml = q.options.map((opt, oIdx) => `
      <label class="flex items-center gap-3 p-3 rounded-lg border border-gray-800 hover:bg-gray-800/60 cursor-pointer transition text-sm text-gray-300">
        <input type="radio" name="quiz-q-${q.id}" value="${oIdx}" class="text-indigo-600 focus:ring-indigo-500">
        <span>${opt}</span>
      </label>
    `).join("");

    card.innerHTML = `
      <div class="flex items-center justify-between text-xs font-semibold text-indigo-400">
        <span>CÂU HỎI ${idx + 1}/10</span>
        <span id="quiz-status-${q.id}" class="text-gray-500">Chưa trả lời</span>
      </div>
      <h4 class="text-base font-bold text-gray-100">${q.question}</h4>
      <div class="space-y-2 mt-2">
        ${optionsHtml}
      </div>
      <div class="pt-2 flex justify-between items-center">
        <button class="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-semibold transition" onclick="checkQuizAnswer(${q.id})">
          Kiểm Tra Đáp Án
        </button>
        <div id="quiz-explain-${q.id}" class="hidden text-xs p-2.5 rounded border max-w-xl"></div>
      </div>
    `;

    container.appendChild(card);
  });
}

window.checkQuizAnswer = function(qid) {
  const qObj = NLP_DATA.quizQuestions.find(q => q.id === qid);
  if (!qObj) return;

  const selectedRadio = document.querySelector(`input[name="quiz-q-${qid}"]:checked`);
  const statusEl = document.getElementById(`quiz-status-${qid}`);
  const explainEl = document.getElementById(`quiz-explain-${qid}`);

  if (!selectedRadio) {
    alert("Vui lòng chọn 1 đáp án trước khi bấm kiểm tra!");
    return;
  }

  const selectedIdx = parseInt(selectedRadio.value);
  const isCorrect = selectedIdx === qObj.correct;

  if (explainEl) {
    explainEl.classList.remove("hidden");
    if (isCorrect) {
      explainEl.className = "text-xs p-2.5 rounded border bg-emerald-950/60 text-emerald-300 border-emerald-800";
      explainEl.innerHTML = `<strong> chính xác!</strong> ${qObj.explanation}`;
      if (statusEl) {
        statusEl.textContent = "Chính xác (+10đ)";
        statusEl.className = "text-emerald-400 font-bold text-xs";
      }
    } else {
      explainEl.className = "text-xs p-2.5 rounded border bg-rose-950/60 text-rose-300 border-rose-800";
      explainEl.innerHTML = `<strong>❌ Chưa đúng!</strong> Đáp án đúng là: <em>"${qObj.options[qObj.correct]}"</em>.<br/>${qObj.explanation}`;
      if (statusEl) {
        statusEl.textContent = "Chưa đúng (0đ)";
        statusEl.className = "text-rose-400 font-bold text-xs";
      }
    }
  }

  updateTotalQuizScore();
};

function updateTotalQuizScore() {
  let score = 0;
  let totalChecked = 0;

  NLP_DATA.quizQuestions.forEach(q => {
    const selectedRadio = document.querySelector(`input[name="quiz-q-${q.id}"]:checked`);
    if (selectedRadio) {
      totalChecked++;
      if (parseInt(selectedRadio.value) === q.correct) {
        score += 10;
      }
    }
  });

  const scoreBadge = document.getElementById("quiz-total-score");
  if (scoreBadge) {
    scoreBadge.textContent = `${score}/100 Điểm (${totalChecked}/10 câu)`;
    if (score >= 80) {
      scoreBadge.className = "px-3 py-1 bg-emerald-900/60 text-emerald-300 border border-emerald-700 rounded-full font-bold text-sm badge-glow-green";
    } else if (score >= 50) {
      scoreBadge.className = "px-3 py-1 bg-amber-900/60 text-amber-300 border border-amber-700 rounded-full font-bold text-sm";
    } else {
      scoreBadge.className = "px-3 py-1 bg-gray-800 text-gray-300 border border-gray-700 rounded-full font-bold text-sm";
    }
  }
}
