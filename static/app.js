const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  dataset: { labels: [], total: 0, training_issue: null },
  status: null,
  models: [],
  selectedLabel: "none",
  stream: null,
  streamOwner: null,
  autoTimer: null,
  captureBusy: false,
  uploadBusy: false,
  testTimer: null,
  predicting: false,
  lastJobSignature: "",
  lastJobStatus: "idle",
};

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
  [13,17],[0,17],[17,18],[18,19],[19,20],
];

const BUILTIN_LABELS = {
  None: "未识别", Closed_Fist: "握拳", Open_Palm: "张开手掌",
  Pointing_Up: "食指向上", Thumb_Down: "点赞向下", Thumb_Up: "点赞",
  Victory: "胜利手势", ILoveYou: "I Love You",
};

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let body = {};
  try { body = await response.json(); } catch (_) { /* no-op */ }
  if (!response.ok) throw new Error(body.error || `请求失败（${response.status}）`);
  return body;
}

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove("show"), 2600);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function formatLabel(label) {
  return BUILTIN_LABELS[label] || label;
}

function switchPanel(panel) {
  $$(".workflow-tab").forEach((button) => button.classList.toggle("active", button.dataset.panel === panel));
  $$(".workspace-panel").forEach((section) => section.classList.toggle("active", section.id === `panel-${panel}`));
  if (panel !== "collect") stopAutoCapture();
  if (panel !== "test") stopTesting();
  if (window.innerWidth <= 1000) {
    document.querySelector(".workflow-nav").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function refreshAll() {
  try {
    const [dataset, status, models] = await Promise.all([
      api("/api/dataset"), api("/api/status"), api("/api/models"),
    ]);
    state.dataset = dataset;
    state.status = status;
    state.models = models.models;
    $(".top-status").classList.add("online");
    $("#topStatusText").textContent = "本地服务已连接";
    renderMetrics();
    renderDataset();
    renderEnvironment();
    renderModels();
    renderJob(status.job);
    await loadGallery();
  } catch (error) {
    $("#topStatusText").textContent = "本地服务连接失败";
    toast(error.message, true);
  }
}

function renderMetrics() {
  const gestures = state.dataset.labels.filter((item) => item.name !== "none").length;
  $("#metricSamples").textContent = `${state.dataset.total} 张样本`;
  $("#metricLabels").textContent = `${gestures} 个手势`;
  $("#metricModels").textContent = `${state.models.length} 个可用`;
  const environment = state.status?.training_mode;
  $("#metricEnvironment").textContent = environment === "docker" ? "容器已就绪" : environment === "native" ? "本机已就绪" : "尚未准备";
}

function renderDataset() {
  const list = $("#labelList");
  if (!state.dataset.labels.some((item) => item.name === state.selectedLabel)) state.selectedLabel = "none";
  list.innerHTML = state.dataset.labels.map((item) => {
    const percent = Math.min(100, Math.round(item.count / item.recommended * 100));
    return `<div class="label-row ${item.name === state.selectedLabel ? "selected" : ""}" data-label="${escapeHtml(item.name)}">
      <button class="label-select" type="button">
        <strong>${escapeHtml(item.name)}${item.background ? " · 背景/必需" : " · 模型类别"}</strong>
        <small>${item.background ? "不属于当前模型全部类别的其他动作" : `当前模型类别 · 建议 ${item.recommended} 张`}
          <span class="progress-mini"><i style="width:${percent}%"></i></span>
        </small>
        <span class="label-count">${item.count}</span>
      </button>
      ${item.required ? "" : '<button class="delete-label" type="button" aria-label="删除标签">×</button>'}
    </div>`;
  }).join("");
  $("#datasetTotal").textContent = state.dataset.total;
  $("#captureLabelBadge").textContent = `当前：${state.selectedLabel}`;
  $("#galleryTitle").textContent = `${state.selectedLabel} 的最近样本`;
  renderReadiness();

  $$(".label-row").forEach((row) => {
    row.querySelector(".label-select").addEventListener("click", async () => {
      state.selectedLabel = row.dataset.label;
      renderDataset();
      await loadGallery();
    });
    row.querySelector(".delete-label")?.addEventListener("click", async () => {
      const label = row.dataset.label;
      if (!confirm(`删除标签“${label}”及其全部样本？此操作无法撤销。`)) return;
      try {
        await api(`/api/labels/${encodeURIComponent(label)}`, { method: "DELETE" });
        state.selectedLabel = "none";
        await refreshDataset();
        toast(`已删除标签 ${label}`);
      } catch (error) { toast(error.message, true); }
    });
  });
}

async function refreshDataset() {
  state.dataset = await api("/api/dataset");
  renderDataset();
  renderMetrics();
  await loadGallery();
}

async function loadGallery() {
  const gallery = $("#sampleGallery");
  try {
    const data = await api(`/api/samples/${encodeURIComponent(state.selectedLabel)}`);
    $("#galleryCount").textContent = `${data.count} 张`;
    if (!data.items.length) {
      gallery.innerHTML = '<p class="empty-state">还没有样本，可开启摄像头拍摄或上传本地照片。</p>';
      return;
    }
    gallery.innerHTML = data.items.map((item) => `<figure class="sample-item ${item.source === "upload" ? "uploaded" : ""}">
      <img src="${item.url}" alt="${escapeHtml(state.selectedLabel)} 手势样本" loading="lazy">
      <button type="button" data-name="${escapeHtml(item.name)}" aria-label="删除这张样本">×</button>
    </figure>`).join("");
    gallery.querySelectorAll("button").forEach((button) => button.addEventListener("click", async () => {
      try {
        await api(`/api/samples/${encodeURIComponent(state.selectedLabel)}/${encodeURIComponent(button.dataset.name)}`, { method: "DELETE" });
        await refreshDataset();
      } catch (error) { toast(error.message, true); }
    }));
  } catch (error) {
    gallery.innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`;
  }
}

async function startCamera(owner) {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("当前浏览器不支持摄像头访问。");
  if (state.stream) stopCamera();
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false,
  });
  state.stream = stream;
  state.streamOwner = owner;
  const video = owner === "capture" ? $("#captureVideo") : $("#testVideo");
  video.srcObject = stream;
  await video.play();
  if (owner === "capture") {
    $("#capturePlaceholder").classList.add("hidden");
    $("#captureOnce").disabled = false;
    $("#autoCaptureToggle").disabled = false;
    $("#startCaptureCamera").textContent = "关闭摄像头";
  } else {
    $("#testPlaceholder").classList.add("hidden");
    $("#startTesting").textContent = "停止测试";
  }
}

function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  if (state.streamOwner === "capture") {
    $("#captureVideo").srcObject = null;
    $("#capturePlaceholder").classList.remove("hidden");
    $("#captureOnce").disabled = true;
    $("#autoCaptureToggle").disabled = true;
    $("#startCaptureCamera").textContent = "开启摄像头";
  }
  if (state.streamOwner === "test") {
    $("#testVideo").srcObject = null;
    $("#testPlaceholder").classList.remove("hidden");
    $("#startTesting").textContent = "开始实时测试";
  }
  state.stream = null;
  state.streamOwner = null;
}

function frameData(video, maxWidth = 960, quality = .84) {
  if (!video.videoWidth) throw new Error("摄像头画面尚未准备好。");
  const scale = Math.min(1, maxWidth / video.videoWidth);
  const canvas = $("#captureCanvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const context = canvas.getContext("2d");
  context.save();
  context.translate(canvas.width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  context.restore();
  return canvas.toDataURL("image/jpeg", quality);
}

async function captureSample(silent = false) {
  if (state.captureBusy || state.streamOwner !== "capture") return;
  state.captureBusy = true;
  try {
    const image = frameData($("#captureVideo"));
    const result = await api(`/api/samples/${encodeURIComponent(state.selectedLabel)}`, {
      method: "POST", body: JSON.stringify({ image }),
    });
    $("#captureFlash").classList.remove("flash");
    void $("#captureFlash").offsetWidth;
    $("#captureFlash").classList.add("flash");
    const label = state.dataset.labels.find((item) => item.name === state.selectedLabel);
    if (label) { label.count = result.count; state.dataset.total += 1; }
    if (result.count % 5 === 0) {
      state.dataset = await api("/api/dataset");
    }
    renderDataset();
    renderMetrics();
    if (result.count % 5 === 0 || !silent) await loadGallery();
    if (!silent) toast(`已保存到 ${state.selectedLabel}`);
  } catch (error) {
    stopAutoCapture();
    toast(error.message, true);
  } finally {
    state.captureBusy = false;
  }
}

function imageFileData(file, maxDimension = 1600, quality = .9) {
  if (!file.type.startsWith("image/")) return Promise.reject(new Error(`${file.name} 不是图片文件。`));
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(objectUrl);
      const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/jpeg", quality));
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error(`${file.name} 无法读取，请换一张图片。`));
    };
    image.src = objectUrl;
  });
}

async function uploadSamples(files) {
  if (state.uploadBusy || !files.length) return;
  state.uploadBusy = true;
  const label = state.selectedLabel;
  const button = $("#uploadSamples");
  button.disabled = true;
  let saved = 0;
  const errors = [];
  try {
    for (const [index, file] of [...files].entries()) {
      button.textContent = `上传中 ${index + 1}/${files.length}`;
      try {
        const image = await imageFileData(file);
        await api(`/api/samples/${encodeURIComponent(label)}`, {
          method: "POST", body: JSON.stringify({ image, source: "upload" }),
        });
        saved += 1;
      } catch (error) {
        errors.push(error.message);
      }
    }
    await refreshDataset();
    if (saved) toast(`已上传 ${saved} 张照片到 ${label}${errors.length ? `，${errors.length} 张失败` : ""}`, Boolean(errors.length));
    else toast(errors[0] || "没有可上传的照片。", true);
  } catch (error) {
    toast(error.message, true);
  } finally {
    state.uploadBusy = false;
    button.disabled = false;
    button.textContent = "上传照片";
    $("#sampleUpload").value = "";
  }
}

function startAutoCapture() {
  if (state.streamOwner !== "capture") return;
  stopAutoCapture(false);
  const interval = Number($("#captureInterval").value);
  state.autoTimer = setInterval(() => captureSample(true), interval);
  $("#autoCaptureToggle").checked = true;
  toast("连续采集已开始");
}

function stopAutoCapture(updateToggle = true) {
  clearInterval(state.autoTimer);
  state.autoTimer = null;
  if (updateToggle) $("#autoCaptureToggle").checked = false;
}

function renderEnvironment() {
  const banner = $("#environmentBanner");
  const title = $("#environmentTitle");
  const description = $("#environmentDescription");
  const button = $("#buildEnvironment");
  banner.className = "environment-banner";
  if (state.status.training_mode === "docker" || state.status.training_mode === "native") {
    banner.classList.add("ready");
    title.textContent = "训练环境已就绪";
    description.textContent = state.status.training_mode === "docker" ? "Linux x86_64 训练容器 · Model Maker 0.2.1.4" : "本机训练环境 · Model Maker 0.2.1.4";
    button.textContent = "环境已就绪";
    button.disabled = true;
  } else if (!state.status.docker_installed) {
    banner.classList.add("blocked");
    title.textContent = "需要安装 Docker Desktop";
    description.textContent = "当前是 Apple Silicon；安装并启动 Docker Desktop 后即可在页面内准备训练环境。";
    button.textContent = "未检测到 Docker";
    button.disabled = true;
  } else {
    banner.classList.add("blocked");
    title.textContent = "训练镜像尚未构建";
    description.textContent = "首次准备会下载 TensorFlow 等依赖，通常需要几分钟。";
    button.textContent = "准备训练环境";
    button.disabled = state.status.job?.status === "running";
  }
  renderReadiness();
}

function renderReadiness() {
  const card = $("#readinessCard");
  const start = $("#startTraining");
  if (!card) return;
  const environmentReady = ["docker", "native"].includes(state.status?.training_mode);
  if (state.dataset.training_issue) {
    card.className = "readiness-card full warn";
    card.innerHTML = `<strong>数据集尚未就绪</strong><br>${escapeHtml(state.dataset.training_issue)}`;
  } else if (!environmentReady) {
    card.className = "readiness-card full warn";
    card.innerHTML = "<strong>训练环境尚未就绪</strong><br>完成上方环境准备后即可开始训练。";
  } else {
    card.className = "readiness-card full ready";
    card.innerHTML = `<strong>可以开始训练</strong><br>${state.dataset.labels.length} 个标签 · ${state.dataset.total} 张样本 · 环境版本已锁定`;
  }
  start.disabled = Boolean(state.dataset.training_issue) || !environmentReady || state.status?.job?.status === "running";
}

function renderModels() {
  const select = $("#modelSelect");
  const previous = select.value;
  if (!state.models.length) {
    select.innerHTML = '<option value="">暂无可用模型</option>';
    $("#startTesting").disabled = true;
    return;
  }
  select.innerHTML = state.models.map((model) => `<option value="${escapeHtml(model.path)}">${escapeHtml(model.name)} · ${model.independent ? "独立分类模型" : "非训练模型/不可测试"} · ${model.size_mb} MB</option>`).join("");
  const previousModel = state.models.find((model) => model.path === previous && model.independent);
  const firstUnified = state.models.find((model) => model.independent);
  if (previousModel) select.value = previousModel.path;
  else if (firstUnified) select.value = firstUnified.path;
  updateTestAvailability();
}

function updateTestAvailability() {
  const model = state.models.find((item) => item.path === $("#modelSelect").value);
  const usable = Boolean(model?.independent);
  $("#startTesting").disabled = !usable;
  $("#modelHint").textContent = usable
    ? `独立分类头：${model.labels.length} 类，类别分数可直接比较`
    : "该文件不含可用的独立分类头，请选择页面训练生成的模型";
}

function renderJob(job) {
  if (!job) return;
  const signature = `${job.id}:${job.status}:${job.progress}:${job.logs.length}`;
  $("#jobStatus").textContent = job.status.toUpperCase();
  $("#progressValue").textContent = `${job.progress}%`;
  $("#progressBar").style.width = `${job.progress}%`;
  $("#jobMessage").textContent = job.message;
  $("#epochValue").textContent = job.total_epochs ? `EPOCH ${job.epoch} / ${job.total_epochs}` : job.kind === "environment" ? "BUILDING IMAGE" : "等待开始";
  $("#jobTime").textContent = job.started_at ? job.started_at.replace("T", " ") : "—";
  $("#cancelJob").disabled = job.status !== "running";
  if (signature !== state.lastJobSignature) {
    const log = $("#jobLogs");
    log.textContent = job.logs.length ? job.logs.join("\n") : "训练日志会显示在这里。";
    log.scrollTop = log.scrollHeight;
    state.lastJobSignature = signature;
  }
  if (state.lastJobStatus === "running" && ["success", "error", "cancelled"].includes(job.status)) {
    toast(job.message, job.status === "error");
    refreshAll();
  }
  state.lastJobStatus = job.status;
}

async function pollJob() {
  try {
    const job = await api("/api/job");
    renderJob(job);
    if (state.status) state.status.job = job;
    renderReadiness();
  } catch (_) { /* server may be restarting */ }
}

async function startTesting() {
  if (state.testTimer) { stopTesting(); return; }
  const model = $("#modelSelect").value;
  if (!model) return toast("请先选择模型。", true);
  try {
    await startCamera("test");
    state.testTimer = setInterval(runPrediction, 230);
    runPrediction();
  } catch (error) { toast(error.message, true); }
}

function stopTesting(stopStream = true) {
  clearInterval(state.testTimer);
  state.testTimer = null;
  state.predicting = false;
  if (stopStream && state.streamOwner === "test") stopCamera();
  $("#startTesting").textContent = "开始实时测试";
  const canvas = $("#landmarkCanvas");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}

async function runPrediction() {
  if (state.predicting || state.streamOwner !== "test") return;
  state.predicting = true;
  const started = performance.now();
  try {
    const image = frameData($("#testVideo"), 720, .72);
    const result = await api("/api/predict", {
      method: "POST",
      body: JSON.stringify({ model: $("#modelSelect").value, image }),
    });
    $("#inferenceSpeed").textContent = `${Math.round(performance.now() - started)} ms`;
    renderPrediction(result.hands);
    drawLandmarks(result.hands);
  } catch (error) {
    stopTesting();
    toast(error.message, true);
  } finally {
    state.predicting = false;
  }
}

function renderPrediction(hands) {
  $("#handCount").textContent = `${hands.length} HAND${hands.length === 1 ? "" : "S"}`;
  const list = $("#predictionList");
  if (!hands.length) {
    list.innerHTML = '<div class="prediction-empty"><span>⌁</span><strong>未检测到手</strong><small>将手掌完整放到镜头中央</small></div>';
    return;
  }
  list.innerHTML = hands.map((hand) => {
    const best = hand.winner || hand.top_gestures[0] || { label: "none", score: 0, source: "model" };
    const baseline = hand.baseline_gesture;
    const modelResult = hand.model_gestures[0];
    const ranking = hand.top_gestures.map((item, index) => `<div class="ranking-row">
      <span class="ranking-index">${String(index + 1).padStart(2, "0")}</span>
      <span class="ranking-label">${escapeHtml(formatLabel(item.label))}</span>
      <span class="source-badge custom">当前</span>
      <span class="ranking-score">${Math.round(item.score * 100)}%</span>
    </div>`).join("");
    return `<div class="hand-result">
      <div class="hand-result-head"><strong>${hand.handedness === "Left" ? "左手" : hand.handedness === "Right" ? "右手" : "手"}</strong><small>当前模型判定</small></div>
      <div class="gesture-name">${escapeHtml(formatLabel(best.label))}</div>
      <div class="confidence-line"><div><i style="width:${Math.round(best.score * 100)}%"></i></div><span>${Math.round(best.score * 100)}%</span></div>
      <div class="classifier-summary">
        ${renderClassifierSummary("当前模型", modelResult)}
        ${renderClassifierSummary("官方基线（仅参考）", baseline)}
      </div>
      <div class="ranking-title"><strong>当前模型真实 TOP 5</strong><span>同一分类空间</span></div>
      <div class="gesture-ranking">${ranking || '<span class="empty-state">暂无候选结果</span>'}</div>
    </div>`;
  }).join("");
}

function renderClassifierSummary(title, item) {
  if (!item) return `<div class="classifier-result"><small>${title}</small><strong>暂无结果</strong><span>—</span></div>`;
  return `<div class="classifier-result"><small>${title}</small><strong>${escapeHtml(formatLabel(item.label))}</strong><span>${Math.round(item.score * 100)}%</span></div>`;
}

function drawLandmarks(hands) {
  const video = $("#testVideo");
  const canvas = $("#landmarkCanvas");
  const width = video.clientWidth;
  const height = video.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const context = canvas.getContext("2d");
  context.scale(dpr, dpr);
  context.clearRect(0, 0, width, height);
  if (!video.videoWidth) return;
  const scale = Math.max(width / video.videoWidth, height / video.videoHeight);
  const drawnWidth = video.videoWidth * scale;
  const drawnHeight = video.videoHeight * scale;
  const offsetX = (width - drawnWidth) / 2;
  const offsetY = (height - drawnHeight) / 2;
  const point = (landmark) => ({ x: offsetX + landmark.x * drawnWidth, y: offsetY + landmark.y * drawnHeight });

  hands.forEach((hand, handIndex) => {
    const color = handIndex === 0 ? "#ccff45" : "#65a5ff";
    context.strokeStyle = color;
    context.lineWidth = 2.5;
    context.lineCap = "round";
    HAND_CONNECTIONS.forEach(([start, end]) => {
      const a = point(hand.landmarks[start]);
      const b = point(hand.landmarks[end]);
      context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
    });
    hand.landmarks.forEach((landmark) => {
      const p = point(landmark);
      context.beginPath(); context.arc(p.x, p.y, 4, 0, Math.PI * 2);
      context.fillStyle = "#ffffff"; context.fill(); context.stroke();
    });
  });
}

function wireEvents() {
  $$(".workflow-tab").forEach((button) => button.addEventListener("click", () => switchPanel(button.dataset.panel)));
  $("#labelForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = $("#newLabel");
    const label = input.value.trim();
    if (!label) return;
    try {
      const result = await api("/api/labels", { method: "POST", body: JSON.stringify({ label }) });
      state.selectedLabel = result.label;
      input.value = "";
      await refreshDataset();
      toast(`标签 ${result.label} 已创建`);
    } catch (error) { toast(error.message, true); }
  });
  $("#startCaptureCamera").addEventListener("click", async () => {
    if (state.streamOwner === "capture") { stopAutoCapture(); stopCamera(); return; }
    try { await startCamera("capture"); } catch (error) { toast(error.message, true); }
  });
  $("#captureOnce").addEventListener("click", () => captureSample());
  $("#uploadSamples").addEventListener("click", () => $("#sampleUpload").click());
  $("#sampleUpload").addEventListener("change", (event) => uploadSamples(event.target.files));
  $("#autoCaptureToggle").addEventListener("change", (event) => event.target.checked ? startAutoCapture() : stopAutoCapture());
  $("#captureInterval").addEventListener("change", () => { if (state.autoTimer) startAutoCapture(); });
  $("#buildEnvironment").addEventListener("click", async () => {
    try {
      const job = await api("/api/environment/build", { method: "POST", body: "{}" });
      if (state.status) state.status.job = job;
      renderJob(job); renderEnvironment(); toast("开始准备训练环境");
    } catch (error) { toast(error.message, true); }
  });
  $("#trainingForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const payload = Object.fromEntries(form.entries());
    try {
      const job = await api("/api/train", { method: "POST", body: JSON.stringify(payload) });
      if (state.status) state.status.job = job;
      renderJob(job); renderReadiness(); toast("训练任务已开始");
    } catch (error) { toast(error.message, true); }
  });
  $("#cancelJob").addEventListener("click", async () => {
    if (!confirm("确定取消当前任务？")) return;
    try { renderJob(await api("/api/job/cancel", { method: "POST", body: "{}" })); }
    catch (error) { toast(error.message, true); }
  });
  $("#refreshModels").addEventListener("click", async () => {
    try { state.models = (await api("/api/models")).models; renderModels(); renderMetrics(); toast("模型列表已刷新"); }
    catch (error) { toast(error.message, true); }
  });
  $("#modelSelect").addEventListener("change", () => {
    if (state.testTimer) stopTesting();
    updateTestAvailability();
  });
  $("#startTesting").addEventListener("click", startTesting);
  window.addEventListener("beforeunload", stopCamera);
}

wireEvents();
refreshAll();
setInterval(pollJob, 1200);
