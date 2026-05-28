const $ = (sel) => document.querySelector(sel);

const els = {
  healthPill: $("#health-pill"),
  modelList: $("#model-list"),
  loadedModel: $("#loaded-model"),
  btnRefreshModels: $("#btn-refresh-models"),
  btnLoad: $("#btn-load"),
  btnPredict: $("#btn-predict"),
  btnReloadData: $("#btn-reload-data"),
  saveSummary: $("#save-summary"),
  resultEmpty: $("#result-empty"),
  result: $("#result"),
  redBalls: $("#red-balls"),
  blueBall: $("#blue-ball"),
  metaModel: $("#meta-model"),
  metaTimestamp: $("#meta-timestamp"),
  metaWindow: $("#meta-window"),
  metaLastIssue: $("#meta-last-issue"),
  metaNormalized: $("#meta-normalized"),
  metaSummaryPath: $("#meta-summary-path"),
  toast: $("#toast"),
};

let models = [];
let selectedModelPath = null;
let loadedModelPath = null;
let toastTimer = null;

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  let body = null;
  const text = await res.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { detail: text };
    }
  }
  if (!res.ok) {
    const detail = body?.detail;
    const msg = typeof detail === "string" ? detail : JSON.stringify(detail ?? res.statusText);
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return body;
}

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.classList.toggle("toast--error", isError);
  els.toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.add("hidden"), 4000);
}

function setHealth(status, text) {
  els.healthPill.className = `pill pill--${status}`;
  els.healthPill.querySelector(".pill__text").textContent = text;
}

function renderBall(num, type, size = "md", delay = 0) {
  const el = document.createElement("span");
  el.className = `ball ball--${type} ball--${size}`;
  el.textContent = String(num).padStart(2, "0");
  el.style.animationDelay = `${delay}ms`;
  return el;
}

function updateLoadedModel(info) {
  const block = els.loadedModel;
  const value = block.querySelector(".loaded-model__value");
  if (!info) {
    loadedModelPath = null;
    block.classList.remove("loaded-model--active");
    value.textContent = "未加载";
    els.btnPredict.disabled = true;
    return;
  }
  loadedModelPath = info.path;
  block.classList.add("loaded-model--active");
  value.textContent = info.path;
  els.btnPredict.disabled = false;
}

async function refreshHealth() {
  try {
    const h = await api("/health");
    const modelNote = h.model_loaded ? " · 模型已加载" : "";
    setHealth("ok", `${h.records.toLocaleString()} 期数据${modelNote}`);
  } catch (e) {
    setHealth("error", "API 不可用");
  }
}

async function fetchCurrentModel() {
  try {
    const info = await api("/models/current");
    updateLoadedModel(info);
  } catch {
    updateLoadedModel(null);
  }
}

function renderModelList() {
  const list = els.modelList;
  list.innerHTML = "";

  if (!models.length) {
    list.innerHTML = '<p class="model-list--empty">暂无模型，请先运行 <code>lottery train</code></p>';
    els.btnLoad.disabled = true;
    return;
  }

  models.forEach((m) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "model-card";
    btn.dataset.path = m.path;
    btn.setAttribute("role", "option");
    btn.innerHTML = `
      <span class="model-card__id">${m.id}</span>
      <span class="model-card__time">${m.timestamp ?? "—"}</span>
    `;
    if (m.path === selectedModelPath) {
      btn.classList.add("model-card--selected");
    }
    btn.addEventListener("click", () => {
      selectedModelPath = m.path;
      list.querySelectorAll(".model-card").forEach((c) => c.classList.remove("model-card--selected"));
      btn.classList.add("model-card--selected");
      els.btnLoad.disabled = false;
    });
    list.appendChild(btn);
  });

  if (selectedModelPath) {
    els.btnLoad.disabled = false;
  }
}

async function loadModels() {
  models = await api("/models");
  if (!selectedModelPath && models[0]) {
    selectedModelPath = models[0].path;
  }
  renderModelList();
}

async function loadSelectedModel() {
  if (!selectedModelPath) return;
  els.btnLoad.disabled = true;
  try {
    const info = await api("/models/load", {
      method: "POST",
      body: JSON.stringify({ model: selectedModelPath }),
    });
    updateLoadedModel(info);
    showToast("模型已加载");
    await refreshHealth();
  } catch (e) {
    showToast(e.message, true);
  } finally {
    els.btnLoad.disabled = !selectedModelPath;
  }
}

function showPrediction(data) {
  els.resultEmpty.classList.add("hidden");
  els.result.classList.remove("hidden");

  els.redBalls.innerHTML = "";
  els.blueBall.innerHTML = "";

  const { red_balls, blue_ball } = data.prediction;
  red_balls.forEach((n, i) => {
    els.redBalls.appendChild(renderBall(n, "red", "md", i * 60));
  });
  els.blueBall.appendChild(renderBall(blue_ball, "blue", "md", 360));

  els.metaModel.textContent = data.model_dir;
  els.metaTimestamp.textContent = data.model_timestamp ?? "—";
  const w = data.input_window;
  els.metaWindow.textContent = `${w.seq_len} 期 · ${w.issues.join(", ")}`;
  els.metaLastIssue.textContent = w.last_issue;
  els.metaNormalized.textContent = JSON.stringify(data.normalized, null, 2);

  if (data.summary_path) {
    els.metaSummaryPath.textContent = `已保存: ${data.summary_path}`;
    els.metaSummaryPath.classList.remove("hidden");
  } else {
    els.metaSummaryPath.classList.add("hidden");
  }
}

async function runPredict() {
  const btn = els.btnPredict;
  btn.classList.add("is-loading");
  btn.disabled = true;
  try {
    const body = {
      save_summary: els.saveSummary.checked,
    };
    if (!loadedModelPath && selectedModelPath) {
      body.model = selectedModelPath;
    }
    const data = await api("/predict", {
      method: "POST",
      body: JSON.stringify(body),
    });
    showPrediction(data);
    if (data.model_dir) {
      updateLoadedModel({ path: data.model_dir, timestamp: data.model_timestamp });
    }
    showToast("预测完成");
    await refreshHealth();
  } catch (e) {
    showToast(e.message, true);
  } finally {
    btn.classList.remove("is-loading");
    btn.disabled = !loadedModelPath && !selectedModelPath;
    if (loadedModelPath) btn.disabled = false;
  }
}

async function reloadData() {
  try {
    const { records } = await api("/data/reload", { method: "POST" });
    showToast(`已刷新 ${records.toLocaleString()} 期数据`);
    await refreshHealth();
  } catch (e) {
    showToast(e.message, true);
  }
}

els.btnRefreshModels.addEventListener("click", () => loadModels().catch((e) => showToast(e.message, true)));
els.btnLoad.addEventListener("click", loadSelectedModel);
els.btnPredict.addEventListener("click", runPredict);
els.btnReloadData.addEventListener("click", reloadData);

async function init() {
  await refreshHealth();
  await Promise.all([loadModels(), fetchCurrentModel()]);
  if (loadedModelPath) {
    selectedModelPath = loadedModelPath;
    renderModelList();
  }
}

init().catch((e) => {
  setHealth("error", "初始化失败");
  showToast(e.message, true);
});
