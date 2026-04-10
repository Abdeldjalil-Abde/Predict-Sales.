// ============================================================
//  SalesCast AI — main.js  (CORRECTED)
//  Front-end controller.
// ============================================================

// ── App state ──────────────────────────────────────────────
let state = {
  rawData: null,
  headers: [],
  numericCols: [],
  targetCol: "",
  featureCols: [],
  results: null,
  selectedModels: new Set(["lr", "dt", "nn"]),
};

// ── Navigation ─────────────────────────────────────────────
// FIX 1: Single authoritative showPage — "predict" title added here,
//         removed the duplicate declaration that caused infinite recursion.
function showPage(id, navEl) {
  document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
  document.getElementById("page-" + id).classList.add("active");
  if (navEl) navEl.classList.add("active");

  const titles = {
    home:    "Dashboard Overview",
    upload:  "Upload Dataset",
    train:   "Model Training",
    results: "Results Dashboard",
    predict: "Make Predictions",   // was missing in original single definition
  };
  document.getElementById("pageTitle").textContent = titles[id] || id;

  // FIX 2: When the predict page is opened via the sidebar nav,
  //         initialise the content/form exactly as showPredictionPage() does.
  if (id === "predict") {
    _initPredictPage();
  }
}

// ── File handling ───────────────────────────────────────────
function handleDrop(e) {
  e.preventDefault();
  document.getElementById("uploadZone").classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) processFile(file);
}

function handleFile(e) {
  processFile(e.target.files[0]);
}

function processFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (ext === "csv") {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (r) => loadDataset(r.data, r.meta.fields),
    });
  } else {
    alert(
      'For XLSX files, please convert to CSV first. You can use "Use Sample Data" to try the platform.',
    );
  }
}

function loadDataset(data, headers) {
  state.rawData = data;
  state.headers = headers;
  state.numericCols = headers.filter((h) => typeof data[0][h] === "number");
  populateSelectors();
  buildPreviewTable(data.slice(0, 10), headers);
  document.getElementById("dataInfo").style.display = "block";
  const ds = document.getElementById("dataStats");
  ds.innerHTML = `
    <div class="tag tag-blue">📋 ${data.length} rows</div>
    <div class="tag tag-green">📊 ${headers.length} columns</div>
    <div class="tag tag-purple">🔢 ${state.numericCols.length} numeric</div>
  `;
  document.getElementById("rowCount").textContent =
    `Showing 10 of ${data.length} rows`;
}

function populateSelectors() {
  const { headers, numericCols } = state;
  const targetSel = document.getElementById("targetCol");
  const dateSel   = document.getElementById("dateCol");
  const featSel   = document.getElementById("featureCols");

  targetSel.innerHTML = '<option value="">Select target column...</option>';
  dateSel.innerHTML   = '<option value="">None</option>';
  featSel.innerHTML   = "";

  numericCols.forEach((c) => {
    targetSel.innerHTML += `<option value="${c}">${c}</option>`;
    const opt = document.createElement("option");
    opt.value    = c;
    opt.textContent = c;
    opt.selected = true;
    featSel.appendChild(opt);
  });
  headers.forEach((c) => {
    dateSel.innerHTML += `<option value="${c}">${c}</option>`;
  });

  const salesLike = numericCols.find((c) =>
    /sales|revenue|amount|target|price|total/i.test(c),
  );
  if (salesLike) targetSel.value = salesLike;
}

function updateConfig() {
  const target  = document.getElementById("targetCol").value;
  const featSel = document.getElementById("featureCols");
  Array.from(featSel.options).forEach((o) => {
    o.selected = o.value !== target;
  });
}

function confirmUpload() {
  const target = document.getElementById("targetCol").value;
  if (!target) {
    alert("Please select a target column!");
    return;
  }

  const featSel = document.getElementById("featureCols");
  state.targetCol  = target;
  state.featureCols = Array.from(featSel.selectedOptions)
    .map((o) => o.value)
    .filter((v) => v !== target);

  if (state.featureCols.length === 0) {
    alert("Please select at least one feature column!");
    return;
  }

  document.getElementById("trainDataStatus").innerHTML =
    `✅ Dataset ready: <span style="color:var(--text)">${state.rawData.length} rows</span>` +
    ` · Target: <span class="tag tag-green">${target}</span>` +
    ` · Features: ${state.featureCols.map((f) => `<span class="tag tag-blue">${f}</span>`).join(" ")}`;

  document.getElementById("trainBtn").disabled = false;
  showPage("train", document.querySelectorAll(".nav-item")[2]);
}

function buildPreviewTable(data, headers) {
  const t = document.getElementById("previewTable");
  t.innerHTML = `<thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>`;
  const tbody = document.createElement("tbody");
  data.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = headers
      .map(
        (h) =>
          `<td>${row[h] !== null && row[h] !== undefined ? row[h] : "—"}</td>`,
      )
      .join("");
    tbody.appendChild(tr);
  });
  t.appendChild(tbody);
}

// ── Sample data ────────────────────────────────────────────
function loadSampleData() {
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const data = [];
  for (let y = 2021; y <= 2024; y++) {
    months.forEach((m, i) => {
      const trend      = (y - 2021) * 1200;
      const season     = Math.sin(((i + 3) * Math.PI) / 6) * 3000;
      const noise      = (Math.random() - 0.5) * 2000;
      const ad_spend   = 1000 + Math.random() * 4000;
      const store_count = 10 + Math.floor(Math.random() * 40);
      const price      = 20 + Math.random() * 30;
      const sales      = Math.round(
        Math.max(500, 8000 + trend + season + noise + ad_spend * 1.2 - price * 80 + store_count * 150),
      );
      data.push({
        month: `${y}-${String(i + 1).padStart(2, "0")}`,
        year: y,
        month_num: i + 1,
        ad_spend: Math.round(ad_spend),
        store_count,
        price: Math.round(price * 10) / 10,
        sales,
      });
    });
  }
  loadDataset(data, Object.keys(data[0]));
  document.getElementById("targetCol").value = "sales";
  updateConfig();
  showPage("upload", document.querySelectorAll(".nav-item")[1]);
}

// ── Model card toggle ───────────────────────────────────────
function toggleModel(card, model) {
  if (state.selectedModels.has(model)) {
    if (state.selectedModels.size <= 1) return;
    state.selectedModels.delete(model);
    card.classList.remove("selected");
  } else {
    state.selectedModels.add(model);
    card.classList.add("selected");
  }
}

// ── Training  (calls Python back-end via /api/train) ───────
let trainCharts = {};

async function startTraining() {
  if (!state.rawData || !state.targetCol) {
    alert("Please upload and configure a dataset first!");
    showPage("upload", document.querySelectorAll(".nav-item")[1]);
    return;
  }

  const btn  = document.getElementById("trainBtn");
  const log  = document.getElementById("trainLog");
  const prog = document.getElementById("progressContent");

  btn.disabled  = true;
  btn.innerHTML = '<span class="spinner"></span> Training…';
  document.getElementById("trainingProgress").style.display = "block";
  log.innerHTML = "";

  const addLog = (msg, cls = "") => {
    log.innerHTML += `<div class="log-line ${cls}">[${new Date().toLocaleTimeString()}] ${msg}</div>`;
    log.scrollTop  = log.scrollHeight;
  };
  const setProgress = (label, pct) => {
    prog.innerHTML =
      `<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px">` +
      `<span>${label}</span><span style="color:var(--text3)">${pct}%</span></div>` +
      `<div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>`;
  };

  const config = {
    test_split:    parseFloat(document.getElementById("testSplit").value),
    random_seed:   parseInt(document.getElementById("randomSeed").value),
    max_depth:     parseInt(document.getElementById("treeDepth").value),
    nn_epochs:     150,
    nn_hidden_units: 64,
  };

  addLog("📦 Sending dataset to server…", "info");
  setProgress("Uploading data", 10);

  const payload = {
    records:      state.rawData,
    feature_cols: state.featureCols,
    target_col:   state.targetCol,
    models:       [...state.selectedModels],
    config,
  };

  try {
    addLog("⚙️  Server is training models (Python / scikit-learn + TensorFlow)…", "info");
    setProgress("Training in progress", 40);

    const response = await fetch("/api/train", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.message || `Server error ${response.status}`);
    }

    setProgress("Processing results", 85);
    const data = await response.json();

    if (data.status !== "success") {
      throw new Error(data.message || "Training failed on server.");
    }

    Object.entries(data.models).forEach(([key, m]) => {
      if (m.error) {
        addLog(`❌ ${key.toUpperCase()} failed: ${m.error}`, "error");
      } else {
        addLog(`✅ ${m.name} · R²=${m.test.r2.toFixed(3)} · MAE=${Math.round(m.test.mae)}`, "success");
      }
    });

    const info = data.dataset_info;
    addLog(
      `📊 Dataset: ${info.n_samples_total} rows · ${info.n_train} train · ${info.n_test} test`,
      "info",
    );

    setProgress("All models trained!", 100);
    addLog("🎉 Training complete!", "success");

    state.results = data.models;

    btn.disabled  = false;
    btn.innerHTML = "✅ Retrain";
    document.getElementById("resultsBadge").style.display = "inline";

    renderResults(state.results);
    showPage("results", document.querySelectorAll(".nav-item")[3]);
  } catch (err) {
    addLog(`❌ Error: ${err.message}`, "error");
    setProgress("Training failed", 0);
    btn.disabled  = false;
    btn.innerHTML = "🔁 Retry";
    console.error(err);
  }
}

// ── Results rendering (Charts) ─────────────────────────────
// FIX 3: Single renderResults — extra logic (show predictNav) merged in here.
//         Removed the second declaration that caused infinite recursion.
function renderResults(results) {
  document.getElementById("noResults").style.display   = "none";
  document.getElementById("resultsContent").style.display = "block";

  const keys = Object.keys(results).filter((k) => !results[k].error);

  if (keys.length === 0) {
    document.getElementById("noResults").style.display   = "block";
    document.getElementById("resultsContent").style.display = "none";
    return;
  }

  const best = keys.reduce((a, b) =>
    results[a].test.r2 > results[b].test.r2 ? a : b,
  );

  document.getElementById("trainSummary").textContent =
    `${keys.length} model${keys.length > 1 ? "s" : ""} trained · Best: ${results[best].name} (R²=${results[best].test.r2.toFixed(3)})`;

  // ── Model cards
  const mc = document.getElementById("modelCards");
  mc.innerHTML = "";
  keys.forEach((k) => {
    const m      = results[k];
    const isBest = k === best;
    const r2cls  = m.test.r2 > 0.8 ? "good" : m.test.r2 > 0.5 ? "med" : "bad";
    mc.innerHTML += `
      <div class="result-card ${isBest ? "best" : ""}">
        ${isBest ? '<div class="best-badge">⭐ BEST MODEL</div>' : ""}
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
          <div style="font-size:24px">${m.icon}</div>
          <div>
            <div style="font-size:14px;font-weight:600">${m.name}</div>
            <div style="font-size:11px;color:var(--text3)">${isBest ? "Best performer" : ""}</div>
          </div>
        </div>
        <div class="metric-row"><span class="metric-name">R² Score</span>         <span class="metric-val ${r2cls}">${m.test.r2.toFixed(3)}</span></div>
        <div class="metric-row"><span class="metric-name">Accuracy (±20 %)</span> <span class="metric-val ${r2cls}">${m.test.acc.toFixed(1)}%</span></div>
        <div class="metric-row"><span class="metric-name">MAE</span>              <span class="metric-val">${Number(m.test.mae).toLocaleString(undefined, {maximumFractionDigits: 2})}</span></div>
        <div class="metric-row"><span class="metric-name">RMSE</span>             <span class="metric-val">${Number(m.test.rmse).toLocaleString(undefined, {maximumFractionDigits: 2})}</span></div>
        <div class="metric-row"><span class="metric-name">Train R²</span>         <span class="metric-val">${m.train.r2.toFixed(3)}</span></div>
      </div>`;
  });

  // Destroy old charts
  Object.values(trainCharts).forEach((c) => { try { c.destroy(); } catch (e) {} });
  trainCharts = {};

  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#8b92a8", font: { size: 11 } } } },
  };
  const gridColor = "#252a35";

  // Comparison bar chart
  trainCharts.comp = new Chart(document.getElementById("comparisonChart"), {
    type: "bar",
    data: {
      labels:   keys.map((k) => results[k].name),
      datasets: [
        {
          label:           "R² Score",
          data:            keys.map((k) => results[k].test.r2),
          backgroundColor: keys.map((k) => results[k].color + "cc"),
          borderColor:     keys.map((k) => results[k].color),
          borderWidth: 2,
          borderRadius: 6,
        },
        {
          label:           "Accuracy %",
          data:            keys.map((k) => results[k].test.acc / 100),
          backgroundColor: keys.map((k) => results[k].color + "44"),
          borderColor:     keys.map((k) => results[k].color),
          borderWidth: 2,
          borderRadius: 6,
        },
      ],
    },
    options: {
      ...chartDefaults,
      scales: {
        y: { beginAtZero: true, max: 1.05, ticks: { color: "#8b92a8" }, grid: { color: gridColor } },
        x: { ticks: { color: "#8b92a8" }, grid: { display: false } },
      },
    },
  });

  // Predictions vs actual
  const pts    = Math.min(20, results[keys[0]].predictions.length);
  const labels = Array.from({ length: pts }, (_, i) => `#${i + 1}`);
  trainCharts.pred = new Chart(document.getElementById("predictionsChart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label:       "Actual",
          data:        results[keys[0]].actuals.slice(0, pts),
          borderColor: "#e8eaf0",
          borderWidth: 2,
          pointRadius: 4,
          tension:     0.3,
          fill:        false,
        },
        ...keys.map((k) => ({
          label:       results[k].name,
          data:        results[k].predictions.slice(0, pts),
          borderColor: results[k].color,
          borderWidth: 2,
          pointRadius: 3,
          tension:     0.3,
          fill:        false,
        })),
      ],
    },
    options: {
      ...chartDefaults,
      scales: {
        y: { ticks: { color: "#8b92a8" }, grid: { color: gridColor } },
        x: { ticks: { color: "#8b92a8" }, grid: { display: false } },
      },
    },
  });

  // MAE / RMSE
  trainCharts.met = new Chart(document.getElementById("metricsChart"), {
    type: "bar",
    data: {
      labels:   keys.map((k) => results[k].name),
      datasets: [
        { label: "MAE",  data: keys.map((k) => results[k].test.mae),  backgroundColor: "#4f8ef7cc", borderRadius: 6 },
        { label: "RMSE", data: keys.map((k) => results[k].test.rmse), backgroundColor: "#7c5cfc99", borderRadius: 6 },
      ],
    },
    options: {
      ...chartDefaults,
      scales: {
        y: { beginAtZero: true, ticks: { color: "#8b92a8" }, grid: { color: gridColor } },
        x: { ticks: { color: "#8b92a8" }, grid: { display: false } },
      },
    },
  });

  // Feature importance
  const impKey   = results.dt ? "dt" : keys[0];
  const impModel = results[impKey];
  const featNames = impModel.feature_names || state.featureCols;
  const impData   = impModel.feature_importance || featNames.map(() => Math.random());

  const sorted = impData
    .map((v, i) => ({ v, l: featNames[i] || `F${i}` }))
    .sort((a, b) => b.v - a.v);

  trainCharts.feat = new Chart(document.getElementById("featureChart"), {
    type: "bar",
    data: {
      labels:   sorted.map((s) => s.l),
      datasets: [{ label: "Importance", data: sorted.map((s) => Math.round(s.v * 1000) / 1000), backgroundColor: "#00d4aacc", borderRadius: 4 }],
    },
    options: {
      indexAxis: "y",
      ...chartDefaults,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8b92a8" }, grid: { color: gridColor } },
        y: { ticks: { color: "#8b92a8" }, grid: { display: false } },
      },
    },
  });

  // Train vs test R²
  trainCharts.split = new Chart(document.getElementById("splitChart"), {
    type: "bar",
    data: {
      labels:   keys.map((k) => results[k].name),
      datasets: [
        { label: "Train R²", data: keys.map((k) => results[k].train.r2), backgroundColor: "#4f8ef766", borderRadius: 6 },
        { label: "Test R²",  data: keys.map((k) => results[k].test.r2),  backgroundColor: "#4f8ef7cc", borderRadius: 6 },
      ],
    },
    options: {
      ...chartDefaults,
      scales: {
        y: { beginAtZero: true, max: 1.05, ticks: { color: "#8b92a8" }, grid: { color: gridColor } },
        x: { ticks: { color: "#8b92a8" }, grid: { display: false } },
      },
    },
  });

  // FIX 4 (was in the removed duplicate): reveal the Predict nav item
  document.getElementById("predictNav").style.display = "flex";
}

function switchChart(id, btn) {
  ["comparison", "predictions", "metrics"].forEach((c) => {
    document.getElementById("chart-" + c).style.display = c === id ? "block" : "none";
  });
  document.querySelectorAll(".toggle-btn").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
}

// ── CSV export ─────────────────────────────────────────────
function downloadCSV() {
  if (!state.results) { alert("Train models first!"); return; }
  const keys = Object.keys(state.results).filter((k) => !state.results[k].error);
  const rows = [[
    "#", "Actual",
    ...keys.map((k) => state.results[k].name + " Pred"),
    ...keys.map((k) => state.results[k].name + " Error"),
  ]];
  const n = state.results[keys[0]].predictions.length;
  for (let i = 0; i < n; i++) {
    rows.push([
      i + 1,
      state.results[keys[0]].actuals[i],
      ...keys.map((k) => Math.round(state.results[k].predictions[i])),
      ...keys.map((k) =>
        Math.round(Math.abs(state.results[k].actuals[i] - state.results[k].predictions[i])),
      ),
    ]);
  }
  rows.push([]);
  rows.push(["Model", "MAE", "RMSE", "R2", "Accuracy%"]);
  keys.forEach((k) => {
    const m = state.results[k];
    rows.push([m.name, m.test.mae, m.test.rmse, m.test.r2, m.test.acc]);
  });
  const csv = rows.map((r) => r.join(",")).join("\n");
  const a   = document.createElement("a");
  a.href    = "data:text/csv;charset=utf-8," + encodeURIComponent(csv);
  a.download = "salescast_predictions.csv";
  a.click();
}

// ── Auth helpers ────────────────────────────────────────────
window.addEventListener("load", async () => {
  try {
    const res  = await fetch("/auth/me");
    const data = await res.json();
    if (data.authenticated) {
      applyUser(data.user);
      enterApp();
    }
  } catch (e) { /* backend not running — stay on login page */ }
});

function applyUser(user) {
  const avatar  = user.avatar;
  const initial = user.initial || (user.name ? user.name[0].toUpperCase() : "?");
  const name    = user.name || "User";
  const el      = (id) => document.getElementById(id);

  if (avatar) {
    ["topbarAvatar", "dropdownAvatar"].forEach((elId) => {
      const div = el(elId);
      div.style.backgroundImage = `url(${avatar})`;
      div.style.backgroundSize  = "cover";
      div.textContent           = "";
    });
  } else {
    el("topbarAvatar").textContent  = initial;
    el("dropdownAvatar").textContent = initial;
  }
  el("topbarName").textContent  = name;
  el("dropdownName").textContent = name;
  if (el("dropdownEmail")) el("dropdownEmail").textContent = user.email || "";
}

function enterApp() {
  const lp  = document.getElementById("loginPage");
  const app = document.getElementById("mainApp");
  lp.classList.add("hide");
  setTimeout(() => {
    lp.style.display  = "none";
    app.style.display = "flex";
  }, 500);
}

function showLoginError(msg) {
  const err = document.getElementById("loginError");
  err.style.display = "block";
  err.innerHTML     = msg;
}

function prefill() {
  document.getElementById("usernameInput").value = "demo";
  document.getElementById("passwordInput").value = "demo123";
}

async function doLogin() {
  const username = document.getElementById("usernameInput").value.trim();
  const password = document.getElementById("passwordInput").value;
  const err      = document.getElementById("loginError");
  const btn      = document.getElementById("loginBtn");
  err.style.display = "none";

  if (!username || !password) {
    showLoginError("Please enter both username and password.");
    return;
  }
  btn.disabled  = true;
  btn.innerHTML = '<span class="spinner" style="border-top-color:#fff;border-color:rgba(255,255,255,.3)"></span> Signing in…';

  try {
    const res  = await fetch("/auth/demo", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ username, password }),
    });
    const data = await res.json();

    if (data.ok) {
      applyUser(data.user);
      enterApp();
    } else {
      showLoginError("Invalid credentials. Try <b>demo</b> / <b>demo123</b>");
      btn.disabled    = false;
      btn.textContent = "Sign In";
      document.getElementById("passwordInput").value = "";
    }
  } catch (e) {
    if (username === "demo" && password === "demo123") {
      applyUser({ name: "Demo User", email: "demo@salescast.ai", initial: "D", avatar: null });
      enterApp();
    } else {
      showLoginError("Cannot reach server. For demo mode use <b>demo</b> / <b>demo123</b>");
      btn.disabled    = false;
      btn.textContent = "Sign In";
    }
  }
}

function socialLogin(provider) {
  const btn      = event.currentTarget;
  const original = btn.innerHTML;
  btn.disabled   = true;
  btn.innerHTML  = `<span class="spinner" style="width:14px;height:14px;border-width:2px;border-color:rgba(0,0,0,.15);border-top-color:#4f8ef7"></span><span>Redirecting…</span>`;
  setTimeout(() => { window.location.href = `/auth/${provider.toLowerCase()}`; }, 400);
}

async function logOut() {
  try { await fetch("/auth/logout", { method: "POST" }); } catch (e) {}
  const app = document.getElementById("mainApp");
  const lp  = document.getElementById("loginPage");
  app.style.display = "none";
  lp.classList.remove("hide");
  lp.style.display = "flex";
  document.getElementById("usernameInput").value = "";
  document.getElementById("passwordInput").value = "";
  document.getElementById("loginError").style.display = "none";
  document.getElementById("loginBtn").textContent     = "Sign In";
  ["topbarAvatar", "dropdownAvatar"].forEach((id) => {
    const el = document.getElementById(id);
    el.style.backgroundImage = "";
    el.textContent           = "?";
  });
}

function togglePwd(btn) {
  const inp = document.getElementById("passwordInput");
  if (inp.type === "password") { inp.type = "text";     btn.textContent = "🙈"; }
  else                          { inp.type = "password"; btn.textContent = "👁"; }
}

// ── Neural-network canvas animation ────────────────────────
(function () {
  const canvas = document.getElementById("neuralCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let W, H, nodes, animId;

  function resize() { W = canvas.width = canvas.offsetWidth; H = canvas.height = canvas.offsetHeight; }
  function initNodes() {
    nodes = [];
    const count = Math.floor((W * H) / 9000) + 30;
    for (let i = 0; i < count; i++) {
      nodes.push({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - 0.5) * 0.4, vy: (Math.random() - 0.5) * 0.4, r: Math.random() * 2 + 1, pulse: Math.random() * Math.PI * 2 });
    }
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const maxDist = 130;
    nodes.forEach((n) => {
      n.x += n.vx; n.y += n.vy; n.pulse += 0.02;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    });
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < maxDist) {
          ctx.beginPath(); ctx.moveTo(nodes[i].x, nodes[i].y); ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(79,142,247,${(1 - dist / maxDist) * 0.35})`; ctx.lineWidth = 0.8; ctx.stroke();
        }
      }
    }
    nodes.forEach((n) => {
      const glow = 0.6 + 0.4 * Math.sin(n.pulse);
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r * glow, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(79,142,247,${0.5 + 0.5 * glow})`; ctx.fill();
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r * glow * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(124,92,252,${0.06 * glow})`; ctx.fill();
    });
    animId = requestAnimationFrame(draw);
  }
  function start() { resize(); initNodes(); draw(); }
  window.addEventListener("resize", () => { cancelAnimationFrame(animId); resize(); initNodes(); draw(); });
  if (document.readyState === "complete") start();
  else window.addEventListener("load", start);
})();

// ══════════════════════════════════════════════════════════
//  PREDICTION FUNCTIONALITY
// ══════════════════════════════════════════════════════════

// FIX 5: _initPredictPage() is the single internal helper called by showPage()
//         when id === 'predict'.  The old showPredictionPage() is kept as a
//         public alias so any existing onclick="showPredictionPage()" still works.
function _initPredictPage() {
  if (!state.results || Object.keys(state.results).length === 0) {
    document.getElementById("noPredictionReady").style.display  = "block";
    document.getElementById("predictionContent").style.display  = "none";
    return;
  }
  document.getElementById("noPredictionReady").style.display  = "none";
  document.getElementById("predictionContent").style.display  = "block";
  document.getElementById("predictionResults").style.display  = "none";
  buildPredictionForm();
}

// Public alias — referenced from the Results page button
function showPredictionPage() {
  showPage("predict", document.querySelector("#predictNav"));
}

function buildPredictionForm() {
  const form = document.getElementById("predictionForm");
  form.innerHTML = "";

  if (!state.featureCols || state.featureCols.length === 0) {
    form.innerHTML = `<div style="padding:20px;text-align:center;color:var(--text3)">No features available</div>`;
    return;
  }

  state.featureCols.forEach((feature) => {
    const group  = document.createElement("div");
    group.className = "prediction-input-group";

    const values = state.rawData.map((row) => parseFloat(row[feature])).filter((v) => !isNaN(v));
    const min    = Math.min(...values);
    const max    = Math.max(...values);
    const avg    = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2);

    group.innerHTML = `
      <label class="prediction-input-label">${feature}</label>
      <input type="number" class="prediction-input" id="pred_${feature}" step="any" placeholder="Enter value"/>
      <div class="prediction-input-value">Min: ${min.toFixed(2)} | Avg: ${avg} | Max: ${max.toFixed(2)}</div>
    `;
    form.appendChild(group);
  });
}

function getPredictionValues() {
  const values = {};
  let isValid  = true;

  state.featureCols.forEach((feature) => {
    const input = document.getElementById(`pred_${feature}`);
    const value = parseFloat(input.value);
    if (isNaN(value)) {
      input.style.borderColor = "var(--danger)";
      isValid = false;
    } else {
      input.style.borderColor = "var(--border2)";
      values[feature] = value;
    }
  });

  return { values, isValid };
}

function resetPredictionForm() {
  state.featureCols.forEach((feature) => {
    const el = document.getElementById(`pred_${feature}`);
    el.value = "";
    el.style.borderColor = "var(--border2)";
  });
  document.getElementById("predictionResults").style.display = "none";
}

async function makePrediction() {
  const { values, isValid } = getPredictionValues();
  if (!isValid) { alert("❌ Please fill in all feature values with valid numbers"); return; }

  const btn          = document.getElementById("predictBtn");
  const originalHTML = btn.innerHTML;
  btn.disabled  = true;
  btn.innerHTML = '<span class="spinner"></span> Making predictions…';

  try {
    const payload = {
      feature_values: state.featureCols.map((f) => values[f]),
      feature_names:  state.featureCols,
      models:         Object.keys(state.results).filter((k) => !state.results[k].error),
    };

    const response = await fetch("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.message || `Server error ${response.status}`);
    }

    const data = await response.json();
    if (data.status !== "success") throw new Error(data.message || "Prediction failed on server.");

    displayPredictionResults(data.predictions);
    btn.disabled  = false;
    btn.innerHTML = originalHTML;
  } catch (err) {
    console.error("Prediction error:", err);
    alert(`❌ Prediction failed: ${err.message}`);
    btn.disabled  = false;
    btn.innerHTML = originalHTML;
  }
}

function displayPredictionResults(predictions) {
  const resultsContainer = document.getElementById("predictionResults");
  const resultsContent   = document.getElementById("predictionResultsContent");
  resultsContent.innerHTML = "";

  // Normalise: server may return object {lr: val, ...} or array
  let predictionDict = {};
  if (Array.isArray(predictions)) {
    Object.keys(state.results).forEach((key, index) => {
      if (!state.results[key].error && predictions[index] !== undefined) {
        predictionDict[key] = predictions[index];
      }
    });
  } else {
    predictionDict = predictions;
  }

  const resultsGrid = document.createElement("div");
  resultsGrid.className = "prediction-results-grid";

  Object.entries(state.results).forEach(([key, model]) => {
    if (model.error) return;

    const prediction  = predictionDict[key] || 0;
    const modelColor  = model.test.r2 > 0.8
      ? "var(--success)"
      : model.test.r2 > 0.5
        ? "var(--warning)"
        : "var(--danger)";

    const card = document.createElement("div");
    card.className = "prediction-card";
    card.innerHTML = `
      <div class="prediction-model-name" style="color:${modelColor}">
        <span class="prediction-model-icon">${model.icon}</span>
        ${model.name}
      </div>
      <div class="prediction-value" style="color:${modelColor}">
        ${Number(prediction).toLocaleString(undefined, { maximumFractionDigits: 2 })}
      </div>
      <div class="prediction-confidence">Model Confidence: R² = ${model.test.r2.toFixed(3)}</div>
      <div class="confidence-bar">
        <div class="confidence-fill" style="width:${model.test.r2 * 100}%;background:${modelColor}"></div>
      </div>`;
    resultsGrid.appendChild(card);
  });

  resultsContent.appendChild(resultsGrid);
  resultsContainer.style.display = "block";
  resultsContainer.scrollIntoView({ behavior: "smooth" });
}
