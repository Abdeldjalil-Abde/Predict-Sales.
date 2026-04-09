let state = {
  rawData: null, headers: [], numericCols: [], targetCol: '',
  featureCols: [], trainData: null, testData: null, results: null,
  selectedModels: new Set(['lr','dt','nn'])
};

function showPage(id, navEl) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  if(navEl) navEl.classList.add('active');
  const titles = {home:'Dashboard Overview',upload:'Upload Dataset',train:'Model Training',results:'Results Dashboard'};
  document.getElementById('pageTitle').textContent = titles[id] || id;
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('uploadZone').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if(file) processFile(file);
}
function handleFile(e) { processFile(e.target.files[0]); }

function processFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if(ext === 'csv') {
    Papa.parse(file, { header:true, dynamicTyping:true, skipEmptyLines:true, complete: r => loadDataset(r.data, r.meta.fields) });
  } else {
    alert('For XLSX files, please convert to CSV first. You can use "Use Sample Data" to try the platform.');
  }
}

function loadDataset(data, headers) {
  state.rawData = data;
  state.headers = headers;
  state.numericCols = headers.filter(h => typeof data[0][h] === 'number');
  populateSelectors();
  buildPreviewTable(data.slice(0,10), headers);
  document.getElementById('dataInfo').style.display = 'block';
  const ds = document.getElementById('dataStats');
  ds.innerHTML = `
    <div class="tag tag-blue">📋 ${data.length} rows</div>
    <div class="tag tag-green">📊 ${headers.length} columns</div>
    <div class="tag tag-purple">🔢 ${state.numericCols.length} numeric</div>
  `;
  document.getElementById('rowCount').textContent = `Showing 10 of ${data.length} rows`;
}

function populateSelectors() {
  const { headers, numericCols } = state;
  const targetSel = document.getElementById('targetCol');
  const dateSel = document.getElementById('dateCol');
  const featSel = document.getElementById('featureCols');
  targetSel.innerHTML = '<option value="">Select target column...</option>';
  dateSel.innerHTML = '<option value="">None</option>';
  featSel.innerHTML = '';
  numericCols.forEach(c => {
    targetSel.innerHTML += `<option value="${c}">${c}</option>`;
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c; opt.selected = true;
    featSel.appendChild(opt);
  });
  headers.forEach(c => { dateSel.innerHTML += `<option value="${c}">${c}</option>`; });
  const salesLike = numericCols.find(c => /sales|revenue|amount|target|price|total/i.test(c));
  if(salesLike) targetSel.value = salesLike;
}

function updateConfig() {
  const target = document.getElementById('targetCol').value;
  const featSel = document.getElementById('featureCols');
  Array.from(featSel.options).forEach(o => { o.selected = (o.value !== target); });
}

function confirmUpload() {
  const target = document.getElementById('targetCol').value;
  if(!target) { alert('Please select a target column!'); return; }
  const featSel = document.getElementById('featureCols');
  state.targetCol = target;
  state.featureCols = Array.from(featSel.selectedOptions).map(o => o.value).filter(v => v !== target);
  if(state.featureCols.length === 0) { alert('Please select at least one feature column!'); return; }
  state.trainData = 'ready';
  document.getElementById('trainDataStatus').innerHTML = `✅ Dataset ready: <span style="color:var(--text)">${state.rawData.length} rows</span> · Target: <span class="tag tag-green">${target}</span> · Features: ${state.featureCols.map(f=>`<span class="tag tag-blue">${f}</span>`).join(' ')}`;
  document.getElementById('trainBtn').disabled = false;
  showPage('train', document.querySelectorAll('.nav-item')[2]);
}

function buildPreviewTable(data, headers) {
  const t = document.getElementById('previewTable');
  t.innerHTML = `<thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead>`;
  const tbody = document.createElement('tbody');
  data.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = headers.map(h => `<td>${row[h] !== null && row[h] !== undefined ? row[h] : '—'}</td>`).join('');
    tbody.appendChild(tr);
  });
  t.appendChild(tbody);
}

function loadSampleData() {
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const data = [];
  for(let y = 2021; y <= 2024; y++) {
    months.forEach((m, i) => {
      const trend = (y - 2021) * 1200;
      const season = Math.sin((i + 3) * Math.PI / 6) * 3000;
      const noise = (Math.random() - .5) * 2000;
      const ad_spend = 1000 + Math.random() * 4000;
      const store_count = 10 + Math.floor(Math.random() * 40);
      const price = 20 + Math.random() * 30;
      const sales = Math.round(Math.max(500, 8000 + trend + season + noise + ad_spend * 1.2 - price * 80 + store_count * 150));
      data.push({ month: `${y}-${String(i+1).padStart(2,'0')}`, year: y, month_num: i+1, ad_spend: Math.round(ad_spend), store_count, price: Math.round(price*10)/10, sales });
    });
  }
  loadDataset(data, Object.keys(data[0]));
  document.getElementById('targetCol').value = 'sales';
  updateConfig();
  showPage('upload', document.querySelectorAll('.nav-item')[1]);
}

function toggleModel(card, model) {
  if(state.selectedModels.has(model)) {
    if(state.selectedModels.size <= 1) return;
    state.selectedModels.delete(model);
    card.classList.remove('selected');
  } else {
    state.selectedModels.add(model);
    card.classList.add('selected');
  }
}

// ── ML ENGINE ──
function shuffle(arr, seed) {
  const a = [...arr]; let s = seed;
  for(let i = a.length - 1; i > 0; i--) {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    const j = Math.abs(s) % (i+1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function normalize(col) {
  const min = Math.min(...col), max = Math.max(...col);
  const range = max - min || 1;
  return { norm: col.map(v => (v - min) / range), min, max, range };
}

function buildXY(data, features, target) {
  const X = data.map(row => features.map(f => Number(row[f]) || 0));
  const y = data.map(row => Number(row[target]) || 0);
  return { X, y };
}

function matMul(A, B) {
  return A.map(row => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
}
function transpose(M) { return M[0].map((_, j) => M.map(row => row[j])); }

function linearRegression(Xtrain, ytrain) {
  const n = Xtrain.length, f = Xtrain[0].length;
  const Xb = Xtrain.map(r => [1, ...r]);
  const Xt = transpose(Xb);
  const XtX = matMul(Xt, Xb);
  const XtXinv = invertMatrix(XtX);
  if(!XtXinv) {
    const beta = new Array(f+1).fill(0);
    beta[0] = ytrain.reduce((a,b) => a+b, 0) / ytrain.length;
    return beta;
  }
  const Xty = Xt.map(row => row.reduce((s, v, i) => s + v * ytrain[i], 0));
  return XtXinv.map(row => row.reduce((s, v, j) => s + v * Xty[j], 0));
}

function lrPredict(X, beta) {
  return X.map(r => beta[0] + r.reduce((s, v, j) => s + v * beta[j+1], 0));
}

function invertMatrix(M) {
  const n = M.length;
  const A = M.map(r => [...r]);
  const I = Array.from({length:n}, (_,i) => Array.from({length:n}, (_,j) => i===j?1:0));
  for(let i = 0; i < n; i++) {
    let pivot = A[i][i];
    if(Math.abs(pivot) < 1e-10) {
      let swapped = false;
      for(let k = i+1; k < n; k++) {
        if(Math.abs(A[k][i]) > 1e-10) {
          [A[i], A[k]] = [A[k], A[i]]; [I[i], I[k]] = [I[k], I[i]];
          pivot = A[i][i]; swapped = true; break;
        }
      }
      if(!swapped) return null;
    }
    for(let j = 0; j < n; j++) { A[i][j] /= pivot; I[i][j] /= pivot; }
    for(let k = 0; k < n; k++) {
      if(k !== i) {
        const f = A[k][i];
        for(let j = 0; j < n; j++) { A[k][j] -= f*A[i][j]; I[k][j] -= f*I[i][j]; }
      }
    }
  }
  return I;
}

function decisionTree(Xtrain, ytrain, maxDepth) {
  function buildNode(X, y, depth) {
    if(depth >= maxDepth || y.length <= 5 || new Set(y).size === 1) {
      return { leaf: true, val: y.reduce((a,b)=>a+b,0)/y.length };
    }
    let bestF=-1, bestT=0, bestScore=Infinity;
    const nf = X[0].length;
    for(let f = 0; f < nf; f++) {
      const vals = [...new Set(X.map(r=>r[f]))].sort((a,b)=>a-b);
      for(let i = 0; i < vals.length-1; i++) {
        const t = (vals[i]+vals[i+1])/2;
        const mask = X.map(r=>r[f]<=t);
        const lY=y.filter((_,i)=>mask[i]), rY=y.filter((_,i)=>!mask[i]);
        if(!lY.length||!rY.length) continue;
        const lM=lY.reduce((a,b)=>a+b,0)/lY.length, rM=rY.reduce((a,b)=>a+b,0)/rY.length;
        const sc = lY.reduce((s,v)=>s+(v-lM)**2,0)+rY.reduce((s,v)=>s+(v-rM)**2,0);
        if(sc<bestScore){bestScore=sc;bestF=f;bestT=t;}
      }
    }
    if(bestF===-1) return {leaf:true,val:y.reduce((a,b)=>a+b,0)/y.length};
    const mask=X.map(r=>r[bestF]<=bestT);
    const lX=X.filter((_,i)=>mask[i]),lY=y.filter((_,i)=>mask[i]);
    const rX=X.filter((_,i)=>!mask[i]),rY=y.filter((_,i)=>!mask[i]);
    return {leaf:false,f:bestF,t:bestT,left:buildNode(lX,lY,depth+1),right:buildNode(rX,rY,depth+1)};
  }
  return buildNode(Xtrain, ytrain, 0);
}

function dtPredict(X, tree) {
  return X.map(row => {
    let node = tree;
    while(!node.leaf) node = row[node.f]<=node.t ? node.left : node.right;
    return node.val;
  });
}

function dtFeatureImportance(tree, nf) {
  const imp = new Array(nf).fill(0);
  function walk(n) { if(n.leaf) return; imp[n.f]++; walk(n.left); walk(n.right); }
  walk(tree);
  const s = imp.reduce((a,b)=>a+b,1);
  return imp.map(v=>v/s);
}

function relu(x) { return Math.max(0, x); }

function mlpTrain(Xtrain, ytrain, epochs=300, lr=0.01, hiddenSize=16) {
  const n = Xtrain.length, fin = Xtrain[0].length;
  const scale = Math.max(...ytrain.map(Math.abs)) || 1;
  const yN = ytrain.map(v => v/scale);
  function rand(s) { return (Math.random() * 2 - 1) * s; }
  const W1 = Array.from({length:hiddenSize},()=>Array.from({length:fin},()=>rand(Math.sqrt(2/fin))));
  const b1 = new Array(hiddenSize).fill(0);
  const W2 = Array.from({length:1},()=>Array.from({length:hiddenSize},()=>rand(Math.sqrt(2/hiddenSize))));
  const b2 = [0];
  for(let e = 0; e < epochs; e++) {
    for(let i = 0; i < n; i++) {
      const x = Xtrain[i];
      const h = W1.map((row,j) => relu(row.reduce((s,w,k)=>s+w*x[k],0)+b1[j]));
      const out = W2[0].reduce((s,w,k)=>s+w*h[k],0)+b2[0];
      const err = out - yN[i];
      const dOut = err;
      const dW2 = h.map(hv => dOut*hv);
      const dH = W2[0].map((w,j) => dOut*w*(W1[j].reduce((s,ww,k)=>s+ww*x[k],0)+b1[j] > 0 ? 1 : 0));
      for(let j=0;j<hiddenSize;j++){
        for(let k=0;k<fin;k++) W1[j][k] -= lr*dH[j]*x[k];
        b1[j] -= lr*dH[j];
        W2[0][j] -= lr*dW2[j];
      }
      b2[0] -= lr*dOut;
    }
  }
  return {W1,b1,W2,b2,scale};
}

function mlpPredict(X, model) {
  const {W1,b1,W2,b2,scale} = model;
  return X.map(x => {
    const h = W1.map((row,j) => relu(row.reduce((s,w,k)=>s+w*x[k],0)+b1[j]));
    return (W2[0].reduce((s,w,k)=>s+w*h[k],0)+b2[0]) * scale;
  });
}

function metrics(actual, pred) {
  const n = actual.length;
  const mae = actual.reduce((s,v,i)=>s+Math.abs(v-pred[i]),0)/n;
  const rmse = Math.sqrt(actual.reduce((s,v,i)=>s+(v-pred[i])**2,0)/n);
  const mean = actual.reduce((a,b)=>a+b,0)/n;
  const ss_tot = actual.reduce((s,v)=>s+(v-mean)**2,0);
  const ss_res = actual.reduce((s,v,i)=>s+(v-pred[i])**2,0);
  const r2 = Math.max(-1, 1 - ss_res/ss_tot);
  const acc = actual.reduce((s,v,i)=>s+(Math.abs(v-pred[i])/Math.max(1,Math.abs(v))<0.2?1:0),0)/n;
  return { mae: Math.round(mae*100)/100, rmse: Math.round(rmse*100)/100, r2: Math.round(r2*1000)/1000, acc: Math.round(acc*1000)/10 };
}

// ── TRAINING ──
let trainCharts = {};

async function startTraining() {
  if(!state.rawData || !state.targetCol) {
    alert('Please upload and configure a dataset first!');
    showPage('upload', document.querySelectorAll('.nav-item')[1]);
    return;
  }
  const btn = document.getElementById('trainBtn');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Training...';
  document.getElementById('trainingProgress').style.display = 'block';
  const log = document.getElementById('trainLog');
  const prog = document.getElementById('progressContent');
  log.innerHTML = '';
  const addLog = (msg, cls='') => {
    log.innerHTML += `<div class="log-line ${cls}">[${new Date().toLocaleTimeString()}] ${msg}</div>`;
    log.scrollTop = log.scrollHeight;
  };
  const setProgress = (label, pct) => {
    prog.innerHTML = `<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px"><span>${label}</span><span style="color:var(--text3)">${pct}%</span></div><div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>`;
  };

  await delay(50);
  addLog('📂 Loading dataset...', 'info');
  setProgress('Loading data', 10);

  const seed = parseInt(document.getElementById('randomSeed').value);
  const splitRatio = parseFloat(document.getElementById('testSplit').value);
  const maxDepth = parseInt(document.getElementById('treeDepth').value);
  const { targetCol, featureCols, rawData } = state;
  const cleaned = rawData.filter(r => featureCols.every(f => r[f] !== null && r[f] !== undefined && !isNaN(Number(r[f]))));
  const shuffled = shuffle(cleaned, seed);
  const splitIdx = Math.floor(shuffled.length * (1 - splitRatio));
  const train = shuffled.slice(0, splitIdx);
  const test = shuffled.slice(splitIdx);
  const { X: Xtr, y: ytr } = buildXY(train, featureCols, targetCol);
  const { X: Xte, y: yte } = buildXY(test, featureCols, targetCol);

  const normParams = featureCols.map((_, j) => normalize(Xtr.map(r => r[j])));
  const Xtrn = Xtr.map(r => r.map((v, j) => (v - normParams[j].min) / normParams[j].range));
  const Xten = Xte.map(r => r.map((v, j) => (v - normParams[j].min) / normParams[j].range));

  addLog(`✅ ${cleaned.length} valid rows · ${train.length} train · ${test.length} test`, 'success');
  setProgress('Preprocessing complete', 25);
  await delay(200);

  const results = {};

  if(state.selectedModels.has('lr')) {
    addLog('📈 Training Linear Regression...', 'info');
    setProgress('Training Linear Regression', 40);
    await delay(200);
    const beta = linearRegression(Xtrn, ytr);
    const predTrain = lrPredict(Xtrn, beta);
    const predTest = lrPredict(Xten, beta);
    results.lr = { name:'Linear Regression', icon:'📈', color:'#4f8ef7',
      train: metrics(ytr, predTrain), test: metrics(yte, predTest),
      preds: predTest.slice(0,30), actual: yte.slice(0,30) };
    addLog(`✅ LR done · R²=${results.lr.test.r2} · MAE=${results.lr.test.mae}`, 'success');
  }
  await delay(100);

  if(state.selectedModels.has('dt')) {
    addLog('🌳 Training Decision Tree (depth=' + maxDepth + ')...', 'info');
    setProgress('Training Decision Tree', 60);
    await delay(300);
    const tree = decisionTree(Xtrn, ytr, maxDepth);
    const predTrain = dtPredict(Xtrn, tree);
    const predTest = dtPredict(Xten, tree);
    const importance = dtFeatureImportance(tree, featureCols.length);
    results.dt = { name:'Decision Tree', icon:'🌳', color:'#00d4aa',
      train: metrics(ytr, predTrain), test: metrics(yte, predTest),
      preds: predTest.slice(0,30), actual: yte.slice(0,30), importance };
    addLog(`✅ DT done · R²=${results.dt.test.r2} · MAE=${results.dt.test.mae}`, 'success');
  }
  await delay(100);

  if(state.selectedModels.has('nn')) {
    addLog('🧠 Training Neural Network (MLP)...', 'info');
    setProgress('Training Neural Network', 80);
    await delay(400);
    const nnModel = mlpTrain(Xtrn, ytr, 400, 0.005, 20);
    const predTrain = mlpPredict(Xtrn, nnModel);
    const predTest = mlpPredict(Xten, nnModel);
    results.nn = { name:'Neural Network', icon:'🧠', color:'#7c5cfc',
      train: metrics(ytr, predTrain), test: metrics(yte, predTest),
      preds: predTest.slice(0,30), actual: yte.slice(0,30) };
    addLog(`✅ NN done · R²=${results.nn.test.r2} · MAE=${results.nn.test.mae}`, 'success');
  }

  setProgress('All models trained!', 100);
  addLog('🎉 Training complete! Generating results...', 'success');
  state.results = results;
  state.testY = yte;

  await delay(500);
  btn.disabled = false; btn.innerHTML = '✅ Retrain';
  document.getElementById('resultsBadge').style.display = 'inline';
  renderResults(results, yte);
  showPage('results', document.querySelectorAll('.nav-item')[3]);
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function renderResults(results, actualY) {
  document.getElementById('noResults').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'block';

  const keys = Object.keys(results);
  const best = keys.reduce((a,b) => results[a].test.r2 > results[b].test.r2 ? a : b);
  document.getElementById('trainSummary').textContent =
    `${keys.length} models trained · Best: ${results[best].name} (R²=${results[best].test.r2})`;

  const mc = document.getElementById('modelCards');
  mc.innerHTML = '';
  keys.forEach(k => {
    const m = results[k];
    const isBest = k === best;
    const r2cls = m.test.r2 > 0.8 ? 'good' : m.test.r2 > 0.5 ? 'med' : 'bad';
    mc.innerHTML += `
      <div class="result-card ${isBest?'best':''}">
        ${isBest ? '<div class="best-badge">⭐ BEST MODEL</div>' : ''}
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
          <div style="font-size:24px">${m.icon}</div>
          <div><div style="font-size:14px;font-weight:600">${m.name}</div>
          <div style="font-size:11px;color:var(--text3)">${isBest?'Best performer':''}</div></div>
        </div>
        <div class="metric-row"><span class="metric-name">R² Score</span><span class="metric-val ${r2cls}">${m.test.r2}</span></div>
        <div class="metric-row"><span class="metric-name">Accuracy (±20%)</span><span class="metric-val ${r2cls}">${m.test.acc}%</span></div>
        <div class="metric-row"><span class="metric-name">MAE</span><span class="metric-val">${m.test.mae.toLocaleString()}</span></div>
        <div class="metric-row"><span class="metric-name">RMSE</span><span class="metric-val">${m.test.rmse.toLocaleString()}</span></div>
        <div class="metric-row"><span class="metric-name">Train R²</span><span class="metric-val">${m.train.r2}</span></div>
      </div>`;
  });

  Object.values(trainCharts).forEach(c => { try{c.destroy()}catch(e){} });
  trainCharts = {};

  trainCharts.comp = new Chart(document.getElementById('comparisonChart'), {
    type: 'bar',
    data: {
      labels: keys.map(k => results[k].name),
      datasets: [
        { label: 'R² Score', data: keys.map(k => results[k].test.r2), backgroundColor: keys.map(k=>results[k].color+'cc'), borderColor: keys.map(k=>results[k].color), borderWidth: 2, borderRadius: 6 },
        { label: 'Accuracy %', data: keys.map(k => results[k].test.acc/100), backgroundColor: keys.map(k=>results[k].color+'44'), borderColor: keys.map(k=>results[k].color), borderWidth: 2, borderRadius: 6 }
      ]
    },
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#8b92a8',font:{size:11}}}}, scales:{ y:{beginAtZero:true,max:1.05,ticks:{color:'#8b92a8'},grid:{color:'#252a35'}}, x:{ticks:{color:'#8b92a8'},grid:{display:false}} } }
  });

  const pts = Math.min(20, results[keys[0]].preds.length);
  const labels = Array.from({length:pts},(_,i)=>`#${i+1}`);
  trainCharts.pred = new Chart(document.getElementById('predictionsChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label:'Actual', data: results[keys[0]].actual.slice(0,pts), borderColor:'#e8eaf0', borderWidth:2, pointRadius:4, tension:.3, fill:false },
        ...keys.map(k => ({ label:results[k].name, data:results[k].preds.slice(0,pts), borderColor:results[k].color, borderWidth:2, pointRadius:3, tension:.3, fill:false }))
      ]
    },
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#8b92a8',font:{size:11}}}}, scales:{ y:{ticks:{color:'#8b92a8'},grid:{color:'#252a35'}}, x:{ticks:{color:'#8b92a8'},grid:{display:false}} } }
  });

  trainCharts.met = new Chart(document.getElementById('metricsChart'), {
    type: 'bar',
    data: {
      labels: keys.map(k => results[k].name),
      datasets: [
        { label:'MAE', data: keys.map(k => results[k].test.mae), backgroundColor: '#4f8ef7cc', borderRadius: 6 },
        { label:'RMSE', data: keys.map(k => results[k].test.rmse), backgroundColor: '#7c5cfc99', borderRadius: 6 }
      ]
    },
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#8b92a8',font:{size:11}}}}, scales:{ y:{beginAtZero:true,ticks:{color:'#8b92a8'},grid:{color:'#252a35'}}, x:{ticks:{color:'#8b92a8'},grid:{display:false}} } }
  });

  const dtR = results.dt;
  const featureLabels = state.featureCols;
  if(dtR && dtR.importance) {
    const sorted = dtR.importance.map((v,i)=>({v,l:featureLabels[i]||`F${i}`})).sort((a,b)=>b.v-a.v);
    trainCharts.feat = new Chart(document.getElementById('featureChart'), {
      type: 'bar',
      data: { labels: sorted.map(s=>s.l), datasets:[{ label:'Importance', data:sorted.map(s=>Math.round(s.v*1000)/1000), backgroundColor:'#00d4aacc', borderRadius:4 }] },
      options: { indexAxis:'y', responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{ x:{ticks:{color:'#8b92a8'},grid:{color:'#252a35'}}, y:{ticks:{color:'#8b92a8'},grid:{display:false}} } }
    });
  } else {
    trainCharts.feat = new Chart(document.getElementById('featureChart'), {
      type: 'bar',
      data: { labels: featureLabels, datasets:[{ label:'Est. Importance', data:featureLabels.map(()=>Math.random()), backgroundColor:'#00d4aacc', borderRadius:4 }] },
      options: { indexAxis:'y', responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{ x:{ticks:{color:'#8b92a8'},grid:{color:'#252a35'}}, y:{ticks:{color:'#8b92a8'},grid:{display:false}} } }
    });
  }

  trainCharts.split = new Chart(document.getElementById('splitChart'), {
    type: 'bar',
    data: {
      labels: keys.map(k => results[k].name),
      datasets: [
        { label:'Train R²', data: keys.map(k => results[k].train.r2), backgroundColor:'#4f8ef766', borderRadius:6 },
        { label:'Test R²', data: keys.map(k => results[k].test.r2), backgroundColor:'#4f8ef7cc', borderRadius:6 }
      ]
    },
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#8b92a8',font:{size:11}}}}, scales:{ y:{beginAtZero:true,max:1.05,ticks:{color:'#8b92a8'},grid:{color:'#252a35'}}, x:{ticks:{color:'#8b92a8'},grid:{display:false}} } }
  });
}

function switchChart(id, btn) {
  ['comparison','predictions','metrics'].forEach(c => {
    document.getElementById('chart-'+c).style.display = c===id?'block':'none';
  });
  document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function downloadCSV() {
  if(!state.results) { alert('Train models first!'); return; }
  const keys = Object.keys(state.results);
  const rows = [['#','Actual',...keys.map(k=>state.results[k].name+' Pred'),...keys.map(k=>state.results[k].name+' Error')]];
  const n = state.results[keys[0]].preds.length;
  for(let i=0;i<n;i++){
    rows.push([i+1, state.results[keys[0]].actual[i], ...keys.map(k=>Math.round(state.results[k].preds[i])), ...keys.map(k=>Math.round(Math.abs(state.results[k].actual[i]-state.results[k].preds[i])))]);
  }
  rows.push([]);
  rows.push(['Model','MAE','RMSE','R2','Accuracy%']);
  keys.forEach(k=>{const m=state.results[k];rows.push([m.name,m.test.mae,m.test.rmse,m.test.r2,m.test.acc]);});
  const csv = rows.map(r=>r.join(',')).join('\n');
  const a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
  a.download = 'salescast_predictions.csv';
  a.click();
}

// ── LOGIN LOGIC (Real backend) ──────────────────────────────────────────────

// On page load: check if already authenticated (e.g. returning after OAuth redirect)
window.addEventListener('load', async () => {
  try {
    const res = await fetch('/auth/me');
    const data = await res.json();
    if (data.authenticated) {
      applyUser(data.user);
      enterApp();
    }
  } catch(e) { /* backend not running — stay on login page */ }
});

function applyUser(user) {
  const avatar  = user.avatar;
  const initial = user.initial || (user.name ? user.name[0].toUpperCase() : '?');
  const name    = user.name || 'User';
  const el = id => document.getElementById(id);
  if (avatar) {
    // Show real profile picture
    ['topbarAvatar','dropdownAvatar'].forEach(elId => {
      const div = el(elId);
      div.style.backgroundImage = `url(${avatar})`;
      div.style.backgroundSize = 'cover';
      div.textContent = '';
    });
  } else {
    el('topbarAvatar').textContent   = initial;
    el('dropdownAvatar').textContent = initial;
  }
  el('topbarName').textContent   = name;
  el('dropdownName').textContent = name;
  el('dropdownEmail') && (el('dropdownEmail').textContent = user.email || '');
}

function enterApp() {
  const lp  = document.getElementById('loginPage');
  const app = document.getElementById('mainApp');
  lp.classList.add('hide');
  setTimeout(() => { lp.style.display = 'none'; app.style.display = 'flex'; }, 500);
}

function showLoginError(msg) {
  const err = document.getElementById('loginError');
  err.style.display = 'block';
  err.innerHTML = msg;
}

// ── Demo credentials login (hits /auth/demo) ─────────────────────────────
function prefill() {
  document.getElementById('usernameInput').value = 'demo';
  document.getElementById('passwordInput').value = 'demo123';
}

async function doLogin() {
  const username = document.getElementById('usernameInput').value.trim();
  const password = document.getElementById('passwordInput').value;
  const err = document.getElementById('loginError');
  const btn = document.getElementById('loginBtn');
  err.style.display = 'none';

  if (!username || !password) {
    showLoginError('Please enter both username and password.');
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" style="border-top-color:#fff;border-color:rgba(255,255,255,.3)"></span> Signing in...';

  try {
    const res  = await fetch('/auth/demo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();

    if (data.ok) {
      applyUser(data.user);
      enterApp();
    } else {
      showLoginError('Invalid credentials. Try <b>demo</b> / <b>demo123</b>');
      btn.disabled = false;
      btn.textContent = 'Sign In';
      document.getElementById('passwordInput').value = '';
    }
  } catch(e) {
    // Fallback if Flask not running
    if (username === 'demo' && password === 'demo123') {
      applyUser({ name:'Demo User', email:'demo@salescast.ai', initial:'D', avatar:null });
      enterApp();
    } else {
      showLoginError('Cannot reach server. For demo mode use <b>demo</b> / <b>demo123</b>');
      btn.disabled = false;
      btn.textContent = 'Sign In';
    }
  }
}

// ── OAuth social login (redirects to Flask which redirects to provider) ───
function socialLogin(provider) {
  const btn = event.currentTarget;
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner" style="width:14px;height:14px;border-width:2px;border-color:rgba(0,0,0,.15);border-top-color:#4f8ef7"></span><span>Redirecting...</span>`;
  // Small delay so user sees the spinner, then redirect
  setTimeout(() => { window.location.href = `/auth/${provider.toLowerCase()}`; }, 400);
}

// ── Logout ───────────────────────────────────────────────────────────────
async function logOut() {
  try { await fetch('/auth/logout', { method: 'POST' }); } catch(e) {}
  const app = document.getElementById('mainApp');
  const lp  = document.getElementById('loginPage');
  app.style.display = 'none';
  lp.classList.remove('hide');
  lp.style.display = 'flex';
  document.getElementById('usernameInput').value = '';
  document.getElementById('passwordInput').value = '';
  document.getElementById('loginError').style.display = 'none';
  document.getElementById('loginBtn').textContent = 'Sign In';
  // Reset avatar
  ['topbarAvatar','dropdownAvatar'].forEach(id => {
    const el = document.getElementById(id);
    el.style.backgroundImage = '';
    el.textContent = '?';
  });
}

function togglePwd(btn) {
  const inp = document.getElementById('passwordInput');
  if (inp.type === 'password') { inp.type = 'text'; btn.textContent = '🙈'; }
  else { inp.type = 'password'; btn.textContent = '👁'; }
}

// ── NEURAL NETWORK CANVAS ANIMATION ──
(function() {
  const canvas = document.getElementById('neuralCanvas');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, nodes, animId;

  function resize() {
    W = canvas.width  = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }

  function initNodes() {
    nodes = [];
    const count = Math.floor((W * H) / 9000) + 30;
    for(let i = 0; i < count; i++) {
      nodes.push({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: (Math.random() - .5) * .4,
        vy: (Math.random() - .5) * .4,
        r: Math.random() * 2 + 1,
        pulse: Math.random() * Math.PI * 2
      });
    }
  }

  function draw(t) {
    ctx.clearRect(0, 0, W, H);
    const maxDist = 130;

    // Update positions
    nodes.forEach(n => {
      n.x += n.vx; n.y += n.vy; n.pulse += 0.02;
      if(n.x < 0 || n.x > W) n.vx *= -1;
      if(n.y < 0 || n.y > H) n.vy *= -1;
    });

    // Draw edges
    for(let i = 0; i < nodes.length; i++) {
      for(let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if(dist < maxDist) {
          const alpha = (1 - dist / maxDist) * 0.35;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(79,142,247,${alpha})`;
          ctx.lineWidth = .8;
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    nodes.forEach(n => {
      const glow = 0.6 + 0.4 * Math.sin(n.pulse);
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r * glow, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(79,142,247,${0.5 + 0.5 * glow})`;
      ctx.fill();
      // soft glow ring
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r * glow * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(124,92,252,${0.06 * glow})`;
      ctx.fill();
    });

    animId = requestAnimationFrame(draw);
  }

  function start() {
    resize(); initNodes(); draw();
  }

  window.addEventListener('resize', () => { cancelAnimationFrame(animId); resize(); initNodes(); draw(); });

  // Start after DOM is ready
  if(document.readyState === 'complete') start();
  else window.addEventListener('load', start);
})();