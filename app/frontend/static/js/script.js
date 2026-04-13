const API = window.location.origin;

// ── Tab switching ────────────────────────────────────────────────────────────

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.add('hidden'));
    tab.classList.add('active');

    const targetPanel = document.getElementById(`panel-${tab.dataset.tab}`);
    if (targetPanel) {
      targetPanel.classList.remove('hidden');
    }
  });
});



function hideError(el) {
  el.textContent = '';
  el.classList.add('hidden');
}

async function checkHealth() {
  const badge = document.getElementById('modelBadge');
  const dot = badge.querySelector('.badge-dot');
  const label = badge.querySelector('.badge-label');
  try {
    const res = await fetch(`${API}/health`);
    if (!res.ok) {
      throw new Error('health not ok');
    }
    dot.className = 'badge-dot ok';
    label.textContent = 'API online';
  } catch {
    dot.className = 'badge-dot err';
    label.textContent = 'API offline';
  }
}

function buildPayload() {
  return {
    text: document.getElementById('reviewText').value.trim(),
    model_version: document.getElementById('modelVersionSelect').value || 'v3',
    rating: parseFloat(document.getElementById('rating').value || '0'),
    helpful_vote: parseFloat(document.getElementById('helpfulVote').value || '0'),
    verified_purchase: document.getElementById('verifiedPurchase').checked ? 1 : 0,
  };
}

function renderResult(data) {
  const card = document.getElementById('resultCard');
  const fakeProbabilityPct = Number(data.fake_probability ?? 0);
  const confidencePct = Math.max(0, Math.min(100, Math.round(fakeProbabilityPct)));
  const label = (data.label || '').toLowerCase();
  const isFake = label === 'fake';

  const verdictText = document.getElementById('verdictText');
  verdictText.textContent = isFake ? '⚠ Fake' : '✓ Genuine';
  verdictText.className = `result-verdict ${label || 'genuine'}`;

  document.getElementById('resultProb').textContent = `fake_probability: ${fakeProbabilityPct.toFixed(2)}%`;
  document.getElementById('resultMeta').textContent = `threshold_percent: ${Number(data.threshold_percent ?? 0).toFixed(2)}`;
  document.getElementById('resultModelMeta').textContent = '';
  const comparisonMeta = document.getElementById('comparisonMeta');
  const recommendationMeta = document.getElementById('recommendationMeta');
  comparisonMeta.classList.remove('agreement', 'disagreement');
  comparisonMeta.textContent = '';
  recommendationMeta.classList.remove('recommendation');
  recommendationMeta.textContent = '';

  const scoreEl = document.getElementById('scoreDisplay');
  scoreEl.textContent = `${confidencePct}%`;
  scoreEl.className = `score-display ${label || 'genuine'}`;

  const bar = document.getElementById('scoreBar');
  bar.style.width = '0%';
  bar.className = `score-bar ${label || 'genuine'}`;
  setTimeout(() => {
    bar.style.width = `${confidencePct}%`;
  }, 40);

  card.classList.remove('hidden');
}

async function analyzeReview() {
  const errEl = document.getElementById('singleError');
  const btn = document.getElementById('analyzeBtn');
  const spinner = document.getElementById('btnSpinner');
  const btnText = btn.querySelector('.btn-text');
  const card = document.getElementById('resultCard');

  hideError(errEl);
  card.classList.add('hidden');

  const payload = buildPayload();
  if (!payload.text) {
    showError(errEl, 'Please enter a review before running detection.');
    return;
  }

  btn.disabled = true;
  spinner.classList.remove('hidden');
  btnText.textContent = 'Analysing…';

  try {
    const res = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok) {
      showError(errEl, data.error || 'Prediction failed.');
      return;
    }

    renderResult(data);
  } catch {
    showError(errEl, 'Cannot connect to the server. Make sure backend/app.py is running.');
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
    btnText.textContent = 'Run Detection';
  }
}


const uploadZone = document.getElementById('uploadZone');
const csvFile = document.getElementById('csvFile');
const fileHint = document.getElementById('fileHint');

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
csvFile.addEventListener('change', () => {
  if (csvFile.files[0]) setFile(csvFile.files[0]);
});

function setFile(file) {
  const dt = new DataTransfer();
  dt.items.add(file);
  csvFile.files = dt.files;
  fileHint.textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
}

// ── Batch upload ──────────────────────────────────────────────────────────────

let _batchResults = [];

async function runBatch() {
  const errEl = document.getElementById('batchError');
  const btn = document.getElementById('batchBtn');
  const spinner = document.getElementById('batchSpinner');
  const btnText = btn.querySelector('.btn-text');
  const batchResultsEl = document.getElementById('batchResults');

  hideError(errEl);
  batchResultsEl.classList.add('hidden');
  _batchResults = [];

  const file = csvFile.files[0];
  if (!file) {
    showError(errEl, 'Please select a CSV file first.');
    return;
  }

  btn.disabled = true;
  spinner.classList.remove('hidden');
  btnText.textContent = 'Analysing…';

  try {
    // ✅ FIX: CREATE formData HERE
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_version', 'v3');

    const res = await fetch(`${API}/batch`, {
      method: 'POST',
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      showError(errEl, data.error || 'Batch prediction failed.');
      return;
    }

    _batchResults = data.results || [];
    renderBatch(data);

  } catch (e) {
    showError(errEl, e.message || 'Cannot connect to server.');
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
    btnText.textContent = 'Analyse CSV';
  }
}

function renderBatch(data) {
  const results = data.results;
  const fakeCount = results.filter(r => r.label === 'fake').length;
  const genuineCount = results.length - fakeCount;

  document.getElementById('batchSummary').innerHTML =
    `Processed <strong>${results.length}</strong> reviews — ` +
    `<span class="fake-count">⚠ ${fakeCount} fake</span> · ` +
    `<span class="genuine-count">✓ ${genuineCount} genuine</span>`;

  const tbody = document.getElementById('resultsBody');
  tbody.innerHTML = '';
  results.forEach((r, i) => {
    const tr = document.createElement('tr');
    const preview = (r.text || '').slice(0, 80) + ((r.text || '').length > 80 ? '…' : '');
    tr.innerHTML = `
      <td style="color:var(--text-muted);font-family:var(--font-mono)">${i + 1}</td>
      <td title="${escapeHtml(r.text || '')}">${escapeHtml(preview)}</td>
      <td><span class="verdict-badge ${r.label}">${r.label.toUpperCase()}</span></td>
      <td style="font-family:var(--font-mono)">${r.fake_percent}%</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById('batchResults').classList.remove('hidden');
}

function downloadCSV() {
  if (!_batchResults.length) return;
  const headers = ['text', 'label', 'fake_probability'];
  const rows = _batchResults.map(r =>
    headers.map(h => JSON.stringify(r[h] ?? '')).join(',')
  );
  const csv = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'fake_review_results.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function showError(el, msg) {
  el.textContent = msg;
  el.classList.remove('hidden');
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

document.getElementById('analyzeBtn')?.addEventListener('click', analyzeReview);
document.getElementById('reviewText')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    analyzeReview();
  }
});

checkHealth();