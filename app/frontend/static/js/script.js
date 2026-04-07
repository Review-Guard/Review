const API = '';

function showError(el, msg) {
  el.textContent = msg;
  el.classList.remove('hidden');
}

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
  document.getElementById('resultMeta').textContent = `threshold_percent: ${Number(data.threshold_percent ?? 0).toFixed(2)} · latency_ms: ${data.latency_ms ?? 'N/A'}`;
  document.getElementById('resultModelMeta').textContent = `model_version: ${data.model_version || 'N/A'}`;
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

async function renderComparison(payload) {
  const comparisonMeta = document.getElementById('comparisonMeta');
  const recommendationMeta = document.getElementById('recommendationMeta');
  try {
    const res = await fetch(`${API}/predict_all`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      return;
    }
    const data = await res.json();
    const preds = data.predictions || {};
    const v1 = preds.v1?.label || 'n/a';
    const v2 = preds.v2?.label || 'n/a';
    const v3 = preds.v3?.label || 'n/a';

    if (data.disagreement) {
      comparisonMeta.classList.add('disagreement');
      comparisonMeta.textContent = `cross-version: disagreement (v1=${v1}, v2=${v2}, v3=${v3})`;
      recommendationMeta.classList.add('recommendation');
      recommendationMeta.textContent = 'recommendation: manual review (models disagree)';
    } else {
      comparisonMeta.classList.add('agreement');
      comparisonMeta.textContent = `cross-version: agreement (${(data.majority_label || 'unknown').toLowerCase()})`;
      recommendationMeta.classList.add('recommendation');
      recommendationMeta.textContent = `recommendation: ${String(data.recommendation || data.majority_label || 'unknown').toLowerCase()}`;
    }
  } catch {
    // Keep silent if comparison endpoint is not reachable.
  }
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
    await renderComparison(payload);
  } catch {
    showError(errEl, 'Cannot connect to the server. Make sure backend/app.py is running.');
  } finally {
    btn.disabled = false;
    spinner.classList.add('hidden');
    btnText.textContent = 'Run Detection';
  }
}

document.getElementById('analyzeBtn')?.addEventListener('click', analyzeReview);
document.getElementById('reviewText')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    analyzeReview();
  }
});

checkHealth();
