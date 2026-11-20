/* ================================================================
   App Logic — Intent-Aware Hybrid Retrieval UI
   ================================================================ */

// --- DOM References ---
const form = document.getElementById('search-form');
const queryInput = document.getElementById('query-input');
const searchButton = document.getElementById('search-button');
const requestStatus = document.getElementById('request-status');
const statusDot = document.getElementById('status-dot');
const resultCount = document.getElementById('result-count');
const resultsList = document.getElementById('results-list');
const strategyOutput = document.getElementById('strategy-output');
const jsonRaw = document.getElementById('json-raw');
const toggleParams = document.getElementById('toggle-params');
const paramsPanel = document.getElementById('params-panel');
const toggleJson = document.getElementById('toggle-json');
const jsonOutput = document.getElementById('json-output');

// --- Advanced Params Toggle ---
toggleParams.addEventListener('click', () => {
  const isCollapsed = paramsPanel.classList.toggle('collapsed');
  toggleParams.classList.toggle('expanded', !isCollapsed);
});

// --- Raw JSON Toggle ---
toggleJson.addEventListener('click', () => {
  const isCollapsed = jsonOutput.classList.toggle('collapsed');
  toggleJson.classList.toggle('expanded', !isCollapsed);
});

// --- Helpers ---
function capitalize(str) {
  return str.replace(/\b\w/g, c => c.toUpperCase());
}

function truncate(str, max) {
  if (!str || str.length <= max) return str || '';
  return str.slice(0, max) + '…';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// --- Render Loading State ---
function renderLoading() {
  statusDot.className = 'status-dot loading';
  requestStatus.textContent = 'Searching…';
  resultCount.textContent = '';
  searchButton.disabled = true;

  const skeletons = Array.from({ length: 5 }, () => `
    <div class="skeleton-card">
      <div class="skeleton-line"></div>
      <div class="skeleton-line"></div>
      <div class="skeleton-line"></div>
      <div class="skeleton-line"></div>
    </div>
  `).join('');

  resultsList.innerHTML = skeletons;
}

// --- Render Strategy Panel ---
function renderStrategy(retrieval) {
  if (!retrieval || !retrieval.strategy) {
    strategyOutput.innerHTML = '<p class="sidebar-placeholder">No strategy data available.</p>';
    return;
  }

  const s = retrieval.strategy;
  const rows = [
    ['Mode', retrieval.mode || s.type || '—'],
    ['Top K', s.top_k ?? '—'],
    ['Pool Size', s.candidate_pool_size ?? '—'],
    ['Lexical Weight', s.lexical_weight ?? '—'],
    ['Semantic Weight', s.semantic_weight ?? '—'],
    ['Model', s.semantic_model || '—'],
  ];

  strategyOutput.innerHTML = rows.map(([key, value]) => `
    <div class="strategy-detail">
      <span class="strategy-key">${key}</span>
      <span class="strategy-value">${escapeHtml(String(value))}</span>
    </div>
  `).join('');
}

// --- Render Pipeline Status ---
function renderPipeline(payload) {
  const modules = {
    'module-analysis': payload.query_analysis,
    'module-reformulation': payload.reformulations,
    'module-retrieval': { status: 'available' },
    'module-graph': payload.graph,
    'module-explain': payload.explanations,
  };

  for (const [id, module] of Object.entries(modules)) {
    const el = document.getElementById(id);
    if (!el) continue;
    const icon = el.querySelector('.module-status-icon');
    if (!icon) continue;

    const activeStatuses = new Set(['available', 'active', 'ready', 'enabled', 'completed']);
    const isActive = module?.status && activeStatuses.has(module.status);
    icon.className = `module-status-icon ${isActive ? 'active' : 'pending'}`;
  }
}

// --- Render Result Cards ---
function renderResults(results) {
  if (!results || results.length === 0) {
    resultsList.innerHTML = `
      <div class="no-results">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48">
          <circle cx="11" cy="11" r="8"/>
          <line x1="21" y1="21" x2="16.65" y2="16.65"/>
          <line x1="8" y1="11" x2="14" y2="11"/>
        </svg>
        <p>No matching candidates found.<br>Try broadening your search query.</p>
      </div>
    `;
    return;
  }

  resultsList.innerHTML = results.map((result, index) => {
    const name = escapeHtml(result.name || `Candidate ${result.id}`);
    const id = escapeHtml(result.id);
    const yoe = result.years_of_experience;
    const experienceLabel = yoe > 0 ? `${yoe} yr${yoe !== 1 ? 's' : ''} exp` : 'Entry-level';

    // Roles
    const rolesHtml = (result.roles || []).slice(0, 4).map(r =>
      `<span class="role-tag">${escapeHtml(capitalize(r))}</span>`
    ).join('');

    // Skills — show core (max 5), secondary (max 3), soft (max 3)
    const coreSkills = (result.core_skills || []).slice(0, 5).map(s =>
      `<span class="skill-tag core">${escapeHtml(capitalize(s))}</span>`
    ).join('');

    const secondarySkills = (result.secondary_skills || []).slice(0, 3).map(s =>
      `<span class="skill-tag secondary">${escapeHtml(capitalize(s))}</span>`
    ).join('');

    const softSkills = (result.soft_skills || []).slice(0, 3).map(s =>
      `<span class="skill-tag soft">${escapeHtml(capitalize(s))}</span>`
    ).join('');

    const allSkills = coreSkills + secondarySkills + softSkills;

    // Summary
    const summary = result.skill_summary ? escapeHtml(result.skill_summary) : '';
    const summaryHtml = summary
      ? `<p class="result-summary">${summary}</p>
         <button class="expand-btn" onclick="toggleExpand(this)" type="button">Show more</button>`
      : '';

    return `
      <div class="result-card" data-result-id="${id}">
        <div class="result-card-top">
          <span class="result-rank">${index + 1}</span>
          <div class="result-card-info">
            <div class="result-name">${name}</div>
            <div class="result-id">ID: ${id}</div>
          </div>
          <span class="result-experience">${experienceLabel}</span>
        </div>
        ${rolesHtml ? `<div class="result-roles">${rolesHtml}</div>` : ''}
        ${allSkills ? `<div class="result-skills">${allSkills}</div>` : ''}
        ${summaryHtml}
      </div>
    `;
  }).join('');
}

// --- Toggle summary expand ---
window.toggleExpand = function (btn) {
  const card = btn.closest('.result-card');
  const isExpanded = card.classList.toggle('expanded');
  btn.textContent = isExpanded ? 'Show less' : 'Show more';
};

// --- Full UI Update ---
function updateUi(payload) {
  // Status
  statusDot.className = 'status-dot';
  const count = payload.results?.length || 0;
  requestStatus.textContent = `Returned ${count} result${count !== 1 ? 's' : ''}`;
  resultCount.textContent = `${count} candidate${count !== 1 ? 's' : ''}`;

  // Results
  renderResults(payload.results);

  // Strategy
  renderStrategy(payload.retrieval);

  // Pipeline status
  renderPipeline(payload);

  // Raw JSON
  jsonRaw.textContent = JSON.stringify(payload, null, 2);
}

// --- Error State ---
function renderError(message) {
  statusDot.className = 'status-dot error';
  requestStatus.textContent = message;
  resultCount.textContent = '';
  searchButton.disabled = false;

  resultsList.innerHTML = `
    <div class="no-results">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="48" height="48">
        <circle cx="12" cy="12" r="10"/>
        <line x1="15" y1="9" x2="9" y2="15"/>
        <line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

// --- Search Handler ---
async function runSearch(event) {
  event.preventDefault();
  renderLoading();

  const formData = new FormData(form);
  const params = new URLSearchParams(formData);

  try {
    const response = await fetch(`/demo/search?${params.toString()}`);
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errText.slice(0, 200)}`);
    }

    const payload = await response.json();
    updateUi(payload);
  } catch (error) {
    renderError(`Search failed: ${error.message}`);
  } finally {
    searchButton.disabled = false;
  }
}

// --- Keyboard shortcut ---
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault();
    queryInput.focus();
    queryInput.select();
  }
});

// --- Init ---
form.addEventListener('submit', runSearch);

// Auto-run initial search on page load
window.addEventListener('DOMContentLoaded', () => {
  form.dispatchEvent(new Event('submit', { cancelable: true }));
});
