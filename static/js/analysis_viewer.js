/**
 * 발음교정기 분석 뷰어 — 탭 전환, API 조회, 테이블/차트 렌더링
 */

// ── 탭 전환 ──────────────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tabId = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${tabId}`).classList.add("active");
  });
});

// ── 전역 상태 ────────────────────────────────────────────────────────────────
let _refRows = [];
let _phonemeRows = [];
let _wordRows = [];
let _latestUserData = null;
let _wordCompareRows = [];
const _charts = {};

// ── 초기 데이터 로딩 ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadOverview();
  loadReferenceQuality();
  loadPhonemeAnalysis();
  loadWordResults();
  loadOutliers();
  loadErrors();
  loadUserResults(true);
  loadWordCompare();

  document.getElementById("user-mode-select")?.addEventListener("change", (e) => {
    loadUserResults(e.target.value === "latest");
  });
  document.getElementById("word-compare-select")?.addEventListener("change", (e) => {
    renderWordCompareByKey(e.target.value);
  });
});

// ── API 호출 헬퍼 ─────────────────────────────────────────────────────────────
async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} — HTTP ${res.status}`);
  return res.json();
}

// ── Overview ──────────────────────────────────────────────────────────────────
async function loadOverview() {
  try {
    const data = await fetchJson("/api/overview");
    renderOverviewCards(data);
    renderOverviewMeta(data.metadata);
  } catch (err) {
    document.getElementById("overview-cards").innerHTML = errorHtml(err.message);
  }
}

function renderOverviewCards(data) {
  const cards = [
    { label: "Reference 발음 수", value: data.reference_phoneme_count, cls: "" },
    { label: "Reference Sample", value: data.reference_sample_count, cls: "" },
    { label: "Reference 단어 수", value: data.reference_word_count, cls: "" },
    { label: "Comparison 전체", value: data.comparison_total, cls: "" },
    { label: "Comparison 성공", value: data.comparison_success, cls: "ok" },
    { label: "Comparison 실패", value: data.comparison_error, cls: data.comparison_error > 0 ? "err" : "" },
    { label: "분석된 발음 수", value: data.analyzed_phoneme_count, cls: "" },
    { label: "Outlier 후보", value: data.outlier_count, cls: data.outlier_count > 0 ? "warn" : "" },
  ];
  document.getElementById("overview-cards").innerHTML = cards.map(({ label, value, cls }) => `
    <div class="stat-card ${cls}">
      <div class="stat-label">${label}</div>
      <div class="stat-value">${value ?? "—"}</div>
    </div>
  `).join("");
}

function renderOverviewMeta(meta) {
  const el = document.getElementById("overview-meta");
  if (!meta || !Object.keys(meta).length) {
    el.innerHTML = "<em>metadata 없음</em>";
    return;
  }
  el.innerHTML = Object.entries(meta)
    .map(([k, v]) => `<strong>${escHtml(k)}:</strong> ${escHtml(v ?? "—")}`)
    .join(" &nbsp;·&nbsp; ");
}

// ── Reference Quality ─────────────────────────────────────────────────────────
async function loadReferenceQuality() {
  try {
    _refRows = await fetchJson("/api/reference-quality");
    renderRefTable(_refRows);
  } catch (err) {
    document.getElementById("ref-tbody").innerHTML = `<tr><td colspan="13">${errorHtml(err.message)}</td></tr>`;
  }
}

function renderRefTable(rows) {
  const tbody = document.getElementById("ref-tbody");
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="13">${missingFileHtml("reference_vectors.json", "uv run python scripts/build_reference.py")}</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((r) => {
    const warnHtml = r.quality_warning
      ? `<span class="pill pill-warn">${escHtml(r.quality_warning)}</span>`
      : `<span class="pill pill-ok">OK</span>`;
    return `
      <tr class="clickable" data-phoneme="${escHtml(r.phoneme)}">
        <td class="phoneme-cell">/${escHtml(r.phoneme)}/</td>
        <td>${escHtml(r.phoneme_type)}</td>
        <td class="num-cell">${r.sample_count}</td>
        <td class="num-cell">${r.word_count}</td>
        <td class="num-cell">${fmtNum(r.duration_ms, 2)}</td>
        <td class="num-cell">${fmtNum(r.duration_std, 2)}</td>
        <td class="num-cell">${fmtNum(r.zcr_mean, 6)}</td>
        <td class="num-cell">${fmtNum(r.zcr_std, 6)}</td>
        <td class="num-cell">${fmtNum(r.rms_mean, 6)}</td>
        <td class="num-cell">${fmtNum(r.rms_std, 6)}</td>
        <td class="num-cell">${fmtNum(r.spectral_centroid_mean, 2)}</td>
        <td class="num-cell">${fmtNum(r.spectral_centroid_std, 2)}</td>
        <td>${warnHtml}</td>
      </tr>`;
  }).join("");
  tbody.querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", () => onRefRowClick(tr)));
}

function onRefRowClick(tr) {
  const phoneme = tr.dataset.phoneme;
  const row = _refRows.find((r) => r.phoneme === phoneme);
  if (!row) return;
  document.querySelectorAll("#ref-tbody tr").forEach((r) => r.classList.remove("selected"));
  tr.classList.add("selected");
  const words = row.test_words || [];
  const koreans = row.korean_pronunciations || {};
  const chips = words.map((w) => {
    const ko = koreans[w] ? ` <span style="color:#6b7c8f;font-size:11px;">(${escHtml(koreans[w])})</span>` : "";
    return `<span class="word-chip">${escHtml(w)}${ko}</span>`;
  }).join("");
  const panel = document.getElementById("ref-detail-panel");
  panel.innerHTML = `<h3>/${escHtml(phoneme)}/ — ${escHtml(row.phoneme_type)}</h3>
    <div class="col-title">Test Words (${words.length}개)</div>
    <div class="word-chip-list">${chips || "<em>없음</em>"}</div>`;
  panel.style.display = "block";
}

// ── Phoneme Analysis ──────────────────────────────────────────────────────────
async function loadPhonemeAnalysis() {
  try {
    _phonemeRows = await fetchJson("/api/phoneme-analysis");
    populatePhonemeFilter(_phonemeRows);
    renderPhonemeTable(_phonemeRows);
  } catch (err) {
    document.getElementById("phoneme-tbody").innerHTML = `<tr><td colspan="10">${errorHtml(err.message)}</td></tr>`;
  }
}

function populatePhonemeFilter(rows) {
  const sel = document.getElementById("filter-phoneme");
  if (!sel) return;
  const existing = new Set([...sel.options].map((o) => o.value));
  [...new Set(rows.map((r) => r.phoneme))].sort().forEach((p) => {
    if (existing.has(p)) return;
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = `/${p}/`;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", applyPhonemeFilter);
  document.getElementById("filter-outlier-only")?.addEventListener("change", applyPhonemeFilter);
}

function applyPhonemeFilter() {
  const phonemeVal = document.getElementById("filter-phoneme").value;
  const outlierOnly = document.getElementById("filter-outlier-only").checked;
  renderPhonemeTable(_phonemeRows.filter((r) => (!phonemeVal || r.phoneme === phonemeVal) && (!outlierOnly || r.outlier_count)));
}

function renderPhonemeTable(rows) {
  const noDataEl = document.getElementById("phoneme-no-data");
  const tbody = document.getElementById("phoneme-tbody");
  if (!rows.length) {
    noDataEl.style.display = "block";
    noDataEl.innerHTML = `comparison 성공 결과가 없습니다.<code>uv run python scripts/compare_en_ko.py</code>`;
    tbody.innerHTML = "";
    return;
  }
  noDataEl.style.display = "none";
  tbody.innerHTML = rows.map((r) => `
    <tr>
      <td class="phoneme-cell">/${escHtml(r.phoneme)}/</td>
      <td class="num-cell">${r.word_count}</td>
      <td>${escHtml(r.dominant_features)}</td>
      <td class="num-cell">${r.avg_mfcc_distance ?? "—"}</td>
      <td class="num-cell">${r.avg_mfcc_cosine_distance ?? "—"}</td>
      <td class="num-cell ${ratioCls(r.avg_duration_ko_en_ratio)}">${r.avg_duration_ko_en_ratio ?? "—"}</td>
      <td class="num-cell ${ratioCls(r.avg_zcr_ko_en_ratio)}">${r.avg_zcr_ko_en_ratio ?? "—"}</td>
      <td class="num-cell ${ratioCls(r.avg_rms_ko_en_ratio)}">${r.avg_rms_ko_en_ratio ?? "—"}</td>
      <td class="num-cell ${ratioCls(r.avg_spectral_centroid_ko_en_ratio)}">${r.avg_spectral_centroid_ko_en_ratio ?? "—"}</td>
      <td class="num-cell">${r.outlier_count ? `<span class="pill pill-warn">${r.outlier_count}</span>` : "0"}</td>
    </tr>`).join("");
}

// ── Word Results ──────────────────────────────────────────────────────────────
async function loadWordResults() {
  try {
    _wordRows = await fetchJson("/api/word-results");
    populateWordPhonemeFilter(_wordRows);
    renderWordTable(_wordRows);
  } catch (err) {
    document.getElementById("word-tbody").innerHTML = `<tr><td colspan="11">${errorHtml(err.message)}</td></tr>`;
  }
}

function populateWordPhonemeFilter(rows) {
  const sel = document.getElementById("filter-word-phoneme");
  const existing = new Set([...sel.options].map((o) => o.value));
  [...new Set(rows.map((r) => r.phoneme))].sort().forEach((p) => {
    if (existing.has(p)) return;
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = `/${p}/`;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", applyWordFilter);
  document.getElementById("filter-word-search").addEventListener("input", applyWordFilter);
}

function applyWordFilter() {
  const phonemeVal = document.getElementById("filter-word-phoneme").value;
  const searchVal = document.getElementById("filter-word-search").value.toLowerCase().trim();
  renderWordTable(_wordRows.filter((r) => (!phonemeVal || r.phoneme === phonemeVal) && (!searchVal || r.word.toLowerCase().includes(searchVal))));
}

function renderWordTable(rows) {
  const tbody = document.getElementById("word-tbody");
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="11" style="padding:24px;text-align:center;color:#6b7c8f;">성공한 비교 결과가 없습니다.</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((r) => `
    <tr class="clickable" data-row='${escJson(r)}'>
      <td><strong>${escHtml(r.word)}</strong></td>
      <td>${escHtml(r.korean_pronunciation)}</td>
      <td class="phoneme-cell">/${escHtml(r.phoneme)}/</td>
      <td class="num-cell">${r.en_duration_ms ?? "—"}</td>
      <td class="num-cell">${r.ko_duration_ms ?? "—"}</td>
      <td class="num-cell ${ratioCls(r.duration_ko_en_ratio)}">${r.duration_ko_en_ratio ?? "—"}</td>
      <td class="num-cell ${ratioCls(r.zcr_ko_en_ratio)}">${r.zcr_ko_en_ratio ?? "—"}</td>
      <td class="num-cell ${ratioCls(r.rms_ko_en_ratio)}">${r.rms_ko_en_ratio ?? "—"}</td>
      <td class="num-cell ${ratioCls(r.spectral_centroid_ko_en_ratio)}">${r.spectral_centroid_ko_en_ratio ?? "—"}</td>
      <td class="num-cell">${r.mfcc_distance ?? "—"}</td>
      <td class="num-cell">${r.mfcc_cosine_distance ?? "—"}</td>
    </tr>`).join("");
  tbody.querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", () => onWordRowClick(tr)));
}

function onWordRowClick(tr) {
  document.querySelectorAll("#word-tbody tr").forEach((r) => r.classList.remove("selected"));
  tr.classList.add("selected");
  const r = JSON.parse(tr.dataset.row);
  const panel = document.getElementById("word-detail-panel");
  const metrics = [
    { name: "Duration", en: r.en_duration_ms, ko: r.ko_duration_ms, ratio: r.duration_ko_en_ratio, unit: "ms" },
    { name: "ZCR", en: r.en_zcr_mean, ko: r.ko_zcr_mean, ratio: r.zcr_ko_en_ratio, unit: "" },
    { name: "RMS", en: r.en_rms_mean, ko: r.ko_rms_mean, ratio: r.rms_ko_en_ratio, unit: "" },
    { name: "Spectral Centroid", en: r.en_spectral_centroid_mean, ko: r.ko_spectral_centroid_mean, ratio: r.spectral_centroid_ko_en_ratio, unit: "Hz" },
  ];
  const metricHtml = metrics.map(({ name, en, ko, ratio, unit }) => `
    <div class="detail-metric-block">
      <div class="metric-name">${name}</div>
      <div class="metric-sub">EN: ${fmtMetric(en, unit)} / KO: ${fmtMetric(ko, unit)}</div>
      <div class="metric-sub">ko/en ratio: <strong class="${ratioCls(ratio)}">${ratio != null ? `${ratio}x` : "—"}</strong></div>
    </div>`).join("");
  panel.innerHTML = `<h3>${escHtml(r.word)} <span style="font-weight:400;font-size:14px;color:#6b7c8f;">/${escHtml(r.phoneme)}/ · ${escHtml(r.korean_pronunciation)}</span></h3>
    <div class="detail-row">
      <div class="detail-col"><div class="col-title">Feature Ratios</div>${metricHtml}</div>
      <div class="detail-col"><div class="col-title">MFCC</div>
        <div class="detail-metric-block">
          <div class="metric-sub">distance: <strong>${r.mfcc_distance ?? "—"}</strong></div>
          <div class="metric-sub">cosine distance: <strong>${r.mfcc_cosine_distance ?? "—"}</strong></div>
        </div>
      </div>
    </div>`;
  panel.style.display = "block";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Outliers / Errors ─────────────────────────────────────────────────────────
async function loadOutliers() {
  try {
    renderOutlierTable(await fetchJson("/api/outliers"));
  } catch (err) {
    document.getElementById("outlier-tbody").innerHTML = `<tr><td colspan="7">${errorHtml(err.message)}</td></tr>`;
  }
}

function renderOutlierTable(rows) {
  const tbody = document.getElementById("outlier-tbody");
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="7" style="padding:20px;text-align:center;color:#6b7c8f;">outlier 후보가 없거나 comparison_analysis_report.json이 없습니다.<br/><span style="font-size:12px;">uv run python scripts/analyze_comparison_results.py</span></td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((r) => `
    <tr><td><strong>${escHtml(r.word)}</strong></td><td class="phoneme-cell">/${escHtml(r.phoneme)}/</td><td>${escHtml(r.metric_name)}</td><td class="num-cell">${r.value ?? "—"}</td><td class="num-cell">${r.average ?? "—"}</td><td class="num-cell">${r.stdev ?? "—"}</td><td class="num-cell ${Math.abs(r.z_score) > 2 ? "ratio-high" : ""}">${r.z_score ?? "—"}</td></tr>
  `).join("");
}

async function loadErrors() {
  try {
    renderErrorTable(await fetchJson("/api/errors"));
  } catch (err) {
    document.getElementById("error-tbody").innerHTML = `<tr><td colspan="6">${errorHtml(err.message)}</td></tr>`;
  }
}

function renderErrorTable(rows) {
  const tbody = document.getElementById("error-tbody");
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6" style="padding:16px;text-align:center;color:#10b981;font-weight:600;">실패한 항목 없음</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((r) => `
    <tr><td><strong>${escHtml(r.word)}</strong></td><td>${escHtml(r.korean_pronunciation)}</td><td class="phoneme-cell">/${escHtml(r.phoneme)}/</td><td class="warn-cell">${escHtml(r.error_message)}</td><td>${r.en_audio_exists ? '<span class="pill pill-ok">✓</span>' : '<span class="pill pill-err">✗</span>'}</td><td>${r.ko_audio_exists ? '<span class="pill pill-ok">✓</span>' : '<span class="pill pill-err">✗</span>'}</td></tr>
  `).join("");
}

// ── Chart helpers ─────────────────────────────────────────────────────────────
function _destroyChart(id) {
  if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
}

function _createChart(id, config) {
  _destroyChart(id);
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  _charts[id] = new Chart(canvas, config);
  return _charts[id];
}

const _USER_CHART_IDS = [
  "chart-label-score", "chart-score-dist", "chart-phoneme-label",
  "chart-penalty", "chart-feature-score", "chart-base-final",
  "chart-row-detail", "chart-row-bar",
];
const _WORD_COMPARE_CHART_IDS = ["chart-word-feature-values", "chart-word-feature-ratios"];

function _destroyUserCharts() { _USER_CHART_IDS.forEach(_destroyChart); }
function _destroyWordCompareCharts() { _WORD_COMPARE_CHART_IDS.forEach(_destroyChart); }

const _LABEL_PALETTE = {
  good:           { bg: "rgba(16,185,129,0.7)",  border: "#10b981" },
  korean_like:    { bg: "rgba(245,158,11,0.7)",  border: "#f59e0b" },
  wrong_or_noisy: { bg: "rgba(239,68,68,0.7)",   border: "#ef4444" },
  unlabeled:      { bg: "rgba(107,124,143,0.5)", border: "#6b7c8f" },
  NULL:           { bg: "rgba(148,163,184,0.5)", border: "#94a3b8" },
};
function _labelColor(label, mode = "bg") {
  const key = label ?? "NULL";
  return (_LABEL_PALETTE[key] || _LABEL_PALETTE.NULL)[mode];
}

const _PENALTY_DEFS = [
  { key: "avg_quality_penalty", label: "quality", color: "#ef4444" },
  { key: "avg_pronunciation_penalty", label: "pronunciation", color: "#f97316" },
  { key: "avg_duration_penalty", label: "duration", color: "#eab308" },
  { key: "avg_volume_penalty", label: "volume", color: "#8b5cf6" },
  { key: "avg_noise_penalty", label: "noise", color: "#6366f1" },
  { key: "avg_total_penalty", label: "total", color: "#dc2626" },
];
const _FEATURE_SCORE_DEFS = [
  { key: "avg_mfcc_score", label: "MFCC", color: "#3b82f6" },
  { key: "avg_duration_score", label: "Duration", color: "#10b981" },
  { key: "avg_rms_score", label: "RMS", color: "#8b5cf6" },
  { key: "avg_zcr_score", label: "ZCR", color: "#f59e0b" },
  { key: "avg_spectral_centroid_score", label: "Spectral", color: "#ec4899" },
];

// ── User Results ──────────────────────────────────────────────────────────────
async function loadUserResults(latestOnly = true) {
  _destroyUserCharts();
  const wrap = document.getElementById("user-results-wrap");
  wrap.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  try {
    const data = await fetchJson(`/api/user-results?latest_only=${latestOnly}`);
    if (latestOnly) _latestUserData = data;
    if (!data.exists) {
      wrap.innerHTML = `<div class="no-data-msg">data/pronunciation.db 파일이 없습니다.<br/>앱에서 녹음 후 분석을 1회 이상 실행하면 자동으로 생성됩니다.</div>`;
      return;
    }
    if (data.error) {
      wrap.innerHTML = `<div class="no-data-msg">DB 읽기 오류: ${escHtml(data.error)}<br/>path: ${escHtml(data.path)}</div>`;
      return;
    }
    if (!data.rows || !data.rows.length) {
      wrap.innerHTML = `<div class="no-data-msg">pronunciation.db는 있지만 저장된 결과가 없습니다.</div>`;
      return;
    }
    const columns = data.columns && data.columns.length ? data.columns : Object.keys(data.rows[0]);
    const dispLabel = latestOnly ? `최신 ${data.rows.length}개 (word+label 중복 제거)` : `최근 ${data.rows.length}개`;
    wrap.innerHTML = `
      ${_buildSummaryCards(data.summary_cards)}
      <div class="meta-box" style="margin-top:8px;"><strong>table:</strong> ${escHtml(data.table ?? "")} &nbsp;·&nbsp; <strong>전체 rows:</strong> ${data.row_count} &nbsp;·&nbsp; <strong>표시:</strong> ${dispLabel}</div>
      <div class="chart-grid" style="margin-top:16px;">
        <div class="chart-box"><div class="chart-title">라벨별 평균 점수</div><div class="chart-canvas-wrap"><canvas id="chart-label-score"></canvas></div></div>
        <div class="chart-box"><div class="chart-title">점수 분포</div><div class="chart-canvas-wrap"><canvas id="chart-score-dist"></canvas></div></div>
      </div>
      <div class="chart-grid single"><div class="chart-box"><div class="chart-title">음소별 라벨 평균 점수</div><div class="chart-canvas-wrap" style="height:300px;"><canvas id="chart-phoneme-label"></canvas></div></div></div>
      <div class="chart-grid">
        <div class="chart-box"><div class="chart-title">라벨별 Penalty 평균</div>${data.penalty_summary ? '<div class="chart-canvas-wrap"><canvas id="chart-penalty"></canvas></div>' : '<p class="no-data-inline">세부 penalty 데이터가 없습니다.</p>'}</div>
        <div class="chart-box"><div class="chart-title">라벨별 Feature Score 평균 (sub-score)</div>${data.feature_score_summary ? '<div class="chart-canvas-wrap"><canvas id="chart-feature-score"></canvas></div>' : '<p class="no-data-inline">feature score 데이터가 없습니다.</p>'}</div>
      </div>
      <div class="chart-grid single"><div class="chart-box"><div class="chart-title">라벨별 Base Score vs Final Score</div><div class="chart-canvas-wrap"><canvas id="chart-base-final"></canvas></div></div></div>
      <div class="table-wrap" style="margin-bottom:16px;"><table id="user-table" class="data-table"><thead><tr>${columns.map(c => `<th>${escHtml(c)}</th>`).join("")}</tr></thead><tbody>${data.rows.map((r, idx) => `<tr class="clickable" data-idx="${idx}">${columns.map(c => `<td>${escHtml(String(r[c] ?? ""))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
      <div id="user-row-detail" class="detail-panel" style="display:none;"></div>`;
    _renderLabelScoreChart(data.label_summary);
    _renderScoreDistChart(data.score_by_label);
    _renderPhonemeLabelChart(data.phoneme_label_breakdown);
    if (data.penalty_summary) _renderPenaltyChart(data.penalty_summary);
    if (data.feature_score_summary) _renderFeatureScoreChart(data.feature_score_summary);
    _renderBaseVsFinalChart(data.score_comparison);
    wrap.querySelectorAll("#user-table tbody tr.clickable").forEach((tr) => {
      const idx = parseInt(tr.dataset.idx, 10);
      tr.addEventListener("click", () => _onUserRowClick(tr, data.rows[idx]));
    });
    if (latestOnly) prepareWordCompareOptions(data.rows);
  } catch (err) {
    wrap.innerHTML = errorHtml(err.message);
  }
}

function _buildSummaryCards(cards) {
  if (!cards) return "";
  const { avg_by_label, ordered_correctly, latest_count } = cards;
  const configs = [
    { key: "good", label: "good 평균" },
    { key: "korean_like", label: "korean_like 평균" },
    { key: "wrong_or_noisy", label: "wrong 평균" },
  ];
  const scoreCards = configs.map(({ key, label }) => `<div class="stat-card"><div class="stat-label">${label}</div><div class="user-stat-value" style="color:${_labelColor(key, "border")};">${avg_by_label[key] != null ? avg_by_label[key] : "—"}</div></div>`).join("");
  const badge = ordered_correctly
    ? `<span class="pill pill-ok" style="font-size:12px;">good &gt; korean_like &gt; wrong ✓</span>`
    : `<span class="pill pill-warn" style="font-size:12px;">순서 불일치 ✗</span>`;
  return `<div class="stat-grid user-summary-grid"><div class="stat-card"><div class="stat-label">Latest Rows</div><div class="stat-value">${latest_count}</div></div>${scoreCards}<div class="stat-card ${ordered_correctly ? "ok" : "warn"}"><div class="stat-label">채점 순서 검증</div><div style="margin-top:10px;">${badge}</div></div></div>`;
}

function _renderLabelScoreChart(labelSummary) {
  if (!labelSummary || !labelSummary.length) return;
  const labels = labelSummary.map(s => s.test_label ?? "NULL");
  _createChart("chart-label-score", {
    type: "bar",
    data: { labels, datasets: [{ label: "평균 점수", data: labelSummary.map(s => s.avg_score ?? 0), backgroundColor: labels.map(l => _labelColor(l, "bg")), borderColor: labels.map(l => _labelColor(l, "border")), borderWidth: 2 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, scales: { x: { min: 0, max: 100, title: { display: true, text: "점수" } } }, plugins: { legend: { display: false } } },
  });
}

function _renderPhonemeLabelChart(breakdown) {
  if (!breakdown || !breakdown.length) return;
  const phonemes = [...new Set(breakdown.map(b => b.phoneme))].sort();
  const labelKeys = [...new Set(breakdown.map(b => b.test_label ?? "NULL"))].sort();
  const datasets = labelKeys.map(label => ({ label, data: phonemes.map(p => (breakdown.find(b => b.phoneme === p && (b.test_label ?? "NULL") === label) || {}).avg_score ?? null), backgroundColor: _labelColor(label, "bg"), borderColor: _labelColor(label, "border"), borderWidth: 1 }));
  _createChart("chart-phoneme-label", { type: "bar", data: { labels: phonemes.map(p => `/${p}/`), datasets }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 100, title: { display: true, text: "점수" } } }, plugins: { legend: { display: true, position: "top" } } } });
}

function _renderScoreDistChart(scoreByLabel) {
  if (!scoreByLabel || !Object.keys(scoreByLabel).length) return;
  const buckets = [{ label: "0–20", min: 0, max: 20 }, { label: "20–40", min: 20, max: 40 }, { label: "40–60", min: 40, max: 60 }, { label: "60–70", min: 60, max: 70 }, { label: "70–80", min: 70, max: 80 }, { label: "80–90", min: 80, max: 90 }, { label: "90–100", min: 90, max: 101 }];
  const labelKeys = Object.keys(scoreByLabel).sort();
  const datasets = labelKeys.map(label => ({ label, data: buckets.map(({ min, max }) => scoreByLabel[label].filter(s => s >= min && s < max).length), backgroundColor: _labelColor(label, "bg"), borderColor: _labelColor(label, "border"), borderWidth: 1 }));
  _createChart("chart-score-dist", { type: "bar", data: { labels: buckets.map(b => b.label), datasets }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } }, plugins: { legend: { display: true, position: "top" } } } });
}

function _renderPenaltyChart(rows) { _renderGroupedAverageChart("chart-penalty", rows, _PENALTY_DEFS, "penalty 점수", null); }
function _renderFeatureScoreChart(rows) { _renderGroupedAverageChart("chart-feature-score", rows, _FEATURE_SCORE_DEFS, "sub-score", { min: 0, max: 100 }); }
function _renderGroupedAverageChart(id, rows, defs, title, range) {
  if (!rows || !rows.length) return;
  const active = defs.filter(def => rows.some(s => s[def.key] != null));
  if (!active.length) return;
  const labels = rows.map(s => s.test_label ?? "NULL");
  const datasets = active.map(def => ({ label: def.label, data: rows.map(s => s[def.key] ?? null), backgroundColor: def.color + "aa", borderColor: def.color, borderWidth: 1 }));
  _createChart(id, { type: "bar", data: { labels, datasets }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, ...(range || {}), title: { display: true, text: title } } }, plugins: { legend: { display: true, position: "top" } } } });
}

function _renderBaseVsFinalChart(rows) {
  if (!rows || !rows.length) return;
  const labels = rows.map(s => s.test_label ?? "NULL");
  const datasets = [];
  if (rows.some(s => s.avg_base_score != null)) datasets.push({ label: "base_score", data: rows.map(s => s.avg_base_score ?? null), backgroundColor: "rgba(99,102,241,0.7)", borderColor: "#6366f1", borderWidth: 2 });
  const hasFinal = rows.some(s => s.avg_final_score != null);
  datasets.push({ label: hasFinal ? "final_score" : "score", data: rows.map(s => (hasFinal ? s.avg_final_score : s.avg_score) ?? null), backgroundColor: "rgba(16,185,129,0.7)", borderColor: "#10b981", borderWidth: 2 });
  _createChart("chart-base-final", { type: "bar", data: { labels, datasets }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 100, title: { display: true, text: "점수" } } }, plugins: { legend: { display: true, position: "top" } } } });
}

function _onUserRowClick(tr, row) {
  document.querySelectorAll("#user-table tbody tr").forEach(r => r.classList.remove("selected"));
  tr.classList.add("selected");
  const panel = document.getElementById("user-row-detail");
  const label = row.test_label ?? "NULL";
  const color = _labelColor(label, "border");
  const score = row.final_score ?? row.score ?? null;
  const base = row.base_score ?? null;
  const featureItems = [
    { name: "MFCC", key: "mfcc_score", type: "score" }, { name: "Duration", key: "duration_score", type: "score" },
    { name: "RMS", key: "rms_score", type: "score" }, { name: "ZCR", key: "zcr_score", type: "score" },
    { name: "Spectral", key: "spectral_centroid_score", type: "score" },
  ].filter(f => row[f.key] != null);
  const penaltyItems = [
    { name: "quality", key: "quality_penalty", type: "penalty" }, { name: "pronunciation", key: "pronunciation_penalty", type: "penalty" },
    { name: "duration pen.", key: "duration_penalty", type: "penalty" }, { name: "volume", key: "volume_penalty", type: "penalty" },
    { name: "noise", key: "noise_penalty", type: "penalty" }, { name: "total", key: "total_penalty", type: "penalty" },
  ].filter(p => row[p.key] != null);
  const barItems = [...featureItems, ...penaltyItems];
  const rawValues = [["Score", score != null ? Number(score).toFixed(1) : "—"], ["Base", base != null ? Number(base).toFixed(1) : "—"], ["Total Pen.", row.total_penalty != null ? Number(row.total_penalty).toFixed(1) : "—"], ["MFCC Dist", row.mfcc_distance != null ? Number(row.mfcc_distance).toFixed(3) : "—"], ["Duration", row.duration_ms != null ? `${row.duration_ms} ms` : "—"], ["Dur. Ratio", row.duration_ratio != null ? Number(row.duration_ratio).toFixed(2) : "—"], ["RMS", row.rms_mean != null ? Number(row.rms_mean).toFixed(4) : "—"], ["ZCR", row.zcr_mean != null ? Number(row.zcr_mean).toFixed(4) : "—"], ["Spectral", row.spectral_centroid_mean != null ? `${Number(row.spectral_centroid_mean).toFixed(0)} Hz` : "—"]];
  panel.innerHTML = `<div class="user-row-detail-inner"><div class="detail-chart-col"><h3>${escHtml(row.word ?? "")} <span>/${escHtml(row.phoneme ?? "")}/</span></h3><div style="position:relative;height:110px;margin:8px 0;"><canvas id="chart-row-detail"></canvas><div style="position:absolute;bottom:0;left:50%;transform:translateX(-50%);font-size:26px;font-weight:700;color:${color};">${score != null ? Number(score).toFixed(1) : "—"}</div></div><div class="grade-badge">${escHtml(row.grade ?? "")}</div><div style="font-size:11px;color:#6b7c8f;margin-top:4px;">${escHtml(label)}</div>${base != null ? `<div style="font-size:12px;color:#6b7c8f;margin-top:4px;">base ${Number(base).toFixed(1)} → final ${score != null ? Number(score).toFixed(1) : "—"}</div>` : ""}</div>${barItems.length ? `<div style="flex:2;min-width:240px;"><div class="col-title">Sub Scores & Penalties</div><div style="height:${Math.max(barItems.length * 28 + 40, 80)}px;margin-top:8px;"><canvas id="chart-row-bar"></canvas></div></div>` : ""}<div class="detail-info-col"><div class="col-title">Raw Values</div>${rawValues.map(([n, v]) => `<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #f0f7ff;"><span style="color:#6b7c8f;font-size:12px;">${n}</span><span style="font-variant-numeric:tabular-nums;">${v}</span></div>`).join("")}${row.feedback ? `<div class="col-title" style="margin-top:10px;">Feedback</div><div style="color:#16324f;font-size:12px;margin-top:4px;line-height:1.5;">${escHtml(row.feedback)}</div>` : ""}</div></div>`;
  panel.style.display = "block";
  if (score != null) _createScoreDoughnut("chart-row-detail", score, color);
  if (barItems.length) _createChart("chart-row-bar", { type: "bar", data: { labels: barItems.map(i => i.name), datasets: [{ label: "값", data: barItems.map(i => row[i.key]), backgroundColor: barItems.map(i => i.type === "score" ? "rgba(59,130,246,0.75)" : "rgba(239,68,68,0.75)"), borderColor: barItems.map(i => i.type === "score" ? "#3b82f6" : "#ef4444"), borderWidth: 1 }] }, options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, scales: { x: { min: 0, max: 100 } }, plugins: { legend: { display: false } } } });
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function _createScoreDoughnut(id, score, color) {
  _createChart(id, { type: "doughnut", data: { labels: ["점수", ""], datasets: [{ data: [Number(score), 100 - Number(score)], backgroundColor: [color, "#f0f7ff"], borderColor: [color, "#d8e9f7"], borderWidth: 1 }] }, options: { responsive: true, maintainAspectRatio: false, cutout: "72%", rotation: -90, circumference: 180, plugins: { legend: { display: false }, tooltip: { filter: item => item.dataIndex === 0 } } } });
}

// ── Word Compare tab ─────────────────────────────────────────────────────────
async function loadWordCompare() {
  const wrap = document.getElementById("word-compare-wrap");
  if (!wrap) return;
  wrap.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';
  try {
    const [userData, refRows] = await Promise.all([
      fetchJson("/api/user-results?latest_only=true"),
      fetchJson("/api/reference-quality"),
    ]);
    _latestUserData = userData;
    _refRows = refRows;
    if (!userData.exists || !userData.rows || !userData.rows.length) {
      wrap.innerHTML = `<div class="no-data-msg">저장된 사용자 녹음이 없습니다.<br/>앱에서 녹음 후 분석을 먼저 실행하세요.</div>`;
      return;
    }
    prepareWordCompareOptions(userData.rows);
  } catch (err) {
    wrap.innerHTML = errorHtml(err.message);
  }
}

function prepareWordCompareOptions(rows) {
  const sel = document.getElementById("word-compare-select");
  if (!sel || !rows || !rows.length) return;
  _wordCompareRows = rows.filter(r => r.word && r.phoneme && hasComparableFeature(r));
  if (!_wordCompareRows.length) {
    sel.innerHTML = `<option value="">비교 가능한 녹음 없음</option>`;
    document.getElementById("word-compare-wrap").innerHTML = `<div class="no-data-msg">duration/zcr/rms/spectral 값이 있는 사용자 녹음이 없습니다.</div>`;
    return;
  }
  sel.innerHTML = _wordCompareRows.map((r, idx) => {
    const score = r.score != null ? Number(r.score).toFixed(1) : "--";
    const label = r.test_label || "app";
    return `<option value="${idx}">${escHtml(r.word)} /${escHtml(r.phoneme)}/ · ${escHtml(label)} · ${score}</option>`;
  }).join("");
  if (sel.value === "" || Number(sel.value) >= _wordCompareRows.length) sel.value = "0";
  renderWordCompareByKey(sel.value);
}

function hasComparableFeature(row) {
  return ["duration_ms", "zcr_mean", "rms_mean", "spectral_centroid_mean"].some(k => row[k] != null);
}

function renderWordCompareByKey(key) {
  _destroyWordCompareCharts();
  const wrap = document.getElementById("word-compare-wrap");
  const idx = Number(key);
  const row = _wordCompareRows[idx];
  if (!row) {
    wrap.innerHTML = `<div class="no-data-msg">비교할 단어를 선택해주세요.</div>`;
    return;
  }
  const ref = _refRows.find(r => r.phoneme === row.phoneme);
  if (!ref) {
    wrap.innerHTML = `<div class="no-data-msg">/${escHtml(row.phoneme)}/ reference 정보가 없습니다.</div>`;
    return;
  }
  const metrics = buildWordCompareMetrics(row, ref);
  const score = row.score != null ? Number(row.score).toFixed(1) : "--";
  const label = row.test_label || "app";
  const ratioRows = metrics.map(m => `
    <tr>
      <td>${escHtml(m.label)}</td>
      <td class="num-cell">${fmtMetric(m.user, m.unit)}</td>
      <td class="num-cell">${fmtMetric(m.ref, m.unit)}</td>
      <td class="num-cell ${ratioCls(m.ratio)}">${m.ratio != null ? m.ratio.toFixed(2) + "x" : "—"}</td>
      <td class="num-cell">${fmtMetric(m.diff, m.unit)}</td>
    </tr>`).join("");

  wrap.innerHTML = `
    <div class="stat-grid user-summary-grid">
      <div class="stat-card"><div class="stat-label">Word</div><div class="user-stat-value">${escHtml(row.word)}</div></div>
      <div class="stat-card"><div class="stat-label">Phoneme</div><div class="user-stat-value">/${escHtml(row.phoneme)}/</div></div>
      <div class="stat-card"><div class="stat-label">Score</div><div class="user-stat-value">${score}</div></div>
      <div class="stat-card"><div class="stat-label">Label</div><div class="user-stat-value" style="font-size:22px;color:${_labelColor(row.test_label ?? "NULL", "border")};">${escHtml(label)}</div></div>
    </div>
    <div class="chart-grid">
      <div class="chart-box"><div class="chart-title">사용자 음성 vs 영어 Reference (항목별 정규화)</div><div class="chart-canvas-wrap" style="height:320px;"><canvas id="chart-word-feature-values"></canvas></div></div>
      <div class="chart-box"><div class="chart-title">Reference 대비 비율</div><div class="chart-canvas-wrap" style="height:320px;"><canvas id="chart-word-feature-ratios"></canvas></div></div>
    </div>
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>Metric</th><th>내 발음</th><th>Reference</th><th>Ratio</th><th>Diff</th></tr></thead>
        <tbody>${ratioRows}</tbody>
      </table>
    </div>
    <div class="meta-box" style="margin-top:16px;">
      <strong>그래프 기준:</strong> 왼쪽 그래프는 단위가 다른 Duration/ZCR/RMS/Spectral을 한 화면에서 볼 수 있도록 각 항목별 최대값을 100으로 정규화합니다.
      &nbsp;·&nbsp; <strong>실제 수치:</strong> 아래 표와 tooltip을 기준으로 확인하세요.
      &nbsp;·&nbsp; <strong>MFCC distance:</strong> ${row.mfcc_distance != null ? Number(row.mfcc_distance).toFixed(3) : "—"}
    </div>
  `;
  renderWordFeatureValueChart(metrics);
  renderWordFeatureRatioChart(metrics);
}

function buildWordCompareMetrics(row, ref) {
  const defs = [
    { key: "duration_ms", refKey: "duration_ms", label: "Duration", unit: "ms" },
    { key: "zcr_mean", refKey: "zcr_mean", label: "ZCR", unit: "" },
    { key: "rms_mean", refKey: "rms_mean", label: "RMS", unit: "" },
    { key: "spectral_centroid_mean", refKey: "spectral_centroid_mean", label: "Spectral", unit: "Hz" },
  ];
  return defs.map(d => {
    const user = numOrNull(row[d.key]);
    const reference = numOrNull(ref[d.refKey]);
    return {
      ...d,
      user,
      ref: reference,
      ratio: user != null && reference ? user / reference : null,
      diff: user != null && reference != null ? user - reference : null,
    };
  }).filter(m => m.user != null && m.ref != null);
}

function renderWordFeatureValueChart(metrics) {
  const labels = metrics.map(m => m.label);
  const userData = metrics.map(m => normalizeMetricValue(m.user, m.ref));
  const refData = metrics.map(m => normalizeMetricValue(m.ref, m.user));
  _createChart("chart-word-feature-values", {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "내 발음", data: userData, backgroundColor: "rgba(59,130,246,0.72)", borderColor: "#3b82f6", borderWidth: 1 },
        { label: "Reference", data: refData, backgroundColor: "rgba(148,163,184,0.72)", borderColor: "#94a3b8", borderWidth: 1 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: "top" },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const metric = metrics[ctx.dataIndex];
              const rawValue = ctx.datasetIndex === 0 ? metric.user : metric.ref;
              return ` ${ctx.dataset.label}: ${fmtMetric(rawValue, metric.unit)} (${Number(ctx.raw).toFixed(1)}%)`;
            },
          },
        },
      },
      scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: "항목별 상대 크기 (%)" } } },
    },
  });
}

function normalizeMetricValue(value, otherValue) {
  const v = numOrNull(value);
  const o = numOrNull(otherValue);
  if (v == null) return null;
  const scale = Math.max(Math.abs(v), Math.abs(o ?? 0), 1e-12);
  return Math.abs(v) / scale * 100;
}

function renderWordFeatureRatioChart(metrics) {
  _createChart("chart-word-feature-ratios", {
    type: "bar",
    data: {
      labels: metrics.map(m => m.label),
      datasets: [{ label: "내 발음 / Reference", data: metrics.map(m => m.ratio), backgroundColor: "rgba(16,185,129,0.72)", borderColor: "#10b981", borderWidth: 1 }],
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, title: { display: true, text: "ratio" } } } },
  });
}

// ── 유틸 ─────────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function escJson(obj) { return JSON.stringify(obj).replace(/'/g, "&#39;").replace(/"/g, "&quot;"); }
function ratioCls(ratio) {
  if (ratio === null || ratio === undefined) return "";
  if (ratio > 1.5) return "ratio-high";
  if (ratio < 0.7) return "ratio-low";
  return "";
}
function errorHtml(msg) { return `<div class="no-data-msg" style="border-color:#fca5a5;color:#ef4444;">오류: ${escHtml(msg)}</div>`; }
function missingFileHtml(filename, cmd) { return `<div class="no-data-msg"><strong>${escHtml(filename)}</strong> 파일이 없습니다.<br/>먼저 실행하세요:<code>${escHtml(cmd)}</code></div>`; }
function numOrNull(value) { const n = Number(value); return Number.isFinite(n) ? n : null; }
function fmtNum(value, digits = 2) { const n = numOrNull(value); return n == null ? "—" : n.toFixed(digits); }
function fmtMetric(value, unit = "") {
  const n = numOrNull(value);
  if (n == null) return "—";
  if (unit === "ms") return `${n.toFixed(0)}ms`;
  if (unit === "Hz") return `${n.toFixed(0)}Hz`;
  if (Math.abs(n) < 1) return n.toFixed(5);
  return n.toFixed(2);
}
