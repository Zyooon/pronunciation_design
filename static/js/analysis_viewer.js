/**
 * 발음교정기 분석 뷰어 — 탭 전환, API 조회, 테이블 렌더링
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

// ── 초기 데이터 로딩 ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadOverview();
  loadReferenceQuality();
  loadPhonemeAnalysis();
  loadWordResults();
  loadOutliers();
  loadErrors();
  loadUserResults();
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
    { label: "Reference 발음 수",    value: data.reference_phoneme_count,  cls: "" },
    { label: "Reference Sample",     value: data.reference_sample_count,   cls: "" },
    { label: "Reference 단어 수",    value: data.reference_word_count,     cls: "" },
    { label: "Comparison 전체",      value: data.comparison_total,         cls: "" },
    { label: "Comparison 성공",      value: data.comparison_success,       cls: "ok" },
    { label: "Comparison 실패",      value: data.comparison_error,         cls: data.comparison_error > 0 ? "err" : "" },
    { label: "분석된 발음 수",        value: data.analyzed_phoneme_count,   cls: "" },
    { label: "Outlier 후보",         value: data.outlier_count,            cls: data.outlier_count > 0 ? "warn" : "" },
  ];

  document.getElementById("overview-cards").innerHTML = cards
    .map(({ label, value, cls }) => `
      <div class="stat-card ${cls}">
        <div class="stat-label">${label}</div>
        <div class="stat-value">${value ?? "—"}</div>
      </div>
    `)
    .join("");
}

function renderOverviewMeta(meta) {
  const el = document.getElementById("overview-meta");
  if (!meta || !Object.keys(meta).length) {
    el.innerHTML = "<em>metadata 없음</em>";
    return;
  }
  const rows = Object.entries(meta)
    .map(([k, v]) => `<strong>${k}:</strong> ${v ?? "—"}`)
    .join(" &nbsp;·&nbsp; ");
  el.innerHTML = rows;
}

// ── Reference Quality ─────────────────────────────────────────────────────────
let _refRows = [];

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
        <td class="num-cell">${r.duration_ms}</td>
        <td class="num-cell">${r.duration_std}</td>
        <td class="num-cell">${r.zcr_mean}</td>
        <td class="num-cell">${r.zcr_std}</td>
        <td class="num-cell">${r.rms_mean}</td>
        <td class="num-cell">${r.rms_std}</td>
        <td class="num-cell">${r.spectral_centroid_mean}</td>
        <td class="num-cell">${r.spectral_centroid_std}</td>
        <td>${warnHtml}</td>
      </tr>`;
  }).join("");

  tbody.querySelectorAll("tr.clickable").forEach((tr) => {
    tr.addEventListener("click", () => onRefRowClick(tr));
  });
}

function onRefRowClick(tr) {
  const phoneme = tr.dataset.phoneme;
  const row = _refRows.find((r) => r.phoneme === phoneme);
  if (!row) return;

  document.querySelectorAll("#ref-tbody tr").forEach((r) => r.classList.remove("selected"));
  tr.classList.add("selected");

  const panel = document.getElementById("ref-detail-panel");
  const words = row.test_words || [];
  const koreans = row.korean_pronunciations || {};

  const wordChips = words
    .map((w) => {
      const ko = koreans[w] ? ` <span style="color:#6b7c8f;font-size:11px;">(${escHtml(koreans[w])})</span>` : "";
      return `<span class="word-chip">${escHtml(w)}${ko}</span>`;
    })
    .join("");

  panel.innerHTML = `
    <h3>/${escHtml(phoneme)}/ — ${escHtml(row.phoneme_type)}</h3>
    <div class="col-title">Test Words (${words.length}개)</div>
    <div class="word-chip-list">${wordChips || "<em>없음</em>"}</div>
  `;
  panel.style.display = "block";
}

// ── Phoneme Analysis ──────────────────────────────────────────────────────────
let _phonemeRows = [];

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
  const phonemes = [...new Set(rows.map((r) => r.phoneme))].sort();
  phonemes.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = `/${p}/`;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", applyPhonemeFilter);
  document.getElementById("filter-outlier-only").addEventListener("change", applyPhonemeFilter);
}

function applyPhonemeFilter() {
  const phonemeVal  = document.getElementById("filter-phoneme").value;
  const outlierOnly = document.getElementById("filter-outlier-only").checked;

  const filtered = _phonemeRows.filter((r) => {
    if (phonemeVal && r.phoneme !== phonemeVal) return false;
    if (outlierOnly && !r.outlier_count) return false;
    return true;
  });

  renderPhonemeTable(filtered);
}

function renderPhonemeTable(rows) {
  const noDataEl = document.getElementById("phoneme-no-data");
  const tbody    = document.getElementById("phoneme-tbody");

  if (!rows.length) {
    noDataEl.style.display = "block";
    noDataEl.innerHTML = `
      comparison 성공 결과가 없습니다.
      <code>uv run python scripts/compare_en_ko.py</code>
    `;
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
    </tr>
  `).join("");
}

// ── Word Results ──────────────────────────────────────────────────────────────
let _wordRows = [];

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
  const phonemes = [...new Set(rows.map((r) => r.phoneme))].sort();
  phonemes.forEach((p) => {
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
  const searchVal  = document.getElementById("filter-word-search").value.toLowerCase().trim();

  const filtered = _wordRows.filter((r) => {
    if (phonemeVal && r.phoneme !== phonemeVal) return false;
    if (searchVal && !r.word.toLowerCase().includes(searchVal)) return false;
    return true;
  });

  renderWordTable(filtered);
}

function renderWordTable(rows) {
  const tbody = document.getElementById("word-tbody");

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="11" style="padding:24px;text-align:center;color:#6b7c8f;">성공한 비교 결과가 없습니다.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map((r) => `
    <tr class="clickable"
        data-word="${escHtml(r.word)}"
        data-phoneme="${escHtml(r.phoneme)}"
        data-row='${escJson(r)}'>
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
    </tr>
  `).join("");

  tbody.querySelectorAll("tr.clickable").forEach((tr) => {
    tr.addEventListener("click", () => onWordRowClick(tr));
  });
}

function onWordRowClick(tr) {
  document.querySelectorAll("#word-tbody tr").forEach((r) => r.classList.remove("selected"));
  tr.classList.add("selected");

  const r = JSON.parse(tr.dataset.row);
  const panel = document.getElementById("word-detail-panel");

  const metrics = [
    { name: "Duration",          en: r.en_duration_ms,              ko: r.ko_duration_ms,              ratio: r.duration_ko_en_ratio,           unit: "ms" },
    { name: "ZCR",               en: r.en_zcr_mean,                 ko: r.ko_zcr_mean,                 ratio: r.zcr_ko_en_ratio,                unit: "" },
    { name: "RMS",               en: r.en_rms_mean,                 ko: r.ko_rms_mean,                 ratio: r.rms_ko_en_ratio,                unit: "" },
    { name: "Spectral Centroid", en: r.en_spectral_centroid_mean,   ko: r.ko_spectral_centroid_mean,   ratio: r.spectral_centroid_ko_en_ratio,  unit: "Hz" },
  ];

  const metricHtml = metrics.map(({ name, en, ko, ratio, unit }) => {
    const ratioStr = ratio !== null && ratio !== undefined ? `${ratio}x` : "—";
    const enStr = en !== null && en !== undefined ? `${en}${unit ? unit : ""}` : "—";
    const koStr = ko !== null && ko !== undefined ? `${ko}${unit ? unit : ""}` : "—";
    const hasEnKo = en !== null && en !== undefined;

    return `
      <div class="detail-metric-block">
        <div class="metric-name">${name}</div>
        ${hasEnKo ? `<div class="metric-sub">EN: ${enStr} / KO: ${koStr}</div>` : ""}
        <div class="metric-sub">ko/en ratio: <strong class="${ratioCls(ratio)}">${ratioStr}</strong></div>
      </div>`;
  }).join("");

  panel.innerHTML = `
    <h3>${escHtml(r.word)} <span style="font-weight:400;font-size:14px;color:#6b7c8f;">/${escHtml(r.phoneme)}/ · ${escHtml(r.korean_pronunciation)}</span></h3>
    <div class="detail-row">
      <div class="detail-col">
        <div class="col-title">Feature Ratios</div>
        ${metricHtml}
      </div>
      <div class="detail-col">
        <div class="col-title">MFCC</div>
        <div class="detail-metric-block">
          <div class="metric-sub">distance: <strong>${r.mfcc_distance ?? "—"}</strong></div>
          <div class="metric-sub">cosine distance: <strong>${r.mfcc_cosine_distance ?? "—"}</strong></div>
        </div>
      </div>
    </div>
  `;
  panel.style.display = "block";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Outliers ──────────────────────────────────────────────────────────────────
async function loadOutliers() {
  try {
    const rows = await fetchJson("/api/outliers");
    renderOutlierTable(rows);
  } catch (err) {
    document.getElementById("outlier-tbody").innerHTML = `<tr><td colspan="7">${errorHtml(err.message)}</td></tr>`;
  }
}

function renderOutlierTable(rows) {
  const tbody = document.getElementById("outlier-tbody");

  if (!rows.length) {
    tbody.innerHTML = `
      <tr><td colspan="7" style="padding:20px;text-align:center;color:#6b7c8f;">
        outlier 후보가 없거나 comparison_analysis_report.json이 없습니다.<br/>
        <span style="font-size:12px;">uv run python scripts/analyze_comparison_results.py</span>
      </td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map((r) => `
    <tr>
      <td><strong>${escHtml(r.word)}</strong></td>
      <td class="phoneme-cell">/${escHtml(r.phoneme)}/</td>
      <td>${escHtml(r.metric_name)}</td>
      <td class="num-cell">${r.value ?? "—"}</td>
      <td class="num-cell">${r.average ?? "—"}</td>
      <td class="num-cell">${r.stdev ?? "—"}</td>
      <td class="num-cell ${Math.abs(r.z_score) > 2 ? "ratio-high" : ""}">${r.z_score ?? "—"}</td>
    </tr>
  `).join("");
}

// ── Errors ────────────────────────────────────────────────────────────────────
async function loadErrors() {
  try {
    const rows = await fetchJson("/api/errors");
    renderErrorTable(rows);
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
    <tr>
      <td><strong>${escHtml(r.word)}</strong></td>
      <td>${escHtml(r.korean_pronunciation)}</td>
      <td class="phoneme-cell">/${escHtml(r.phoneme)}/</td>
      <td class="warn-cell">${escHtml(r.error_message)}</td>
      <td>${r.en_audio_exists ? '<span class="pill pill-ok">✓</span>' : '<span class="pill pill-err">✗</span>'}</td>
      <td>${r.ko_audio_exists ? '<span class="pill pill-ok">✓</span>' : '<span class="pill pill-err">✗</span>'}</td>
    </tr>
  `).join("");
}

// ── User Results ──────────────────────────────────────────────────────────────
async function loadUserResults() {
  const wrap = document.getElementById("user-results-wrap");

  try {
    const data = await fetchJson("/api/user-results");

    if (!data.exists) {
      wrap.innerHTML = `
        <div class="no-data-msg">
          data/results.csv 파일이 없습니다.<br/>
          실제 녹음 후 결과 저장이 되는지 확인하세요.
        </div>`;
      return;
    }

    if (data.error) {
      wrap.innerHTML = `
        <div class="no-data-msg">
          results.csv 읽기 오류: ${escHtml(data.error)}<br/>
          path: ${escHtml(data.path)}
        </div>`;
      return;
    }

    if (!data.rows || data.rows.length === 0) {
      wrap.innerHTML = `
        <div class="no-data-msg">
          results.csv 파일은 있지만 저장된 row가 없습니다.<br/>
          path: ${escHtml(data.path)}
        </div>`;
      return;
    }

    const columns = data.columns && data.columns.length
      ? data.columns
      : Object.keys(data.rows[0]);

    const theadHtml = columns.map((c) => `<th>${escHtml(c)}</th>`).join("");
    const tbodyHtml = data.rows.map((r) =>
      `<tr>${columns.map((c) => `<td>${escHtml(String(r[c] ?? ""))}</td>`).join("")}</tr>`
    ).join("");

    wrap.innerHTML = `
      <div class="meta-box">
        <strong>path:</strong> ${escHtml(data.path)}
        &nbsp;·&nbsp;
        <strong>rows:</strong> ${data.row_count}
        &nbsp;·&nbsp;
        <strong>columns:</strong> ${columns.length}
      </div>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr>${theadHtml}</tr></thead>
          <tbody>${tbodyHtml}</tbody>
        </table>
      </div>`;
  } catch (err) {
    wrap.innerHTML = errorHtml(err.message);
  }
}

// ── 유틸 ─────────────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escJson(obj) {
  return JSON.stringify(obj).replace(/'/g, "&#39;").replace(/"/g, "&quot;");
}

function ratioCls(ratio) {
  if (ratio === null || ratio === undefined) return "";
  if (ratio > 1.5) return "ratio-high";
  if (ratio < 0.7) return "ratio-low";
  return "";
}

function errorHtml(msg) {
  return `<div class="no-data-msg" style="border-color:#fca5a5;color:#ef4444;">오류: ${escHtml(msg)}</div>`;
}

function missingFileHtml(filename, cmd) {
  return `
    <div class="no-data-msg">
      <strong>${escHtml(filename)}</strong> 파일이 없습니다.<br/>먼저 실행하세요:
      <code>${escHtml(cmd)}</code>
    </div>`;
}
