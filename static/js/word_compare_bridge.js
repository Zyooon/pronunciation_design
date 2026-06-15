// Label Review -> Word Compare 연결 전용 스크립트
// 기존 analysis_viewer.js의 드롭다운 기반 Word Compare 초기화를 덮어쓰고,
// Label Review에서 선택한 row만 Word Compare에 렌더링한다.

(function () {
  const WORD_COMPARE_PLACEHOLDER = `<div class="no-data-msg">Label Review 탭에서 비교할 단어명을 선택하세요.</div>`;
  let referenceRows = [];

  window.loadWordCompare = function loadWordCompareFromLabelReviewOnly() {
    const wrap = document.getElementById("word-compare-wrap");
    if (!wrap) return;
    wrap.innerHTML = WORD_COMPARE_PLACEHOLDER;
  };

  window.openWordCompareFromLabelReview = async function openWordCompareFromLabelReview(row) {
    if (!row) return;
    const wrap = document.getElementById("word-compare-wrap");
    if (!wrap) return;

    activateAnalysisTab("word-compare");
    wrap.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';

    try {
      if (!referenceRows.length) {
        referenceRows = await fetchJson("/api/reference-quality");
      }

      const comparableRow = normalizeLabelReviewRow(row);
      if (!hasComparableFeature(comparableRow)) {
        wrap.innerHTML = `<div class="no-data-msg">${escHtml(comparableRow.word || "선택한 단어")} row에 비교 가능한 feature 값이 없습니다.</div>`;
        return;
      }

      renderWordCompareFromRow(comparableRow);
    } catch (err) {
      wrap.innerHTML = errorHtml(err.message);
    }
  };

  function activateAnalysisTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabId);
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
    const panel = document.getElementById(`tab-${tabId}`);
    panel?.classList.add("active");
    panel?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function normalizeLabelReviewRow(row) {
    return {
      ...row,
      score: row.score ?? row.final_score ?? null,
      final_score: row.final_score ?? row.score ?? null,
      test_label: row.test_label || null,
    };
  }

  function renderWordCompareFromRow(row) {
    if (typeof _destroyWordCompareCharts === "function") {
      _destroyWordCompareCharts();
    }
    const wrap = document.getElementById("word-compare-wrap");
    const ref = referenceRows.find((r) => r.phoneme === row.phoneme);
    if (!ref) {
      wrap.innerHTML = `<div class="no-data-msg">/${escHtml(row.phoneme || "")}/ reference 정보가 없습니다.</div>`;
      return;
    }

    const metrics = buildWordCompareMetrics(row, ref);
    if (!metrics.length) {
      wrap.innerHTML = `<div class="no-data-msg">${escHtml(row.word || "선택한 단어")} row와 reference 사이에 비교 가능한 공통 feature가 없습니다.</div>`;
      return;
    }

    const score = row.score != null ? Number(row.score).toFixed(1) : "--";
    const label = row.test_label || "unlabeled";
    const ratioRows = metrics.map((m) => `
      <tr>
        <td>${escHtml(m.label)}</td>
        <td class="num-cell">${fmtMetric(m.user, m.unit)}</td>
        <td class="num-cell">${fmtMetric(m.ref, m.unit)}</td>
        <td class="num-cell ${ratioCls(m.ratio)}">${m.ratio != null ? m.ratio.toFixed(2) + "x" : "—"}</td>
        <td class="num-cell">${fmtMetric(m.diff, m.unit)}</td>
      </tr>`).join("");

    wrap.innerHTML = `
      <div class="stat-grid user-summary-grid">
        <div class="stat-card"><div class="stat-label">Word</div><div class="user-stat-value">${escHtml(row.word || "")}</div></div>
        <div class="stat-card"><div class="stat-label">Phoneme</div><div class="user-stat-value">/${escHtml(row.phoneme || "")}/</div></div>
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
        <strong>진입 경로:</strong> Label Review에서 선택한 녹음 row 기준입니다.
        &nbsp;·&nbsp; <strong>녹음 ID:</strong> ${row.id ?? "—"}
        &nbsp;·&nbsp; <strong>MFCC distance:</strong> ${row.mfcc_distance != null ? Number(row.mfcc_distance).toFixed(3) : "—"}
      </div>
    `;

    renderWordFeatureValueChart(metrics);
    renderWordFeatureRatioChart(metrics);
  }
})();
