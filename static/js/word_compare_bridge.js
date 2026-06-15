// Label Review -> Word Compare 연결 전용 스크립트
// 기존 analysis_viewer.js의 드롭다운 기반 Word Compare 초기화를 덮어쓰고,
// Label Review에서 선택한 row만 Word Compare에 렌더링한다.

(function () {
  const WORD_COMPARE_PLACEHOLDER = `<div class="no-data-msg">Label Review 탭에서 비교할 단어명을 선택하세요.</div>`;

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
      if (!Array.isArray(window._refRows) || !window._refRows.length) {
        window._refRows = await fetchJson("/api/reference-quality");
      }

      const comparableRow = normalizeLabelReviewRow(row);
      if (!hasComparableFeature(comparableRow)) {
        wrap.innerHTML = `<div class="no-data-msg">${escHtml(comparableRow.word || "선택한 단어")} row에 비교 가능한 feature 값이 없습니다.</div>`;
        return;
      }

      if (typeof window.renderWordCompareByRow === "function") {
        window.renderWordCompareByRow(comparableRow);
        return;
      }

      renderWordCompareRowFallback(comparableRow);
    } catch (err) {
      wrap.innerHTML = errorHtml(err.message);
    }
  };

  window.renderWordCompareByRow = function renderWordCompareByRow(row) {
    window._wordCompareRows = [row];
    renderWordCompareByKey(0);
  };

  function activateAnalysisTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabId);
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
    document.getElementById(`tab-${tabId}`)?.classList.add("active");
    document.getElementById(`tab-${tabId}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function normalizeLabelReviewRow(row) {
    return {
      ...row,
      score: row.score ?? row.final_score ?? null,
      final_score: row.final_score ?? row.score ?? null,
      test_label: row.test_label || null,
    };
  }

  function renderWordCompareRowFallback(row) {
    const wrap = document.getElementById("word-compare-wrap");
    const ref = (window._refRows || []).find((r) => r.phoneme === row.phoneme);
    if (!ref) {
      wrap.innerHTML = `<div class="no-data-msg">/${escHtml(row.phoneme || "")}/ reference 정보가 없습니다.</div>`;
      return;
    }
    const metrics = buildWordCompareMetrics(row, ref);
    const rows = metrics.map((m) => `
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
        <div class="stat-card"><div class="stat-label">Score</div><div class="user-stat-value">${row.score != null ? Number(row.score).toFixed(1) : "--"}</div></div>
        <div class="stat-card"><div class="stat-label">Label</div><div class="user-stat-value" style="font-size:22px;">${escHtml(row.test_label || "unlabeled")}</div></div>
      </div>
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Metric</th><th>내 발음</th><th>Reference</th><th>Ratio</th><th>Diff</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }
})();
