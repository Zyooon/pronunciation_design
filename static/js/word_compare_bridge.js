// Label Review -> Word Compare 연결 전용 스크립트
// Word Compare는 analysis_viewer 전용 진단 화면으로 사용한다.
// Label Review에서 선택한 row를 종합 요약 / 영어 reference 비교 / 한국어 패턴 비교 3섹션으로 렌더링한다.

(function () {
  const WORD_COMPARE_PLACEHOLDER = `<div class="no-data-msg">Label Review 탭에서 비교할 단어명을 선택하세요.</div>`;

  const ENGLISH_GROUPS = [
    {
      title: "MFCC",
      subtitle: "음색 / 입 모양",
      keys: [
        ["mfcc_mean_dist", "Mean", "영어 ref 대비 평균 거리"],
        ["mfcc_max_dist", "Max", "영어 ref 대비 최대 오차"],
        ["mfcc_std_dist", "Std", "변화 안정성"],
      ],
    },
    {
      title: "ZCR",
      subtitle: "자음 명확성",
      keys: [
        ["zcr_mean_dist", "Mean", "자음 평균 차이"],
        ["zcr_max_dist", "Max", "자음 최대 오차"],
        ["zcr_std_dist", "Std", "자음 변화 안정성"],
      ],
    },
    {
      title: "RMS",
      subtitle: "강세 / 볼륨",
      keys: [
        ["rms_mean_dist", "Mean", "강세 평균 차이"],
        ["rms_max_dist", "Max", "강세 최대 오차"],
        ["rms_std_dist", "Std", "강세 변화 안정성"],
      ],
    },
  ];

  const KOREAN_PATTERN_KEYS = [
    ["ko_mfcc_mean_dist", "KO MFCC", "입 모양이 한국어식 reference와 얼마나 가까운가"],
    ["ko_zcr_mean_dist", "KO ZCR", "자음이 한국어식 뭉개짐과 얼마나 가까운가"],
    ["ko_rms_mean_dist", "KO RMS", "강세가 한국어식 평탄함과 얼마나 가까운가"],
  ];

  window.loadWordCompare = function loadWordCompareFromLabelReviewOnly() {
    const wrap = document.getElementById("word-compare-wrap");
    if (!wrap) return;
    wrap.innerHTML = WORD_COMPARE_PLACEHOLDER;
  };

  window.openWordCompareFromLabelReview = function openWordCompareFromLabelReview(row) {
    if (!row) return;
    activateAnalysisTab("word-compare");
    renderWordCompareFromRow(normalizeLabelReviewRow(row));
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
    const details = row.details && typeof row.details === "object" ? row.details : {};
    return {
      ...details,
      ...row,
      score: row.score ?? row.final_score ?? details.final_score ?? null,
      final_score: row.final_score ?? row.score ?? details.final_score ?? null,
      test_label: row.test_label || null,
      duration_ratio: row.duration_ratio ?? details.duration_ratio ?? null,
    };
  }

  function renderWordCompareFromRow(row) {
    if (typeof _destroyWordCompareCharts === "function") {
      _destroyWordCompareCharts();
    }
    const wrap = document.getElementById("word-compare-wrap");
    if (!wrap) return;

    const englishMetrics = collectEnglishMetrics(row);
    const koreanMetrics = collectKoreanPatternMetrics(row);
    const hasAnyDetailedMetric = englishMetrics.some((m) => m.value != null) || koreanMetrics.some((m) => m.value != null);

    wrap.innerHTML = `
      ${renderSummarySection(row, englishMetrics, koreanMetrics)}
      ${renderEnglishReferenceSection(englishMetrics)}
      ${renderKoreanPatternSection(koreanMetrics)}
      ${!hasAnyDetailedMetric ? renderMissingMetricNotice() : ""}
    `;

    renderMetricChart("chart-en-reference-dist", englishMetrics.filter((m) => m.value != null), "english");
    renderMetricChart("chart-ko-pattern-dist", koreanMetrics.filter((m) => m.value != null), "korean");
  }

  function collectEnglishMetrics(row) {
    return ENGLISH_GROUPS.flatMap((group) => group.keys.map(([key, shortLabel, description]) => ({
      key,
      group: group.title,
      groupSubtitle: group.subtitle,
      label: `${group.title} ${shortLabel}`,
      description,
      value: numOrNull(row[key]),
    })));
  }

  function collectKoreanPatternMetrics(row) {
    return KOREAN_PATTERN_KEYS.map(([key, label, description]) => ({
      key,
      label,
      description,
      value: numOrNull(row[key]),
    }));
  }

  function renderSummarySection(row, englishMetrics, koreanMetrics) {
    const score = row.final_score ?? row.score;
    const label = row.test_label || "unlabeled";
    const durationRatio = row.duration_ratio;
    const topEnglish = smallestOrLargestMetric(englishMetrics, "max");
    const topKorean = smallestOrLargestMetric(koreanMetrics, "min");

    return `
      <section class="word-compare-section">
        <div class="word-compare-section-head">
          <div>
            <h3>종합 요약</h3>
            <p>Label Review에서 선택한 녹음 row 기준입니다.</p>
          </div>
          <span class="pill pill-muted">Recording ID ${row.id ?? "—"}</span>
        </div>
        <div class="stat-grid user-summary-grid">
          <div class="stat-card"><div class="stat-label">Word</div><div class="user-stat-value">${escHtml(row.word || "")}</div></div>
          <div class="stat-card"><div class="stat-label">Phoneme</div><div class="user-stat-value">/${escHtml(row.phoneme || "")}/</div></div>
          <div class="stat-card"><div class="stat-label">Score</div><div class="user-stat-value">${formatValue(score, 1)}</div></div>
          <div class="stat-card"><div class="stat-label">Label</div><div class="user-stat-value" style="font-size:22px;color:${_labelColor(row.test_label ?? "NULL", "border")};">${escHtml(label)}</div></div>
          <div class="stat-card"><div class="stat-label">Duration Ratio</div><div class="user-stat-value">${formatRatio(durationRatio)}</div></div>
          <div class="stat-card"><div class="stat-label">English Distance Top</div><div class="word-compare-summary-value">${topEnglish ? `${escHtml(topEnglish.label)}<br><strong>${formatValue(topEnglish.value)}</strong>` : "—"}</div></div>
          <div class="stat-card"><div class="stat-label">Korean Pattern Closest</div><div class="word-compare-summary-value ko">${topKorean ? `${escHtml(topKorean.label)}<br><strong>${formatValue(topKorean.value)}</strong>` : "—"}</div></div>
        </div>
      </section>
    `;
  }

  function renderEnglishReferenceSection(metrics) {
    return `
      <section class="word-compare-section">
        <div class="word-compare-section-head">
          <div>
            <h3>영어 Reference 비교</h3>
            <p>낮을수록 영어 reference와 가깝습니다. MFCC/ZCR/RMS 각각 mean · max · std distance를 확인합니다.</p>
          </div>
        </div>
        <div class="word-compare-group-grid">
          ${ENGLISH_GROUPS.map((group) => renderEnglishGroupCard(group, metrics)).join("")}
        </div>
        <div class="chart-box word-compare-chart-box">
          <div class="chart-title">English Reference Distance</div>
          <div class="chart-canvas-wrap" style="height:320px;"><canvas id="chart-en-reference-dist"></canvas></div>
        </div>
      </section>
    `;
  }

  function renderEnglishGroupCard(group, metrics) {
    const rows = group.keys.map(([key, shortLabel, description]) => {
      const metric = metrics.find((m) => m.key === key);
      return `
        <div class="word-compare-metric-row">
          <div><strong>${escHtml(shortLabel)}</strong><span>${escHtml(description)}</span></div>
          <em>${formatValue(metric?.value)}</em>
        </div>`;
    }).join("");
    return `
      <div class="word-compare-card english">
        <div class="word-compare-card-title">${escHtml(group.title)} <span>${escHtml(group.subtitle)}</span></div>
        ${rows}
      </div>
    `;
  }

  function renderKoreanPatternSection(metrics) {
    return `
      <section class="word-compare-section">
        <div class="word-compare-section-head">
          <div>
            <h3>한국어 패턴 비교</h3>
            <p>낮을수록 한국어식 reference 패턴과 가깝습니다. 분석용 지표이며 사용자 화면에는 아직 노출하지 않습니다.</p>
          </div>
        </div>
        <div class="word-compare-group-grid korean-grid">
          ${metrics.map(renderKoreanMetricCard).join("")}
        </div>
        <div class="chart-box word-compare-chart-box korean">
          <div class="chart-title">Korean Pattern Distance</div>
          <div class="chart-canvas-wrap" style="height:280px;"><canvas id="chart-ko-pattern-dist"></canvas></div>
        </div>
      </section>
    `;
  }

  function renderKoreanMetricCard(metric) {
    return `
      <div class="word-compare-card korean">
        <div class="word-compare-card-title">${escHtml(metric.label)}</div>
        <div class="word-compare-korean-value">${formatValue(metric.value)}</div>
        <p>${escHtml(metric.description)}</p>
      </div>
    `;
  }

  function renderMissingMetricNotice() {
    return `
      <div class="no-data-msg" style="margin-top:16px;">
        이 row에는 새 Word Compare metric이 아직 저장되어 있지 않습니다.<br>
        scorer/features 쪽에서 duration_ratio, mfcc/zcr/rms distance, ko_* distance를 details_json에 저장한 뒤 다시 확인하세요.
      </div>
    `;
  }

  function renderMetricChart(canvasId, metrics, mode) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !metrics.length || typeof Chart === "undefined") return;
    const color = mode === "korean" ? "rgba(249,115,22,0.72)" : "rgba(59,130,246,0.72)";
    const border = mode === "korean" ? "#f97316" : "#3b82f6";
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: metrics.map((m) => m.label),
        datasets: [{ label: mode === "korean" ? "KO pattern distance" : "English reference distance", data: metrics.map((m) => m.value), backgroundColor: color, borderColor: border, borderWidth: 1 }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { beginAtZero: true, title: { display: true, text: "Distance" } } },
        plugins: { legend: { display: false } },
      },
    });
  }

  function smallestOrLargestMetric(metrics, mode) {
    const valid = metrics.filter((m) => m.value != null && Number.isFinite(Number(m.value)));
    if (!valid.length) return null;
    return valid.sort((a, b) => mode === "min" ? a.value - b.value : b.value - a.value)[0];
  }

  function formatValue(value, digits = 3) {
    const numeric = numOrNull(value);
    if (numeric == null) return "—";
    return Number(numeric).toFixed(digits).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
  }

  function formatRatio(value) {
    const numeric = numOrNull(value);
    if (numeric == null) return "—";
    return `${formatValue(numeric, 2)}x`;
  }
})();
