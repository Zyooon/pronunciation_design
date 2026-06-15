// Label Review -> Word Compare 연결 전용 스크립트
// Word Compare는 analysis_viewer 전용 진단 화면으로 사용한다.
// 메인 비교 기준은 선택한 녹음 row vs ElevenLabs 영어 reference 평균이다.

(function () {
  const WORD_COMPARE_PLACEHOLDER = `<div class="no-data-msg">Label Review 탭에서 비교할 단어명을 선택하세요.</div>`;
  const CONSONANT_PHONEMES = new Set(["θ", "f", "v", "r", "l", "s", "z", "ʃ", "tʃ", "dʒ", "p", "b", "t", "d", "k", "g"]);
  let referenceRows = [];

  const RAW_REFERENCE_METRICS = [
    ["duration_ms", "duration_ms", "Duration", "ms", "낮거나 높음보다 reference 평균에 가까운지가 중요합니다."],
    ["zcr_mean", "zcr_mean", "ZCR", "raw", "자음 명확성 관련 평균값입니다."],
    ["rms_mean", "rms_mean", "RMS", "raw", "강세/볼륨 관련 평균값입니다."],
    ["spectral_centroid_mean", "spectral_centroid_mean", "Spectral centroid", "Hz", "소리 밝기/주파수 중심 관련 평균값입니다."],
  ];

  const SCORE_METRICS = [
    ["mfcc_distance", "MFCC distance", "낮을수록 ElevenLabs reference와 가깝습니다.", "lower"],
    ["mfcc_score", "MFCC score", "높을수록 음색/입 모양이 reference와 가깝습니다.", "higher"],
    ["duration_score", "Duration score", "높을수록 길이가 reference와 가깝습니다.", "higher"],
    ["zcr_score", "ZCR score", "높을수록 자음 명확성이 reference와 가깝습니다.", "higher"],
    ["rms_score", "RMS score", "높을수록 강세/볼륨이 reference와 가깝습니다.", "higher"],
    ["spectral_centroid_score", "Spectral score", "높을수록 주파수 중심이 reference와 가깝습니다.", "higher"],
  ];

  const ONSET_METRICS = [
    ["onset_mfcc_mean", "Onset MFCC", "자음 시작 구간의 MFCC 관찰값입니다."],
    ["onset_zcr_mean", "Onset ZCR", "자음 시작 구간의 ZCR 관찰값입니다."],
    ["onset_rms_mean", "Onset RMS", "자음 시작 구간의 RMS 관찰값입니다."],
    ["onset_spectral_centroid_mean", "Onset spectral", "자음 시작 구간의 spectral centroid 관찰값입니다."],
  ];

  const KOREAN_PATTERN_METRICS = [
    ["en_distance", "English distance", "낮을수록 영어 reference와 가깝습니다.", "lower"],
    ["ko_distance", "Korean-like distance", "높을수록 한국어식 reference와 멀어져 더 좋습니다.", "higher"],
    ["relative_distance_score", "Relative score", "높을수록 영어 reference 쪽에 가까운 경향입니다.", "higher"],
    ["korean_like_penalty", "Korean-like penalty", "낮을수록 좋습니다. 한국어식 패턴 감지 시 감점됩니다.", "lower"],
  ];

  window.loadWordCompare = function loadWordCompareFromLabelReviewOnly() {
    const wrap = document.getElementById("word-compare-wrap");
    if (!wrap) return;
    wrap.innerHTML = WORD_COMPARE_PLACEHOLDER;
  };

  window.openWordCompareFromLabelReview = async function openWordCompareFromLabelReview(row) {
    if (!row) return;
    activateAnalysisTab("word-compare");
    const wrap = document.getElementById("word-compare-wrap");
    if (wrap) wrap.innerHTML = '<div class="spinner-wrap"><div class="spinner"></div></div>';

    try {
      if (!referenceRows.length) {
        referenceRows = await fetchJson("/api/reference-quality");
      }
      const normalizedRow = normalizeLabelReviewRow(row);
      const englishReference = findEnglishReference(normalizedRow.phoneme);
      renderWordCompareFromRow(normalizedRow, englishReference);
    } catch (err) {
      if (wrap) wrap.innerHTML = errorHtml(err.message);
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
    const details = row.details && typeof row.details === "object" ? row.details : {};
    return {
      ...details,
      ...row,
      score: row.score ?? row.final_score ?? details.final_score ?? null,
      final_score: row.final_score ?? row.score ?? details.final_score ?? null,
      test_label: row.test_label || null,
      duration_ratio: row.duration_ratio ?? details.duration_ratio ?? null,
      recording_quality_status: row.recording_quality_status ?? details.recording_quality_status ?? null,
      korean_like_penalty: row.korean_like_penalty ?? details.korean_like_penalty ?? null,
      korean_pattern_diagnosis: row.korean_pattern_diagnosis ?? details.korean_pattern_diagnosis ?? null,
      korean_pattern_status: row.korean_pattern_status ?? details.korean_pattern_status ?? null,
    };
  }

  function findEnglishReference(phoneme) {
    return (referenceRows || []).find((ref) => String(ref.phoneme || "") === String(phoneme || "")) || null;
  }

  function renderWordCompareFromRow(row, englishReference) {
    if (typeof _destroyWordCompareCharts === "function") {
      _destroyWordCompareCharts();
    }
    const wrap = document.getElementById("word-compare-wrap");
    if (!wrap) return;

    const rawMetrics = collectRawReferenceMetrics(row, englishReference);
    const scoreMetrics = collectScoreMetrics(row);
    const onsetMetrics = collectOnsetMetrics(row);
    const koreanMetrics = collectKoreanPatternMetrics(row);
    const hasAnyMetric = [...rawMetrics, ...scoreMetrics, ...onsetMetrics, ...koreanMetrics].some((m) => m.userValue != null || m.value != null);

    wrap.innerHTML = `
      ${renderSummarySection(row, englishReference)}
      ${renderEnglishReferenceSection(row, englishReference, rawMetrics, scoreMetrics, onsetMetrics)}
      ${renderKoreanPatternSection(row, koreanMetrics)}
      ${!hasAnyMetric ? renderMissingMetricNotice() : ""}
    `;

    renderReferenceComparisonChart("chart-en-reference-raw", rawMetrics.filter((m) => m.userValue != null || m.refValue != null));
    renderScoreChart("chart-en-reference-scores", scoreMetrics.filter((m) => m.value != null));
    renderScoreChart("chart-ko-pattern", koreanMetrics.filter((m) => m.value != null), "korean");
  }

  function collectRawReferenceMetrics(row, reference) {
    return RAW_REFERENCE_METRICS.map(([userKey, refKey, label, unit, description]) => {
      const userValue = numOrNull(row[userKey]);
      const refValue = numOrNull(reference?.[refKey]);
      const ratio = userValue != null && refValue != null && Math.abs(refValue) > 0 ? userValue / refValue : null;
      return { userKey, refKey, label, unit, description, userValue, refValue, ratio };
    });
  }

  function collectScoreMetrics(row) {
    return SCORE_METRICS.map(([key, label, description, goodDirection]) => ({
      key,
      label,
      description,
      goodDirection,
      value: numOrNull(row[key]),
    }));
  }

  function collectOnsetMetrics(row) {
    return ONSET_METRICS.map(([key, label, description]) => ({
      key,
      label,
      description,
      goodDirection: "observe",
      value: numOrNull(row[key]),
    }));
  }

  function collectKoreanPatternMetrics(row) {
    return KOREAN_PATTERN_METRICS.map(([key, label, description, goodDirection]) => ({
      key,
      label,
      description,
      goodDirection,
      value: numOrNull(row[key]),
    }));
  }

  function renderSummarySection(row, englishReference) {
    const label = row.test_label || "unlabeled";
    const score = row.final_score ?? row.score;
    return `
      <section class="word-compare-section">
        <div class="word-compare-section-head">
          <div>
            <h3>종합 요약</h3>
            <p>Label Review에서 선택한 녹음 row와 ElevenLabs 영어 reference 평균을 비교합니다.</p>
          </div>
          <span class="pill pill-muted">Recording ID ${row.id ?? "—"}</span>
        </div>
        <div class="stat-grid user-summary-grid">
          <div class="stat-card"><div class="stat-label">Word</div><div class="user-stat-value">${escHtml(row.word || "")}</div></div>
          <div class="stat-card"><div class="stat-label">Phoneme</div><div class="user-stat-value">/${escHtml(row.phoneme || "")}/</div></div>
          <div class="stat-card"><div class="stat-label">Reference</div><div class="user-stat-value" style="font-size:22px;">${englishReference ? "ElevenLabs" : "없음"}</div></div>
          <div class="stat-card"><div class="stat-label">Label</div><div class="user-stat-value" style="font-size:22px;color:${_labelColor(row.test_label ?? "NULL", "border")};">${escHtml(label)}</div></div>
          <div class="stat-card"><div class="stat-label">Score</div><div class="user-stat-value">${formatValue(score, 1)}</div></div>
          <div class="stat-card"><div class="stat-label">Duration Ratio</div><div class="user-stat-value">${formatRatio(row.duration_ratio)}</div></div>
          <div class="stat-card"><div class="stat-label">Recording Quality</div><div class="user-stat-value" style="font-size:22px;">${escHtml(row.recording_quality_status || "—")}</div></div>
          <div class="stat-card"><div class="stat-label">Korean-like Penalty</div><div class="user-stat-value" style="color:#f97316;">${formatValue(row.korean_like_penalty, 1)}</div></div>
        </div>
      </section>
    `;
  }

  function renderEnglishReferenceSection(row, englishReference, rawMetrics, scoreMetrics, onsetMetrics) {
    const shouldShowOnset = isConsonantPhoneme(row.phoneme) && onsetMetrics.some((m) => m.value != null);
    return `
      <section class="word-compare-section">
        <div class="word-compare-section-head">
          <div>
            <h3>영어 Reference 비교</h3>
            <p>현재 녹음 row의 raw feature를 같은 음소의 ElevenLabs 영어 reference 평균과 나란히 비교합니다.</p>
          </div>
          <span class="pill ${englishReference ? "pill-ok" : "pill-warn"}">${englishReference ? "reference found" : "reference missing"}</span>
        </div>
        ${!englishReference ? `<div class="no-data-msg" style="margin-bottom:14px;">/${escHtml(row.phoneme || "")}/에 해당하는 ElevenLabs reference 평균을 찾지 못했습니다.</div>` : ""}
        <div class="word-compare-group-grid">
          ${rawMetrics.map(renderRawReferenceCard).join("")}
        </div>
        <div class="chart-box word-compare-chart-box">
          <div class="chart-title">Recording row vs ElevenLabs average</div>
          <div class="chart-canvas-wrap" style="height:320px;"><canvas id="chart-en-reference-raw"></canvas></div>
        </div>
        <div class="word-compare-section-head" style="margin-top:18px;">
          <div>
            <h3 style="font-size:14px;">Reference 기반 점수/거리</h3>
            <p>MFCC distance는 낮을수록 좋고, score 계열은 높을수록 좋습니다.</p>
          </div>
        </div>
        <div class="word-compare-group-grid">
          ${scoreMetrics.map(renderScoreMetricCard).join("")}
        </div>
        <div class="chart-box word-compare-chart-box">
          <div class="chart-title">Reference-derived scores and distance</div>
          <div class="chart-canvas-wrap" style="height:320px;"><canvas id="chart-en-reference-scores"></canvas></div>
        </div>
        ${shouldShowOnset ? renderOnsetSection(onsetMetrics) : ""}
      </section>
    `;
  }

  function renderRawReferenceCard(metric) {
    return `
      <div class="word-compare-card english">
        <div class="word-compare-card-title">${escHtml(metric.label)} <span>${escHtml(metric.description)}</span></div>
        <div class="word-compare-metric-row"><div><strong>Recording row</strong><span>현재 선택한 녹음</span></div><em>${formatMetricValue(metric.userValue, metric.unit)}</em></div>
        <div class="word-compare-metric-row"><div><strong>ElevenLabs avg</strong><span>영어 reference 평균</span></div><em>${formatMetricValue(metric.refValue, metric.unit)}</em></div>
        <div class="word-compare-metric-row"><div><strong>Ratio</strong><span>recording / reference</span></div><em>${metric.ratio != null ? formatValue(metric.ratio, 2) + "x" : "—"}</em></div>
      </div>
    `;
  }

  function renderScoreMetricCard(metric) {
    return `
      <div class="word-compare-card english">
        <div class="word-compare-card-title">${escHtml(metric.label)} <span>${escHtml(metric.description)}</span></div>
        <div class="word-compare-korean-value" style="color:${metric.goodDirection === "lower" ? "#1677c7" : "#10b981"};">${formatValue(metric.value)}</div>
        <p style="color:#6b7c8f;font-size:12px;line-height:1.6;">${escHtml(directionText(metric.goodDirection))}</p>
      </div>
    `;
  }

  function renderOnsetSection(onsetMetrics) {
    return `
      <div class="word-compare-section-head" style="margin-top:18px;">
        <div>
          <h3 style="font-size:14px;">Onset 지표</h3>
          <p>자음 시작 구간의 관찰값입니다. 현재는 좋고 나쁨을 직접 판정하지 않고 해석 보조용으로만 사용합니다.</p>
        </div>
      </div>
      <div class="word-compare-group-grid">
        ${onsetMetrics.map(renderScoreMetricCard).join("")}
      </div>
    `;
  }

  function renderKoreanPatternSection(row, metrics) {
    return `
      <section class="word-compare-section">
        <div class="word-compare-section-head">
          <div>
            <h3>한국어 패턴 비교</h3>
            <p>한국어식 reference는 점수로 더하지 않고 penalty 트랩으로만 사용합니다.</p>
          </div>
          <span class="pill ${getPatternPillClass(row.korean_pattern_status)}">${escHtml(row.korean_pattern_status || "unknown")}</span>
        </div>
        ${renderKoreanDiagnosis(row)}
        <div class="word-compare-group-grid korean-grid">
          ${metrics.map(renderKoreanMetricCard).join("")}
        </div>
        <div class="chart-box word-compare-chart-box korean">
          <div class="chart-title">Korean pattern metrics</div>
          <div class="chart-canvas-wrap" style="height:280px;"><canvas id="chart-ko-pattern"></canvas></div>
        </div>
      </section>
    `;
  }

  function renderKoreanDiagnosis(row) {
    const diagnosis = row.korean_pattern_diagnosis || buildKoreanDiagnosis(row);
    const policy = row.korean_pattern_penalty_policy || "penalty_only";
    return `
      <div class="meta-box" style="margin-bottom:14px;border-color:#fed7aa;background:#fff7ed;color:#9a3412;">
        <strong>진단:</strong> ${escHtml(diagnosis)}<br>
        <strong>정책:</strong> ${escHtml(policy)} · 한국어 reference는 최종 점수에 직접 더하지 않고 korean_like_penalty로만 반영합니다.
      </div>
    `;
  }

  function renderKoreanMetricCard(metric) {
    return `
      <div class="word-compare-card korean">
        <div class="word-compare-card-title">${escHtml(metric.label)}</div>
        <div class="word-compare-korean-value">${formatValue(metric.value)}</div>
        <p>${escHtml(metric.description)}</p>
        <p style="margin-top:6px;font-weight:700;color:#9a3412;">${escHtml(directionText(metric.goodDirection))}</p>
      </div>
    `;
  }

  function renderMissingMetricNotice() {
    return `
      <div class="no-data-msg" style="margin-top:16px;">
        이 row에는 Word Compare에 표시할 metric이 아직 저장되어 있지 않습니다.<br>
        새 녹음 또는 재채점 후 다시 확인하세요.
      </div>
    `;
  }

  function renderReferenceComparisonChart(canvasId, metrics) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !metrics.length || typeof Chart === "undefined") return;
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: metrics.map((m) => m.label),
        datasets: [
          { label: "Recording row", data: metrics.map((m) => m.userValue), backgroundColor: "rgba(59,130,246,0.72)", borderColor: "#3b82f6", borderWidth: 1 },
          { label: "ElevenLabs avg", data: metrics.map((m) => m.refValue), backgroundColor: "rgba(16,185,129,0.62)", borderColor: "#10b981", borderWidth: 1 },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { beginAtZero: true } },
      },
    });
  }

  function renderScoreChart(canvasId, metrics, mode = "english") {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !metrics.length || typeof Chart === "undefined") return;
    const color = mode === "korean" ? "rgba(249,115,22,0.72)" : "rgba(59,130,246,0.72)";
    const border = mode === "korean" ? "#f97316" : "#3b82f6";
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: metrics.map((m) => m.label),
        datasets: [{ label: mode === "korean" ? "Korean pattern" : "Score / distance", data: metrics.map((m) => m.value), backgroundColor: color, borderColor: border, borderWidth: 1 }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { beginAtZero: true } },
        plugins: { legend: { display: false } },
      },
    });
  }

  function isConsonantPhoneme(phoneme) {
    return CONSONANT_PHONEMES.has(String(phoneme || ""));
  }

  function getPatternPillClass(status) {
    if (status === "korean_like") return "pill-err";
    if (status === "borderline_korean_like") return "pill-warn";
    if (status === "english_like") return "pill-ok";
    return "pill-muted";
  }

  function buildKoreanDiagnosis(row) {
    const enDistance = numOrNull(row.en_distance);
    const koDistance = numOrNull(row.ko_distance);
    const phoneme = row.phoneme ? `/${row.phoneme}/` : "영어 reference";
    if (enDistance == null || koDistance == null) {
      return "한국어식 reference 비교 데이터가 아직 없습니다.";
    }
    if (koDistance < enDistance) {
      return `현재 발음은 영어 ${phoneme} reference보다 한국어식 패턴에 더 가까운 경향이 있습니다.`;
    }
    return `현재 발음은 한국어식 reference보다 영어 ${phoneme} reference에 더 가까운 경향이 있습니다.`;
  }

  function directionText(direction) {
    if (direction === "lower") return "낮을수록 좋음";
    if (direction === "higher") return "높을수록 좋음";
    return "관찰값";
  }

  function formatMetricValue(value, unit) {
    const numeric = numOrNull(value);
    if (numeric == null) return "—";
    if (unit === "ms") return `${formatValue(numeric, 1)}ms`;
    if (unit === "Hz") return `${formatValue(numeric, 1)}Hz`;
    return formatValue(numeric, 6);
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
