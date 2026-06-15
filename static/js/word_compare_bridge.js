// Label Review -> Word Compare 연결 전용 스크립트
// Word Compare는 analysis_viewer 전용 진단 화면으로 사용한다.
// 메인 비교 기준은 선택한 녹음 row vs ElevenLabs 영어 reference 평균이다.

(function () {
  const WORD_COMPARE_PLACEHOLDER = `<div class="no-data-msg">Label Review 탭에서 비교할 단어명을 선택하세요.</div>`;
  const LIQUID_PHONEMES = new Set(["r", "l"]);
  const CONSONANT_PHONEMES = new Set(["θ", "f", "v", "r", "l", "s", "z", "ʃ", "tʃ", "dʒ", "p", "b", "t", "d", "k", "g"]);
  const LIQUID_PHONEMES = new Set(["r", "l"]);
  let referenceRows = [];

  const RAW_REFERENCE_METRICS = [
    ["duration_ms", "duration_ms", "Duration", "ms", "낮거나 높음보다 reference 평균에 가까운지가 중요합니다."],
    ["zcr_mean", "zcr_mean", "ZCR", "raw", "자음 명확성 관련 평균값입니다."],
    ["rms_mean", "rms_mean", "RMS", "raw", "강세/볼륨 관련 평균값입니다."],
    ["spectral_centroid_mean", "spectral_centroid_mean", "Spectral centroid", "Hz", "소리 밝기/주파수 중심 관련 평균값입니다."],
  ];

  const SCORE_METRICS = [
<<<<<<< HEAD
    ["mfcc_score", "MFCC score", "높을수록 음색/입 모양이 reference와 가깝습니다.", "higher", "음색·입 모양"],
    ["duration_score", "Duration score", "높을수록 길이가 reference와 가깝습니다.", "higher", "발음 길이·속도"],
    ["zcr_score", "ZCR score", "높을수록 자음 명확성이 reference와 가깝습니다.", "higher", "자음 명확성"],
    ["rms_score", "RMS score", "높을수록 강세/볼륨이 reference와 가깝습니다.", "higher", "강세·볼륨"],
    ["spectral_centroid_score", "Spectral score", "높을수록 주파수 중심이 reference와 가깝습니다.", "higher", "음색 주파수"],
=======
    ["mfcc_score", "입모양/음색 (MFCC)", "높을수록 음색과 입 모양이 ElevenLabs reference와 가깝습니다.", "higher"],
    ["duration_score", "박자/속도 (Duration)", "높을수록 발음 길이와 박자가 ElevenLabs reference와 가깝습니다.", "higher"],
    ["zcr_score", "자음 명확성 (ZCR)", "높을수록 자음의 마찰/명확성이 ElevenLabs reference와 가깝습니다.", "higher"],
    ["rms_score", "강세/에너지 (RMS)", "높을수록 강세와 에너지 패턴이 ElevenLabs reference와 가깝습니다.", "higher"],
    ["spectral_centroid_score", "소리 밝기 (Spectral)", "높을수록 소리의 밝기와 주파수 중심이 ElevenLabs reference와 가깝습니다.", "higher"],
>>>>>>> cceaca7113463d4bcb9b11e2d30659c628fb5a36
  ];

  const ONSET_METRICS = [
    ["onset_mfcc_mean", "Onset MFCC", "자음 시작 구간의 MFCC 관찰값입니다."],
    ["onset_zcr_mean", "Onset ZCR", "자음 시작 구간의 ZCR 관찰값입니다."],
    ["onset_rms_mean", "Onset RMS", "자음 시작 구간의 RMS 관찰값입니다."],
    ["onset_spectral_centroid_mean", "Onset spectral", "자음 시작 구간의 spectral centroid 관찰값입니다."],
  ];

  const KOREAN_PATTERN_METRICS = [
    ["en_distance", "English distance", "낮을수록 영어 reference와 가깝습니다.", "lower"],
    ["ko_distance", "Korean-like distance", "낮을수록 한국어식 reference와 가깝습니다. 이 값은 점수가 아니라 거리입니다.", "lower"],
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
    const scoreGroups = groupScoreMetrics(row.phoneme, scoreMetrics);
    const mfccDistance = collectMfccDistance(row);
    const onsetMetrics = collectOnsetMetrics(row);
    const koreanMetrics = collectKoreanPatternMetrics(row);
    const shouldShowOnset = isConsonantPhoneme(row.phoneme) && onsetMetrics.some((m) => m.value != null);
    const hasAnyMetric = [...rawMetrics, ...scoreMetrics, ...onsetMetrics, ...koreanMetrics].some((m) => m.userValue != null || m.value != null) || mfccDistance.value != null;

    wrap.innerHTML = `
      ${renderSummarySection(row, englishReference)}
<<<<<<< HEAD
      ${renderEnglishReferenceSection(row, englishReference, rawMetrics, onsetMetrics)}
      ${renderElevenLabsScoreSection(row, scoreMetrics, mfccDistance)}
=======
      ${renderEnglishReferenceSection(row, englishReference, rawMetrics)}
      ${renderElevenLabsScoreSection(row, scoreGroups, mfccDistance)}
      ${shouldShowOnset ? renderOnsetSection(onsetMetrics) : ""}
>>>>>>> cceaca7113463d4bcb9b11e2d30659c628fb5a36
      ${renderKoreanPatternSection(row, koreanMetrics)}
      ${!hasAnyMetric ? renderMissingMetricNotice() : ""}
    `;

    renderReferenceComparisonCharts(rawMetrics.filter((m) => m.userValue != null || m.refValue != null));
<<<<<<< HEAD
    const { usedScores } = groupScoreMetrics(scoreMetrics, row.phoneme);
    renderScoreChart("chart-en-reference-scores", usedScores);
=======
    renderScoreChart("chart-en-reference-scores", scoreGroups.usedScores);
>>>>>>> cceaca7113463d4bcb9b11e2d30659c628fb5a36
    renderScoreChart("chart-ko-pattern", koreanMetrics.filter((m) => m.value != null), "korean");
  }

  function collectRawReferenceMetrics(row, reference) {
    return RAW_REFERENCE_METRICS.map(([userKey, refKey, label, unit, description]) => {
      const userValue = numOrNull(row[userKey]);
      const refValue = numOrNull(reference?.[refKey]);
      const ratio = userValue != null && refValue != null && Math.abs(refValue) > 0 ? userValue / refValue : null;
      return { userKey, refKey, label, unit, description, userValue, refValue, ratio, canvasId: `chart-raw-${userKey}` };
    });
  }

  function collectScoreMetrics(row) {
    return SCORE_METRICS.map(([key, label, description, goodDirection, koreanLabel]) => ({
      key,
      label,
      description,
      goodDirection,
      koreanLabel: koreanLabel || label,
      value: numOrNull(row[key]),
    }));
  }

  function collectMfccDistance(row) {
    return {
      key: "mfcc_distance",
      label: "MFCC distance",
      description: "현재 녹음과 ElevenLabs reference 사이 거리입니다. 0~100 점수가 아니므로 score 그래프에는 넣지 않습니다.",
      goodDirection: "lower",
      value: numOrNull(row["mfcc_distance"]),
    };
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

  function groupScoreMetrics(phoneme, scoreMetrics) {
    const expectedScoreKeys = getExpectedScoreKeys(phoneme);
    return {
      usedScores: scoreMetrics.filter((metric) => expectedScoreKeys.has(metric.key) && metric.value != null),
      unusedScores: scoreMetrics.filter((metric) => !expectedScoreKeys.has(metric.key)),
      missingExpectedScores: scoreMetrics.filter((metric) => expectedScoreKeys.has(metric.key) && metric.value == null),
    };
  }

  function getExpectedScoreKeys(phoneme) {
    const normalizedPhoneme = String(phoneme || "");
    if (LIQUID_PHONEMES.has(normalizedPhoneme)) {
      return new Set(["mfcc_score", "duration_score", "zcr_score", "spectral_centroid_score"]);
    }
    if (CONSONANT_PHONEMES.has(normalizedPhoneme)) {
      return new Set(["mfcc_score", "zcr_score", "spectral_centroid_score"]);
    }
    return new Set(["mfcc_score", "duration_score", "rms_score", "spectral_centroid_score"]);
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

<<<<<<< HEAD
  function renderEnglishReferenceSection(row, englishReference, rawMetrics, onsetMetrics) {
    const shouldShowOnset = isConsonantPhoneme(row.phoneme) && onsetMetrics.some((m) => m.value != null);
=======
  function renderEnglishReferenceSection(row, englishReference, rawMetrics) {
>>>>>>> cceaca7113463d4bcb9b11e2d30659c628fb5a36
    return `
      <section class="word-compare-section">
        <div class="word-compare-section-head">
          <div>
            <h3>영어 Reference 비교</h3>
            <p>현재 녹음 row의 raw feature를 같은 음소의 ElevenLabs 영어 reference 평균과 나란히 비교합니다. 항목마다 스케일이 달라 2×2 개별 그래프로 분리했습니다.</p>
          </div>
          <span class="pill ${englishReference ? "pill-ok" : "pill-warn"}">${englishReference ? "reference found" : "reference missing"}</span>
        </div>
        ${!englishReference ? `<div class="no-data-msg" style="margin-bottom:14px;">/${escHtml(row.phoneme || "")}/에 해당하는 ElevenLabs reference 평균을 찾지 못했습니다.</div>` : ""}
        <div class="chart-grid raw-reference-chart-grid">
          ${rawMetrics.map(renderRawReferenceChartBox).join("")}
        </div>
        <div class="word-compare-group-grid">
          ${rawMetrics.map(renderRawReferenceCard).join("")}
        </div>
<<<<<<< HEAD
        ${shouldShowOnset ? renderOnsetSection(onsetMetrics) : ""}
=======
>>>>>>> cceaca7113463d4bcb9b11e2d30659c628fb5a36
      </section>
    `;
  }

  function renderRawReferenceChartBox(metric) {
    return `
      <div class="chart-box word-compare-chart-box">
        <div class="chart-title">${escHtml(metric.label)} · ${escHtml(metric.unit)}</div>
        <div class="chart-canvas-wrap" style="height:220px;"><canvas id="${escHtml(metric.canvasId)}"></canvas></div>
      </div>
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

<<<<<<< HEAD
  function renderElevenLabsScoreSection(row, scoreMetrics, mfccDistance) {
    const { usedScores, unusedScores, missingExpectedScores } = groupScoreMetrics(scoreMetrics, row.phoneme);
    const finalScore = numOrNull(row.final_score ?? row.score);
    const weakest = findWeakestScore(usedScores);
    const chartHeight = Math.max(160, usedScores.length * 64);
=======
  function renderElevenLabsScoreSection(row, scoreGroups, mfccDistance) {
    const { usedScores, unusedScores, missingExpectedScores } = scoreGroups;
    const chartHeight = Math.max(160, usedScores.length * 60);
>>>>>>> cceaca7113463d4bcb9b11e2d30659c628fb5a36
    return `
      <section class="word-compare-section">
        <div class="word-compare-section-head">
          <div>
            <h3>ElevenLabs 기준 점수</h3>
<<<<<<< HEAD
            <p>scorer가 ElevenLabs reference와 비교해 계산한 결과입니다. score 계열은 0~100점이며 높을수록 좋습니다.</p>
          </div>
        </div>
        ${renderFinalScoreBanner(finalScore)}
        ${weakest ? renderWeakestBanner(weakest) : ""}
        ${usedScores.length ? `
          <div class="chart-box word-compare-chart-box">
            <div class="chart-title">항목별 점수 (0~100)</div>
            <div class="chart-canvas-wrap" style="height:${chartHeight}px;"><canvas id="chart-en-reference-scores"></canvas></div>
          </div>
        ` : ""}
        <details style="margin-top:14px;">
          <summary style="cursor:pointer;font-size:13px;color:#64748b;padding:6px 2px;user-select:none;">개발자 상세 정보 (MFCC distance · 미사용 지표)</summary>
          <div style="margin-top:8px;">
            ${renderMfccDistanceCard(mfccDistance)}
            ${unusedScores.length ? renderUnusedScoresSection(unusedScores) : ""}
            ${missingExpectedScores.length ? renderMissingExpectedScoresSection(missingExpectedScores) : ""}
          </div>
        </details>
      </section>
    `;
  }

  function renderFinalScoreBanner(score) {
    const color = scoreColor(score);
    const label = scoreLabel(score);
    return `
      <div style="display:flex;align-items:center;gap:20px;padding:16px 22px;border-radius:10px;background:#f8fafc;border:2px solid ${color}33;margin-bottom:14px;">
        <div style="font-size:56px;font-weight:800;color:${color};line-height:1;letter-spacing:-2px;">${formatValue(score, 0)}</div>
        <div>
          <div style="font-size:12px;color:#94a3b8;margin-bottom:2px;">종합 점수 / 100</div>
          <div style="font-size:15px;font-weight:700;color:${color};">${escHtml(label)}</div>
        </div>
      </div>
=======
            <p>scorer가 ElevenLabs reference와 비교해 계산한 0~100 점수입니다. 이 섹션은 한국어식 reference 지표와 섞지 않으며, 표시된 score는 모두 높을수록 좋습니다.</p>
          </div>
          <span class="pill pill-ok">higher is better</span>
        </div>
        ${renderMfccDistanceCard(mfccDistance)}
        ${usedScores.length ? `
          <div class="chart-box word-compare-chart-box">
            <div class="chart-title">Score (0~100) · 원어민 일치도</div>
            <div class="chart-canvas-wrap" style="height:${chartHeight}px;"><canvas id="chart-en-reference-scores"></canvas></div>
          </div>
        ` : renderNoUsedScoresNotice(row.phoneme)}
        ${unusedScores.length ? renderUnusedScoresSection(unusedScores) : ""}
        ${missingExpectedScores.length ? renderMissingExpectedScoresSection(missingExpectedScores) : ""}
      </section>
>>>>>>> cceaca7113463d4bcb9b11e2d30659c628fb5a36
    `;
  }

  function renderWeakestBanner(weakest) {
    return `
      <div style="padding:9px 14px;border-radius:8px;background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;font-size:13px;margin-bottom:12px;">
        ⚠ 가장 약한 부분: <strong>${escHtml(weakest.koreanLabel)}</strong> (${escHtml(weakest.label)} ${formatValue(weakest.value, 0)}점)
      </div>
    `;
  }

  function findWeakestScore(usedScores) {
    if (!usedScores.length) return null;
    return usedScores.reduce((min, m) => (m.value < min.value ? m : min), usedScores[0]);
  }

  function renderMfccDistanceCard(mfccDistance) {
    return `
      <div class="word-compare-group-grid" style="margin-bottom:12px;">
        <div class="word-compare-card english">
          <div class="word-compare-card-title">MFCC distance <span>${escHtml(mfccDistance.description)}</span></div>
          <div class="word-compare-korean-value" style="color:#1677c7;">${formatValue(mfccDistance.value)}</div>
          <p style="color:#6b7c8f;font-size:12px;line-height:1.6;">낮을수록 좋음 · 0~100 score 아님</p>
        </div>
      </div>
    `;
  }

  function renderUnusedScoresSection(unusedScores) {
    return `
      <div class="meta-box" style="margin-top:10px;border-color:#e2e8f0;background:#f8fafc;color:#64748b;">
        <strong>사용하지 않은 지표</strong>
        <p style="margin:6px 0 0 0;font-size:12px;line-height:1.7;">이 음소의 scoring recipe에는 포함되지 않아 그래프에 넣지 않습니다.</p>
        <ul style="margin:6px 0 0 0;padding-left:18px;font-size:12px;line-height:1.8;">
          ${unusedScores.map((m) => `<li>${escHtml(m.label)} <code>${escHtml(m.key)}</code></li>`).join("")}
        </ul>
      </div>
    `;
  }

  function renderMissingExpectedScoresSection(missingExpectedScores) {
    return `
<<<<<<< HEAD
      <div class="meta-box" style="margin-top:10px;border-color:#fde68a;background:#fffbeb;color:#92400e;">
        <strong>계산 누락 가능 지표</strong>
        <ul style="margin:6px 0 0 0;padding-left:18px;font-size:12px;line-height:1.8;">
          ${missingExpectedScores.map((m) => `<li>${escHtml(m.label)}: 이 음소 recipe상 사용 대상이지만 row/details_json에 값이 없음</li>`).join("")}
=======
      <div class="meta-box" style="margin-top:10px;border-color:#fecaca;background:#fef2f2;color:#991b1b;">
        <strong>계산 누락 가능 지표</strong>
        <p style="margin:6px 0 0 0;font-size:12px;line-height:1.7;">이 음소의 scoring recipe에서는 사용해야 하지만 현재 row에 값이 없습니다. 재채점 또는 저장 로직 확인이 필요할 수 있습니다.</p>
        <ul style="margin:6px 0 0 0;padding-left:18px;font-size:12px;line-height:1.8;">
          ${missingExpectedScores.map((m) => `<li>${escHtml(m.label)} <code>${escHtml(m.key)}</code></li>`).join("")}
>>>>>>> cceaca7113463d4bcb9b11e2d30659c628fb5a36
        </ul>
      </div>
    `;
  }

<<<<<<< HEAD
=======
  function renderNoUsedScoresNotice(phoneme) {
    return `
      <div class="no-data-msg" style="margin-top:10px;">
        /${escHtml(phoneme || "")}/에 대해 표시할 ElevenLabs 기준 score가 없습니다.
      </div>
    `;
  }

>>>>>>> cceaca7113463d4bcb9b11e2d30659c628fb5a36
  function renderOnsetSection(onsetMetrics) {
    return `
      <section class="word-compare-section">
        <div class="word-compare-section-head">
          <div>
            <h3>Onset 지표</h3>
            <p>자음 시작 구간의 관찰값입니다. 현재는 좋고 나쁨을 직접 판정하지 않고 해석 보조용으로만 사용합니다.</p>
          </div>
          <span class="pill pill-muted">observe only</span>
        </div>
        <div class="word-compare-group-grid">
          ${onsetMetrics.map(renderScoreMetricCard).join("")}
        </div>
      </section>
    `;
  }

  function renderKoreanPatternSection(row, metrics) {
    return `
      <section class="word-compare-section">
        <div class="word-compare-section-head">
          <div>
            <h3>한국어 패턴 비교</h3>
            <p>한국어식 reference는 점수로 더하지 않고 penalty 트랩으로만 사용합니다. 거리와 penalty는 ElevenLabs 0~100 score 그래프와 분리해서 해석합니다.</p>
          </div>
          <span class="pill ${getPatternPillClass(row.korean_pattern_status)}">${escHtml(row.korean_pattern_status || "unknown")}</span>
        </div>
        ${renderKoreanDiagnosis(row)}
        <div class="word-compare-group-grid korean-grid">
          ${metrics.map(renderKoreanMetricCard).join("")}
        </div>
        <div class="chart-box word-compare-chart-box korean">
          <div class="chart-title">Korean pattern metrics · 낮을수록 좋은 값과 높을수록 좋은 값이 섞여 있음</div>
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

  function renderReferenceComparisonCharts(metrics) {
    metrics.forEach((metric) => renderSingleReferenceChart(metric));
  }

  function renderSingleReferenceChart(metric) {
    const canvas = document.getElementById(metric.canvasId);
    if (!canvas || typeof Chart === "undefined") return;
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: ["Recording row", "ElevenLabs avg"],
        datasets: [{
          label: metric.label,
          data: [metric.userValue, metric.refValue],
          backgroundColor: ["rgba(59,130,246,0.72)", "rgba(16,185,129,0.62)"],
          borderColor: ["#3b82f6", "#10b981"],
          borderWidth: 1,
          barPercentage: 0.3,
          categoryPercentage: 0.6,
          maxBarThickness: 22,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { display: false } },
      },
    });
  }

  function renderScoreChart(canvasId, metrics, mode = "english") {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !metrics.length || typeof Chart === "undefined") return;
<<<<<<< HEAD

    let labels, backgroundColors, borderColors, datasetLabel, maxX;
    if (mode === "korean") {
      labels = metrics.map((m) => m.label);
      backgroundColors = "rgba(249,115,22,0.72)";
      borderColors = "#f97316";
      datasetLabel = "Korean pattern";
      maxX = undefined;
    } else {
      labels = metrics.map((m) => [m.koreanLabel || m.label, m.label]);
      backgroundColors = metrics.map((m) => scoreColorAlpha(m.value));
      borderColors = metrics.map((m) => scoreColor(m.value));
      datasetLabel = "Score";
      maxX = 100;
    }

    new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: [{ label: datasetLabel, data: metrics.map((m) => m.value), backgroundColor: backgroundColors, borderColor: borderColors, borderWidth: 1 }],
=======
    const color = mode === "korean" ? "rgba(249,115,22,0.72)" : "rgba(59,130,246,0.72)";
    const border = mode === "korean" ? "#f97316" : "#3b82f6";
    const label = mode === "korean" ? "Korean pattern" : "ElevenLabs score";
    const xScale = mode === "english" ? { beginAtZero: true, max: 100 } : { beginAtZero: true };
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: metrics.map((m) => m.label),
        datasets: [{ label, data: metrics.map((m) => m.value), backgroundColor: color, borderColor: border, borderWidth: 1 }],
>>>>>>> cceaca7113463d4bcb9b11e2d30659c628fb5a36
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
<<<<<<< HEAD
        scales: { x: { beginAtZero: true, ...(maxX != null ? { max: maxX } : {}) } },
=======
        scales: { x: xScale },
>>>>>>> cceaca7113463d4bcb9b11e2d30659c628fb5a36
        plugins: { legend: { display: false } },
      },
    });
  }

  function getExpectedScoreKeys(phoneme) {
    if (LIQUID_PHONEMES.has(String(phoneme || ""))) {
      return new Set(["mfcc_score", "duration_score", "zcr_score", "spectral_centroid_score"]);
    }
    if (CONSONANT_PHONEMES.has(String(phoneme || ""))) {
      return new Set(["mfcc_score", "zcr_score", "spectral_centroid_score"]);
    }
    return new Set(["mfcc_score", "duration_score", "rms_score", "spectral_centroid_score"]);
  }

  function groupScoreMetrics(scoreMetrics, phoneme) {
    const expectedKeys = getExpectedScoreKeys(phoneme);
    const usedScores = scoreMetrics.filter((m) => expectedKeys.has(m.key) && m.value != null);
    const unusedScores = scoreMetrics.filter((m) => !expectedKeys.has(m.key));
    const missingExpectedScores = scoreMetrics.filter((m) => expectedKeys.has(m.key) && m.value == null);
    return { usedScores, unusedScores, missingExpectedScores };
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

  function scoreColor(value) {
    if (value == null) return "#94a3b8";
    if (value < 40) return "#ef4444";
    if (value < 70) return "#f59e0b";
    return "#22c55e";
  }

  function scoreColorAlpha(value) {
    if (value == null) return "rgba(148,163,184,0.5)";
    if (value < 40) return "rgba(239,68,68,0.72)";
    if (value < 70) return "rgba(245,158,11,0.72)";
    return "rgba(34,197,94,0.72)";
  }

  function scoreLabel(value) {
    if (value == null) return "—";
    if (value < 40) return "개선 필요";
    if (value < 70) return "보통";
    return "좋음";
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
