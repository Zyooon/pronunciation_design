/**
 * 화면 이벤트 연결과 상태 관리를 담당한다.
 */

// ── 상태 ──────────────────────────────────────────────
let selectedWord = null;   // { word, korean_pronunciation, phoneme }
let recordedBlob = null;   // Blob | null
let uploadedFile = null;   // File | null
let allWords     = [];     // 전체 단어 목록 (발음 필터링용)

// ── DOM 참조 ──────────────────────────────────────────
const phonemeSelect    = document.getElementById("phoneme-select");
const wordSelect       = document.getElementById("word-select");
const targetWordEl     = document.getElementById("target-word");
const targetPhonemeEl  = document.getElementById("target-phoneme");
const targetKoreanEl   = document.getElementById("target-korean");
const audioPreview     = document.getElementById("audio-preview");
const recZone          = document.getElementById("record-zone");
const recStatus        = document.getElementById("rec-status");
const btnRecord        = document.getElementById("btn-record");
const btnRecordLabel   = document.getElementById("btn-record-label");
const fileUpload       = document.getElementById("file-upload");
const audioSourceBar   = document.getElementById("audio-source-bar");
const sourceBadge      = document.getElementById("source-badge");
const sourceName       = document.getElementById("source-name");
const btnAnalyze       = document.getElementById("btn-analyze");
const resultSection    = document.getElementById("result-section");
const loadingOverlay   = document.getElementById("loading-overlay");
const errorMessage     = document.getElementById("error-message");
const btnRetry         = document.getElementById("btn-retry");

// ── 초기화 ────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", initApp);

async function initApp() {
  checkBrowserSupport();
  await loadWordList();
  bindEvents();
}

function checkBrowserSupport() {
  const hasMediaDevices = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  const hasMediaRecorder = typeof MediaRecorder !== "undefined";
  if (!hasMediaDevices || !hasMediaRecorder) {
    btnRecord.disabled = true;
    btnRecord.title = "이 브라우저는 마이크 녹음을 지원하지 않습니다";
    recStatus.textContent = "녹음 미지원 — 파일 업로드를 이용해주세요";
  }
}

// ── 단어 목록 로딩 ────────────────────────────────────
async function loadWordList() {
  try {
    const words = await fetchWords();
    if (!words.length) {
      phonemeSelect.innerHTML = '<option value="">발음 없음</option>';
      showError("data/words.txt에 단어가 없습니다. 파일을 확인해주세요.");
      return;
    }
    allWords = words;
    renderPhonemeOptions(words);
  } catch (err) {
    phonemeSelect.innerHTML = '<option value="">불러오기 실패</option>';
    showError(err.message || "단어 목록을 불러오지 못했습니다. 서버를 확인해주세요.");
  }
}

/** 전체 단어에서 발음을 추출해 왼쪽 셀렉트를 채운다. */
function renderPhonemeOptions(words) {
  const phonemes = [...new Set(words.map((w) => w.phoneme))];

  phonemeSelect.innerHTML = '<option value="">— 발음 선택 —</option>';
  phonemes.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = `/${p}/`;
    phonemeSelect.appendChild(opt);
  });

  // 오른쪽은 발음 선택 전까지 비활성
  wordSelect.innerHTML = '<option value="">← 발음을 먼저 선택하세요</option>';
  wordSelect.disabled = true;
}

/** 발음 선택 시 오른쪽 단어 셀렉트를 해당 발음 단어로 채운다. */
function onPhonemeSelect() {
  const phoneme = phonemeSelect.value;

  selectedWord = null;
  updateTargetCard(null);
  resetResult();

  if (!phoneme) {
    wordSelect.innerHTML = '<option value="">← 발음을 먼저 선택하세요</option>';
    wordSelect.disabled = true;
    return;
  }

  const filtered = allWords.filter((w) => w.phoneme === phoneme);

  wordSelect.innerHTML = '<option value="">— 단어 선택 —</option>';
  filtered.forEach((w) => {
    const opt = document.createElement("option");
    opt.value       = w.word;
    opt.textContent = w.word;
    opt.dataset.word   = w.word;
    opt.dataset.korean = w.korean_pronunciation;
    wordSelect.appendChild(opt);
  });
  wordSelect.disabled = false;
}

// ── 단어 선택 이벤트 ──────────────────────────────────
function onWordSelect() {
  const opt = wordSelect.selectedOptions[0];
  if (!opt || !opt.dataset.word) {
    selectedWord = null;
    updateTargetCard(null);
    return;
  }

  selectedWord = {
    word: opt.dataset.word,
    korean_pronunciation: opt.dataset.korean,
    phoneme: phonemeSelect.value,   // 발음은 왼쪽 셀렉트에서 가져온다
  };
  updateTargetCard(selectedWord);
  resetResult();
}

function updateTargetCard(word) {
  if (!word) {
    targetWordEl.textContent    = "—";
    targetPhonemeEl.textContent = "";
    targetKoreanEl.textContent  = "";
    return;
  }
  targetWordEl.textContent    = word.word;
  targetPhonemeEl.textContent = `/${word.phoneme}/`;
  targetKoreanEl.textContent  = `한국어 힌트: ${word.korean_pronunciation}`;
}

// ── 이벤트 바인딩 ─────────────────────────────────────
function bindEvents() {
  phonemeSelect.addEventListener("change", onPhonemeSelect);
  wordSelect.addEventListener("change", onWordSelect);
  fileUpload.addEventListener("change", onFileUpload);
  btnRecord.addEventListener("click", onRecordClick);
  btnAnalyze.addEventListener("click", onAnalyzeClick);
  btnRetry.addEventListener("click", onRetryClick);
  document.addEventListener("recordingcomplete", onRecordingComplete);
}

// ── 파일 업로드 ───────────────────────────────────────
function onFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  if (!isValidAudioFile(file)) {
    showError("지원하지 않는 파일 형식입니다. wav, mp3, m4a, webm, ogg 파일만 사용할 수 있습니다.");
    fileUpload.value = "";
    return;
  }

  // 녹음 중이면 먼저 중지한다
  if (isRecording()) {
    stopRecording();
    resetRecording();
    setRecordingUi(false);
  }

  recordedBlob = null;
  uploadedFile = file;

  setAudioSource("upload", file.name);
  setAudioPreview(URL.createObjectURL(file));
  updateAnalyzeButton();
  hideError();
  resetResult();
}

function isValidAudioFile(file) {
  const allowedTypes = [
    "audio/wav", "audio/x-wav",
    "audio/mpeg",
    "audio/mp4", "audio/x-m4a",
    "audio/webm",
    "audio/ogg",
  ];
  const allowedExts = /\.(wav|mp3|m4a|webm|ogg)$/i;
  return allowedTypes.includes(file.type) || allowedExts.test(file.name);
}

// ── 녹음 토글 ─────────────────────────────────────────
async function onRecordClick() {
  if (isRecording()) {
    stopRecording();
    // UI는 recordingcomplete 이벤트에서 갱신한다
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("이 브라우저는 마이크 녹음을 지원하지 않습니다. Chrome, Firefox, Safari를 사용해주세요.");
    return;
  }

  hideError();
  uploadedFile = null;
  recordedBlob = null;
  fileUpload.value = "";
  hideAudioSource();
  audioPreview.style.display = "none";

  try {
    await startRecording();
    setRecordingUi(true);
    resetResult();
  } catch (err) {
    const isDenied = err.name === "NotAllowedError" || err.name === "PermissionDeniedError";
    showError(
      isDenied
        ? "마이크 권한이 거부됐습니다. 브라우저 설정에서 마이크를 허용해주세요."
        : `녹음을 시작할 수 없습니다: ${err.message}`
    );
  }
}

function onRecordingComplete(e) {
  recordedBlob = e.detail.blob;
  setRecordingUi(false);
  setAudioSource("record", "방금 녹음된 파일");
  setAudioPreview(URL.createObjectURL(recordedBlob));
  updateAnalyzeButton();
}

function setRecordingUi(active) {
  if (active) {
    btnRecord.classList.add("is-recording");
    btnRecordLabel.textContent = "녹음 중지";
    recStatus.textContent = "● 녹음 중...";
    recStatus.classList.add("is-recording");
    recZone.classList.add("is-recording");
  } else {
    btnRecord.classList.remove("is-recording");
    btnRecordLabel.textContent = "녹음 시작";
    recStatus.textContent = "녹음 완료 — 아래에서 미리 들어보세요";
    recStatus.classList.remove("is-recording");
    recZone.classList.remove("is-recording");
  }
}

// ── 분석 버튼 활성 여부 ───────────────────────────────
function updateAnalyzeButton() {
  const hasAudio = recordedBlob !== null || uploadedFile !== null;
  btnAnalyze.disabled = !hasAudio;
}

// ── 분석 요청 ─────────────────────────────────────────
async function onAnalyzeClick() {
  if (!selectedWord) {
    showError("단어를 먼저 선택해주세요.");
    return;
  }

  const audioBlob = recordedBlob ?? uploadedFile;
  if (!audioBlob) {
    showError("녹음하거나 파일을 업로드한 뒤 분석해주세요.");
    return;
  }

  hideError();
  showLoading(true);

  try {
    const formData = new FormData();
    formData.append("word",    selectedWord.word);
    formData.append("phoneme", selectedWord.phoneme);
    formData.append("audio_file", audioBlob, getAudioFilename(audioBlob));

    const result = await analyzePronunciation(formData);
    showResult(result);
  } catch (err) {
    showError(err.message || "분석 중 오류가 발생했습니다.");
  } finally {
    showLoading(false);
  }
}

function getAudioFilename(blob) {
  if (blob instanceof File) return blob.name;
  return "recording.wav";
}

// ── 재시도 ────────────────────────────────────────────
function onRetryClick() {
  resetResult();
  resetAudio();
}

// ── 결과 표시 ─────────────────────────────────────────
function showResult(result) {
  document.getElementById("result-word").textContent         = result.word;
  document.getElementById("result-phoneme-pill").textContent = `/${result.phoneme}/`;
  document.getElementById("feedback-body").textContent       = result.feedback;

  const score = result.score;
  const { grade, gradeSub, gradeClass } = resolveGrade(score);

  document.getElementById("score-grade").textContent     = grade;
  document.getElementById("score-grade-sub").textContent = gradeSub;

  const scoreCard = document.getElementById("score-card");
  scoreCard.className = `score-card ${gradeClass}`;

  renderMetrics(result.details || {});

  // 숨겨진 상태에서 표시 → 애니메이션 트리거
  resultSection.style.display = "block";

  // 점수 카운트업 + 스크롤
  animateScoreNumber(0, Math.round(score), 700);
  setTimeout(() => {
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}

function resolveGrade(score) {
  if (score >= 85) {
    return { grade: "Excellent ✓", gradeSub: "기준 발음과 상당히 유사합니다.", gradeClass: "score-card--excellent" };
  }
  if (score >= 70) {
    return { grade: "Good", gradeSub: "전반적으로 좋지만 더 다듬을 수 있습니다.", gradeClass: "score-card--good" };
  }
  return { grade: "Needs Practice", gradeSub: "조금 더 반복 연습해보세요.", gradeClass: "score-card--practice" };
}

function animateScoreNumber(from, to, durationMs) {
  const el = document.getElementById("score-number");
  const startTime = performance.now();

  function step(now) {
    const elapsed  = now - startTime;
    const progress = Math.min(elapsed / durationMs, 1);
    // easeOutQuart
    const eased = 1 - Math.pow(1 - progress, 4);
    el.textContent = Math.round(from + (to - from) * eased);
    if (progress < 1) requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}

function renderMetrics(details) {
  const card = document.getElementById("metrics-card");
  const labelMap = {
    mfcc_score:              "MFCC",
    duration_score:          "Duration",
    rms_score:               "RMS",
    zcr_score:               "ZCR",
    spectral_centroid_score: "Spectral",
  };

  const entries = Object.entries(labelMap).filter(([key]) => details[key] != null);

  if (!entries.length) {
    card.innerHTML = "";
    return;
  }

  // width를 0으로 먼저 렌더링한 뒤 rAF에서 실제 값을 적용해 CSS transition을 유발한다
  card.innerHTML =
    `<div class="metrics-label">DETAILED METRICS</div>` +
    entries.map(([key, label]) => {
      const val = details[key];
      return `
        <div class="m-row">
          <div class="m-name">${label}</div>
          <div class="m-track"><div class="m-fill" id="mf-${key}" style="width:0%"></div></div>
          <div class="m-val">${val}</div>
        </div>`;
    }).join("");

  // 두 프레임 뒤에 실제 너비 적용 → transition 발동
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      entries.forEach(([key]) => {
        const el = document.getElementById(`mf-${key}`);
        if (el) el.style.width = `${details[key]}%`;
      });
    });
  });
}

// ── 상태 초기화 ───────────────────────────────────────
function resetResult() {
  resultSection.style.display = "none";
  document.getElementById("score-number").textContent = "0";
  const scoreCard = document.getElementById("score-card");
  if (scoreCard) scoreCard.className = "score-card";
  hideError();
}

function resetAudio() {
  resetRecording();
  recordedBlob = null;
  uploadedFile = null;
  fileUpload.value = "";
  hideAudioSource();
  audioPreview.style.display = "none";
  audioPreview.src = "";
  recStatus.textContent = "마이크 버튼을 눌러 녹음을 시작하세요";
  recStatus.className = "rec-status";
  btnRecord.className = "btn btn-record";
  btnRecordLabel.textContent = "녹음 시작";
  recZone.classList.remove("is-recording");
  updateAnalyzeButton();
}

// ── 오디오 소스 표시 ──────────────────────────────────
function setAudioSource(type, name) {
  sourceBadge.textContent = type === "record" ? "🎙" : "📂";
  sourceName.textContent  = name;
  audioSourceBar.style.display = "flex";
}

function hideAudioSource() {
  audioSourceBar.style.display = "none";
  sourceBadge.textContent = "";
  sourceName.textContent  = "";
}

// ── 오디오 미리듣기 ───────────────────────────────────
function setAudioPreview(url) {
  audioPreview.src = url;
  audioPreview.style.display = "block";
}

// ── 로딩 / 에러 ──────────────────────────────────────
function showLoading(visible) {
  loadingOverlay.style.display = visible ? "flex" : "none";
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.style.display = "block";
}

function hideError() {
  errorMessage.style.display = "none";
  errorMessage.textContent = "";
}
