/**
 * 화면 이벤트 연결과 상태 관리를 담당한다.
 */

// ── 상태 ──────────────────────────────────────────────
let selectedWord = null;   // { word, korean_pronunciation, phoneme }
let recordedBlob = null;   // Blob | null
let uploadedFile = null;   // File | null

// ── DOM 참조 ──────────────────────────────────────────
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
const fileNameEl       = document.getElementById("file-name");
const btnAnalyze       = document.getElementById("btn-analyze");
const resultSection    = document.getElementById("result-section");
const loadingOverlay   = document.getElementById("loading-overlay");
const errorMessage     = document.getElementById("error-message");
const btnRetry         = document.getElementById("btn-retry");

// ── 초기화 ────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", initApp);

async function initApp() {
  await loadWordList();
  bindEvents();
}

// ── 단어 목록 로딩 ────────────────────────────────────
async function loadWordList() {
  try {
    const words = await fetchWords();
    renderWordOptions(words);
  } catch (err) {
    showError("단어 목록을 불러오지 못했습니다. 서버를 확인해주세요.");
  }
}

function renderWordOptions(words) {
  wordSelect.innerHTML = '<option value="">— 단어를 선택하세요 —</option>';

  words.forEach((w, idx) => {
    const option = document.createElement("option");
    option.value = idx;
    option.textContent = `${w.word}  /${w.phoneme}/`;
    option.dataset.word   = w.word;
    option.dataset.korean = w.korean_pronunciation;
    option.dataset.phoneme = w.phoneme;
    wordSelect.appendChild(option);
  });
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
    phoneme: opt.dataset.phoneme,
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
  wordSelect.addEventListener("change", onWordSelect);
  fileUpload.addEventListener("change", onFileUpload);
  btnRecord.addEventListener("click", onRecordClick);
  btnAnalyze.addEventListener("click", onAnalyzeClick);
  btnRetry.addEventListener("click", onRetryClick);
}

// ── 파일 업로드 ───────────────────────────────────────
function onFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const allowed = ["audio/wav", "audio/mpeg", "audio/mp4", "audio/webm",
                   "audio/ogg", "audio/x-m4a", "audio/x-wav"];
  if (!allowed.includes(file.type) && !file.name.match(/\.(wav|mp3|m4a|webm|ogg)$/i)) {
    showError("지원하지 않는 파일 형식입니다. (wav, mp3, m4a, webm, ogg)");
    fileUpload.value = "";
    return;
  }

  uploadedFile = file;
  recordedBlob = null;
  fileNameEl.textContent = `선택된 파일: ${file.name}`;
  setAudioPreview(URL.createObjectURL(file));
  updateAnalyzeButton();
  hideError();
  resetResult();
}

// ── 녹음 버튼 (placeholder — recorder.js에서 구현) ───
function onRecordClick() {
  if (typeof toggleRecording === "function") {
    toggleRecording();
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
  return "recording.webm";
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
  document.getElementById("score-number").textContent        = Math.round(result.score);
  document.getElementById("feedback-body").textContent       = result.feedback;

  const score = result.score;
  let grade, gradeSub;
  if (score >= 85) {
    grade = "Excellent";  gradeSub = "기준 발음과 상당히 유사합니다.";
  } else if (score >= 70) {
    grade = "Good";       gradeSub = "전반적으로 좋지만 더 다듬을 수 있습니다.";
  } else {
    grade = "Needs Practice"; gradeSub = "조금 더 반복 연습해보세요.";
  }
  document.getElementById("score-grade").textContent     = grade;
  document.getElementById("score-grade-sub").textContent = gradeSub;

  renderMetrics(result.details || {});

  resultSection.style.display = "block";
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

  const rows = Object.entries(labelMap)
    .filter(([key]) => details[key] != null)
    .map(([key, label]) => {
      const val = details[key];
      return `
        <div class="m-row">
          <div class="m-name">${label}</div>
          <div class="m-track"><div class="m-fill" style="width:${val}%"></div></div>
          <div class="m-val">${val}</div>
        </div>`;
    }).join("");

  if (!rows) {
    card.innerHTML = "";
    return;
  }
  card.innerHTML = `<div class="metrics-label">DETAILED METRICS</div>${rows}`;
}

// ── 상태 초기화 ───────────────────────────────────────
function resetResult() {
  resultSection.style.display = "none";
  hideError();
}

function resetAudio() {
  recordedBlob = null;
  uploadedFile = null;
  fileUpload.value = "";
  fileNameEl.textContent = "";
  audioPreview.style.display = "none";
  audioPreview.src = "";
  recStatus.textContent = "마이크 버튼을 눌러 녹음을 시작하세요";
  recStatus.className = "rec-status";
  btnRecord.className = "btn btn-record";
  btnRecordLabel.textContent = "녹음 시작";
  if (recZone) recZone.classList.remove("is-recording");
  updateAnalyzeButton();
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
