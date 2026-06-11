import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import gradio as gr

from pipeline.audio import load_and_trim_audio
from pipeline.features import extract_features
from pipeline.reference import (
    build_target_index,
    get_available_targets,
    get_gradio_choices,
    get_reference_for_target_id,
    load_reference_vectors,
)
from pipeline.scorer import score_pronunciation


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "data" / "results.csv"


# ── Waveform bars (decorative) ────────────────────────────────────────────────
_WAVE_H = [8, 14, 22, 28, 18, 34, 24, 38, 26, 18, 32, 20, 28, 36, 22, 16, 30, 24, 38, 12]
_WAVE_BARS = "".join(
    f'<span style="height:{h}px;animation-delay:{i*0.07:.2f}s"></span>'
    for i, h in enumerate(_WAVE_H)
)
_REC_VISUAL_HTML = f"""
<div class="rec-visual">
  <div class="rec-circles">
    <div class="rec-ring1"></div>
    <div class="rec-ring2"></div>
    <div class="rec-core">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
           stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
        <line x1="12" y1="19" x2="12" y2="23"/>
        <line x1="8" y1="23" x2="16" y2="23"/>
      </svg>
    </div>
  </div>
  <div class="waveform">{_WAVE_BARS}</div>
</div>
"""

_LOADING_HTML = """
<div class="loading-wrap">
  <div class="spinner-ring"></div>
  <div class="loading-text">발음을 분석하고 있습니다...</div>
</div>
"""

CUSTOM_CSS = """
/* ── Base ───────────────────────────────────── */
body,
.gradio-container {
  background: linear-gradient(160deg, #e8f4fd 0%, #dbeafe 100%) !important;
  background-attachment: fixed !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
               "Apple SD Gothic Neo", sans-serif !important;
  min-height: 100vh;
  color: #1e293b;
}
footer { display: none !important; }

/* ── Center shell ───────────────────────────── */
.app-shell {
  max-width: 420px !important;
  margin: 0 auto !important;
  padding: 24px 0 80px !important;
}
.app-shell > .form,
.app-shell > div {
  gap: 0 !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}

/* ── White screen card ──────────────────────── */
.screen-card {
  background: #ffffff;
  border-radius: 24px;
  box-shadow: 0 4px 28px rgba(30, 64, 175, 0.10);
  overflow: hidden;
  margin: 0 12px;
}
.screen-card > .form,
.screen-card > div {
  gap: 0 !important;
  padding: 0 !important;
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  border-radius: 0 !important;
}

/* ── Welcome — gradient header ──────────────── */
.w-header {
  background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
  padding: 36px 24px 28px;
  text-align: center;
  color: #ffffff;
}
.w-logo      { font-size: 28px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 6px; }
.w-subtitle  { font-size: 13.5px; opacity: 0.90; letter-spacing: 0.01em; }
.w-desc      { font-size: 12.5px; opacity: 0.78; margin-top: 8px; line-height: 1.5; }

/* ── Welcome — 3-step cards ─────────────────── */
.steps-row {
  display: flex;
  gap: 10px;
  padding: 20px 18px 4px;
}
.step-item {
  flex: 1;
  background: #f0f7ff;
  border: 1px solid #bfdbfe;
  border-radius: 16px;
  padding: 14px 8px;
  text-align: center;
}
.step-num {
  width: 28px; height: 28px;
  border-radius: 50%;
  background: #3b82f6;
  color: #ffffff;
  font-size: 13px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 8px;
}
.step-title { font-size: 12.5px; font-weight: 700; color: #1e40af; margin-bottom: 4px; }
.step-desc  { font-size: 11px; color: #64748b; line-height: 1.4; }

/* ── Welcome — body area ────────────────────── */
.w-body > .form,
.w-body > div {
  padding: 14px 18px 6px !important;
  gap: 10px !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.w-preview {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 11px 14px;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 13px;
  min-height: 46px;
}
.w-preview-word    { font-size: 20px; font-weight: 800; color: #1e40af; }
.w-preview-phoneme {
  font-size: 13px; font-weight: 700; color: #1d4ed8;
  background: #dbeafe; border-radius: 999px; padding: 3px 10px;
}
.w-error {
  color: #ef4444; font-size: 13px; font-weight: 600;
  padding: 2px 0; min-height: 20px;
}

/* ── Buttons ────────────────────────────────── */
#btn-start {
  background: #3b82f6 !important; color: #ffffff !important;
  border: none !important; border-radius: 14px !important;
  height: 52px !important; font-size: 16px !important; font-weight: 600 !important;
  width: 100% !important;
  box-shadow: 0 4px 16px rgba(59,130,246,0.32) !important;
  transition: filter .15s, transform .12s !important;
  margin: 4px 0 18px;
}
#btn-start:hover { filter: brightness(1.08) !important; transform: translateY(-1px) !important; }

#btn-back {
  background: transparent !important; border: none !important;
  color: #3b82f6 !important; font-size: 14px !important; font-weight: 600 !important;
  padding: 14px 20px 6px !important; height: auto !important;
  box-shadow: none !important; text-align: left !important; width: auto !important;
}

#btn-analyze {
  background: #3b82f6 !important; color: #ffffff !important;
  border: none !important; border-radius: 14px !important;
  height: 52px !important; font-size: 16px !important; font-weight: 600 !important;
  width: 100% !important;
  box-shadow: 0 4px 16px rgba(59,130,246,0.32) !important;
  transition: filter .15s, transform .12s !important;
}
#btn-analyze:hover { filter: brightness(1.08) !important; transform: translateY(-1px) !important; }

#btn-try-again {
  background: #ffffff !important; border: 1.5px solid #3b82f6 !important;
  color: #3b82f6 !important; border-radius: 14px !important;
  height: 48px !important; font-size: 14px !important; font-weight: 600 !important;
}
#btn-change-word {
  background: #3b82f6 !important; color: #ffffff !important;
  border: none !important; border-radius: 14px !important;
  height: 48px !important; font-size: 14px !important; font-weight: 600 !important;
}

/* ── Practice — word display ────────────────── */
.pw-section {
  padding: 16px 20px 14px;
  border-bottom: 1px solid #f1f5f9;
}
.pw-label   { font-size: 11px; font-weight: 700; letter-spacing: .10em; text-transform: uppercase; color: #94a3b8; margin-bottom: 6px; }
.pw-word    { font-size: 48px; font-weight: 700; color: #1e293b; line-height: 1; margin-bottom: 8px; letter-spacing: -0.02em; }
.pw-phoneme {
  display: inline-block; background: #eff6ff; color: #1d4ed8;
  border-radius: 999px; font-size: 20px; font-weight: 700;
  padding: 8px 18px; margin-bottom: 10px;
}
.pw-hint { font-size: 14px; color: #64748b; line-height: 1.6; }

/* ── Practice — recording zone (dark) ──────── */
#recording-zone {
  background: linear-gradient(160deg, #1e3058 0%, #18264a 100%) !important;
  border-radius: 0 !important;
}
#recording-zone > .form,
#recording-zone > div {
  background: transparent !important;
  border: none !important;
  gap: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
}

.rec-visual {
  display: flex; flex-direction: column; align-items: center;
  padding: 28px 20px 18px;
}
.rec-circles {
  position: relative; width: 144px; height: 144px;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 22px;
}
.rec-ring1 {
  position: absolute; width: 144px; height: 144px; border-radius: 50%;
  background: rgba(59,130,246,0.12);
  animation: pulseRing 2.2s ease-out infinite;
}
.rec-ring2 {
  position: absolute; width: 112px; height: 112px; border-radius: 50%;
  background: rgba(59,130,246,0.18);
  animation: pulseRing 2.2s ease-out infinite .45s;
}
.rec-core {
  position: relative; width: 80px; height: 80px; border-radius: 50%;
  background: linear-gradient(145deg, #4f9cf0, #2563eb);
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 8px 24px rgba(37,99,235,.55);
}
@keyframes pulseRing {
  0%   { transform: scale(.85); opacity: .8; }
  65%  { transform: scale(1);   opacity: .25; }
  100% { transform: scale(1);   opacity: 0; }
}
.waveform { display: flex; align-items: center; gap: 3px; height: 38px; }
.waveform span {
  display: inline-block; width: 3px;
  background: rgba(148,196,255,.55); border-radius: 2px;
  animation: waveAnim 1.6s ease-in-out infinite;
}
@keyframes waveAnim {
  0%, 100% { transform: scaleY(.35); }
  50%       { transform: scaleY(1); }
}

/* Style the Gradio Audio inside the dark zone */
#audio-practice label        { color: rgba(255,255,255,0.55) !important; font-size: 12px !important; }
#audio-practice .wrap        { background: rgba(15,30,70,.55) !important; border-color: rgba(59,130,246,.28) !important; border-radius: 16px !important; margin: 0 16px 18px !important; }
#audio-practice .icon-button { color: rgba(255,255,255,.8) !important; }
#audio-practice .waveform    { color: #60a5fa !important; }

/* Practice status / error */
.p-status > p, .p-status p {
  color: #ef4444; font-size: 13px; font-weight: 600;
  padding: 6px 20px 0; margin: 0;
}

/* Analyze btn wrapper padding */
.analyze-wrap > .form,
.analyze-wrap > div {
  padding: 0 18px 20px !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* ── Result screen ──────────────────────────── */
.result-target {
  display: flex; align-items: center; gap: 10px;
  padding: 18px 18px 0;
}
.result-word { font-size: 26px; font-weight: 800; color: #1e293b; }
.result-phoneme-pill {
  background: #eff6ff; color: #1d4ed8;
  border: 1px solid #bfdbfe; border-radius: 999px;
  font-size: 14px; font-weight: 700; padding: 5px 12px;
}

.score-card {
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  border-radius: 20px; margin: 12px 16px 0;
  padding: 26px 20px; text-align: center; color: #ffffff;
}
.score-lbl  { font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; opacity: .78; margin-bottom: 8px; }
.score-num  { font-size: 72px; font-weight: 800; line-height: 1; letter-spacing: -0.03em; }
.score-max  { font-size: 18px; opacity: .65; }
.score-grade { font-size: 20px; font-weight: 700; margin-top: 8px; }
.score-grade-sub { font-size: 13.5px; opacity: .85; margin-top: 3px; }

.feedback-card {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 18px;
  padding: 18px; margin: 12px 16px 0;
}
.feedback-lbl  { font-size: 11px; font-weight: 700; letter-spacing: .10em; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px; }
.feedback-body { font-size: 14.5px; color: #334155; line-height: 1.75; }

.metrics-card {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 18px;
  padding: 18px; margin: 12px 16px 0;
}
.metrics-lbl { font-size: 11px; font-weight: 700; letter-spacing: .10em; text-transform: uppercase; color: #94a3b8; margin-bottom: 12px; }
.m-row       { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.m-name      { font-size: 12px; font-weight: 600; color: #64748b; width: 80px; flex-shrink: 0; text-align: right; }
.m-track     { flex: 1; height: 7px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
.m-fill      { height: 100%; background: linear-gradient(90deg, #3b82f6, #6366f1); border-radius: 999px; }
.m-val       { font-size: 13px; font-weight: 700; color: #1e40af; width: 36px; text-align: right; flex-shrink: 0; }

.result-actions > .form,
.result-actions > div {
  padding: 14px 16px 24px !important;
  gap: 10px !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* ── Loading spinner ────────────────────────── */
.loading-wrap {
  display: flex; flex-direction: column; align-items: center;
  padding: 52px 20px; gap: 16px;
}
.spinner-ring {
  width: 48px; height: 48px;
  border: 4px solid rgba(59,130,246,.18);
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin .8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text { font-size: 15px; color: #64748b; font-weight: 500; }

/* ── Mobile ─────────────────────────────────── */
@media (max-width: 480px) {
  .app-shell { padding: 0 0 60px !important; }
  .screen-card { margin: 0; border-radius: 0 0 20px 20px; }
  .w-header { border-radius: 0; }
  .steps-row { flex-direction: column; gap: 8px; }
  .pw-word  { font-size: 40px; }
  .score-num { font-size: 60px; }
}
"""


# ── CSV helpers ───────────────────────────────────────────────────────────────

def ensure_results_csv() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        return
    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow([
            "timestamp", "target_word", "korean_pronunciation", "phoneme",
            "score", "feedback",
            "mfcc_score", "duration_score", "rms_score", "zcr_score", "spectral_centroid_score",
        ])


def save_result(
    target_word: str,
    korean_pronunciation: str,
    phoneme: str,
    result: dict[str, Any],
) -> None:
    ensure_results_csv()
    d = result.get("details", {})
    with RESULTS_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            target_word, korean_pronunciation, phoneme,
            result.get("score"), result.get("feedback"),
            d.get("mfcc_score", ""), d.get("duration_score", ""),
            d.get("rms_score", ""), d.get("zcr_score", ""),
            d.get("spectral_centroid_score", ""),
        ])


# ── App data init ─────────────────────────────────────────────────────────────

def _init() -> tuple[dict, list, dict, list, str | None]:
    try:
        ref   = load_reference_vectors()
        words = get_available_targets()
        index = build_target_index(words)
        choices = get_gradio_choices(words)
        if not choices:
            raise ValueError("사용 가능한 단어가 없습니다.")
        return ref, words, index, choices, None
    except Exception as e:
        return {}, [], {}, [], f"{type(e).__name__}: {e}"


REFERENCE_VECTORS, TARGET_WORDS, TARGET_INDEX, GRADIO_CHOICES, STARTUP_ERROR = _init()


# ── Render helpers ────────────────────────────────────────────────────────────

def render_welcome_preview(target_id: str | None) -> str:
    if not target_id or target_id not in TARGET_INDEX:
        return '<div class="w-preview" style="opacity:.45;">단어를 선택하면 미리보기가 표시됩니다</div>'
    t = TARGET_INDEX[target_id]
    return f"""
    <div class="w-preview">
      <span class="w-preview-word">{t.word}</span>
      <span class="w-preview-phoneme">/{t.phoneme}/</span>
    </div>
    """


def render_practice_header(target_id: str | None) -> str:
    if not target_id or target_id not in TARGET_INDEX:
        return ""
    t = TARGET_INDEX[target_id]
    return f"""
    <div class="pw-section">
      <div class="pw-label">Target Word</div>
      <div class="pw-word">{t.word}</div>
      <div class="pw-phoneme">/{t.phoneme}/</div>
      <div class="pw-hint">한국어 힌트: {t.korean_pronunciation}</div>
    </div>
    """


def render_result_content(target: Any, result: dict[str, Any]) -> str:
    score    = float(result["score"])
    feedback = result["feedback"]
    details  = result.get("details", {})

    if score >= 85:
        grade, grade_sub = "Excellent", "기준 발음과 상당히 유사합니다."
    elif score >= 70:
        grade, grade_sub = "Good", "전반적으로 좋지만 더 다듬을 수 있습니다."
    else:
        grade, grade_sub = "Needs Practice", "조금 더 반복 연습해보세요."

    label_map = {
        "mfcc_score": "MFCC",
        "duration_score": "Duration",
        "rms_score": "RMS",
        "zcr_score": "ZCR",
        "spectral_centroid_score": "Spectral",
    }
    rows = "".join(
        f'<div class="m-row">'
        f'<div class="m-name">{lbl}</div>'
        f'<div class="m-track"><div class="m-fill" style="width:{details[k]}%"></div></div>'
        f'<div class="m-val">{details[k]}</div>'
        f'</div>'
        for k, lbl in label_map.items()
        if details.get(k) not in ("", None)
    )

    return f"""
    <div class="result-target">
      <div class="result-word">{target.word}</div>
      <div class="result-phoneme-pill">/{target.phoneme}/</div>
    </div>
    <div class="score-card">
      <div class="score-lbl">Pronunciation Score</div>
      <div><span class="score-num">{score:.0f}</span><span class="score-max"> / 100</span></div>
      <div class="score-grade">{grade}</div>
      <div class="score-grade-sub">{grade_sub}</div>
    </div>
    <div class="feedback-card">
      <div class="feedback-lbl">Feedback</div>
      <div class="feedback-body">{feedback}</div>
    </div>
    {"" if not rows else f'<div class="metrics-card"><div class="metrics-lbl">Detailed Metrics</div>{rows}</div>'}
    """


# ── Event handlers ────────────────────────────────────────────────────────────

def on_target_change(target_id: str) -> str:
    # outputs(1): [welcome_word_preview]
    return render_welcome_preview(target_id)


def start_practice(target_id: str) -> tuple:
    # outputs(5): [screen_welcome, screen_practice, screen_result, practice_header, welcome_error]
    if not target_id:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            '<div class="w-error">단어를 먼저 선택해주세요.</div>',
        )
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        render_practice_header(target_id),
        "",
    )


def go_to_welcome() -> tuple:
    # outputs(3): [screen_welcome, screen_practice, screen_result]
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def analyze_pronunciation(
    target_id: str,
    audio_file: str | None,
) -> Generator:
    # Generator — outputs(4): [screen_practice, screen_result, result_content, practice_status]

    if not audio_file:
        yield (
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            "음성을 녹음하거나 파일을 업로드해주세요.",
        )
        return

    # Show loading screen
    yield (
        gr.update(visible=False),
        gr.update(visible=True),
        _LOADING_HTML,
        "",
    )

    try:
        target, reference = get_reference_for_target_id(
            target_id=target_id,
            target_index=TARGET_INDEX,
            reference_vectors=REFERENCE_VECTORS,
        )
        y, sr = load_and_trim_audio(audio_file)
        features = extract_features(y, sr)
        result = score_pronunciation(
            user_features=features,
            reference=reference,
            phoneme=target.phoneme,
        )
        save_result(
            target_word=target.word,
            korean_pronunciation=target.korean_pronunciation,
            phoneme=target.phoneme,
            result=result,
        )
        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            render_result_content(target, result),
            "",
        )

    except Exception as e:
        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            f'<div class="loading-wrap"><div class="loading-text">오류: {type(e).__name__} — {e}</div></div>',
            "",
        )


def try_again(target_id: str) -> tuple:
    # outputs(5): [screen_practice, screen_result, practice_header, audio_input, practice_status]
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        render_practice_header(target_id),
        None,
        "",
    )


def change_word() -> tuple:
    # outputs(4): [screen_welcome, screen_practice, screen_result, audio_input]
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        None,
    )


# ── UI ────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
    )
    default_id = GRADIO_CHOICES[0][1] if GRADIO_CHOICES else None

    with gr.Blocks(title="PronounceAI", theme=theme, css=CUSTOM_CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):

            # ════════════════════════════════════════
            # SCREEN 1 — Welcome
            # ════════════════════════════════════════
            with gr.Column(
                visible=True, elem_id="screen_welcome", elem_classes=["screen-card"]
            ) as screen_welcome:

                gr.HTML("""
                <div class="w-header">
                  <div class="w-logo">PronounceAI</div>
                  <div class="w-subtitle">Korean English Pronunciation Coach</div>
                  <div class="w-desc">한국인 학습자를 위한 AI 발음 채점 앱</div>
                </div>
                <div class="steps-row">
                  <div class="step-item">
                    <div class="step-num">1</div>
                    <div class="step-title">단어 선택</div>
                    <div class="step-desc">연습할 단어를 선택하세요</div>
                  </div>
                  <div class="step-item">
                    <div class="step-num">2</div>
                    <div class="step-title">녹음</div>
                    <div class="step-desc">단어를 발음하고 녹음하세요</div>
                  </div>
                  <div class="step-item">
                    <div class="step-num">3</div>
                    <div class="step-title">분석</div>
                    <div class="step-desc">AI가 채점하고 피드백을 드립니다</div>
                  </div>
                </div>
                """)

                with gr.Column(elem_classes=["w-body"]):
                    target_dropdown = gr.Dropdown(
                        choices=GRADIO_CHOICES,
                        value=default_id,
                        label="연습할 단어 선택",
                        interactive=not bool(STARTUP_ERROR),
                    )
                    welcome_word_preview = gr.HTML(
                        value=render_welcome_preview(default_id)
                    )
                    welcome_error = gr.HTML("")

                start_btn = gr.Button(
                    "Start Practice →", elem_id="btn-start", variant="primary"
                )

            # ════════════════════════════════════════
            # SCREEN 2 — Practice
            # ════════════════════════════════════════
            with gr.Column(
                visible=False, elem_id="screen_practice", elem_classes=["screen-card"]
            ) as screen_practice:

                back_btn = gr.Button("← Back", elem_id="btn-back")
                practice_header = gr.HTML("")

                with gr.Column(elem_id="recording-zone"):
                    gr.HTML(_REC_VISUAL_HTML)
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="녹음 또는 파일 업로드",
                        elem_id="audio-practice",
                        interactive=not bool(STARTUP_ERROR),
                    )

                practice_status = gr.Markdown(
                    value="", elem_classes=["p-status"]
                )

                with gr.Column(elem_classes=["analyze-wrap"]):
                    analyze_btn = gr.Button(
                        "Analyze Pronunciation",
                        elem_id="btn-analyze",
                        variant="primary",
                        interactive=not bool(STARTUP_ERROR),
                    )

            # ════════════════════════════════════════
            # SCREEN 3 — Result
            # ════════════════════════════════════════
            with gr.Column(
                visible=False, elem_id="screen_result", elem_classes=["screen-card"]
            ) as screen_result:

                result_content = gr.HTML("")

                with gr.Row(elem_classes=["result-actions"]):
                    try_again_btn = gr.Button(
                        "Try Again", elem_id="btn-try-again", variant="secondary"
                    )
                    change_word_btn = gr.Button(
                        "Change Word", elem_id="btn-change-word", variant="primary"
                    )

            # ── Event wiring ───────────────────────────────────────────────
            target_dropdown.change(
                fn=on_target_change,
                inputs=[target_dropdown],
                outputs=[welcome_word_preview],
            )

            start_btn.click(
                fn=start_practice,
                inputs=[target_dropdown],
                outputs=[screen_welcome, screen_practice, screen_result,
                         practice_header, welcome_error],
            )

            back_btn.click(
                fn=go_to_welcome,
                outputs=[screen_welcome, screen_practice, screen_result],
            )

            analyze_btn.click(
                fn=analyze_pronunciation,
                inputs=[target_dropdown, audio_input],
                outputs=[screen_practice, screen_result,
                         result_content, practice_status],
            )

            try_again_btn.click(
                fn=try_again,
                inputs=[target_dropdown],
                outputs=[screen_practice, screen_result,
                         practice_header, audio_input, practice_status],
            )

            change_word_btn.click(
                fn=change_word,
                outputs=[screen_welcome, screen_practice, screen_result, audio_input],
            )

    return demo


if __name__ == "__main__":
    ensure_results_csv()
    app = build_app()
    app.launch()
