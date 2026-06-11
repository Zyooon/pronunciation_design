import csv
from datetime import datetime
from pathlib import Path
from typing import Any

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


CUSTOM_CSS = """
:root {
  --bg-1: #e8f6ff;
  --bg-2: #d4efff;
  --card: rgba(255, 255, 255, 0.90);
  --card-strong: rgba(255, 255, 255, 0.97);
  --line: rgba(76, 152, 210, 0.16);
  --text-1: #0f2a40;
  --text-2: #4a6a88;
  --text-3: #7a99b8;
  --primary: #3eaeff;
  --primary-dark: #1a8fe0;
  --accent: #7c6fff;
  --success: #12b8b8;
  --warn: #ff7c3e;
  --shadow-sm: 0 4px 16px rgba(60, 120, 180, 0.10);
  --shadow-md: 0 12px 36px rgba(60, 120, 180, 0.14);
  --shadow-lg: 0 24px 60px rgba(60, 120, 180, 0.18);
  --radius-xl: 28px;
  --radius-lg: 22px;
  --radius-md: 16px;
  --radius-sm: 12px;
}

/* ── Base ──────────────────────────────────────── */
body, .gradio-container {
  background: linear-gradient(155deg, #f0faff 0%, #e4f5ff 40%, #ddf0ff 100%) !important;
  min-height: 100vh;
  color: var(--text-1);
  font-family: "Segoe UI", "Apple SD Gothic Neo", "Pretendard", sans-serif;
}

/* ── Shell ─────────────────────────────────────── */
.app-shell {
  max-width: 1160px;
  margin: 0 auto;
  padding: 16px 10px 40px;
}

/* ── Hero ──────────────────────────────────────── */
.hero-card {
  background: linear-gradient(130deg, #2baeff 0%, #5b88ff 55%, #9b7aff 100%);
  border-radius: 32px;
  padding: 34px 36px 32px;
  margin-bottom: 24px;
  box-shadow: 0 28px 64px rgba(43, 110, 255, 0.28);
  position: relative;
  overflow: hidden;
}

.hero-card::before {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 80% 20%, rgba(255,255,255,0.12) 0%, transparent 60%);
  pointer-events: none;
}

.hero-badge {
  display: inline-block;
  background: rgba(255, 255, 255, 0.20);
  border: 1px solid rgba(255, 255, 255, 0.28);
  color: rgba(255, 255, 255, 0.95);
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.10em;
  margin-bottom: 14px;
  text-transform: uppercase;
}

.hero-title {
  font-size: 36px;
  font-weight: 800;
  color: #ffffff;
  line-height: 1.15;
  margin-bottom: 10px;
}

.hero-subtitle {
  font-size: 14.5px;
  color: rgba(255, 255, 255, 0.88);
  line-height: 1.75;
  max-width: 720px;
}

/* ── Generic panel card ────────────────────────── */
.panel-card {
  background: var(--card);
  backdrop-filter: blur(14px);
  border: 1px solid var(--line);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-md);
  padding: 24px;
}

.section-title {
  font-size: 11.5px;
  font-weight: 800;
  letter-spacing: 0.10em;
  color: var(--text-3);
  margin-bottom: 14px;
  text-transform: uppercase;
}

/* ── Target card ───────────────────────────────── */
.target-card {
  background: linear-gradient(160deg, #ffffff 0%, #f0f9ff 100%);
  border: 1px solid rgba(62, 174, 255, 0.20);
  border-radius: 26px;
  padding: 24px 22px;
  min-height: 248px;
  box-shadow: var(--shadow-sm);
}

.target-eyebrow {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.12em;
  color: var(--text-3);
  text-transform: uppercase;
  margin-bottom: 10px;
}

.target-word {
  font-size: 44px;
  font-weight: 800;
  color: var(--text-1);
  line-height: 1.1;
  margin-bottom: 12px;
  letter-spacing: -0.01em;
}

.target-phoneme {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: linear-gradient(135deg, rgba(62, 174, 255, 0.14), rgba(124, 111, 255, 0.14));
  color: #1a5a8a;
  border: 1px solid rgba(62, 174, 255, 0.26);
  border-radius: 999px;
  font-size: 19px;
  font-weight: 800;
  padding: 10px 20px;
  margin-bottom: 14px;
}

.target-hint {
  color: var(--text-2);
  font-size: 14px;
  line-height: 1.75;
}

.target-hint strong {
  color: var(--text-1);
}

/* ── Info card ─────────────────────────────────── */
.info-card {
  background: linear-gradient(160deg, #ffffff 0%, #edf7ff 100%);
  border: 1px solid rgba(62, 174, 255, 0.16);
  border-radius: 24px;
  padding: 22px 20px;
  min-height: 248px;
  box-shadow: var(--shadow-sm);
}

.info-card ul {
  margin: 0;
  padding-left: 18px;
  color: var(--text-2);
  line-height: 1.9;
  font-size: 14.5px;
}

.info-card li {
  margin-bottom: 6px;
}

/* ── Audio panel ───────────────────────────────── */
#audio-panel {
  margin-top: 22px;
}

#audio-panel .wrap,
#audio-panel .block {
  border-radius: 20px !important;
}

.compact-note {
  font-size: 13px;
  color: var(--text-3);
  line-height: 1.7;
  padding: 2px 0;
}

/* ── Buttons ───────────────────────────────────── */
#score-btn {
  background: linear-gradient(130deg, #3eaeff, #7c6fff) !important;
  border: none !important;
  border-radius: 14px !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  color: #ffffff !important;
  padding: 12px 28px !important;
  box-shadow: 0 8px 24px rgba(62, 174, 255, 0.32) !important;
  transition: filter 0.15s, transform 0.12s, box-shadow 0.15s !important;
}

#score-btn:hover {
  filter: brightness(1.08) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 12px 32px rgba(62, 174, 255, 0.40) !important;
}

#score-btn:active {
  transform: translateY(0) !important;
}

#clear-btn {
  border-radius: 14px !important;
  font-weight: 600 !important;
}

/* ── Status ────────────────────────────────────── */
#status-markdown p {
  margin: 6px 0;
  color: var(--warn);
  font-weight: 600;
}

/* ── Results shell ─────────────────────────────── */
.results-shell {
  margin-top: 24px;
}

/* ── Score card ────────────────────────────────── */
.score-card {
  background: linear-gradient(160deg, #ffffff 0%, #eef9ff 100%);
  border: 1px solid rgba(62, 174, 255, 0.18);
  border-radius: 28px;
  padding: 28px 22px;
  min-height: 370px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow-md);
}

.score-kicker {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 11px;
  font-weight: 800;
  color: var(--text-3);
  text-align: center;
  margin-bottom: 18px;
}

.score-ring {
  width: 220px;
  height: 220px;
  border-radius: 50%;
  background: conic-gradient(from 190deg, #3eaeff 0%, #7c6fff 55%, #a8f0ff 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 20px;
  box-shadow: 0 16px 48px rgba(62, 174, 255, 0.30);
}

.score-ring-inner {
  width: 180px;
  height: 180px;
  border-radius: 50%;
  background: linear-gradient(180deg, #ffffff, #eef8ff);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.score-number {
  font-size: 52px;
  font-weight: 800;
  color: var(--text-1);
  line-height: 1;
  letter-spacing: -0.02em;
}

.score-label {
  margin-top: 6px;
  font-size: 13px;
  color: var(--text-3);
  font-weight: 600;
}

.score-caption {
  text-align: center;
  color: var(--text-2);
  font-size: 14px;
  line-height: 1.75;
}

.score-caption strong {
  color: var(--text-1);
  font-size: 16px;
}

/* ── Feedback card ─────────────────────────────── */
.feedback-card {
  background: linear-gradient(160deg, #ffffff 0%, #f0f8ff 100%);
  border: 1px solid rgba(62, 174, 255, 0.18);
  border-radius: 28px;
  padding: 26px 24px;
  min-height: 370px;
  box-shadow: var(--shadow-md);
}

#feedback-markdown h3 {
  margin-top: 0;
  font-size: 15px;
  color: var(--text-1);
  font-weight: 800;
}

#feedback-markdown h4 {
  margin-top: 14px;
  font-size: 13px;
  color: var(--text-2);
  font-weight: 700;
}

#feedback-markdown p,
#feedback-markdown li {
  color: var(--text-2);
  line-height: 1.80;
  font-size: 14px;
}

#feedback-markdown code {
  background: rgba(62, 174, 255, 0.12);
  color: #1a5a8a;
  border-radius: 6px;
  padding: 2px 7px;
  font-size: 13px;
  font-weight: 600;
}

/* ── Metric grid ───────────────────────────────── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 16px;
}

.metric-card {
  background: linear-gradient(160deg, #ffffff, #eef7ff);
  border: 1px solid rgba(62, 174, 255, 0.15);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: var(--shadow-sm);
}

.metric-label {
  color: var(--text-3);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.metric-value {
  color: var(--text-1);
  font-size: 26px;
  font-weight: 800;
  letter-spacing: -0.01em;
}

/* ── Footer ────────────────────────────────────── */
.footer-note {
  margin-top: 28px;
  text-align: center;
  color: var(--text-3);
  font-size: 12.5px;
  line-height: 1.7;
}

/* ── Mobile ────────────────────────────────────── */
@media (max-width: 860px) {
  .hero-title { font-size: 28px; }
  .hero-card  { padding: 24px 20px; }

  .target-word { font-size: 36px; }

  .score-ring       { width: 190px; height: 190px; }
  .score-ring-inner { width: 154px; height: 154px; }
  .score-number     { font-size: 44px; }

  .metric-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 480px) {
  .hero-title { font-size: 24px; }
  .score-ring       { width: 160px; height: 160px; }
  .score-ring-inner { width: 128px; height: 128px; }
  .score-number     { font-size: 36px; }
  .metric-grid { grid-template-columns: 1fr 1fr; }
}
"""


def ensure_results_csv() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        return
    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
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
    details = result.get("details", {})
    with RESULTS_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            target_word, korean_pronunciation, phoneme,
            result.get("score"), result.get("feedback"),
            details.get("mfcc_score", ""),
            details.get("duration_score", ""),
            details.get("rms_score", ""),
            details.get("zcr_score", ""),
            details.get("spectral_centroid_score", ""),
        ])


REFERENCE_VECTORS = load_reference_vectors()
TARGET_WORDS = get_available_targets()
TARGET_INDEX = build_target_index(TARGET_WORDS)
GRADIO_CHOICES = get_gradio_choices(TARGET_WORDS)


def render_target_card(target_id: str | None) -> str:
    if not target_id or target_id not in TARGET_INDEX:
        return """
        <div class="target-card">
          <div class="target-eyebrow">Target Word</div>
          <div class="target-word">—</div>
          <div class="target-phoneme">/ - /</div>
          <div class="target-hint">드롭다운에서 연습할 단어를 선택해주세요.</div>
        </div>
        """
    target = TARGET_INDEX[target_id]
    return f"""
    <div class="target-card">
      <div class="target-eyebrow">Target Word</div>
      <div class="target-word">{target.word}</div>
      <div class="target-phoneme">/{target.phoneme}/</div>
      <div class="target-hint">
        <strong>한국어 힌트</strong> &nbsp;·&nbsp; {target.korean_pronunciation}<br>
        발음기호를 확인한 뒤 아래에서 녹음하거나 파일을 업로드하세요.
      </div>
    </div>
    """


def render_score_card(score: float) -> str:
    if score >= 85:
        caption, sub = "Excellent", "기준 발음과 상당히 유사합니다."
    elif score >= 70:
        caption, sub = "Good", "대체로 좋지만 조금 더 다듬을 수 있습니다."
    else:
        caption, sub = "Needs Practice", "조금 더 반복 연습이 필요합니다."

    return f"""
    <div class="score-card">
      <div class="score-kicker">Pronunciation Score</div>
      <div class="score-ring">
        <div class="score-ring-inner">
          <div class="score-number">{score:.1f}</div>
          <div class="score-label">/ 100</div>
        </div>
      </div>
      <div class="score-caption">
        <strong>{caption}</strong><br>{sub}
      </div>
    </div>
    """


def render_details_html(details: dict[str, Any]) -> str:
    label_map = {
        "mfcc_score": "MFCC",
        "duration_score": "Duration",
        "rms_score": "RMS",
        "zcr_score": "ZCR",
        "spectral_centroid_score": "Spectral Centroid",
    }
    cards = [
        f"""<div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{details[key]}</div>
            </div>"""
        for key, label in label_map.items()
        if key in details and details[key] != ""
    ]
    if not cards:
        return ""
    return f'<div class="metric-grid">{"".join(cards)}</div>'


def render_feedback_markdown(
    target_word: str, phoneme: str, korean_hint: str, feedback: str
) -> str:
    return f"""
### 분석 결과

- **타겟 단어**: `{target_word}`
- **타겟 발음기호**: `/{phoneme}/`
- **한국어 힌트**: `{korean_hint}`

### 피드백

{feedback}
"""


def on_target_change(target_id: str) -> tuple[str, str, gr.update]:
    return render_target_card(target_id), "", gr.update(visible=False)


def score_audio(
    target_id: str,
    audio_file: str | None,
) -> tuple[str, gr.update, str, str, str]:
    if not target_id:
        return "먼저 타겟 단어를 선택해주세요.", gr.update(visible=False), "", "", ""

    if audio_file is None:
        return "먼저 음성을 녹음하거나 업로드해주세요.", gr.update(visible=False), "", "", ""

    try:
        target, reference = get_reference_for_target_id(
            target_id=target_id,
            target_index=TARGET_INDEX,
            reference_vectors=REFERENCE_VECTORS,
        )
        y, sr = load_and_trim_audio(audio_file)
        user_features = extract_features(y, sr)
        result = score_pronunciation(
            user_features=user_features,
            reference=reference,
            phoneme=target.phoneme,
        )
        save_result(
            target_word=target.word,
            korean_pronunciation=target.korean_pronunciation,
            phoneme=target.phoneme,
            result=result,
        )
        return (
            "",
            gr.update(visible=True),
            render_score_card(result["score"]),
            render_feedback_markdown(
                target_word=target.word,
                phoneme=target.phoneme,
                korean_hint=target.korean_pronunciation,
                feedback=result["feedback"],
            ),
            render_details_html(result.get("details", {})),
        )
    except Exception as e:
        return (
            f"채점 중 오류가 발생했습니다: {type(e).__name__} — {e}",
            gr.update(visible=False),
            "", "", "",
        )


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    default_target_id = GRADIO_CHOICES[0][1] if GRADIO_CHOICES else None

    with gr.Blocks(title="Pronunciation Design", theme=theme, css=CUSTOM_CSS) as demo:
        with gr.Column(elem_classes=["app-shell"]):

            # ── Hero ──────────────────────────────────────
            gr.HTML("""
            <div class="hero-card">
              <div class="hero-badge">PRONUNCIATION DESIGN</div>
              <div class="hero-title">English Pronunciation Scoring</div>
              <div class="hero-subtitle">
                원어민 기준 벡터와 사용자의 발음을 비교해
                MFCC · ZCR · Duration · RMS 기반 유사도 점수를 제공합니다.<br>
                단어 하나를 선택한 뒤 음성을 녹음하거나 업로드해보세요.
              </div>
            </div>
            """)

            # ── Target selection + Guide ──────────────────
            with gr.Row(equal_height=True):
                with gr.Column(scale=5):
                    gr.HTML('<div class="section-title">Target Selection</div>')
                    target_dropdown = gr.Dropdown(
                        choices=GRADIO_CHOICES,
                        value=default_target_id,
                        label="연습할 단어 선택",
                        info="단어와 타겟 발음기호를 확인한 뒤 채점해보세요.",
                    )
                    target_card = gr.HTML(value=render_target_card(default_target_id))

                with gr.Column(scale=4):
                    gr.HTML('<div class="section-title">Guide</div>')
                    gr.HTML("""
                    <div class="info-card">
                      <ul>
                        <li>타겟 단어를 선택한 뒤 발음기호를 먼저 확인하세요.</li>
                        <li>아래 녹음 영역에서 직접 말하거나 mp3 / wav 파일을 업로드하세요.</li>
                        <li>점수와 피드백은 음성 입력 후 <strong>채점하기</strong>를 눌러야 표시됩니다.</li>
                        <li>마이크가 안 되면 브라우저 마이크 권한을 허용해주세요.</li>
                      </ul>
                    </div>
                    """)

            # ── Audio input ───────────────────────────────
            with gr.Column(elem_id="audio-panel", elem_classes=["panel-card"]):
                gr.HTML('<div class="section-title">Record or Upload</div>')
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    format="wav",
                    label="음성 녹음 또는 파일 업로드",
                )
                gr.Markdown(
                    "<div class='compact-note'>"
                    "로컬 환경에서는 <strong>Chrome 또는 Edge</strong>를 권장합니다. &nbsp;"
                    "마이크 사용 불가 시 브라우저 주소창 자물쇠 아이콘 → 마이크 <strong>허용</strong>."
                    "</div>"
                )
                with gr.Row():
                    score_button = gr.Button(
                        "발음 채점하기", variant="primary", elem_id="score-btn"
                    )
                    clear_button = gr.ClearButton(
                        components=[audio_input],
                        value="초기화",
                        elem_id="clear-btn",
                    )

            # ── Status ────────────────────────────────────
            status_output = gr.Markdown(elem_id="status-markdown")

            # ── Results (hidden until scored) ─────────────
            with gr.Column(visible=False, elem_classes=["results-shell"]) as result_section:
                gr.HTML('<div class="section-title">Result</div>')
                with gr.Row(equal_height=True):
                    score_html = gr.HTML()
                    with gr.Column():
                        feedback_output = gr.Markdown(elem_id="feedback-markdown")
                        details_html = gr.HTML()

            # ── Footer ────────────────────────────────────
            gr.HTML("""
            <div class="footer-note">
              현재 버전은 원어민 기준 벡터와의 음향적 유사도를 점수화하는 실험용 MVP입니다.<br>
              완벽한 발음 판정기가 아니므로 참고용으로 활용해주세요.
            </div>
            """)

            # ── Event wiring ──────────────────────────────
            target_dropdown.change(
                fn=on_target_change,
                inputs=[target_dropdown],
                outputs=[target_card, status_output, result_section],
            )

            score_button.click(
                fn=score_audio,
                inputs=[target_dropdown, audio_input],
                outputs=[status_output, result_section, score_html, feedback_output, details_html],
            )

    return demo


if __name__ == "__main__":
    ensure_results_csv()
    app = build_app()
    app.launch()
