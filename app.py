import csv
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import gradio as gr

from pipeline.audio import load_trimmed_audio
from pipeline.compare import (
    list_recording_choices,
    render_onset_analysis_html,
    render_reference_comparison_html,
    render_saved_recording_analysis,
)
from pipeline.db import save_user_recording_result
from pipeline.features import extract_features
from pipeline.quality import evaluate_recording_quality
from pipeline.reference import (
    ReferenceVector,
    TargetWord,
    build_target_index,
    get_available_targets,
    get_gradio_choices,
    get_reference_for_target_id,
    load_reference_vectors,
)
from pipeline.scorer import score_pronunciation
from pipeline.word_targets import attach_word_target_features, load_word_targets, should_extract_onset


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_PATH = DATA_DIR / "results.csv"
USER_RECORDINGS_DIR = DATA_DIR / "user_recordings"
ANALYSIS_DETAIL_KEYS = (
    "duration_ms",
    "zcr_mean",
    "rms_mean",
    "spectral_centroid_mean",
    "mfcc_mean",
    "target_id",
    "target_position",
    "target_phoneme",
    "onset_window_ms",
    "onset_mfcc_mean",
    "onset_zcr_mean",
    "onset_rms_mean",
    "onset_spectral_centroid_mean",
)


CUSTOM_CSS = """
body, .gradio-container {
  background: linear-gradient(160deg, #e8f4fd 0%, #dbeafe 100%) !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Apple SD Gothic Neo", sans-serif !important;
  color: #1e293b;
}
footer { display: none !important; }
.app-shell { max-width: 430px !important; margin: 0 auto !important; padding: 18px 0 70px !important; }
.analysis-shell { max-width: 720px !important; margin: 0 auto !important; padding: 18px 0 70px !important; }
.screen-card { background: #ffffff; border-radius: 24px; box-shadow: 0 4px 28px rgba(30,64,175,.10); overflow: hidden; margin: 0 12px; }
.hero { background: linear-gradient(135deg, #3b82f6, #6366f1); padding: 34px 24px 28px; text-align: center; color: #fff; }
.hero-title { font-size: 28px; font-weight: 800; letter-spacing: -.02em; }
.hero-sub { font-size: 13px; opacity: .88; margin-top: 6px; line-height: 1.5; }
.body-pad { padding: 18px; }
.target-preview, .target-card { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 16px; padding: 14px 16px; margin: 12px 0; }
.target-word { font-size: 30px; font-weight: 800; color: #1e40af; }
.target-phoneme { display: inline-block; margin-top: 6px; background: #dbeafe; color: #1d4ed8; border-radius: 999px; padding: 4px 12px; font-weight: 700; }
.target-hint { margin-top: 10px; color: #64748b; font-size: 13px; }
button { border-radius: 14px !important; }
#btn-start, #btn-analyze, #btn-change-word, #btn-refresh-analysis { background: #3b82f6 !important; color: #fff !important; border: 0 !important; height: 50px !important; font-weight: 700 !important; }
#btn-back, #btn-try-again { background: #fff !important; color: #2563eb !important; border: 1px solid #bfdbfe !important; height: 46px !important; font-weight: 700 !important; }
.recording-zone { background: linear-gradient(160deg, #1e3058, #18264a); padding: 20px 16px; }
.recording-title { color: #dbeafe; text-align: center; font-weight: 700; margin-bottom: 12px; }
.status-text p { color: #ef4444; font-size: 13px; font-weight: 700; }
.loading-wrap { display: flex; flex-direction: column; align-items: center; padding: 50px 20px; gap: 16px; }
.spinner-ring { width: 48px; height: 48px; border: 4px solid rgba(59,130,246,.18); border-top-color: #3b82f6; border-radius: 50%; animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text { font-size: 15px; color: #64748b; font-weight: 600; }
.result-target { display: flex; align-items: center; gap: 10px; padding: 18px 18px 0; }
.result-word { font-size: 28px; font-weight: 800; }
.result-phoneme { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; border-radius: 999px; font-size: 14px; font-weight: 700; padding: 5px 12px; }
.score-card { background: linear-gradient(135deg, #3b82f6, #6366f1); border-radius: 20px; margin: 14px 16px 0; padding: 26px 20px; text-align: center; color: #fff; }
.score-card.bad { background: linear-gradient(135deg, #f97316, #ef4444); }
.score-lbl { font-size: 11px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; opacity: .78; margin-bottom: 8px; }
.score-num { font-size: 68px; font-weight: 800; line-height: 1; letter-spacing: -.03em; }
.score-max { font-size: 18px; opacity: .68; }
.score-grade { font-size: 20px; font-weight: 800; margin-top: 8px; }
.score-grade-sub { font-size: 13px; opacity: .88; margin-top: 4px; line-height: 1.5; }
.feedback-card, .metrics-card, .issue-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 18px; margin: 12px 16px 0; }
.issue-card { background: #fff7ed; border-color: #fed7aa; color: #9a3412; }
.card-lbl { font-size: 11px; font-weight: 800; letter-spacing: .10em; text-transform: uppercase; color: #94a3b8; margin-bottom: 8px; }
.card-body { font-size: 14.5px; color: #334155; line-height: 1.75; }
.issue-card .card-lbl, .issue-card .card-body { color: #9a3412; }
.metric-row { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.metric-name { font-size: 12px; font-weight: 700; color: #64748b; width: 78px; text-align: right; }
.metric-track { flex: 1; height: 7px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
.metric-fill { height: 100%; background: linear-gradient(90deg, #3b82f6, #6366f1); border-radius: 999px; }
.metric-val { width: 38px; text-align: right; font-size: 13px; font-weight: 800; color: #1e40af; }
.actions { padding: 14px 16px 24px; gap: 10px; }
@media (max-width: 480px) { .app-shell, .analysis-shell { padding: 0 0 60px !important; } .screen-card { margin: 0; border-radius: 0 0 20px 20px; } }
"""

_LOADING_HTML = """
<div class="loading-wrap">
  <div class="spinner-ring"></div>
  <div class="loading-text">발음을 분석하고 있습니다...</div>
</div>
"""

_EMPTY_ANALYSIS_HTML = """
<div class="metrics-card">
  <div class="card-lbl">Analysis</div>
  <div class="card-body">저장된 녹음을 선택하면 내 음성과 영어 reference의 수치 비교가 표시됩니다.</div>
</div>
"""

_ISSUE_LABELS = {
    "too_short": "녹음이 너무 짧습니다",
    "too_long": "녹음이 너무 깁니다",
    "too_quiet": "소리가 너무 작습니다",
    "almost_silent": "거의 무음에 가깝습니다",
    "high_noise": "잡음이 높게 감지됐습니다",
    "extreme_zcr": "비정상적인 고주파/노이즈 패턴이 감지됐습니다",
    "word_mismatch": "목표 단어와 다른 단어가 감지됐습니다",
}


def ensure_results_csv() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        return
    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow([
            "timestamp",
            "target_word",
            "korean_pronunciation",
            "phoneme",
            "score",
            "pronunciation_score",
            "recording_quality_status",
            "issue_flags",
            "feedback",
            "mfcc_score",
            "duration_score",
            "rms_score",
            "zcr_score",
            "spectral_centroid_score",
        ])


def _csv_value(value: Any) -> str | float | int | None:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return value


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attach_analysis_details(result: dict[str, Any], features: dict[str, Any]) -> None:
    details = dict(result.get("details") or {})
    for key in ANALYSIS_DETAIL_KEYS:
        if key in features:
            details[key] = features[key]
    result["details"] = details


def _safe_filename_part(value: str | None, fallback: str = "unknown") -> str:
    text = (value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def _standardize_recording_file(audio_file: str, target: TargetWord) -> str:
    """Gradio 임시 녹음 파일을 표준 위치에 복사하고 프로젝트 기준 상대경로를 반환한다.

    새 녹음은 data/user_recordings/YYYY-MM-DD/{word}_{phoneme}_{HHMMSS}_{uuid}.ext 형태로 저장한다.
    기존 DB row는 건드리지 않는다.
    """
    source_path = Path(audio_file)
    if not source_path.exists() or not source_path.is_file():
        return audio_file

    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    time_part = now.strftime("%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    suffix = source_path.suffix.lower() or ".wav"
    filename = "_".join([
        _safe_filename_part(target.word),
        _safe_filename_part(target.phoneme, "phoneme"),
        time_part,
        short_id,
    ]) + suffix

    destination_dir = USER_RECORDINGS_DIR / date_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / filename
    shutil.copy2(source_path, destination_path)

    return destination_path.relative_to(PROJECT_ROOT).as_posix()


def save_result(
    *,
    target_word: str,
    korean_pronunciation: str,
    phoneme: str,
    result: dict[str, Any],
    features: dict[str, Any],
    recording_path: str | None,
) -> None:
    """CSV와 SQLite에 사용자 분석 결과를 저장한다."""
    ensure_results_csv()
    details = result.get("details", {})
    issue_flags = result.get("issue_flags", [])

    with RESULTS_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            target_word,
            korean_pronunciation,
            phoneme,
            result.get("score"),
            result.get("pronunciation_score"),
            result.get("recording_quality_status"),
            _csv_value(issue_flags),
            result.get("feedback"),
            details.get("mfcc_score", ""),
            details.get("duration_score", ""),
            details.get("rms_score", ""),
            details.get("zcr_score", ""),
            details.get("spectral_centroid_score", ""),
        ])

    save_user_recording_result(
        word=target_word,
        phoneme=phoneme,
        score=_float_or_none(result.get("score")),
        grade=_grade_for_storage(result),
        feedback=result.get("feedback"),
        recording_path=recording_path,
        duration_ms=_float_or_none(features.get("duration_ms")),
        rms_mean=_float_or_none(features.get("rms_mean")),
        zcr_mean=_float_or_none(features.get("zcr_mean")),
        spectral_centroid_mean=_float_or_none(features.get("spectral_centroid_mean")),
        details=details,
    )


def _init() -> tuple[dict[str, ReferenceVector], list[TargetWord], dict[str, TargetWord], list, dict, str | None]:
    try:
        reference_vectors = load_reference_vectors()
        target_words = get_available_targets()
        target_index = build_target_index(target_words)
        gradio_choices = get_gradio_choices(target_words)
        word_targets = load_word_targets()
        if not gradio_choices:
            raise ValueError("사용 가능한 단어가 없습니다.")
        return reference_vectors, target_words, target_index, gradio_choices, word_targets, None
    except Exception as exc:
        return {}, [], {}, [], {}, f"{type(exc).__name__}: {exc}"


REFERENCE_VECTORS, TARGET_WORDS, TARGET_INDEX, GRADIO_CHOICES, WORD_TARGETS, STARTUP_ERROR = _init()


def render_welcome_preview(target_id: str | None) -> str:
    if not target_id or target_id not in TARGET_INDEX:
        return '<div class="target-preview" style="opacity:.55;">단어를 선택하면 미리보기가 표시됩니다.</div>'
    target = TARGET_INDEX[target_id]
    return f"""
    <div class="target-preview">
      <div class="target-word">{target.word}</div>
      <div class="target-phoneme">/{target.phoneme}/</div>
    </div>
    """


def render_practice_header(target_id: str | None) -> str:
    if not target_id or target_id not in TARGET_INDEX:
        return ""
    target = TARGET_INDEX[target_id]
    return f"""
    <div class="target-card">
      <div class="target-word">{target.word}</div>
      <div class="target-phoneme">/{target.phoneme}/</div>
      <div class="target-hint">한국어 힌트: {target.korean_pronunciation}</div>
    </div>
    """


def _grade_for_score(score: float) -> tuple[str, str]:
    if score >= 85:
        return "Excellent", "기준 발음과 상당히 유사합니다."
    if score >= 70:
        return "Good", "전반적으로 좋지만 더 다듬을 수 있습니다."
    return "Needs Practice", "조금 더 반복 연습해보세요."


def _grade_for_storage(result: dict[str, Any]) -> str | None:
    score = result.get("score")
    if score is None:
        return None
    grade, _ = _grade_for_score(float(score))
    return grade


def _render_issue_card(issue_flags: list[str]) -> str:
    if not issue_flags:
        return ""
    issue_items = "<br>".join(f"• {_ISSUE_LABELS.get(flag, flag)}" for flag in issue_flags)
    return f"""
    <div class="issue-card">
      <div class="card-lbl">Recording Check</div>
      <div class="card-body">{issue_items}</div>
    </div>
    """


def _render_metric_rows(details: dict[str, Any]) -> str:
    label_map = {
        "mfcc_score": "MFCC",
        "duration_score": "Duration",
        "rms_score": "RMS",
        "zcr_score": "ZCR",
        "spectral_centroid_score": "Spectral",
    }
    rows: list[str] = []
    for key, label in label_map.items():
        value = details.get(key)
        if value in ("", None):
            continue
        numeric_value = _float_or_none(value)
        width = 0 if numeric_value is None else max(0, min(100, numeric_value))
        rows.append(
            f'<div class="metric-row">'
            f'<div class="metric-name">{label}</div>'
            f'<div class="metric-track"><div class="metric-fill" style="width:{width}%"></div></div>'
            f'<div class="metric-val">{value}</div>'
            f'</div>'
        )
    if not rows:
        return ""
    return f'<div class="metrics-card"><div class="card-lbl">Detailed Metrics</div>{"".join(rows)}</div>'


def render_result_content(target: TargetWord, result: dict[str, Any], reference: ReferenceVector) -> str:
    score = result.get("score")
    feedback = result.get("feedback", "")
    details = result.get("details", {})
    issue_flags = result.get("issue_flags") or details.get("issue_flags") or []

    if score is None:
        score_html = """
        <div class="score-card bad">
          <div class="score-lbl">Pronunciation Score</div>
          <div class="score-num">--</div>
          <div class="score-grade">재녹음 필요</div>
          <div class="score-grade-sub">녹음 상태 또는 단어 일치 여부를 먼저 확인해주세요.</div>
        </div>
        """
    else:
        numeric_score = float(score)
        grade, grade_sub = _grade_for_score(numeric_score)
        score_html = f"""
        <div class="score-card">
          <div class="score-lbl">Pronunciation Score</div>
          <div><span class="score-num">{numeric_score:.0f}</span><span class="score-max"> / 100</span></div>
          <div class="score-grade">{grade}</div>
          <div class="score-grade-sub">{grade_sub}</div>
        </div>
        """

    comparison_html = render_reference_comparison_html(details, reference)
    onset_html = render_onset_analysis_html(details)

    return f"""
    <div class="result-target">
      <div class="result-word">{target.word}</div>
      <div class="result-phoneme">/{target.phoneme}/</div>
    </div>
    {_render_issue_card(issue_flags)}
    {score_html}
    <div class="feedback-card">
      <div class="card-lbl">Feedback</div>
      <div class="card-body">{feedback}</div>
    </div>
    {_render_metric_rows(details)}
    {comparison_html}
    {onset_html}
    """


def on_target_change(target_id: str) -> str:
    return render_welcome_preview(target_id)


def start_practice(target_id: str) -> tuple:
    if not target_id:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            '<div style="color:#ef4444;font-weight:700;">단어를 먼저 선택해주세요.</div>',
        )
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        render_practice_header(target_id),
        "",
    )


def go_to_welcome() -> tuple:
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


def render_analysis_for_recording(recording_id: int | None) -> str:
    return render_saved_recording_analysis(recording_id, REFERENCE_VECTORS)


def refresh_analysis_options() -> tuple:
    choices = list_recording_choices()
    selected = choices[0][1] if choices else None
    html = render_analysis_for_recording(selected) if selected else _EMPTY_ANALYSIS_HTML
    return gr.update(choices=choices, value=selected), html


def analyze_pronunciation(target_id: str, audio_file: str | None) -> Generator:
    if not audio_file:
        yield gr.update(visible=True), gr.update(visible=False), "", "음성을 녹음하거나 파일을 업로드해주세요."
        return

    yield gr.update(visible=False), gr.update(visible=True), _LOADING_HTML, ""

    try:
        target, reference = get_reference_for_target_id(
            target_id=target_id,
            target_index=TARGET_INDEX,
            reference_vectors=REFERENCE_VECTORS,
        )
        waveform, sample_rate = load_trimmed_audio(audio_file)
        include_onset = should_extract_onset(target.word, target.phoneme, WORD_TARGETS)
        features = extract_features(waveform, sample_rate, include_onset=include_onset)
        attach_word_target_features(features, target.word, target.phoneme, WORD_TARGETS)
        quality_result = evaluate_recording_quality(
            features=features,
            reference=reference,
            audio_path=audio_file,
            target_word=target.word,
        )
        result = score_pronunciation(
            user_features=features,
            reference=reference,
            phoneme=target.phoneme,
            recording_quality_result=quality_result,
        )
        _attach_analysis_details(result, features)
        standardized_recording_path = _standardize_recording_file(audio_file, target)
        save_result(
            target_word=target.word,
            korean_pronunciation=target.korean_pronunciation,
            phoneme=target.phoneme,
            result=result,
            features=features,
            recording_path=standardized_recording_path,
        )
        yield gr.update(visible=False), gr.update(visible=True), render_result_content(target, result, reference), ""
    except Exception as exc:
        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            f'<div class="loading-wrap"><div class="loading-text">오류: {type(exc).__name__} — {exc}</div></div>',
            "",
        )


def try_again(target_id: str) -> tuple:
    return gr.update(visible=True), gr.update(visible=False), render_practice_header(target_id), None, ""


def change_word() -> tuple:
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), None


def build_practice_tab() -> None:
    default_id = GRADIO_CHOICES[0][1] if GRADIO_CHOICES else None
    with gr.Column(elem_classes=["app-shell"]):
        with gr.Column(visible=True, elem_id="screen_welcome", elem_classes=["screen-card"]) as screen_welcome:
            gr.HTML("""
            <div class="hero">
              <div class="hero-title">PronounceAI</div>
              <div class="hero-sub">한국인 학습자를 위한 AI 영어 발음 코치</div>
            </div>
            """)
            with gr.Column(elem_classes=["body-pad"]):
                target_dropdown = gr.Dropdown(
                    choices=GRADIO_CHOICES,
                    value=default_id,
                    label="연습할 단어 선택",
                    interactive=not bool(STARTUP_ERROR),
                )
                welcome_word_preview = gr.HTML(value=render_welcome_preview(default_id))
                welcome_error = gr.HTML(STARTUP_ERROR or "")
                start_btn = gr.Button("Start Practice →", elem_id="btn-start", variant="primary")

        with gr.Column(visible=False, elem_id="screen_practice", elem_classes=["screen-card"]) as screen_practice:
            with gr.Column(elem_classes=["body-pad"]):
                back_btn = gr.Button("← Back", elem_id="btn-back")
                practice_header = gr.HTML("")
            with gr.Column(elem_classes=["recording-zone"]):
                gr.HTML('<div class="recording-title">단어를 발음하고 녹음해주세요</div>')
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="녹음 또는 파일 업로드",
                    interactive=not bool(STARTUP_ERROR),
                )
            with gr.Column(elem_classes=["body-pad"]):
                practice_status = gr.Markdown(value="", elem_classes=["status-text"])
                analyze_btn = gr.Button(
                    "Analyze Pronunciation",
                    elem_id="btn-analyze",
                    variant="primary",
                    interactive=not bool(STARTUP_ERROR),
                )

        with gr.Column(visible=False, elem_id="screen_result", elem_classes=["screen-card"]) as screen_result:
            result_content = gr.HTML("")
            with gr.Row(elem_classes=["actions"]):
                try_again_btn = gr.Button("Try Again", elem_id="btn-try-again", variant="secondary")
                change_word_btn = gr.Button("Change Word", elem_id="btn-change-word", variant="primary")

        target_dropdown.change(fn=on_target_change, inputs=[target_dropdown], outputs=[welcome_word_preview])
        start_btn.click(
            fn=start_practice,
            inputs=[target_dropdown],
            outputs=[screen_welcome, screen_practice, screen_result, practice_header, welcome_error],
        )
        back_btn.click(fn=go_to_welcome, outputs=[screen_welcome, screen_practice, screen_result])
        analyze_btn.click(
            fn=analyze_pronunciation,
            inputs=[target_dropdown, audio_input],
            outputs=[screen_practice, screen_result, result_content, practice_status],
        )
        try_again_btn.click(
            fn=try_again,
            inputs=[target_dropdown],
            outputs=[screen_practice, screen_result, practice_header, audio_input, practice_status],
        )
        change_word_btn.click(
            fn=change_word,
            outputs=[screen_welcome, screen_practice, screen_result, audio_input],
        )


def build_analysis_tab() -> None:
    choices = list_recording_choices()
    selected = choices[0][1] if choices else None
    with gr.Column(elem_classes=["analysis-shell"]):
        with gr.Column(elem_classes=["screen-card"]):
            gr.HTML("""
            <div class="hero">
              <div class="hero-title">Analysis</div>
              <div class="hero-sub">저장된 녹음을 선택해 영어 reference와 수치 비교합니다</div>
            </div>
            """)
            with gr.Column(elem_classes=["body-pad"]):
                recording_dropdown = gr.Dropdown(
                    choices=choices,
                    value=selected,
                    label="분석할 저장 녹음 선택",
                    interactive=True,
                )
                refresh_btn = gr.Button("Refresh recordings", elem_id="btn-refresh-analysis", variant="primary")
                analysis_html = gr.HTML(value=render_analysis_for_recording(selected) if selected else _EMPTY_ANALYSIS_HTML)

        recording_dropdown.change(
            fn=render_analysis_for_recording,
            inputs=[recording_dropdown],
            outputs=[analysis_html],
        )
        refresh_btn.click(
            fn=refresh_analysis_options,
            outputs=[recording_dropdown, analysis_html],
        )


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo", neutral_hue="slate")
    with gr.Blocks(title="PronounceAI", theme=theme, css=CUSTOM_CSS) as demo:
        with gr.Tabs():
            with gr.Tab("Practice"):
                build_practice_tab()
            with gr.Tab("Analysis"):
                build_analysis_tab()
    return demo


if __name__ == "__main__":
    ensure_results_csv()
    app = build_app()
    app.launch()
