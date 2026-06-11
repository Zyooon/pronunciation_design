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


def ensure_results_csv() -> None:
    """
    결과 저장용 CSV 파일이 없으면 새로 만듭니다.
    """
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RESULTS_PATH.exists():
        return

    with RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "target_word",
                "korean_pronunciation",
                "phoneme",
                "score",
                "feedback",
                "mfcc_score",
                "duration_score",
                "rms_score",
                "zcr_score",
                "spectral_centroid_score",
            ]
        )


def save_result(
    target_word: str,
    korean_pronunciation: str,
    phoneme: str,
    result: dict[str, Any],
) -> None:
    """
    채점 결과를 data/results.csv에 저장합니다.

    음성 원본은 저장하지 않습니다.
    MVP에서는 개인정보와 파일 관리 부담을 줄이기 위해 점수 결과만 저장합니다.
    """
    ensure_results_csv()

    details = result.get("details", {})

    with RESULTS_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                target_word,
                korean_pronunciation,
                phoneme,
                result.get("score"),
                result.get("feedback"),
                details.get("mfcc_score", ""),
                details.get("duration_score", ""),
                details.get("rms_score", ""),
                details.get("zcr_score", ""),
                details.get("spectral_centroid_score", ""),
            ]
        )


# 앱 시작 시 reference와 target 목록을 한 번만 로드합니다.
REFERENCE_VECTORS = load_reference_vectors()
TARGET_WORDS = get_available_targets()
TARGET_INDEX = build_target_index(TARGET_WORDS)
GRADIO_CHOICES = get_gradio_choices(TARGET_WORDS)


def format_details(details: dict[str, Any]) -> str:
    """
    세부 점수를 Markdown으로 보기 좋게 변환합니다.
    """
    lines = ["### 세부 점수"]

    label_map = {
        "mfcc_score": "MFCC",
        "duration_score": "Duration",
        "rms_score": "RMS",
        "zcr_score": "ZCR",
        "spectral_centroid_score": "Spectral Centroid",
    }

    for key, label in label_map.items():
        if key in details:
            lines.append(f"- **{label}**: {details[key]}점")

    return "\n".join(lines)


def score_audio(target_id: str, audio_file: str | None) -> tuple[str, str, str]:
    """
    Gradio 버튼 클릭 시 실행되는 메인 함수입니다.

    Args:
        target_id: Gradio Dropdown에서 선택한 값. 예: "think|θ"
        audio_file: 사용자가 녹음하거나 업로드한 오디오 파일 경로

    Returns:
        score_text: 최종 점수 문자열
        feedback_text: 피드백 문자열
        detail_markdown: 세부 점수 Markdown
    """
    if not target_id:
        return "점수 없음", "먼저 타겟 단어를 선택해주세요.", ""

    if audio_file is None:
        return "점수 없음", "먼저 음성을 녹음하거나 업로드해주세요.", ""

    try:
        target, reference = get_reference_for_target_id(
            target_id=target_id,
            target_index=TARGET_INDEX,
            reference_vectors=REFERENCE_VECTORS,
        )

        # 1. 오디오 로드 + VAD 묵음 제거
        y, sr = load_and_trim_audio(audio_file)

        # 2. 음향 특징 추출
        user_features = extract_features(y, sr)

        # 3. reference vector와 비교해 채점
        result = score_pronunciation(
            user_features=user_features,
            reference=reference,
            phoneme=target.phoneme,
        )

        # 4. CSV 저장
        save_result(
            target_word=target.word,
            korean_pronunciation=target.korean_pronunciation,
            phoneme=target.phoneme,
            result=result,
        )

        score_text = f"{result['score']}점"
        feedback_text = (
            f"단어: {target.word}\n"
            f"타겟 음소: /{target.phoneme}/\n"
            f"한국어식 힌트: {target.korean_pronunciation}\n\n"
            f"{result['feedback']}"
        )
        detail_markdown = format_details(result["details"])

        return score_text, feedback_text, detail_markdown

    except Exception as e:
        return "오류", f"채점 중 오류가 발생했습니다.\n\n{type(e).__name__}: {e}", ""


def build_app() -> gr.Blocks:
    """
    Gradio UI를 생성합니다.
    """
    with gr.Blocks(title="Pronunciation Design") as demo:
        gr.Markdown(
            """
            # Korean Pronunciation Scoring System

            원어민 기준 벡터와 사용자의 단어 발음을 비교해  
            MFCC, ZCR, duration, RMS 기반 유사도 점수를 계산합니다.

            음성을 녹음하거나 mp3/wav 파일을 업로드한 뒤 채점 버튼을 눌러주세요.
            """
        )

        with gr.Row():
            target_dropdown = gr.Dropdown(
                choices=GRADIO_CHOICES,
                label="타겟 단어",
                value=GRADIO_CHOICES[0][1] if GRADIO_CHOICES else None,
            )

        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="음성 녹음 또는 업로드",
        )

        score_button = gr.Button("채점하기")

        with gr.Row():
            score_output = gr.Textbox(label="최종 점수", interactive=False)
            feedback_output = gr.Textbox(label="피드백", lines=6, interactive=False)

        detail_output = gr.Markdown(label="세부 점수")

        score_button.click(
            fn=score_audio,
            inputs=[target_dropdown, audio_input],
            outputs=[score_output, feedback_output, detail_output],
        )

        gr.Markdown(
            """
            ---
            현재 버전은 완벽한 발음 판정기가 아니라,  
            원어민 기준 벡터와의 음향적 유사도를 점수화하는 MVP 실험입니다.
            """
        )

    return demo


if __name__ == "__main__":
    ensure_results_csv()
    app = build_app()
    app.launch()