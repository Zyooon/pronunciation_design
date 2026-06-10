from pathlib import Path

import librosa
import numpy as np


DEFAULT_SAMPLE_RATE = 16_000


def load_audio(file_path: str | Path, sr: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    오디오 파일을 로드합니다.

    Args:
        file_path: wav, mp3 등 오디오 파일 경로
        sr: 샘플링 레이트. MVP에서는 16kHz로 통일합니다.

    Returns:
        y: 오디오 파형 데이터
        sr: 샘플링 레이트
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # mono=True: 스테레오 음성을 모노로 변환합니다.
    # sr=16000: 모든 입력 음성을 같은 샘플링 레이트로 맞춥니다.
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    if len(y) == 0:
        raise ValueError("Loaded audio is empty.")

    return y, sr


def trim_silence(
    y: np.ndarray,
    top_db: int = 20,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    VAD 방식으로 앞뒤 묵음을 제거합니다.

    librosa.effects.split은 음성 구간을 interval로 반환합니다.
    여기서는 여러 음성 구간이 있으면 하나로 이어 붙이지 않고,
    첫 음성 시작점부터 마지막 음성 끝점까지 잘라냅니다.

    Args:
        y: 오디오 파형 데이터
        top_db: 묵음 판단 기준. 값이 작을수록 더 엄격하게 자릅니다.
        frame_length: 분석 프레임 길이
        hop_length: 프레임 이동 간격

    Returns:
        trimmed_y: 묵음이 제거된 오디오 파형
    """
    if len(y) == 0:
        raise ValueError("Input audio is empty.")

    intervals = librosa.effects.split(
        y,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # 전부 묵음으로 판단된 경우, 원본을 그대로 반환합니다.
    # MVP에서는 앱이 죽지 않게 하는 것이 더 중요합니다.
    if len(intervals) == 0:
        return y

    start = intervals[0][0]
    end = intervals[-1][1]

    return y[start:end]


def load_and_trim_audio(
    file_path: str | Path,
    sr: int = DEFAULT_SAMPLE_RATE,
    top_db: int = 20,
) -> tuple[np.ndarray, int]:
    """
    오디오 로드와 VAD 묵음 제거를 한 번에 수행합니다.

    Args:
        file_path: 오디오 파일 경로
        sr: 샘플링 레이트
        top_db: VAD 묵음 제거 기준

    Returns:
        trimmed_y: 묵음이 제거된 오디오 파형
        sr: 샘플링 레이트
    """
    y, sr = load_audio(file_path, sr=sr)
    trimmed_y = trim_silence(y, top_db=top_db)

    return trimmed_y, sr


def get_duration_ms(y: np.ndarray, sr: int) -> float:
    """
    오디오 길이를 밀리초(ms) 단위로 계산합니다.

    Args:
        y: 오디오 파형 데이터
        sr: 샘플링 레이트

    Returns:
        duration_ms: 오디오 길이(ms)
    """
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    return round(len(y) / sr * 1000, 2)