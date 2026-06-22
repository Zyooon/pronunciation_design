"""오디오 로드와 전처리를 담당하는 모듈.

사용자 녹음 또는 reference 음성을 로드하고, sampling rate를 정규화한다.
VAD 기반 앞뒤 묵음 제거를 수행해 feature 추출에 사용할 waveform을 만든다.
"""

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
        waveform: 오디오 파형 데이터
        sr: 샘플링 레이트
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    waveform, sr = librosa.load(file_path, sr=sr, mono=True)

    if len(waveform) == 0:
        raise ValueError("Loaded audio is empty.")

    return waveform, sr


def trim_silence(
    waveform: np.ndarray,
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
        waveform: 오디오 파형 데이터
        top_db: 묵음 판단 기준. 값이 작을수록 더 엄격하게 자릅니다.
        frame_length: 분석 프레임 길이
        hop_length: 프레임 이동 간격

    Returns:
        trimmed_waveform: 묵음이 제거된 오디오 파형
    """
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    intervals = librosa.effects.split(
        waveform,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # 전부 묵음으로 판단된 경우, 원본을 그대로 반환합니다.
    # MVP에서는 앱이 죽지 않게 하는 것이 더 중요합니다.
    if len(intervals) == 0:
        return waveform

    start = intervals[0][0]
    end = intervals[-1][1]

    return waveform[start:end]


def load_trimmed_audio(
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
        trimmed_waveform: 묵음이 제거된 오디오 파형
        sr: 샘플링 레이트
    """
    waveform, sr = load_audio(file_path, sr=sr)
    trimmed_waveform = trim_silence(waveform, top_db=top_db)

    return trimmed_waveform, sr


def get_duration_ms(waveform: np.ndarray, sr: int) -> float:
    """
    오디오 길이를 밀리초(ms) 단위로 계산합니다.

    Args:
        waveform: 오디오 파형 데이터
        sr: 샘플링 레이트
    """
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    return round(len(waveform) / sr * 1000, 2)
