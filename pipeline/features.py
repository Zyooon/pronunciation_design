import librosa
import numpy as np


N_MFCC = 13
MFCC_SEGMENT_NAMES = ("start", "middle", "end")


def _as_list(values: np.ndarray) -> list[float]:
    return values.astype(float).tolist()


def extract_mfcc(waveform: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    MFCC 특징을 추출합니다.

    MFCC는 발음의 전체적인 음색/조음 차이를 숫자 벡터로 표현합니다.
    MVP에서는 시간축 전체를 평균내서 13차원 벡터로 사용합니다.
    """
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def extract_mfcc_time_features(waveform: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> dict[str, list[float]]:
    """MFCC의 전체 평균, 구간 평균, 변화량 평균을 함께 추출합니다."""
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    segments = np.array_split(mfcc, 3, axis=1)
    segment_features: dict[str, list[float]] = {}
    for name, segment in zip(MFCC_SEGMENT_NAMES, segments, strict=True):
        if segment.shape[1] == 0:
            segment_mean = mfcc_mean
        else:
            segment_mean = np.mean(segment, axis=1)
        segment_features[f"mfcc_{name}_mean"] = _as_list(segment_mean)

    if mfcc.shape[1] <= 1:
        delta_mfcc_mean = np.zeros(n_mfcc, dtype=float)
    else:
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1)

    return {
        "mfcc_mean": _as_list(mfcc_mean),
        "mfcc_std": _as_list(mfcc_std),
        "delta_mfcc_mean": _as_list(delta_mfcc_mean),
        **segment_features,
    }


def extract_zcr(waveform: np.ndarray) -> float:
    """Zero Crossing Rate를 추출합니다."""
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    zcr = librosa.feature.zero_crossing_rate(waveform)
    return float(np.mean(zcr))


def extract_duration_ms(waveform: np.ndarray, sr: int) -> float:
    """오디오 길이를 ms 단위로 계산합니다."""
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    return float(len(waveform) / sr * 1000)


def extract_rms(waveform: np.ndarray) -> float:
    """RMS energy를 추출합니다."""
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    rms = librosa.feature.rms(y=waveform)
    return float(np.mean(rms))


def extract_spectral_centroid(waveform: np.ndarray, sr: int) -> float:
    """Spectral Centroid를 추출합니다."""
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")

    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
    return float(np.mean(centroid))


def extract_features(waveform: np.ndarray, sr: int) -> dict[str, float | list[float]]:
    """채점과 reference vector 생성에 사용할 특징 dict를 반환합니다."""
    mfcc_features = extract_mfcc_time_features(waveform, sr)

    return {
        **mfcc_features,
        "zcr_mean": extract_zcr(waveform),
        "duration_ms": extract_duration_ms(waveform, sr),
        "rms_mean": extract_rms(waveform),
        "spectral_centroid_mean": extract_spectral_centroid(waveform, sr),
    }
