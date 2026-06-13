import librosa
import numpy as np


N_MFCC = 13
MFCC_SEGMENT_NAMES = ("start", "middle", "end")
DEFAULT_ONSET_WINDOW_MS = 200.0

FeatureValue = float | list[float]


def _to_float_list(values: np.ndarray) -> list[float]:
    return values.astype(float).tolist()


def _extract_mfcc_frames(waveform: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")
    return librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)


def _slice_onset_window(waveform: np.ndarray, sr: int, window_ms: float = DEFAULT_ONSET_WINDOW_MS) -> np.ndarray:
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    window_samples = int(sr * window_ms / 1000)
    window_samples = max(1, min(len(waveform), window_samples))
    return waveform[:window_samples]


def extract_mfcc(waveform: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """시간축 전체를 평균낸 MFCC 벡터를 반환합니다."""
    mfcc = _extract_mfcc_frames(waveform, sr, n_mfcc)
    return np.mean(mfcc, axis=1)


def extract_mfcc_time_features(waveform: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> dict[str, list[float]]:
    """MFCC 전체 평균, 앞/중간/끝 구간 평균, 변화량 평균을 추출합니다."""
    mfcc = _extract_mfcc_frames(waveform, sr, n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_frame_std = np.std(mfcc, axis=1)

    features: dict[str, list[float]] = {
        "mfcc_mean": _to_float_list(mfcc_mean),
        "mfcc_frame_std": _to_float_list(mfcc_frame_std),
    }

    for name, segment in zip(MFCC_SEGMENT_NAMES, np.array_split(mfcc, 3, axis=1), strict=True):
        segment_mean = mfcc_mean if segment.shape[1] == 0 else np.mean(segment, axis=1)
        features[f"mfcc_{name}_mean"] = _to_float_list(segment_mean)

    if mfcc.shape[1] <= 1:
        delta_mfcc_mean = np.zeros(n_mfcc, dtype=float)
    else:
        delta_mfcc_mean = np.mean(np.diff(mfcc, axis=1), axis=1)
    features["delta_mfcc_mean"] = _to_float_list(delta_mfcc_mean)

    return features


def extract_zcr(waveform: np.ndarray) -> float:
    """Zero Crossing Rate 평균값을 반환합니다."""
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")
    return float(np.mean(librosa.feature.zero_crossing_rate(waveform)))


def extract_duration_ms(waveform: np.ndarray, sr: int) -> float:
    """오디오 길이를 ms 단위로 반환합니다."""
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    return float(len(waveform) / sr * 1000)


def extract_rms(waveform: np.ndarray) -> float:
    """RMS energy 평균값을 반환합니다."""
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")
    return float(np.mean(librosa.feature.rms(y=waveform)))


def extract_spectral_centroid(waveform: np.ndarray, sr: int) -> float:
    """Spectral centroid 평균값을 반환합니다."""
    if len(waveform) == 0:
        raise ValueError("Input audio is empty.")
    return float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr)))


def extract_onset_window_features(
    waveform: np.ndarray,
    sr: int,
    window_ms: float = DEFAULT_ONSET_WINDOW_MS,
) -> dict[str, FeatureValue]:
    """trim 이후 오디오의 앞쪽 고정 시간 window 특징을 추출합니다.

    단어 전체 평균 feature로 희석되는 onset 자음(/r/, /l/, /θ/, /f/, /v/)의
    분리 신호를 확인하기 위한 분석용 feature입니다. 현재 scorer 점수에는
    직접 반영하지 않고 CSV/DB details 분석에 사용합니다.
    """
    onset_waveform = _slice_onset_window(waveform, sr, window_ms)
    actual_window_ms = extract_duration_ms(onset_waveform, sr)

    return {
        "onset_window_ms": actual_window_ms,
        "onset_mfcc_mean": _to_float_list(extract_mfcc(onset_waveform, sr)),
        "onset_zcr_mean": extract_zcr(onset_waveform),
        "onset_rms_mean": extract_rms(onset_waveform),
        "onset_spectral_centroid_mean": extract_spectral_centroid(onset_waveform, sr),
    }


def extract_features(
    waveform: np.ndarray,
    sr: int,
    *,
    include_onset: bool = False,
    onset_window_ms: float = DEFAULT_ONSET_WINDOW_MS,
) -> dict[str, FeatureValue]:
    """채점과 reference vector 생성에 사용할 특징 dict를 반환합니다.

    include_onset=False를 기본값으로 둬 기존 reference vector 생성 기준선을 유지합니다.
    onset feature는 재채점/분석 루프에서 onset target 단어에만 명시적으로 켭니다.
    """
    features: dict[str, FeatureValue] = {
        **extract_mfcc_time_features(waveform, sr),
        "zcr_mean": extract_zcr(waveform),
        "duration_ms": extract_duration_ms(waveform, sr),
        "rms_mean": extract_rms(waveform),
        "spectral_centroid_mean": extract_spectral_centroid(waveform, sr),
    }
    if include_onset:
        features.update(extract_onset_window_features(waveform, sr, window_ms=onset_window_ms))
    return features
