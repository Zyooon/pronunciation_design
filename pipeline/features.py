import librosa
import numpy as np


N_MFCC = 13
MFCC_SEGMENT_NAMES = ("start", "middle", "end")
DEFAULT_ONSET_WINDOW_MS = 200.0
ONSET_RMS_NOISE_THRESHOLD = 0.005
ONSET_RMS_WINDOW_MS = 150.0
_ONSET_HOP_LENGTH = 512

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


def _find_valid_onset_sample(
    waveform: np.ndarray,
    sr: int,
    rms_threshold: float = ONSET_RMS_NOISE_THRESHOLD,
) -> int:
    """RMS 임계값을 넘는 첫 onset 샘플 인덱스를 반환한다.

    틱 노이즈·숨소리처럼 임계값 미만인 피크는 건너뛴다.
    유효한 onset을 찾지 못하면 0을 반환해 오디오 시작을 기준으로 삼는다.
    """
    onset_samples = librosa.onset.onset_detect(
        y=waveform, sr=sr, units="samples", hop_length=_ONSET_HOP_LENGTH
    )
    for raw_idx in onset_samples:
        sample_idx = int(raw_idx)
        frame_end = min(sample_idx + _ONSET_HOP_LENGTH, len(waveform))
        frame_rms = float(np.sqrt(np.mean(waveform[sample_idx:frame_end] ** 2)))
        if frame_rms >= rms_threshold:
            return sample_idx
    return 0


def extract_onset_window_features(
    waveform: np.ndarray,
    sr: int,
    window_ms: float = DEFAULT_ONSET_WINDOW_MS,
) -> dict[str, FeatureValue]:
    """trim 이후 오디오의 Onset 구간 특징을 추출합니다.

    librosa.onset.onset_detect로 유효 onset 시점을 감지한다.
    틱 노이즈·숨소리 방어: RMS 임계값 미만 피크는 건너뛴다.
    onset_rms_mean은 onset 감지 시점부터 150ms 구간의 평균 RMS다.
    나머지 특징(MFCC·ZCR·Spectral)은 onset 시점부터 window_ms 구간을 사용한다.
    """
    onset_sample = _find_valid_onset_sample(waveform, sr)
    onset_waveform = _slice_onset_window(waveform[onset_sample:], sr, window_ms)
    actual_window_ms = extract_duration_ms(onset_waveform, sr)

    rms_window_samples = max(1, int(sr * ONSET_RMS_WINDOW_MS / 1000))
    rms_window = waveform[onset_sample : onset_sample + rms_window_samples]
    if len(rms_window) == 0:
        rms_window = onset_waveform
    onset_rms_mean = extract_rms(rms_window)

    return {
        "onset_window_ms": actual_window_ms,
        "onset_mfcc_mean": _to_float_list(extract_mfcc(onset_waveform, sr)),
        "onset_zcr_mean": extract_zcr(onset_waveform),
        "onset_rms_mean": onset_rms_mean,
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
