from __future__ import annotations

from typing import Any

import librosa
import numpy as np

LIQUID_PHONEMES: frozenset[str] = frozenset({"r", "l"})

ANALYSIS_WINDOW_MS = 250.0
ENERGY_FRAME_MS = 10.0
ENERGY_HOP_MS = 5.0
ONSET_MFCC_WINDOW_MS = 50.0
VOWEL_MFCC_START_MS = 100.0
VOWEL_MFCC_WINDOW_MS = 80.0
DEFAULT_N_MELS = 64
DEFAULT_N_MFCC = 13

LOW_BAND_HZ = (0.0, 1000.0)
MID_BAND_HZ = (1000.0, 2000.0)
F3_BAND_HZ = (2000.0, 3000.0)
HIGH_BAND_HZ = (3000.0, 5000.0)

# /r/ 감지 임계값: F3 Drop 누락 판별
# R_LOW_STRONG_THRESHOLD: 0.002 → 0.001 로 완화 (good/road f3_to_low≈0.0017 오탐 방어)
R_LOW_STRONG_THRESHOLD = 0.001    # Strong AND 조건 — f3_to_low
R_MID_STRONG_THRESHOLD = 0.11     # Strong AND 조건 — f3_to_mid
R_LOW_WEAK_THRESHOLD = 0.0010     # Weak 단일 지표 — f3_to_low

# /l/ 감지 임계값: MFCC 전이 부재 판별
L_STRONG_THRESHOLD = 320.0        # Strong AND 조건 — transition_distance
L_C0_STRONG_THRESHOLD = 300.0     # Strong AND 조건 — c0_delta
L_WEAK_THRESHOLD = 340.0          # Weak OR 조건 — transition_distance
L_C0_WEAK_THRESHOLD = 330.0       # Weak OR 조건 — c0_delta

LIQUID_ACOUSTIC_STRONG_PENALTY = 7.0
LIQUID_ACOUSTIC_WEAK_PENALTY = 5.0
LIQUID_ACOUSTIC_MAX_PENALTY = 9.5

LiquidFeatureValue = float | None
LiquidFeatureDict = dict[str, LiquidFeatureValue]


def _slice_window(waveform: np.ndarray, sr: int, start_ms: float, window_ms: float) -> np.ndarray:
    start_sample = max(0, int(sr * start_ms / 1000.0))
    window_samples = max(1, int(sr * window_ms / 1000.0))
    end_sample = min(len(waveform), start_sample + window_samples)
    if end_sample <= start_sample:
        return np.array([], dtype=float)
    return waveform[start_sample:end_sample]


def _slice_analysis_window(waveform: np.ndarray, sr: int, window_ms: float = ANALYSIS_WINDOW_MS) -> np.ndarray:
    window_samples = max(1, int(sr * window_ms / 1000.0))
    return waveform[: min(len(waveform), window_samples)]


def _safe_n_fft(waveform: np.ndarray) -> int:
    if len(waveform) <= 1:
        return 64
    return min(512, max(64, 2 ** int(np.floor(np.log2(max(1, len(waveform)))))))


def compute_energy_features(
    waveform: np.ndarray,
    sr: int,
    *,
    frame_ms: float = ENERGY_FRAME_MS,
    hop_ms: float = ENERGY_HOP_MS,
) -> LiquidFeatureDict:
    frame_length = max(1, int(sr * frame_ms / 1000.0))
    hop_length = max(1, int(sr * hop_ms / 1000.0))
    empty = {
        "liquid_energy_frame_count": 0.0,
        "liquid_energy_rms_mean": None,
        "liquid_energy_rms_min": None,
        "liquid_energy_rms_max": None,
        "liquid_energy_delta_min": None,
        "liquid_energy_delta_max": None,
        "liquid_energy_dip_depth": None,
        "liquid_energy_peak_after_dip": None,
        "liquid_energy_v_shape_score": None,
        "liquid_energy_roughness": None,
    }
    if len(waveform) < frame_length:
        return empty

    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length, center=False)[0]
    if len(rms) == 0:
        return empty

    deltas = np.diff(rms) if len(rms) > 1 else np.array([0.0])
    min_idx = int(np.argmin(rms))
    pre_peak = float(np.max(rms[: min_idx + 1])) if min_idx >= 0 else float(rms[0])
    post_peak = float(np.max(rms[min_idx:])) if min_idx < len(rms) else float(rms[-1])
    dip = float(rms[min_idx])
    dip_depth = max(0.0, min(pre_peak, post_peak) - dip)
    peak_after_dip = max(0.0, post_peak - dip)
    rms_mean = float(np.mean(rms))
    v_shape_score = dip_depth / (rms_mean + 1e-8)
    roughness = float(np.mean(np.abs(deltas)) / (rms_mean + 1e-8))

    return {
        "liquid_energy_frame_count": float(len(rms)),
        "liquid_energy_rms_mean": round(rms_mean, 8),
        "liquid_energy_rms_min": round(float(np.min(rms)), 8),
        "liquid_energy_rms_max": round(float(np.max(rms)), 8),
        "liquid_energy_delta_min": round(float(np.min(deltas)), 8),
        "liquid_energy_delta_max": round(float(np.max(deltas)), 8),
        "liquid_energy_dip_depth": round(float(dip_depth), 8),
        "liquid_energy_peak_after_dip": round(float(peak_after_dip), 8),
        "liquid_energy_v_shape_score": round(float(v_shape_score), 6),
        "liquid_energy_roughness": round(float(roughness), 6),
    }


def _band_energy(mel_power: np.ndarray, mel_frequencies: np.ndarray, band: tuple[float, float]) -> float | None:
    low, high = band
    mask = (mel_frequencies >= low) & (mel_frequencies < high)
    if not np.any(mask):
        return None
    return float(np.mean(mel_power[mask, :]))


def compute_mel_band_features(waveform: np.ndarray, sr: int, *, n_mels: int = DEFAULT_N_MELS) -> LiquidFeatureDict:
    if len(waveform) == 0:
        return {
            "liquid_mel_low_band_energy": None,
            "liquid_mel_mid_band_energy": None,
            "liquid_mel_f3_band_energy": None,
            "liquid_mel_high_band_energy": None,
            "liquid_mel_f3_to_low_ratio": None,
            "liquid_mel_f3_to_mid_ratio": None,
            "liquid_mel_high_to_low_ratio": None,
            "liquid_mel_f3_band_log_energy": None,
        }

    n_fft = _safe_n_fft(waveform)
    hop_length = max(1, n_fft // 4)
    fmax = min(float(sr) / 2.0, HIGH_BAND_HZ[1])
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
        power=2.0,
    )
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fmax)

    low = _band_energy(mel, mel_frequencies, LOW_BAND_HZ)
    mid = _band_energy(mel, mel_frequencies, MID_BAND_HZ)
    f3 = _band_energy(mel, mel_frequencies, F3_BAND_HZ)
    high = _band_energy(mel, mel_frequencies, HIGH_BAND_HZ)

    return {
        "liquid_mel_low_band_energy": round(low, 8) if low is not None else None,
        "liquid_mel_mid_band_energy": round(mid, 8) if mid is not None else None,
        "liquid_mel_f3_band_energy": round(f3, 8) if f3 is not None else None,
        "liquid_mel_high_band_energy": round(high, 8) if high is not None else None,
        "liquid_mel_f3_to_low_ratio": round(f3 / (low + 1e-10), 6) if low is not None and f3 is not None else None,
        "liquid_mel_f3_to_mid_ratio": round(f3 / (mid + 1e-10), 6) if mid is not None and f3 is not None else None,
        "liquid_mel_high_to_low_ratio": round(high / (low + 1e-10), 6) if low is not None and high is not None else None,
        "liquid_mel_f3_band_log_energy": round(float(np.log10((f3 or 0.0) + 1e-10)), 6) if f3 is not None else None,
    }


def _mean_mfcc(waveform: np.ndarray, sr: int, n_mfcc: int) -> np.ndarray | None:
    if len(waveform) == 0:
        return None
    n_fft = _safe_n_fft(waveform)
    hop_length = max(1, n_fft // 4)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    if mfcc.shape[1] == 0:
        return None
    return np.mean(mfcc, axis=1)


def compute_mfcc_transition_features(waveform: np.ndarray, sr: int, *, n_mfcc: int = DEFAULT_N_MFCC) -> LiquidFeatureDict:
    onset = _slice_window(waveform, sr, 0.0, ONSET_MFCC_WINDOW_MS)
    vowel = _slice_window(waveform, sr, VOWEL_MFCC_START_MS, VOWEL_MFCC_WINDOW_MS)
    onset_mfcc = _mean_mfcc(onset, sr, n_mfcc)
    vowel_mfcc = _mean_mfcc(vowel, sr, n_mfcc)
    if onset_mfcc is None or vowel_mfcc is None:
        return {
            "liquid_transition_mfcc_distance": None,
            "liquid_transition_mfcc_slope": None,
            "liquid_transition_mfcc_delta_norm": None,
            "liquid_transition_mfcc_c0_delta": None,
            "liquid_transition_mfcc_c1_delta": None,
            "liquid_transition_mfcc_c2_delta": None,
        }

    delta = vowel_mfcc - onset_mfcc
    distance = float(np.linalg.norm(delta))
    slope = distance / max(1.0, VOWEL_MFCC_START_MS)
    return {
        "liquid_transition_mfcc_distance": round(distance, 6),
        "liquid_transition_mfcc_slope": round(float(slope), 6),
        "liquid_transition_mfcc_delta_norm": round(float(np.mean(np.abs(delta))), 6),
        "liquid_transition_mfcc_c0_delta": round(float(delta[0]), 6),
        "liquid_transition_mfcc_c1_delta": round(float(delta[1]), 6) if len(delta) > 1 else None,
        "liquid_transition_mfcc_c2_delta": round(float(delta[2]), 6) if len(delta) > 2 else None,
    }


def extract_liquid_acoustic_features(
    waveform: np.ndarray,
    sr: int,
    *,
    analysis_window_ms: float = ANALYSIS_WINDOW_MS,
    n_mels: int = DEFAULT_N_MELS,
    n_mfcc: int = DEFAULT_N_MFCC,
) -> LiquidFeatureDict:
    analysis_waveform = _slice_analysis_window(waveform, sr, analysis_window_ms)
    features: LiquidFeatureDict = {
        "liquid_analysis_window_ms": round(float(len(analysis_waveform) / sr * 1000.0), 3) if sr > 0 else None,
    }
    features.update(compute_energy_features(analysis_waveform, sr))
    features.update(compute_mel_band_features(analysis_waveform, sr, n_mels=n_mels))
    features.update(compute_mfcc_transition_features(analysis_waveform, sr, n_mfcc=n_mfcc))
    return features


def _compute_r_penalty(features: dict[str, Any]) -> tuple[str, float]:
    """F3 Drop 누락 여부를 두 지표의 AND/단일 조건으로 판별한다."""
    f3_to_low = features.get("liquid_mel_f3_to_low_ratio")
    if f3_to_low is None:
        return "missing_mel_f3_to_low_ratio", 0.0

    f3_to_low = float(f3_to_low)
    f3_to_mid_raw = features.get("liquid_mel_f3_to_mid_ratio")
    f3_to_mid = float(f3_to_mid_raw) if f3_to_mid_raw is not None else None

    if f3_to_low < R_LOW_STRONG_THRESHOLD and f3_to_mid is not None and f3_to_mid < R_MID_STRONG_THRESHOLD:
        return "korean_like", LIQUID_ACOUSTIC_STRONG_PENALTY

    if f3_to_low < R_LOW_WEAK_THRESHOLD:
        return "borderline_korean_like", LIQUID_ACOUSTIC_WEAK_PENALTY

    return "ok", 0.0


def _compute_l_penalty(features: dict[str, Any]) -> tuple[str, float]:
    """MFCC 전이 부재 여부를 AND(강한) / OR(약한) 조건으로 판별한다."""
    distance_raw = features.get("liquid_transition_mfcc_distance")
    c0_raw = features.get("liquid_transition_mfcc_c0_delta")

    if distance_raw is None and c0_raw is None:
        return "missing_transition_features", 0.0

    distance = float(distance_raw) if distance_raw is not None else None
    c0_delta = float(c0_raw) if c0_raw is not None else None

    if distance is not None and distance < L_STRONG_THRESHOLD and c0_delta is not None and c0_delta < L_C0_STRONG_THRESHOLD:
        return "korean_like", LIQUID_ACOUSTIC_STRONG_PENALTY

    weak_distance = distance is not None and distance < L_WEAK_THRESHOLD
    weak_c0 = c0_delta is not None and c0_delta < L_C0_WEAK_THRESHOLD
    if weak_distance or weak_c0:
        return "borderline_korean_like", LIQUID_ACOUSTIC_WEAK_PENALTY

    return "ok", 0.0


def compute_liquid_acoustic_penalty(features: dict[str, Any], phoneme: str) -> dict[str, float | str | bool | None]:
    """유음 음소의 한국어식 오독 패널티를 계산한다."""
    if phoneme not in LIQUID_PHONEMES:
        return {
            "liquid_acoustic_status": "not_applicable",
            "liquid_acoustic_penalty": 0.0,
            "liquid_acoustic_penalty_applied": False,
        }

    if phoneme == "r":
        status, penalty = _compute_r_penalty(features)
    else:
        status, penalty = _compute_l_penalty(features)

    capped_penalty = round(float(min(penalty, LIQUID_ACOUSTIC_MAX_PENALTY)), 1)
    return {
        "liquid_acoustic_status": status,
        "liquid_acoustic_penalty": capped_penalty,
        "liquid_acoustic_penalty_applied": penalty > 0.0,
    }
