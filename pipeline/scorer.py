from typing import TypedDict

import numpy as np


EPSILON = 1e-8

AudioFeatures = dict[str, float | list[float]]
ReferenceVector = dict[str, float | list[float] | str]

_LIQUID_PHONEMES: frozenset[str] = frozenset({"r", "l"})

_MIN_DURATION_MS       = 150.0
_RMS_SILENT            = 0.005
_RMS_VERY_QUIET        = 0.015
_ZCR_ACTIVE_NOISE      = 0.35
_RMS_LOW_FOR_ZCR_CHECK = 0.020
_ZCR_EXTREME           = 0.50

_MFCC_LOW_THRESHOLD     = 55.0
_DURATION_LOW_THRESHOLD = 45.0
_CENTROID_LOW_THRESHOLD = 45.0
_VOWEL_MFCC_DOUBLE      = 60.0

_KO_RELATIVE_PENALTY_START = 50.0
_KO_RELATIVE_PENALTY_STRONG = 45.0
_KO_RELATIVE_PENALTY_SEVERE = 40.0
_KO_RELATIVE_PENALTY_MULTIPLIER = 0.60
_KO_RELATIVE_PENALTY_MAX = 15.0

_MISMATCH_BASE_SCORE_START = 80.0
_MISMATCH_BASE_SCORE_STRONG = 82.0
_MISMATCH_RELATIVE_SCORE_START = 50.0
_MISMATCH_RELATIVE_SCORE_STRONG = 47.0
_MISMATCH_RELATIVE_SCORE_SEVERE = 44.0
_MISMATCH_PENALTY_MAX = 14.0


class ScoreResult(TypedDict):
    score: float
    feedback: str
    details: dict[str, float]


def sigmoid_score(diff_ratio: float, steepness: float = 5.0, tolerance: float = 0.3) -> float:
    score = 100 / (1 + np.exp(steepness * (diff_ratio - tolerance)))
    return float(np.clip(score, 0, 100))


def z_score_distance_score(
    user_values: list[float] | np.ndarray,
    ref_mean: list[float] | np.ndarray,
    ref_std: list[float] | np.ndarray,
    scale: float = 10.0,
) -> float:
    user_values = np.array(user_values, dtype=float)
    ref_mean = np.array(ref_mean, dtype=float)
    ref_std = np.array(ref_std, dtype=float)
    z_scores = (user_values - ref_mean) / (ref_std + EPSILON)
    distance = np.mean(np.abs(z_scores))
    score = 100 - np.clip(distance * scale, 0, 100)
    return float(np.clip(score, 0, 100))


def ratio_feature_score(user_value: float, ref_value: float) -> float:
    diff_ratio = abs(user_value - ref_value) / (abs(ref_value) + EPSILON)
    return sigmoid_score(diff_ratio)


def _mfcc_distance(user_features: AudioFeatures, reference: ReferenceVector) -> float | None:
    user_mfcc = user_features.get("mfcc_mean")
    ref_mfcc = reference.get("mfcc_mean")
    if user_mfcc is None or ref_mfcc is None:
        return None
    try:
        return float(np.linalg.norm(np.array(user_mfcc, dtype=float) - np.array(ref_mfcc, dtype=float)))
    except Exception:
        return None


def compute_ko_reference_metrics(
    user_features: AudioFeatures,
    en_reference: ReferenceVector,
    ko_reference: ReferenceVector | None,
) -> dict[str, float]:
    """원어민 reference와 한국어식 reference 사이의 상대 위치 지표를 계산한다."""
    en_distance = _mfcc_distance(user_features, en_reference)
    ko_distance = _mfcc_distance(user_features, ko_reference) if ko_reference else None

    if en_distance is None or ko_distance is None:
        return {}

    relative_distance_score = ko_distance / (en_distance + ko_distance + EPSILON) * 100
    korean_like_penalty = max(
        0.0,
        (_KO_RELATIVE_PENALTY_START - relative_distance_score) * _KO_RELATIVE_PENALTY_MULTIPLIER,
    )
    if relative_distance_score < _KO_RELATIVE_PENALTY_STRONG:
        korean_like_penalty += 3.0
    if relative_distance_score < _KO_RELATIVE_PENALTY_SEVERE:
        korean_like_penalty += 5.0
    korean_like_penalty = float(np.clip(korean_like_penalty, 0.0, _KO_RELATIVE_PENALTY_MAX))

    return {
        "en_distance": round(en_distance, 4),
        "ko_distance": round(ko_distance, 4),
        "relative_distance_score": round(float(relative_distance_score), 1),
        "korean_like_penalty": round(korean_like_penalty, 1),
    }


def compute_mismatch_penalty(base_score: float, ko_metrics: dict[str, float]) -> float:
    """base_score가 높은데 한국어식 reference 쪽에 가까운 경우 추가 감점한다."""
    relative_score = ko_metrics.get("relative_distance_score")
    if relative_score is None or base_score < _MISMATCH_BASE_SCORE_START:
        return 0.0

    penalty = 0.0

    if relative_score < _MISMATCH_RELATIVE_SCORE_START:
        penalty += (_MISMATCH_RELATIVE_SCORE_START - relative_score) * 0.8

    if base_score >= _MISMATCH_BASE_SCORE_STRONG and relative_score < _MISMATCH_RELATIVE_SCORE_STRONG:
        penalty += 3.0

    if base_score >= _MISMATCH_BASE_SCORE_STRONG and relative_score < _MISMATCH_RELATIVE_SCORE_SEVERE:
        penalty += 5.0

    return round(float(np.clip(penalty, 0.0, _MISMATCH_PENALTY_MAX)), 1)


def score_vowel(user_features: AudioFeatures, reference: ReferenceVector) -> dict[str, float]:
    mfcc_score = z_score_distance_score(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    duration_score = ratio_feature_score(float(user_features["duration_ms"]), float(reference["duration_ms"]))
    centroid_score = ratio_feature_score(float(user_features["spectral_centroid_mean"]), float(reference["spectral_centroid_mean"]))
    rms_score = ratio_feature_score(float(user_features["rms_mean"]), float(reference["rms_mean"]))
    final_score = mfcc_score * 0.70 + duration_score * 0.15 + centroid_score * 0.10 + rms_score * 0.05
    return {
        "score": round(float(final_score), 1),
        "mfcc_score": round(float(mfcc_score), 1),
        "duration_score": round(float(duration_score), 1),
        "spectral_centroid_score": round(float(centroid_score), 1),
        "rms_score": round(float(rms_score), 1),
    }


def score_consonant(user_features: AudioFeatures, reference: ReferenceVector) -> dict[str, float]:
    mfcc_score = z_score_distance_score(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    zcr_score = ratio_feature_score(float(user_features["zcr_mean"]), float(reference["zcr_mean"]))
    centroid_score = ratio_feature_score(float(user_features["spectral_centroid_mean"]), float(reference["spectral_centroid_mean"]))
    final_score = mfcc_score * 0.55 + zcr_score * 0.35 + centroid_score * 0.10
    return {
        "score": round(float(final_score), 1),
        "mfcc_score": round(float(mfcc_score), 1),
        "zcr_score": round(float(zcr_score), 1),
        "spectral_centroid_score": round(float(centroid_score), 1),
    }


def score_liquid(user_features: AudioFeatures, reference: ReferenceVector) -> dict[str, float]:
    mfcc_score = z_score_distance_score(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    duration_score = ratio_feature_score(float(user_features["duration_ms"]), float(reference["duration_ms"]))
    centroid_score = ratio_feature_score(float(user_features["spectral_centroid_mean"]), float(reference["spectral_centroid_mean"]))
    zcr_score = ratio_feature_score(float(user_features["zcr_mean"]), float(reference["zcr_mean"]))
    final_score = mfcc_score * 0.75 + duration_score * 0.15 + centroid_score * 0.10
    return {
        "score": round(float(final_score), 1),
        "mfcc_score": round(float(mfcc_score), 1),
        "duration_score": round(float(duration_score), 1),
        "spectral_centroid_score": round(float(centroid_score), 1),
        "zcr_score": round(float(zcr_score), 1),
    }


def compute_quality_penalty(duration_ms: float, rms_mean: float, zcr_mean: float, ref_duration_ms: float) -> tuple[float, float, float]:
    if duration_ms < _MIN_DURATION_MS:
        duration_penalty = 30.0
    else:
        duration_ratio = duration_ms / (ref_duration_ms + EPSILON)
        if duration_ratio < 0.25 or duration_ratio > 4.0:
            duration_penalty = 20.0
        elif duration_ratio < 0.35 or duration_ratio > 3.0:
            duration_penalty = 10.0
        elif duration_ratio < 0.5 or duration_ratio > 2.5:
            duration_penalty = 4.0
        else:
            duration_penalty = 0.0

    if rms_mean < _RMS_SILENT:
        volume_penalty = 35.0
    elif rms_mean < _RMS_VERY_QUIET:
        volume_penalty = 10.0
    else:
        volume_penalty = 0.0

    if zcr_mean > _ZCR_ACTIVE_NOISE and rms_mean < _RMS_LOW_FOR_ZCR_CHECK:
        noise_penalty = 20.0
    elif zcr_mean > _ZCR_EXTREME:
        noise_penalty = 10.0
    else:
        noise_penalty = 0.0

    return duration_penalty, volume_penalty, noise_penalty


def compute_pronunciation_penalty(sub_scores: dict[str, float], phoneme_type: str) -> float:
    mfcc_score = sub_scores.get("mfcc_score", 100.0)
    duration_score = sub_scores.get("duration_score", 100.0)
    centroid_score = sub_scores.get("spectral_centroid_score", 100.0)
    penalty = 0.0
    if mfcc_score < _MFCC_LOW_THRESHOLD:
        penalty += 8.0
    if duration_score < _DURATION_LOW_THRESHOLD:
        penalty += 4.0
    if centroid_score < _CENTROID_LOW_THRESHOLD:
        penalty += 4.0
    if phoneme_type == "vowel" and mfcc_score < _VOWEL_MFCC_DOUBLE and centroid_score < _VOWEL_MFCC_DOUBLE:
        penalty += 6.0
    return penalty


def get_feedback(score: float, phoneme: str, phoneme_type: str) -> str:
    if score >= 85:
        return f"/{phoneme}/ 발음이 기준 발음과 꽤 비슷합니다."
    if score >= 70:
        return f"/{phoneme}/ 발음은 괜찮지만 조금 더 또렷하게 연습하면 좋습니다."
    phoneme_tips = {
        "θ": "혀끝을 윗니와 아랫니 사이에 가볍게 두고 공기를 빼보세요.",
        "f": "윗니를 아랫입술에 가볍게 대고 바람을 내보내세요.",
        "v": "윗니와 아랫입술을 가볍게 대고 목의 울림을 함께 내보내세요.",
        "i": "짧고 가볍게 발음하세요. 너무 길게 끌지 않는 것이 중요합니다.",
        "iː": "입꼬리를 옆으로 당기고 소리를 조금 더 길게 유지해보세요.",
        "æ": "입을 조금 더 크게 벌리고 턱을 낮춰서 발음해보세요.",
        "ə": "강하게 말하지 말고 짧고 약하게 지나가듯 발음해보세요.",
        "oʊ": "입술을 둥글게 모으며 뒤로 미끄러지듯 발음해보세요.",
        "r": "혀끝을 입천장에 붙이지 말고 뒤로 살짝 말아보세요.",
        "l": "혀끝을 윗잇몸 뒤쪽에 가볍게 붙여보세요.",
    }
    tip = phoneme_tips.get(phoneme)
    if tip:
        return f"/{phoneme}/ 발음 차이가 큽니다. {tip}"
    if phoneme_type == "vowel":
        return f"/{phoneme}/ 모음의 입 모양과 길이를 다시 확인해보세요."
    return f"/{phoneme}/ 자음의 조음 위치를 다시 확인해보세요."


def _build_phoneme_score_details(user_features: AudioFeatures, reference: ReferenceVector, phoneme_type: str, phoneme: str) -> dict[str, float]:
    if phoneme in _LIQUID_PHONEMES:
        return score_liquid(user_features, reference)
    if phoneme_type == "vowel":
        return score_vowel(user_features, reference)
    if phoneme_type == "consonant":
        return score_consonant(user_features, reference)
    mfcc_score = z_score_distance_score(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    return {"score": round(float(mfcc_score), 1), "mfcc_score": round(float(mfcc_score), 1)}


def score_pronunciation(
    user_features: AudioFeatures,
    reference: ReferenceVector,
    phoneme: str,
    ko_reference: ReferenceVector | None = None,
) -> ScoreResult:
    phoneme_type = str(reference.get("phoneme_type", "unknown"))
    sub_scores = _build_phoneme_score_details(user_features, reference, phoneme_type, phoneme)
    base_score = float(sub_scores.pop("score"))

    duration_ms = float(user_features.get("duration_ms", 0))
    rms_mean = float(user_features.get("rms_mean", 0))
    zcr_mean = float(user_features.get("zcr_mean", 0))
    spectral_centroid_mean = float(user_features.get("spectral_centroid_mean", 0))
    ref_duration_ms = float(reference.get("duration_ms", 500))

    duration_penalty, volume_penalty, noise_penalty = compute_quality_penalty(duration_ms, rms_mean, zcr_mean, ref_duration_ms)
    quality_penalty = duration_penalty + volume_penalty + noise_penalty
    pronunciation_penalty = compute_pronunciation_penalty(sub_scores, phoneme_type)
    ko_metrics = compute_ko_reference_metrics(user_features, reference, ko_reference)
    korean_like_penalty = ko_metrics.get("korean_like_penalty", 0.0)
    mismatch_penalty = compute_mismatch_penalty(base_score, ko_metrics)

    total_penalty = quality_penalty + pronunciation_penalty + korean_like_penalty + mismatch_penalty
    duration_ratio = duration_ms / (ref_duration_ms + EPSILON)
    final_score = float(np.clip(base_score - total_penalty, 0.0, 100.0))
    feedback = get_feedback(final_score, phoneme, phoneme_type)

    details: dict[str, float] = {
        **sub_scores,
        **ko_metrics,
        "base_score": round(base_score, 1),
        "quality_penalty": round(quality_penalty, 1),
        "duration_penalty": round(duration_penalty, 1),
        "volume_penalty": round(volume_penalty, 1),
        "noise_penalty": round(noise_penalty, 1),
        "pronunciation_penalty": round(pronunciation_penalty, 1),
        "mismatch_penalty": round(mismatch_penalty, 1),
        "total_penalty": round(total_penalty, 1),
        "final_score": round(final_score, 1),
        "duration_ratio": round(duration_ratio, 3),
        "rms_mean": round(rms_mean, 6),
        "zcr_mean": round(zcr_mean, 6),
        "spectral_centroid_mean": round(spectral_centroid_mean, 2),
    }

    return {"score": round(final_score, 1), "feedback": feedback, "details": details}
