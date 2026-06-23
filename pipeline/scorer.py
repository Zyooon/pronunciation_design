"""발음 점수 계산과 피드백 생성을 담당하는 모듈.

사용자 음성 feature와 reference vector를 비교해 pronunciation score를 계산한다.
음소별 scoring rule, 한국어식 reference penalty, quality gate 결과를 반영해 최종 피드백과 세부 지표를 생성한다.
"""

from typing import Any, TypedDict

import numpy as np

from pipeline.liquid_features import compute_liquid_acoustic_penalty
from pipeline.quality import RecordingQualityResult, evaluate_recording_quality


EPSILON = 1e-8

AudioFeatures = dict[str, Any]
ReferenceVector = dict[str, float | list[float] | str]
ScoreDetailValue = float | str | list[str] | bool | None

_LIQUID_PHONEMES: frozenset[str] = frozenset({"r", "l"})
_DURATION_FOCUSED_PHONEMES: frozenset[str] = frozenset({"i", "iː"})
_KO_REFERENCE_PHONEMES: frozenset[str] = frozenset({"θ", "v", "æ", "f", "i"})
_SCHWA_PHONEMES: frozenset[str] = frozenset({"ə"})
_VOWEL_PHONEMES: frozenset[str] = frozenset({"i", "iː", "æ", "ə", "oʊ"})

_MFCC_LOW_THRESHOLD = 55.0
_DURATION_LOW_THRESHOLD = 45.0
_DURATION_VERY_LOW_THRESHOLD = 35.0
_CENTROID_LOW_THRESHOLD = 45.0
_ZCR_LOW_THRESHOLD = 45.0

_RMS_SCORE_FLOOR = 0.01
_RMS_SCORE_TOLERANCE = 1.2
_RMS_SCORE_STEEPNESS = 3.0

_ZCR_SCORE_FLOOR = 0.08
_ZCR_SCORE_TOLERANCE = 0.55
_ZCR_SCORE_STEEPNESS = 4.0
_ZCR_OVERSHOOT_PENALTY_SCALE = 0.3

_KO_RELATIVE_PENALTY_START = 62.0
_KO_RELATIVE_PENALTY_STRONG = 52.0
_KO_RELATIVE_PENALTY_MULTIPLIER = 0.8
_KO_RELATIVE_PENALTY_MAX = 15.0

_SCHWA_ONSET_RMS_FLOOR = 0.001
_SCHWA_ONSET_RMS_RATIO_START = 1.20
_SCHWA_ONSET_TO_TOTAL_RATIO_START = 1.05
_SCHWA_ONSET_RMS_MULTIPLIER = 28.0
_SCHWA_ONSET_TO_TOTAL_MULTIPLIER = 28.0
_SCHWA_OVERSTRESS_MAX_PENALTY = 12.0

_LIQUID_ALT_SCORE_START = 53.0
_LIQUID_ALT_SCORE_STRONG = 46.0
_LIQUID_ALT_PENALTY_MULTIPLIER = 0.55
_LIQUID_ALT_PENALTY_MAX = 10.0

_LIQUID_ONSET_SCORE_START = 56.0
_LIQUID_ONSET_SCORE_STRONG = 48.0
_LIQUID_ONSET_PENALTY_MULTIPLIER = 0.5
_LIQUID_ONSET_PENALTY_MAX = 8.0

# 데이터 수집 단계의 보수적 가설치 — 충분한 샘플 확보 후 재조정 필요
_F_ONSET_MFCC_DISTANCE_THRESHOLD = 20.0

# good/road(≈0.0017) 대비 korean_like/road(0.0001~0.0002)의 8~10배 격차를 이용한 핀셋 조건
_R_MICRO_F3_TO_LOW_THRESHOLD = 0.0003

_QUALITY_GATE_FEEDBACK = "녹음 품질이 낮아 발음 점수를 신뢰하기 어렵습니다. 다시 녹음해주세요."
_WORD_MISMATCH_FEEDBACK = "녹음된 단어가 목표 단어와 달라 발음 점수를 신뢰하기 어렵습니다. 목표 단어를 다시 녹음해주세요."


class ScoreResult(TypedDict):
    score: float | None
    pronunciation_score: float | None
    recording_quality_status: str
    issue_flags: list[str]
    feedback: str
    details: dict[str, ScoreDetailValue]


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


def z_score_distance_score_without_c0(
    user_values: list[float] | np.ndarray,
    ref_mean: list[float] | np.ndarray,
    ref_std: list[float] | np.ndarray,
    scale: float = 10.0,
) -> float:
    """MFCC 0번 계수(C0)를 제외한 z-score 거리 점수를 계산한다.

    C0는 전체 spectral energy 성격이 강해 pitch/개인 음색 차이에 민감하다.
    길이가 2 미만이면 기존 z_score_distance_score로 fallback한다.
    """
    user_arr = np.array(user_values, dtype=float)
    ref_mean_arr = np.array(ref_mean, dtype=float)
    ref_std_arr = np.array(ref_std, dtype=float)
    if len(user_arr) < 2:
        return z_score_distance_score(user_arr, ref_mean_arr, ref_std_arr, scale)
    return z_score_distance_score(user_arr[1:], ref_mean_arr[1:], ref_std_arr[1:], scale)


def ratio_feature_score(user_value: float, ref_value: float) -> float:
    diff_ratio = abs(user_value - ref_value) / (abs(ref_value) + EPSILON)
    return sigmoid_score(diff_ratio)


def rms_feature_score(user_value: float, ref_value: float) -> float:
    """RMS는 녹음 볼륨 차이에 민감하므로 floor가 있는 log-ratio로 완화한다."""
    adjusted_user = max(float(user_value), 0.0) + _RMS_SCORE_FLOOR
    adjusted_ref = max(float(ref_value), 0.0) + _RMS_SCORE_FLOOR
    log_diff = abs(np.log(adjusted_user / adjusted_ref))
    return sigmoid_score(
        float(log_diff),
        steepness=_RMS_SCORE_STEEPNESS,
        tolerance=_RMS_SCORE_TOLERANCE,
    )


def zcr_feature_score(user_value: float, ref_value: float) -> float:
    user_value = float(user_value)
    ref_value = float(ref_value)
    baseline = max(abs(ref_value), _ZCR_SCORE_FLOOR)

    if user_value >= ref_value:
        diff_ratio = ((user_value - ref_value) / (baseline + EPSILON)) * _ZCR_OVERSHOOT_PENALTY_SCALE
    else:
        diff_ratio = (ref_value - user_value) / (baseline + EPSILON)

    return sigmoid_score(
        float(diff_ratio),
        steepness=_ZCR_SCORE_STEEPNESS,
        tolerance=_ZCR_SCORE_TOLERANCE,
    )


def _mfcc_distance(user_features: AudioFeatures, reference: ReferenceVector | None) -> float | None:
    return _vector_distance(user_features, reference, "mfcc_mean")


def _vector_distance(user_features: AudioFeatures, reference: ReferenceVector | None, key: str) -> float | None:
    if reference is None:
        return None
    user_values = user_features.get(key)
    ref_values = reference.get(key)
    if user_values is None or ref_values is None:
        return None
    try:
        return float(np.linalg.norm(np.array(user_values, dtype=float) - np.array(ref_values, dtype=float)))
    except Exception:
        return None


def _round_sub_scores(**values: float) -> dict[str, float]:
    return {key: round(float(value), 1) for key, value in values.items()}


def _get_korean_pattern_status(en_distance: float, ko_distance: float, relative_distance_score: float) -> str:
    if ko_distance < en_distance:
        return "korean_like"
    if relative_distance_score < _KO_RELATIVE_PENALTY_START:
        return "borderline_korean_like"
    return "english_like"


def _get_korean_pattern_diagnosis(status: str) -> str:
    if status == "korean_like":
        return "영어 reference보다 한국어식 reference에 더 가까운 경향이 있습니다."
    if status == "borderline_korean_like":
        return "영어 reference에 더 가깝지만 한국어식 패턴도 일부 감지됩니다."
    return "한국어식 reference보다 영어 reference에 더 가까운 경향입니다."


def compute_ko_reference_metrics(
    user_features: AudioFeatures,
    en_reference: ReferenceVector,
    ko_reference: ReferenceVector | None,
) -> dict[str, ScoreDetailValue]:
    """한국어식 reference는 점수로 더하지 않고 penalty 트랩으로만 사용한다."""
    en_distance = _mfcc_distance(user_features, en_reference)
    ko_distance = _mfcc_distance(user_features, ko_reference)
    if en_distance is None or ko_distance is None:
        return {
            "korean_pattern_status": "unavailable",
            "korean_pattern_penalty_policy": "penalty_only",
            "korean_pattern_diagnosis": "한국어식 reference 비교를 계산할 수 없습니다.",
        }

    relative_distance_score = ko_distance / (en_distance + ko_distance + EPSILON) * 100
    korean_like_penalty = max(
        0.0,
        (_KO_RELATIVE_PENALTY_START - relative_distance_score) * _KO_RELATIVE_PENALTY_MULTIPLIER,
    )
    if relative_distance_score < _KO_RELATIVE_PENALTY_STRONG:
        korean_like_penalty += 3.0
    korean_like_penalty = float(np.clip(korean_like_penalty, 0.0, _KO_RELATIVE_PENALTY_MAX))

    status = _get_korean_pattern_status(en_distance, ko_distance, float(relative_distance_score))
    is_korean_reference_closer = ko_distance < en_distance

    return {
        "en_distance": round(en_distance, 4),
        "ko_distance": round(ko_distance, 4),
        "ko_minus_en_distance": round(ko_distance - en_distance, 4),
        "relative_distance_score": round(float(relative_distance_score), 1),
        "korean_like_penalty": round(korean_like_penalty, 1),
        "korean_pattern_status": status,
        "korean_pattern_is_closer": is_korean_reference_closer,
        "korean_pattern_penalty_applied": korean_like_penalty > 0.0,
        "korean_pattern_penalty_policy": "penalty_only",
        "korean_pattern_diagnosis": _get_korean_pattern_diagnosis(status),
    }


def compute_schwa_overstress_metrics(
    user_features: AudioFeatures,
    reference: ReferenceVector,
    phoneme: str,
) -> dict[str, ScoreDetailValue]:
    """/ə/는 단어 초반(Onset)에 과도한 에너지가 몰리는 것을 감점한다."""
    if phoneme not in _SCHWA_PHONEMES:
        return {
            "schwa_overstress_status": "not_applicable",
            "schwa_overstress_penalty": 0.0,
        }

    user_onset_rms = float(user_features.get("onset_rms_mean", 0.0))
    user_total_rms = float(user_features.get("rms_mean", 0.0))
    ref_onset_rms = float(reference.get("onset_rms_mean", 0.0))

    if user_onset_rms == 0.0 or ref_onset_rms == 0.0:
        return {"schwa_overstress_status": "missing_onset_features", "schwa_overstress_penalty": 0.0}

    onset_rms_ratio = (user_onset_rms + _SCHWA_ONSET_RMS_FLOOR) / (ref_onset_rms + _SCHWA_ONSET_RMS_FLOOR)
    onset_to_total_ratio = (user_onset_rms + _SCHWA_ONSET_RMS_FLOOR) / (user_total_rms + _SCHWA_ONSET_RMS_FLOOR)

    onset_rms_excess = max(0.0, float(np.log(onset_rms_ratio)) - float(np.log(_SCHWA_ONSET_RMS_RATIO_START)))
    onset_to_total_excess = max(0.0, onset_to_total_ratio - _SCHWA_ONSET_TO_TOTAL_RATIO_START)

    penalty = onset_rms_excess * _SCHWA_ONSET_RMS_MULTIPLIER + onset_to_total_excess * _SCHWA_ONSET_TO_TOTAL_MULTIPLIER
    penalty = float(np.clip(penalty, 0.0, _SCHWA_OVERSTRESS_MAX_PENALTY))

    status = "overstressed" if penalty > 0.0 else "ok"
    return {
        "schwa_overstress_status": status,
        "schwa_overstress_penalty": round(penalty, 1),
        "schwa_onset_rms": round(user_onset_rms, 6),
        "schwa_total_rms": round(user_total_rms, 6),
        "schwa_ref_onset_rms": round(ref_onset_rms, 6),
        "schwa_onset_rms_ratio": round(onset_rms_ratio, 4),
        "schwa_onset_to_total_rms_ratio": round(onset_to_total_ratio, 4),
        "schwa_onset_rms_excess": round(onset_rms_excess, 4),
        "schwa_onset_to_total_excess": round(onset_to_total_excess, 4),
    }


def compute_liquid_alt_metrics(
    user_features: AudioFeatures,
    target_reference: ReferenceVector,
    alt_reference: ReferenceVector | None,
) -> dict[str, ScoreDetailValue]:
    target_distance = _mfcc_distance(user_features, target_reference)
    alt_distance = _mfcc_distance(user_features, alt_reference)
    if target_distance is None or alt_distance is None:
        return {
            "liquid_alt_status": "unavailable",
            "liquid_alt_penalty": 0.0,
        }

    alt_relative_score = alt_distance / (target_distance + alt_distance + EPSILON) * 100
    liquid_alt_penalty = max(
        0.0,
        (_LIQUID_ALT_SCORE_START - alt_relative_score) * _LIQUID_ALT_PENALTY_MULTIPLIER,
    )
    if alt_relative_score < _LIQUID_ALT_SCORE_STRONG:
        liquid_alt_penalty += 3.0
    liquid_alt_penalty = float(np.clip(liquid_alt_penalty, 0.0, _LIQUID_ALT_PENALTY_MAX))

    status = "korean_like" if alt_distance < target_distance else "english_like"
    if liquid_alt_penalty > 0.0 and status == "english_like":
        status = "borderline_korean_like"

    return {
        "liquid_alt_status": status,
        "liquid_target_distance": round(target_distance, 4),
        "liquid_alt_distance": round(alt_distance, 4),
        "liquid_alt_relative_score": round(float(alt_relative_score), 1),
        "liquid_alt_penalty": round(liquid_alt_penalty, 1),
    }


def compute_liquid_onset_metrics(
    user_features: AudioFeatures,
    target_reference: ReferenceVector,
    alt_reference: ReferenceVector | None,
    phoneme: str,
) -> dict[str, ScoreDetailValue]:
    """/r/, /l/은 전체 단어보다 onset MFCC가 한국어식 ㄹ 계열에 가까운지 별도로 본다."""
    if phoneme not in _LIQUID_PHONEMES:
        return {
            "liquid_onset_status": "not_applicable",
            "liquid_onset_penalty": 0.0,
        }

    target_distance = _vector_distance(user_features, target_reference, "onset_mfcc_mean")
    alt_distance = _vector_distance(user_features, alt_reference, "onset_mfcc_mean")
    if target_distance is None or alt_distance is None:
        return {
            "liquid_onset_status": "missing_onset_features",
            "liquid_onset_penalty": 0.0,
        }

    relative_score = alt_distance / (target_distance + alt_distance + EPSILON) * 100
    if _LIQUID_ONSET_PENALTY_MULTIPLIER <= 0.0:
        penalty = 0.0
    else:
        penalty = max(
            0.0,
            (_LIQUID_ONSET_SCORE_START - relative_score) * _LIQUID_ONSET_PENALTY_MULTIPLIER,
        )
        if relative_score < _LIQUID_ONSET_SCORE_STRONG:
            penalty += 2.0
        penalty = float(np.clip(penalty, 0.0, _LIQUID_ONSET_PENALTY_MAX))

    status = "korean_like" if alt_distance < target_distance else "english_like"
    if penalty > 0.0 and status == "english_like":
        status = "borderline_korean_like"

    return {
        "liquid_onset_status": status,
        "liquid_onset_en_distance": round(target_distance, 4),
        "liquid_onset_ko_distance": round(alt_distance, 4),
        "liquid_onset_relative_score": round(float(relative_score), 1),
        "liquid_onset_penalty": round(penalty, 1),
    }


def score_vowel(user_features: AudioFeatures, reference: ReferenceVector) -> dict[str, float]:
    mfcc_score = z_score_distance_score(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    mfcc_no_c0_score = z_score_distance_score_without_c0(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    duration_score = ratio_feature_score(float(user_features["duration_ms"]), float(reference["duration_ms"]))
    centroid_score = ratio_feature_score(float(user_features["spectral_centroid_mean"]), float(reference["spectral_centroid_mean"]))
    rms_score = rms_feature_score(float(user_features["rms_mean"]), float(reference["rms_mean"]))
    final_score = mfcc_score * 0.70 + duration_score * 0.15 + centroid_score * 0.10 + rms_score * 0.05

    return _round_sub_scores(
        score=final_score,
        mfcc_score=mfcc_score,
        mfcc_no_c0_score=mfcc_no_c0_score,
        duration_score=duration_score,
        spectral_centroid_score=centroid_score,
        rms_score=rms_score,
    )


def score_duration_focused_vowel(user_features: AudioFeatures, reference: ReferenceVector) -> dict[str, float]:
    mfcc_score = z_score_distance_score(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    mfcc_no_c0_score = z_score_distance_score_without_c0(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    duration_score = ratio_feature_score(float(user_features["duration_ms"]), float(reference["duration_ms"]))
    centroid_score = ratio_feature_score(float(user_features["spectral_centroid_mean"]), float(reference["spectral_centroid_mean"]))
    rms_score = rms_feature_score(float(user_features["rms_mean"]), float(reference["rms_mean"]))
    final_score = mfcc_score * 0.50 + duration_score * 0.35 + centroid_score * 0.10 + rms_score * 0.05

    return _round_sub_scores(
        score=final_score,
        mfcc_score=mfcc_score,
        mfcc_no_c0_score=mfcc_no_c0_score,
        duration_score=duration_score,
        spectral_centroid_score=centroid_score,
        rms_score=rms_score,
    )


def score_consonant(user_features: AudioFeatures, reference: ReferenceVector) -> dict[str, float]:
    mfcc_score = z_score_distance_score(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    mfcc_no_c0_score = z_score_distance_score_without_c0(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    zcr_score = zcr_feature_score(float(user_features["zcr_mean"]), float(reference["zcr_mean"]))
    centroid_score = ratio_feature_score(float(user_features["spectral_centroid_mean"]), float(reference["spectral_centroid_mean"]))
    final_score = mfcc_score * 0.55 + zcr_score * 0.35 + centroid_score * 0.10

    return _round_sub_scores(
        score=final_score,
        mfcc_score=mfcc_score,
        mfcc_no_c0_score=mfcc_no_c0_score,
        zcr_score=zcr_score,
        spectral_centroid_score=centroid_score,
    )


def score_liquid(user_features: AudioFeatures, reference: ReferenceVector) -> dict[str, float]:
    mfcc_score = z_score_distance_score(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    mfcc_no_c0_score = z_score_distance_score_without_c0(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    duration_score = ratio_feature_score(float(user_features["duration_ms"]), float(reference["duration_ms"]))
    centroid_score = ratio_feature_score(float(user_features["spectral_centroid_mean"]), float(reference["spectral_centroid_mean"]))
    zcr_score = zcr_feature_score(float(user_features["zcr_mean"]), float(reference["zcr_mean"]))
    final_score = mfcc_score * 0.75 + duration_score * 0.15 + centroid_score * 0.10

    return _round_sub_scores(
        score=final_score,
        mfcc_score=mfcc_score,
        mfcc_no_c0_score=mfcc_no_c0_score,
        duration_score=duration_score,
        spectral_centroid_score=centroid_score,
        zcr_score=zcr_score,
        liquid_zcr_score_weight=0.0,
    )


def compute_pronunciation_penalty(sub_scores: dict[str, float], phoneme_type: str, phoneme: str) -> float:
    mfcc_score = sub_scores.get("mfcc_score", 100.0)
    duration_score = sub_scores.get("duration_score", 100.0)
    centroid_score = sub_scores.get("spectral_centroid_score", 100.0)
    zcr_score = sub_scores.get("zcr_score", 100.0)

    if phoneme in _LIQUID_PHONEMES:
        penalty = 0.0
        if mfcc_score < _MFCC_LOW_THRESHOLD:
            penalty += 5.0
        return round(float(penalty), 1)

    penalty = 0.0
    if mfcc_score < _MFCC_LOW_THRESHOLD:
        penalty += 7.0
    if duration_score < _DURATION_LOW_THRESHOLD:
        penalty += 8.0 if phoneme_type == "vowel" else 4.0
        if phoneme_type == "vowel" and duration_score < _DURATION_VERY_LOW_THRESHOLD:
            penalty += 3.0
    centroid_is_low = centroid_score < _CENTROID_LOW_THRESHOLD
    zcr_is_low = zcr_score < _ZCR_LOW_THRESHOLD

    if phoneme == "f":
        # /f/는 후행 자음 환경에서 centroid가 왜곡될 수 있으므로 ZCR과 AND 교차 검증
        if centroid_is_low and zcr_is_low:
            penalty += 3.0
    elif centroid_is_low and phoneme_type != "vowel":
        penalty += 3.0

    if phoneme_type == "consonant" and zcr_is_low:
        penalty += 4.0
    return round(float(penalty), 1)


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


def _build_phoneme_score_details(
    user_features: AudioFeatures,
    reference: ReferenceVector,
    phoneme_type: str,
    phoneme: str,
) -> dict[str, float]:
    if phoneme in _DURATION_FOCUSED_PHONEMES:
        return score_duration_focused_vowel(user_features, reference)
    if phoneme in _LIQUID_PHONEMES:
        return score_liquid(user_features, reference)
    if phoneme_type == "vowel":
        return score_vowel(user_features, reference)
    if phoneme_type == "consonant":
        return score_consonant(user_features, reference)

    mfcc_score = z_score_distance_score(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    mfcc_no_c0_score = z_score_distance_score_without_c0(user_features["mfcc_mean"], reference["mfcc_mean"], reference["mfcc_std"])
    return {
        "score": round(float(mfcc_score), 1),
        "mfcc_score": round(float(mfcc_score), 1),
        "mfcc_no_c0_score": round(float(mfcc_no_c0_score), 1),
    }


def _coerce_issue_flags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(flag) for flag in value]
    return []


def _get_optional_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _get_quality_detail_fields(quality_result: RecordingQualityResult) -> dict[str, ScoreDetailValue]:
    return {
        "recording_quality_status": quality_result["status"],
        "issue_flags": quality_result["issue_flags"],
        "word_match": quality_result["word_match"],
        "stt_status": quality_result.get("stt_status"),
        "transcript": quality_result.get("transcript"),
        "prompted_transcript": quality_result.get("prompted_transcript"),
    }


def _get_recording_quality_result(
    user_features: AudioFeatures,
    reference: ReferenceVector,
    recording_quality_result: RecordingQualityResult | None,
) -> RecordingQualityResult:
    if recording_quality_result is not None:
        return recording_quality_result

    status = user_features.get("recording_quality_status")
    if status in ("ok", "bad"):
        word_match = user_features.get("word_match")
        return {
            "status": status,
            "issue_flags": _coerce_issue_flags(user_features.get("issue_flags")),
            "word_match": word_match if isinstance(word_match, bool) else None,
            "stt_status": _get_optional_string(user_features.get("stt_status")) or "unavailable",
            "transcript": _get_optional_string(user_features.get("transcript")),
            "prompted_transcript": _get_optional_string(user_features.get("prompted_transcript")),
        }

    return evaluate_recording_quality(user_features, reference)


def _get_quality_gate_feedback(issue_flags: list[str]) -> str:
    if "word_mismatch" in issue_flags:
        return _WORD_MISMATCH_FEEDBACK
    return _QUALITY_GATE_FEEDBACK


def _build_quality_gate_result(quality_result: RecordingQualityResult) -> ScoreResult:
    issue_flags = quality_result["issue_flags"]
    return {
        "score": None,
        "pronunciation_score": None,
        "recording_quality_status": quality_result["status"],
        "issue_flags": issue_flags,
        "feedback": _get_quality_gate_feedback(issue_flags),
        "details": _get_quality_detail_fields(quality_result),
    }


def _get_ko_metrics(
    phoneme: str,
    user_features: AudioFeatures,
    reference: ReferenceVector,
    ko_reference: ReferenceVector | None,
) -> dict[str, ScoreDetailValue]:
    if phoneme not in _KO_REFERENCE_PHONEMES:
        return {
            "korean_pattern_status": "not_applicable",
            "korean_pattern_penalty_policy": "penalty_only",
            "korean_pattern_diagnosis": "이 음소는 한국어식 reference penalty 대상이 아닙니다.",
        }
    return compute_ko_reference_metrics(user_features, reference, ko_reference)


def _get_liquid_metrics(
    phoneme: str,
    user_features: AudioFeatures,
    reference: ReferenceVector,
    liquid_alt_reference: ReferenceVector | None,
) -> dict[str, ScoreDetailValue]:
    if phoneme not in _LIQUID_PHONEMES:
        return {}
    metrics = {
        **compute_liquid_alt_metrics(user_features, reference, liquid_alt_reference),
        **compute_liquid_onset_metrics(user_features, reference, liquid_alt_reference, phoneme),
    }
    metrics["liquid_penalty_policy"] = "diagnostic_only"
    return metrics


def _get_liquid_acoustic_metrics(phoneme: str, user_features: AudioFeatures) -> dict[str, ScoreDetailValue]:
    if phoneme not in _LIQUID_PHONEMES:
        return {
            "liquid_acoustic_status": "not_applicable",
            "liquid_acoustic_penalty": 0.0,
            "liquid_acoustic_penalty_applied": False,
        }
    return compute_liquid_acoustic_penalty(user_features, phoneme)


def _apply_f_korean_like_penalty_adjustment(
    phoneme: str,
    korean_pattern_status: str,
    centroid_is_low: bool,
    zcr_is_low: bool,
    base_penalty: float,
) -> float:
    """/f/ 한국어식 오독이 경계선일 때 물리 지표 교차 검증 후 1.5점 추가 감점."""
    if phoneme == "f" and korean_pattern_status == "borderline_korean_like" and centroid_is_low and zcr_is_low:
        return base_penalty + 1.5
    return base_penalty


def _compute_f_onset_zcr_penalty(
    phoneme: str,
    korean_pattern_status: str,
    user_onset_zcr_mean: float | None,
    ref_onset_zcr_mean: float | None,
) -> tuple[float, float, str]:
    """유저 Onset ZCR / 원어민 Ref Onset ZCR 비율로 /f/ 마찰음 실종 오독을 탐지한다.

    단어 전체 평균 대신 초반 onset 구간만 비교하므로 fix의 /ks/ 오염에 무관하다.
    """
    if phoneme != "f":
        return 0.0, 1.0, "none"
    if not user_onset_zcr_mean or not ref_onset_zcr_mean or ref_onset_zcr_mean <= 0.0:
        return 0.0, 1.0, "missing_onset_zcr"

    ratio = float(user_onset_zcr_mean) / (float(ref_onset_zcr_mean) + EPSILON)
    if korean_pattern_status in ("borderline_korean_like", "ko_error") and ratio < 0.65:
        return 2.0, ratio, "applied"
    return 0.0, ratio, "none"


def _compute_f_onset_mfcc_metrics(
    phoneme: str,
    korean_pattern_status: str,
    user_onset_mfcc: list[float] | None,
    ref_onset_mfcc: list[float] | None,
    user_onset_centroid: float | None,
    ref_onset_centroid: float | None,
) -> tuple[float, float, float, str, float]:
    """Onset 구간 13차원 MFCC 유클리드 거리로 /f/ 순수 음색 오독을 탐지한다.

    후속 복합 자음(/ks/ 등)의 오염을 원천 차단하고, 초반 onset 음색 공간만 비교한다.
    f_onset_centroid_ratio는 패널티 없이 진단 데이터로만 수집한다.

    Returns:
        (penalty, distance, score, status, centroid_ratio)
    """
    centroid_ratio = (
        float(user_onset_centroid) / (float(ref_onset_centroid) + EPSILON)
        if user_onset_centroid is not None and ref_onset_centroid is not None
        else 0.0
    )

    if phoneme != "f":
        return 0.0, 0.0, 100.0, "none", centroid_ratio
    if not user_onset_mfcc or not ref_onset_mfcc:
        return 0.0, 0.0, 100.0, "missing_onset_mfcc", centroid_ratio

    try:
        distance = float(
            np.linalg.norm(np.array(user_onset_mfcc, dtype=float) - np.array(ref_onset_mfcc, dtype=float))
        )
    except Exception:
        return 0.0, 0.0, 100.0, "compute_error", centroid_ratio

    score = float(np.clip(100.0 - distance * 2.5, 0.0, 100.0))

    if korean_pattern_status in ("borderline_korean_like", "ko_error") and distance > _F_ONSET_MFCC_DISTANCE_THRESHOLD:
        return 2.0, distance, score, "applied", centroid_ratio
    return 0.0, distance, score, "none", centroid_ratio


def _compute_f_onset_crest_penalty(
    phoneme: str,
    korean_pattern_status: str,
    onset_rms_max: float | None,
    onset_rms_mean: float | None,
) -> tuple[float, float, str]:
    """/f/ Onset 구간의 RMS Crest Factor(파고율)로 파열음 오독을 탐지한다.

    /f/ 마찰음은 에너지 엔벨로프가 평탄(낮은 파고율),
    한국어 ㅍ 파열음은 첫 burst에 에너지가 집중(높은 파고율).
    crest_factor >= 4.0은 보수적 임계값으로, good/fix 오탐 방지에 중점을 둔다.
    """
    if phoneme != "f":
        return 0.0, 0.0, "none"
    if onset_rms_max is None or onset_rms_mean is None:
        return 0.0, 0.0, "missing_onset_rms"

    crest_factor = float(onset_rms_max) / (float(onset_rms_mean) + EPSILON)
    if korean_pattern_status in ("borderline_korean_like", "ko_error") and crest_factor >= 4.0:
        return 2.0, crest_factor, "applied"
    return 0.0, crest_factor, "none"


def _compute_r_micro_penalty(
    phoneme: str,
    liquid_alt_status: str,
    user_features: AudioFeatures,
) -> tuple[float, str]:
    """/r/ F3 극저값(< 0.0003) × liquid_alt_status 교차 핀셋 패널티.

    /r/, /l/은 _KO_REFERENCE_PHONEMES 비대상이라 korean_pattern_status가 항상
    "not_applicable"이므로 liquid_alt_status를 조건 변수로 사용한다.
    good/road(f3_to_low ≈ 0.0017)와 8~20배 격차를 이용해 오독 샘플만 타격한다.
    """
    if phoneme != "r":
        return 0.0, "none"
    if liquid_alt_status not in ("borderline_korean_like", "korean_like"):
        return 0.0, "none"
    f3_to_low_raw = user_features.get("liquid_mel_f3_to_low_ratio")
    if f3_to_low_raw is None:
        return 0.0, "missing_f3_to_low"
    if float(f3_to_low_raw) < _R_MICRO_F3_TO_LOW_THRESHOLD:
        return 2.0, "applied"
    return 0.0, "none"


def _compute_l_micro_penalty(
    phoneme: str,
    liquid_alt_status: str,
) -> tuple[float, str]:
    """/l/ 경계선(borderline_korean_like) liquid_alt_status 마이크로 패널티.

    /l/은 _KO_REFERENCE_PHONEMES 비대상이라 korean_pattern_status가 항상
    "not_applicable"이므로 liquid_alt_status를 조건 변수로 사용한다.
    borderline 생존 오독 샘플만 낙제선 아래로 밀어낸다.
    """
    if phoneme != "l":
        return 0.0, "none"
    if liquid_alt_status == "borderline_korean_like":
        return 2.0, "applied"
    return 0.0, "none"


def _compute_l_acoustic_borderline_penalty(
    phoneme: str,
    liquid_acoustic_status: str,
) -> tuple[float, str]:
    """/l/ liquid_acoustic_status == "borderline_korean_like" 전용 마이크로 패널티.

    liquid_acoustic_status는 MFCC 전이 거리(c0_delta 등) 기반으로 판별되므로
    liquid_alt_status(MFCC vs 대조 레퍼런스)와는 독립적인 채널이다.
    두 채널이 모두 정상인 good/load 샘플은 어느 쪽도 borderline으로 찍히지 않아
    이 패널티가 발동되지 않는다.
    """
    if phoneme != "l":
        return 0.0, "none"
    if liquid_acoustic_status == "borderline_korean_like":
        return 2.0, "applied"
    return 0.0, "none"


def score_pronunciation(
    user_features: AudioFeatures,
    reference: ReferenceVector,
    phoneme: str,
    ko_reference: ReferenceVector | None = None,
    liquid_alt_reference: ReferenceVector | None = None,
    recording_quality_result: RecordingQualityResult | None = None,
) -> ScoreResult:
    quality_result = _get_recording_quality_result(user_features, reference, recording_quality_result)
    if quality_result["status"] == "bad":
        return _build_quality_gate_result(quality_result)

    phoneme_type = str(reference.get("phoneme_type", "unknown"))
    sub_scores = _build_phoneme_score_details(user_features, reference, phoneme_type, phoneme)
    base_score = float(sub_scores.pop("score"))
    mfcc_c0_score_gap = round(
        float(sub_scores.get("mfcc_no_c0_score", 0.0)) - float(sub_scores.get("mfcc_score", 0.0)),
        1,
    )
    centroid_is_low = sub_scores.get("spectral_centroid_score", 100.0) < _CENTROID_LOW_THRESHOLD
    zcr_is_low = sub_scores.get("zcr_score", 100.0) < _ZCR_LOW_THRESHOLD

    duration_ms = float(user_features.get("duration_ms", 0))
    rms_mean = float(user_features.get("rms_mean", 0))
    zcr_mean = float(user_features.get("zcr_mean", 0))
    spectral_centroid_mean = float(user_features.get("spectral_centroid_mean", 0))
    ref_duration_ms = float(reference.get("duration_ms", 500))

    pronunciation_penalty = compute_pronunciation_penalty(sub_scores, phoneme_type, phoneme)
    ko_metrics = _get_ko_metrics(phoneme, user_features, reference, ko_reference)
    liquid_alt_metrics = _get_liquid_metrics(phoneme, user_features, reference, liquid_alt_reference)
    liquid_acoustic_metrics = _get_liquid_acoustic_metrics(phoneme, user_features)
    schwa_metrics = compute_schwa_overstress_metrics(user_features, reference, phoneme)

    korean_like_penalty = float(ko_metrics.get("korean_like_penalty") or 0.0)
    korean_pattern_status = str(ko_metrics.get("korean_pattern_status", ""))
    korean_like_penalty = _apply_f_korean_like_penalty_adjustment(
        phoneme, korean_pattern_status, centroid_is_low, zcr_is_low, korean_like_penalty
    )
    f_onset_penalty, f_onset_zcr_ratio, f_onset_penalty_status = _compute_f_onset_zcr_penalty(
        phoneme,
        korean_pattern_status,
        user_features.get("onset_zcr_mean"),
        reference.get("onset_zcr_mean"),
    )
    f_onset_crest_penalty, f_onset_rms_crest_factor, f_onset_rms_crest_status = _compute_f_onset_crest_penalty(
        phoneme,
        korean_pattern_status,
        user_features.get("onset_rms_max"),
        user_features.get("onset_rms_mean"),
    )
    f_onset_mfcc_penalty, f_onset_mfcc_distance, f_onset_mfcc_score, f_onset_mfcc_status, f_onset_centroid_ratio = (
        _compute_f_onset_mfcc_metrics(
            phoneme,
            korean_pattern_status,
            user_features.get("onset_mfcc_mean"),
            reference.get("onset_mfcc_mean"),
            user_features.get("onset_spectral_centroid_mean"),
            reference.get("onset_spectral_centroid_mean"),
        )
    )
    liquid_alt_status = str(liquid_alt_metrics.get("liquid_alt_status") or "")
    liquid_acoustic_status = str(liquid_acoustic_metrics.get("liquid_acoustic_status") or "")
    r_micro_penalty, r_micro_status = _compute_r_micro_penalty(phoneme, liquid_alt_status, user_features)
    l_micro_penalty, l_micro_status = _compute_l_micro_penalty(phoneme, liquid_alt_status)
    l_borderline_penalty, l_borderline_status = _compute_l_acoustic_borderline_penalty(phoneme, liquid_acoustic_status)
    liquid_alt_penalty = float(liquid_alt_metrics.get("liquid_alt_penalty") or 0.0)
    liquid_onset_penalty = float(liquid_alt_metrics.get("liquid_onset_penalty") or 0.0)
    liquid_acoustic_penalty = float(liquid_acoustic_metrics.get("liquid_acoustic_penalty") or 0.0)
    schwa_overstress_penalty = float(schwa_metrics.get("schwa_overstress_penalty") or 0.0)
    vowel_core_mfcc_distance = _vector_distance(user_features, reference, "vowel_core_mfcc_mean")

    active_liquid_alt_penalty = 0.0
    active_liquid_onset_penalty = 0.0
    total_penalty = (
        pronunciation_penalty
        + korean_like_penalty
        + f_onset_penalty
        + f_onset_crest_penalty
        + f_onset_mfcc_penalty
        + r_micro_penalty
        + l_micro_penalty
        + l_borderline_penalty
        + active_liquid_alt_penalty
        + active_liquid_onset_penalty
        + liquid_acoustic_penalty
        + schwa_overstress_penalty
    )

    duration_ratio = duration_ms / (ref_duration_ms + EPSILON)
    final_score = float(np.clip(base_score - total_penalty, 0.0, 100.0))
    feedback = get_feedback(final_score, phoneme, phoneme_type)

    penalty_breakdown: dict[str, float] = {
        "pronunciation_penalty": round(pronunciation_penalty, 1),
        "korean_like_penalty": round(korean_like_penalty, 1),
        "f_onset_penalty": round(f_onset_penalty, 1),
        "f_onset_crest_penalty": round(f_onset_crest_penalty, 1),
        "f_onset_mfcc_penalty": round(f_onset_mfcc_penalty, 1),
        "r_micro_penalty": round(r_micro_penalty, 1),
        "l_micro_penalty": round(l_micro_penalty, 1),
        "l_borderline_penalty": round(l_borderline_penalty, 1),
        "liquid_acoustic_penalty": round(liquid_acoustic_penalty, 1),
        "schwa_overstress_penalty": round(schwa_overstress_penalty, 1),
    }

    details: dict[str, ScoreDetailValue] = {
        **sub_scores,
        **ko_metrics,
        **liquid_alt_metrics,
        **liquid_acoustic_metrics,
        **schwa_metrics,
        **_get_quality_detail_fields(quality_result),
        "similarity_score": round(base_score, 1),
        "mfcc_score_used": round(float(sub_scores.get("mfcc_score", 0.0)), 1),
        "penalty_breakdown": penalty_breakdown,
        "mfcc_c0_score_gap": mfcc_c0_score_gap,
        "base_score": round(base_score, 1),
        "quality_penalty": 0.0,
        "pronunciation_penalty": pronunciation_penalty,
        "korean_like_penalty": round(korean_like_penalty, 1),
        "liquid_alt_penalty": round(liquid_alt_penalty, 1),
        "liquid_onset_penalty": round(liquid_onset_penalty, 1),
        "active_liquid_alt_penalty": round(active_liquid_alt_penalty, 1),
        "active_liquid_onset_penalty": round(active_liquid_onset_penalty, 1),
        "liquid_acoustic_penalty": round(liquid_acoustic_penalty, 1),
        "r_micro_penalty": r_micro_penalty,
        "r_micro_status": r_micro_status,
        "l_micro_penalty": l_micro_penalty,
        "l_micro_status": l_micro_status,
        "l_borderline_penalty": l_borderline_penalty,
        "l_borderline_status": l_borderline_status,
        "vowel_i_zcr_duration_penalty": 0.0,
        "vowel_i_zcr_duration_status": "disabled",
        "vowel_i_duration_ms": round(duration_ms, 2) if phoneme == "i" else None,
        "vowel_i_zcr_mean": round(zcr_mean, 6) if phoneme == "i" else None,
        "schwa_overstress_penalty": round(schwa_overstress_penalty, 1),
        "f_onset_penalty": f_onset_penalty,
        "f_onset_zcr_ratio": round(f_onset_zcr_ratio, 4),
        "f_onset_penalty_status": f_onset_penalty_status,
        "f_onset_crest_penalty": f_onset_crest_penalty,
        "f_onset_rms_crest_factor": round(f_onset_rms_crest_factor, 4),
        "f_onset_rms_crest_status": f_onset_rms_crest_status,
        "onset_rms_max": round(float(user_features.get("onset_rms_max") or 0.0), 6),
        "f_onset_mfcc_distance": round(f_onset_mfcc_distance, 4),
        "f_onset_mfcc_score": round(f_onset_mfcc_score, 1),
        "f_onset_mfcc_penalty": f_onset_mfcc_penalty,
        "f_onset_mfcc_status": f_onset_mfcc_status,
        "f_onset_centroid_ratio": round(f_onset_centroid_ratio, 4),
        "vowel_core_peak_width_ms": user_features.get("vowel_core_peak_width_ms"),
        "vowel_core_mfcc_delta_mean": user_features.get("vowel_core_mfcc_delta_mean"),
        "vowel_core_mfcc_std_mean": user_features.get("vowel_core_mfcc_std_mean"),
        "vowel_core_mfcc_distance": round(vowel_core_mfcc_distance, 4) if vowel_core_mfcc_distance is not None else None,
        "mismatch_penalty": 0.0,
        "total_penalty": round(total_penalty, 1),
        "final_score": round(final_score, 1),
        "duration_ratio": round(duration_ratio, 3),
        "rms_mean": round(rms_mean, 6),
        "zcr_mean": round(zcr_mean, 6),
        "spectral_centroid_mean": round(spectral_centroid_mean, 2),
    }

    return {
        "score": round(final_score, 1),
        "pronunciation_score": round(final_score, 1),
        "recording_quality_status": quality_result["status"],
        "issue_flags": quality_result["issue_flags"],
        "feedback": feedback,
        "details": details,
    }
