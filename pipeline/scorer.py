from typing import TypedDict

import numpy as np


EPSILON = 1e-8

AudioFeatures = dict[str, float | list[float]]
ReferenceVector = dict[str, float | list[float] | str]

# /r/, /l/은 마찰음 특성이 약하므로 일반 자음 scoring과 분리한다.
_LIQUID_PHONEMES: frozenset[str] = frozenset({"r", "l"})

# ── Quality penalty thresholds ────────────────────────────────────────────────
_MIN_DURATION_MS       = 150.0   # 이보다 짧으면 의미 있는 발음이 아닐 가능성이 높다
_RMS_SILENT            = 0.005   # 사실상 무음
_RMS_VERY_QUIET        = 0.015   # 목소리가 매우 작음
_ZCR_ACTIVE_NOISE      = 0.35    # 고주파 잡음 의심 수준
_RMS_LOW_FOR_ZCR_CHECK = 0.020   # ZCR 잡음 체크 시 함께 보는 RMS 임계치
_ZCR_EXTREME           = 0.50    # 발화와 무관한 극단적 잡음

# ── Pronunciation penalty thresholds ─────────────────────────────────────────
_MFCC_LOW_THRESHOLD     = 55.0   # MFCC 점수가 이 미만이면 감점
_DURATION_LOW_THRESHOLD = 45.0   # duration 점수가 이 미만이면 감점
_CENTROID_LOW_THRESHOLD = 45.0   # spectral centroid 점수가 이 미만이면 감점
_VOWEL_MFCC_DOUBLE      = 60.0   # 모음에서 MFCC + centroid 동시 저조 시 추가 감점 기준


class ScoreResult(TypedDict):
    score: float
    feedback: str
    details: dict[str, float]


def sigmoid_score(diff_ratio: float, steepness: float = 5.0, tolerance: float = 0.3) -> float:
    """
    차이 비율을 0~100점 점수로 변환합니다.

    diff_ratio가 작을수록 기준 발음과 비슷하다는 뜻입니다.
    tolerance보다 차이가 작으면 높은 점수가 나오고,
    tolerance보다 차이가 커질수록 점수가 빠르게 낮아집니다.

    Args:
        diff_ratio: 기준값 대비 차이 비율
        steepness: 점수 하락 곡선의 가파름
        tolerance: 허용 오차 기준

    Returns:
        0~100 사이 점수
    """
    score = 100 / (1 + np.exp(steepness * (diff_ratio - tolerance)))
    return float(np.clip(score, 0, 100))


def z_score_distance_score(
    user_values: list[float] | np.ndarray,
    ref_mean: list[float] | np.ndarray,
    ref_std: list[float] | np.ndarray,
    scale: float = 10.0,
) -> float:
    """
    사용자 벡터와 기준 벡터의 z-score 기반 유사도 점수를 계산합니다.

    z-score:
        기준 평균에서 사용자 값이 표준편차 몇 개만큼 떨어져 있는지 보는 방식입니다.

    Args:
        user_values: 사용자 음성에서 추출한 벡터
        ref_mean: 기준 음성들의 평균 벡터
        ref_std: 기준 음성들의 표준편차 벡터
        scale: z-score 차이를 점수 감점으로 바꾸는 비율

    Returns:
        0~100 사이 점수
    """
    user_values = np.array(user_values, dtype=float)
    ref_mean = np.array(ref_mean, dtype=float)
    ref_std = np.array(ref_std, dtype=float)

    z_scores = (user_values - ref_mean) / (ref_std + EPSILON)
    distance = np.mean(np.abs(z_scores))

    score = 100 - np.clip(distance * scale, 0, 100)
    return float(np.clip(score, 0, 100))


def ratio_feature_score(user_value: float, ref_value: float) -> float:
    """
    duration, RMS, ZCR, spectral centroid처럼 단일 숫자 특징을 비교합니다.

    Args:
        user_value: 사용자 음성 특징값
        ref_value: 기준 음성 평균값

    Returns:
        0~100 사이 점수
    """
    diff_ratio = abs(user_value - ref_value) / (abs(ref_value) + EPSILON)
    return sigmoid_score(diff_ratio)


def score_vowel(user_features: AudioFeatures, reference: ReferenceVector) -> dict[str, float]:
    """
    모음 음소를 채점합니다.

    가중치:
        MFCC 70% · duration 15% · spectral_centroid 10% · RMS 5%

    Returns:
        세부 점수와 최종 점수
    """
    mfcc_score = z_score_distance_score(
        user_values=user_features["mfcc_mean"],
        ref_mean=reference["mfcc_mean"],
        ref_std=reference["mfcc_std"],
    )

    duration_score = ratio_feature_score(
        user_value=float(user_features["duration_ms"]),
        ref_value=float(reference["duration_ms"]),
    )

    centroid_score = ratio_feature_score(
        user_value=float(user_features["spectral_centroid_mean"]),
        ref_value=float(reference["spectral_centroid_mean"]),
    )

    rms_score = ratio_feature_score(
        user_value=float(user_features["rms_mean"]),
        ref_value=float(reference["rms_mean"]),
    )

    final_score = (
        mfcc_score * 0.70
        + duration_score * 0.15
        + centroid_score * 0.10
        + rms_score * 0.05
    )

    return {
        "score":                   round(float(final_score), 1),
        "mfcc_score":              round(float(mfcc_score), 1),
        "duration_score":          round(float(duration_score), 1),
        "spectral_centroid_score": round(float(centroid_score), 1),
        "rms_score":               round(float(rms_score), 1),
    }


def score_consonant(user_features: AudioFeatures, reference: ReferenceVector) -> dict[str, float]:
    """
    자음 음소를 채점합니다 (/θ/, /f/, /v/ 계열).

    가중치:
        MFCC 55% · ZCR 35% · spectral_centroid 10%

    Returns:
        세부 점수와 최종 점수
    """
    mfcc_score = z_score_distance_score(
        user_values=user_features["mfcc_mean"],
        ref_mean=reference["mfcc_mean"],
        ref_std=reference["mfcc_std"],
    )

    zcr_score = ratio_feature_score(
        user_value=float(user_features["zcr_mean"]),
        ref_value=float(reference["zcr_mean"]),
    )

    centroid_score = ratio_feature_score(
        user_value=float(user_features["spectral_centroid_mean"]),
        ref_value=float(reference["spectral_centroid_mean"]),
    )

    # 자음은 마찰음 특성이 중요하므로 ZCR 비중을 높게 둡니다.
    final_score = (
        mfcc_score * 0.55
        + zcr_score * 0.35
        + centroid_score * 0.10
    )

    return {
        "score":                   round(float(final_score), 1),
        "mfcc_score":              round(float(mfcc_score), 1),
        "zcr_score":               round(float(zcr_score), 1),
        "spectral_centroid_score": round(float(centroid_score), 1),
    }


def score_liquid(user_features: AudioFeatures, reference: ReferenceVector) -> dict[str, float]:
    """
    /r/, /l/ 음소를 채점합니다.

    유음(liquid)은 마찰음 특성이 약해 ZCR이 채점 지표로 적합하지 않습니다.
    조음 위치를 잘 반영하는 MFCC와 스펙트럼 중심 위주로 채점합니다.

    가중치:
        MFCC 75% · duration 15% · spectral_centroid 10%

    Returns:
        세부 점수와 최종 점수
    """
    mfcc_score = z_score_distance_score(
        user_values=user_features["mfcc_mean"],
        ref_mean=reference["mfcc_mean"],
        ref_std=reference["mfcc_std"],
    )

    duration_score = ratio_feature_score(
        user_value=float(user_features["duration_ms"]),
        ref_value=float(reference["duration_ms"]),
    )

    centroid_score = ratio_feature_score(
        user_value=float(user_features["spectral_centroid_mean"]),
        ref_value=float(reference["spectral_centroid_mean"]),
    )

    # ZCR은 참고용으로만 계산하고 채점에는 포함하지 않습니다.
    zcr_score = ratio_feature_score(
        user_value=float(user_features["zcr_mean"]),
        ref_value=float(reference["zcr_mean"]),
    )

    final_score = (
        mfcc_score * 0.75
        + duration_score * 0.15
        + centroid_score * 0.10
    )

    return {
        "score":                   round(float(final_score), 1),
        "mfcc_score":              round(float(mfcc_score), 1),
        "duration_score":          round(float(duration_score), 1),
        "spectral_centroid_score": round(float(centroid_score), 1),
        "zcr_score":               round(float(zcr_score), 1),
    }


def compute_quality_penalty(
    duration_ms: float,
    rms_mean: float,
    zcr_mean: float,
    ref_duration_ms: float,
) -> tuple[float, float, float]:
    """녹음 품질 기반 감점을 계산한다.

    test_label을 사용하지 않으며, 추출된 feature 값만으로 판단한다.

    Args:
        duration_ms: 사용자 녹음 길이(ms)
        rms_mean: 사용자 녹음 평균 에너지
        zcr_mean: 사용자 녹음 평균 ZCR
        ref_duration_ms: reference 기준 발화 길이(ms)

    Returns:
        (duration_penalty, volume_penalty, noise_penalty) — 각 항목별 감점
    """
    # 1. Duration penalty
    if duration_ms < _MIN_DURATION_MS:
        # 너무 짧은 발화 — 클릭음·잡음일 가능성이 높다
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

    # 2. Volume penalty
    if rms_mean < _RMS_SILENT:
        # 사실상 무음
        volume_penalty = 35.0
    elif rms_mean < _RMS_VERY_QUIET:
        # 목소리가 매우 작음
        volume_penalty = 10.0
    else:
        volume_penalty = 0.0

    # 3. Noise penalty: 고주파 잡음 + 낮은 에너지 → 발화 없는 잡음
    if zcr_mean > _ZCR_ACTIVE_NOISE and rms_mean < _RMS_LOW_FOR_ZCR_CHECK:
        noise_penalty = 20.0
    elif zcr_mean > _ZCR_EXTREME:
        # 극단적으로 높은 ZCR — 발화와 무관한 잡음
        noise_penalty = 10.0
    else:
        noise_penalty = 0.0

    return duration_penalty, volume_penalty, noise_penalty


def compute_pronunciation_penalty(
    sub_scores: dict[str, float],
    phoneme_type: str,
) -> float:
    """발음 유사도 세부 점수 기반 감점을 계산한다.

    녹음 품질과 무관하게, 발음 유사도 점수 자체가 낮은데
    base_score가 높게 나오는 경우를 방어하기 위한 패널티다.
    test_label을 사용하지 않는다.

    Args:
        sub_scores: score_vowel / score_consonant / score_liquid 반환값 (score 키 제거 후)
        phoneme_type: "vowel" | "consonant" | "unknown"

    Returns:
        0 이상의 감점값
    """
    # 없는 항목은 중립값(100)으로 처리해 불필요한 감점을 막는다
    mfcc_score    = sub_scores.get("mfcc_score", 100.0)
    duration_score = sub_scores.get("duration_score", 100.0)
    centroid_score = sub_scores.get("spectral_centroid_score", 100.0)

    penalty = 0.0

    if mfcc_score < _MFCC_LOW_THRESHOLD:
        penalty += 8.0

    if duration_score < _DURATION_LOW_THRESHOLD:
        penalty += 4.0

    if centroid_score < _CENTROID_LOW_THRESHOLD:
        penalty += 4.0

    # 모음에서 MFCC와 spectral centroid가 동시에 저조하면 추가 감점
    if (
        phoneme_type == "vowel"
        and mfcc_score < _VOWEL_MFCC_DOUBLE
        and centroid_score < _VOWEL_MFCC_DOUBLE
    ):
        penalty += 6.0

    return penalty


def get_feedback(score: float, phoneme: str, phoneme_type: str) -> str:
    """
    점수와 음소에 따라 간단한 피드백 문장을 반환합니다.

    MVP에서는 규칙 기반 피드백만 제공합니다.
    """
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
    if phoneme in _LIQUID_PHONEMES:
        return score_liquid(user_features, reference)
    if phoneme_type == "vowel":
        return score_vowel(user_features, reference)
    if phoneme_type == "consonant":
        return score_consonant(user_features, reference)
    # 타입 불명 — MFCC만으로 최소 채점
    mfcc_score = z_score_distance_score(
        user_values=user_features["mfcc_mean"],
        ref_mean=reference["mfcc_mean"],
        ref_std=reference["mfcc_std"],
    )
    return {
        "score":      round(float(mfcc_score), 1),
        "mfcc_score": round(float(mfcc_score), 1),
    }


def score_pronunciation(
    user_features: AudioFeatures,
    reference: ReferenceVector,
    phoneme: str,
) -> ScoreResult:
    """
    사용자 음성 특징과 reference vector를 비교해 최종 채점 결과를 반환합니다.

    quality_penalty(녹음 품질)와 pronunciation_penalty(발음 유사도)를
    base_score에서 차감해 final_score를 산출한다.
    test_label은 채점에 사용하지 않는다.

    Args:
        user_features: 음성에서 추출한 특징 벡터
        reference: 타겟 음소의 기준 벡터
        phoneme: 타겟 음소

    Returns:
        {
            "score": 75.2,
            "feedback": "...",
            "details": {
                "mfcc_score": 82.0,
                "duration_score": 70.0,
                "spectral_centroid_score": 65.0,
                "rms_score": 80.0,
                "base_score": 78.5,
                "quality_penalty": 0.0,
                "duration_penalty": 0.0,
                "volume_penalty": 0.0,
                "noise_penalty": 0.0,
                "pronunciation_penalty": 0.0,
                "total_penalty": 0.0,
                "final_score": 78.5,
                "duration_ratio": 1.05,
                "rms_mean": 0.042,
                "zcr_mean": 0.18,
                "spectral_centroid_mean": 3200.5,
            }
        }
    """
    phoneme_type = str(reference.get("phoneme_type", "unknown"))
    sub_scores = _build_phoneme_score_details(user_features, reference, phoneme_type, phoneme)
    base_score = float(sub_scores.pop("score"))

    duration_ms            = float(user_features.get("duration_ms", 0))
    rms_mean               = float(user_features.get("rms_mean", 0))
    zcr_mean               = float(user_features.get("zcr_mean", 0))
    spectral_centroid_mean = float(user_features.get("spectral_centroid_mean", 0))
    ref_duration_ms        = float(reference.get("duration_ms", 500))

    duration_penalty, volume_penalty, noise_penalty = compute_quality_penalty(
        duration_ms=duration_ms,
        rms_mean=rms_mean,
        zcr_mean=zcr_mean,
        ref_duration_ms=ref_duration_ms,
    )
    quality_penalty = duration_penalty + volume_penalty + noise_penalty

    pronunciation_penalty = compute_pronunciation_penalty(sub_scores, phoneme_type)

    total_penalty = quality_penalty + pronunciation_penalty
    duration_ratio = duration_ms / (ref_duration_ms + EPSILON)
    final_score = float(np.clip(base_score - total_penalty, 0.0, 100.0))

    feedback = get_feedback(final_score, phoneme, phoneme_type)

    details: dict[str, float] = {
        **sub_scores,
        "base_score":             round(base_score, 1),
        "quality_penalty":        round(quality_penalty, 1),
        "duration_penalty":       round(duration_penalty, 1),
        "volume_penalty":         round(volume_penalty, 1),
        "noise_penalty":          round(noise_penalty, 1),
        "pronunciation_penalty":  round(pronunciation_penalty, 1),
        "total_penalty":          round(total_penalty, 1),
        "final_score":            round(final_score, 1),
        "duration_ratio":         round(duration_ratio, 3),
        "rms_mean":               round(rms_mean, 6),
        "zcr_mean":               round(zcr_mean, 6),
        "spectral_centroid_mean": round(spectral_centroid_mean, 2),
    }

    return {
        "score":    round(final_score, 1),
        "feedback": feedback,
        "details":  details,
    }
