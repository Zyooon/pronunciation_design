from typing import TypedDict

import numpy as np


EPSILON = 1e-8

AudioFeatures = dict[str, float | list[float]]
ReferenceVector = dict[str, float | list[float] | str]

# ── Quality penalty thresholds ────────────────────────────────────────────────
_MIN_DURATION_MS       = 150.0   # 이보다 짧으면 의미 있는 발음이 아닐 가능성이 높다
_RMS_SILENT            = 0.005   # 사실상 무음
_RMS_VERY_QUIET        = 0.015   # 목소리가 매우 작음
_ZCR_ACTIVE_NOISE      = 0.35    # 고주파 잡음 의심 수준
_RMS_LOW_FOR_ZCR_CHECK = 0.020   # ZCR 잡음 체크 시 함께 보는 RMS 임계치
_ZCR_EXTREME           = 0.50    # 발화와 무관한 극단적 잡음


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

    MVP 기준:
        - MFCC: 발음 음색/조음 차이
        - duration: 장단 차이 프록시
        - RMS: 너무 강하거나 약한 발음 여부

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

    rms_score = ratio_feature_score(
        user_value=float(user_features["rms_mean"]),
        ref_value=float(reference["rms_mean"]),
    )

    # MVP에서는 MFCC를 가장 중요하게 보고,
    # duration과 RMS는 보조 지표로 사용합니다.
    final_score = (
        mfcc_score * 0.65
        + duration_score * 0.25
        + rms_score * 0.10
    )

    return {
        "score": round(float(final_score), 1),
        "mfcc_score": round(float(mfcc_score), 1),
        "duration_score": round(float(duration_score), 1),
        "rms_score": round(float(rms_score), 1),
    }


def score_consonant(user_features: AudioFeatures, reference: ReferenceVector) -> dict[str, float]:
    """
    자음 음소를 채점합니다.

    MVP 기준:
        - MFCC: 전체 음색/조음 차이
        - ZCR: 마찰음 계열 감지
        - spectral centroid: 고주파 중심성 비교

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
        "score": round(float(final_score), 1),
        "mfcc_score": round(float(mfcc_score), 1),
        "zcr_score": round(float(zcr_score), 1),
        "spectral_centroid_score": round(float(centroid_score), 1),
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
) -> dict[str, float]:
    if phoneme_type == "vowel":
        return score_vowel(user_features, reference)
    if phoneme_type == "consonant":
        return score_consonant(user_features, reference)
    # 모음/자음 타입을 알 수 없으면 MFCC만으로 최소 채점합니다.
    mfcc_score = z_score_distance_score(
        user_values=user_features["mfcc_mean"],
        ref_mean=reference["mfcc_mean"],
        ref_std=reference["mfcc_std"],
    )
    return {
        "score": round(float(mfcc_score), 1),
        "mfcc_score": round(float(mfcc_score), 1),
    }

def require_float(features: AudioFeatures, key: str) -> float:
    value = features.get(key)
    if value is None:
        raise KeyError(f"필수 feature가 없습니다: {key}")
    return float(value)


def score_pronunciation(
    user_features: AudioFeatures,
    reference: ReferenceVector,
    phoneme: str,
) -> ScoreResult:
    """
    사용자 음성 특징과 reference vector를 비교해 최종 채점 결과를 반환합니다.

    품질 패널티를 base score에서 차감해 final score를 산출한다.
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
                ...,
                "base_score": 78.5,
                "quality_penalty": 3.0,
                "duration_penalty": 0.0,
                "volume_penalty": 0.0,
                "noise_penalty": 3.0,
                "duration_ratio": 1.05,
                "rms_mean": 0.042,
                "zcr_mean": 0.18,
                "spectral_centroid_mean": 3200.5,
            }
        }
    """
    phoneme_type = str(reference.get("phoneme_type", "unknown"))
    sub_scores = _build_phoneme_score_details(user_features, reference, phoneme_type)
    base_score = float(sub_scores.pop("score"))

    duration_ms = require_float(user_features, "duration_ms")
    rms_mean = require_float(user_features, "rms_mean")
    zcr_mean = require_float(user_features, "zcr_mean")
    spectral_centroid_mean = require_float(user_features, "spectral_centroid_mean")
    ref_duration_ms = float(reference.get("duration_ms", 500))

    duration_penalty, volume_penalty, noise_penalty = compute_quality_penalty(
        duration_ms=duration_ms,
        rms_mean=rms_mean,
        zcr_mean=zcr_mean,
        ref_duration_ms=ref_duration_ms,
    )
    quality_penalty = duration_penalty + volume_penalty + noise_penalty

    duration_ratio = duration_ms / (ref_duration_ms + EPSILON)
    final_score = float(np.clip(base_score - quality_penalty, 0.0, 100.0))

    feedback = get_feedback(final_score, phoneme, phoneme_type)

    details: dict[str, float] = {
        **sub_scores,
        "base_score":             round(base_score, 1),
        "quality_penalty":        round(quality_penalty, 1),
        "duration_penalty":       round(duration_penalty, 1),
        "volume_penalty":         round(volume_penalty, 1),
        "noise_penalty":          round(noise_penalty, 1),
        "duration_ratio":         round(duration_ratio, 3),
        "rms_mean":               round(rms_mean, 6),
        "zcr_mean":               round(zcr_mean, 6),
        "spectral_centroid_mean": round(spectral_centroid_mean, 2),
        "final_score":            round(final_score, 1),
    }

    return {
        "score": round(final_score, 1),
        "feedback": feedback,
        "details": details,
    }
