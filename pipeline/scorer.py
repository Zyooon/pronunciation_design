from typing import Any

import numpy as np


EPSILON = 1e-8


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

    z = (user_values - ref_mean) / (ref_std + EPSILON)

    # 평균 절대 z-score가 클수록 기준 발음에서 멀다는 뜻입니다.
    distance = np.mean(np.abs(z))

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


def score_vowel(user_features: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
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


def score_consonant(user_features: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
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
        "v": "윗니와 아랫입술을 가볍게 대고 목의 울림을 함께 내보세요.",
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


def score_pronunciation(
    user_features: dict[str, Any],
    reference: dict[str, Any],
    phoneme: str,
) -> dict[str, Any]:
    """
    사용자 음성 특징과 reference vector를 비교해 최종 채점 결과를 반환합니다.

    Args:
        user_features: extract_features() 결과
        reference: reference_vectors.json에서 가져온 특정 음소 기준 벡터
        phoneme: 타겟 음소

    Returns:
        {
            "score": 82.5,
            "feedback": "...",
            "details": {...}
        }
    """
    phoneme_type = reference.get("phoneme_type", "unknown")

    if phoneme_type == "vowel":
        details = score_vowel(user_features, reference)
    elif phoneme_type == "consonant":
        details = score_consonant(user_features, reference)
    else:
        # 모음/자음 타입을 알 수 없으면 MFCC만으로 최소 채점합니다.
        mfcc_score = z_score_distance_score(
            user_values=user_features["mfcc_mean"],
            ref_mean=reference["mfcc_mean"],
            ref_std=reference["mfcc_std"],
        )
        details = {
            "score": round(float(mfcc_score), 1),
            "mfcc_score": round(float(mfcc_score), 1),
        }

    final_score = float(details["score"])
    feedback = get_feedback(final_score, phoneme, phoneme_type)

    return {
        "score": final_score,
        "feedback": feedback,
        "details": details,
    }