from typing import Literal, TypedDict


AudioFeatures = dict[str, float | list[float]]
ReferenceVector = dict[str, float | list[float] | str]
QualityStatus = Literal["ok", "bad"]

EPSILON = 1e-8

_MIN_DURATION_MS = 150.0
_TOO_SHORT_RATIO = 0.35
_TOO_LONG_RATIO = 3.0
_RMS_ALMOST_SILENT = 0.005
_RMS_TOO_QUIET = 0.015
_ZCR_HIGH_NOISE = 0.35
_RMS_LOW_FOR_NOISE_CHECK = 0.020
_ZCR_EXTREME = 0.50


class RecordingQualityResult(TypedDict):
    status: QualityStatus
    issue_flags: list[str]


def _get_float_value(values: AudioFeatures | ReferenceVector, key: str) -> float | None:
    value = values.get(key)
    if value is None or isinstance(value, list):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_bad_duration(duration_ms: float, reference_duration_ms: float | None) -> list[str]:
    issue_flags: list[str] = []

    if duration_ms < _MIN_DURATION_MS:
        issue_flags.append("too_short")
        return issue_flags

    if reference_duration_ms is None or reference_duration_ms <= 0:
        return issue_flags

    duration_ratio = duration_ms / (reference_duration_ms + EPSILON)
    if duration_ratio < _TOO_SHORT_RATIO:
        issue_flags.append("too_short")
    elif duration_ratio > _TOO_LONG_RATIO:
        issue_flags.append("too_long")

    return issue_flags


def _has_bad_volume(rms_mean: float | None) -> list[str]:
    if rms_mean is None:
        return []
    if rms_mean < _RMS_ALMOST_SILENT:
        return ["almost_silent"]
    if rms_mean < _RMS_TOO_QUIET:
        return ["too_quiet"]
    return []


def _has_bad_noise(zcr_mean: float | None, rms_mean: float | None) -> list[str]:
    if zcr_mean is None:
        return []

    issue_flags: list[str] = []
    if zcr_mean > _ZCR_EXTREME:
        issue_flags.append("extreme_zcr")
    if rms_mean is not None and zcr_mean > _ZCR_HIGH_NOISE and rms_mean < _RMS_LOW_FOR_NOISE_CHECK:
        issue_flags.append("high_noise")
    return issue_flags


def evaluate_recording_quality(
    features: AudioFeatures,
    reference: ReferenceVector | None = None,
) -> RecordingQualityResult:
    """녹음 품질 문제를 발음 점수와 분리해서 판단합니다."""
    duration_ms = _get_float_value(features, "duration_ms")
    rms_mean = _get_float_value(features, "rms_mean")
    zcr_mean = _get_float_value(features, "zcr_mean")
    reference_duration_ms = None if reference is None else _get_float_value(reference, "duration_ms")

    issue_flags: list[str] = []
    if duration_ms is not None:
        issue_flags.extend(_has_bad_duration(duration_ms, reference_duration_ms))
    issue_flags.extend(_has_bad_volume(rms_mean))
    issue_flags.extend(_has_bad_noise(zcr_mean, rms_mean))

    if issue_flags:
        return {"status": "bad", "issue_flags": issue_flags}
    return {"status": "ok", "issue_flags": []}
