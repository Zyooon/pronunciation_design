from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


log = logging.getLogger(__name__)

AudioFeatureValue = float | list[float] | str | list[str] | bool | dict[str, str | None] | None
AudioFeatures = dict[str, AudioFeatureValue]
ReferenceVector = dict[str, float | list[float] | str]
QualityStatus = Literal["ok", "bad"]
SttStatus = Literal["matched", "mismatched", "uncertain", "unavailable"]

EPSILON = 1e-8

_MIN_DURATION_MS = 150.0
_TOO_SHORT_RATIO = 0.35
_TOO_LONG_RATIO = 3.0
_RMS_ALMOST_SILENT = 0.005
_RMS_TOO_QUIET = 0.015
_ZCR_HIGH_NOISE = 0.35
_RMS_LOW_FOR_NOISE_CHECK = 0.020
_ZCR_EXTREME = 0.50

_SHORT_WORD_FUZZY_THRESHOLD = 0.65
_LONG_WORD_FUZZY_THRESHOLD = 0.75
_CONFIDENT_OTHER_WORD_THRESHOLD = 0.45
_SHORT_WORD_MAX_LEN = 4

_whisper_model: WhisperModel | None = None


class WordMatchResult(TypedDict):
    word_match: bool | None
    stt_status: SttStatus
    transcript: str | None
    prompted_transcript: str | None


class RecordingQualityResult(TypedDict):
    status: QualityStatus
    issue_flags: list[str]
    word_match: bool | None
    stt_status: SttStatus
    transcript: str | None
    prompted_transcript: str | None


def _get_float_value(values: AudioFeatures | ReferenceVector, key: str) -> float | None:
    value = values.get(key)
    if value is None or isinstance(value, list) or isinstance(value, dict):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel as _WhisperModel
        _whisper_model = _WhisperModel("tiny", device="cpu", compute_type="int8")
    return _whisper_model


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z\s]", " ", text.lower()).strip()


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def _get_fuzzy_threshold(target_word: str) -> float:
    if len(target_word) <= _SHORT_WORD_MAX_LEN:
        return _SHORT_WORD_FUZZY_THRESHOLD
    return _LONG_WORD_FUZZY_THRESHOLD


def _has_exact_token_match(target_word: str, transcript_tokens: list[str]) -> bool:
    return target_word in transcript_tokens


def _has_fuzzy_token_match(target_word: str, transcript_tokens: list[str]) -> bool:
    if not transcript_tokens:
        return False
    threshold = _get_fuzzy_threshold(target_word)
    best_similarity = max(_similarity(target_word, token) for token in transcript_tokens)
    return best_similarity >= threshold


def _is_confident_other_word(target_word: str, transcript_tokens: list[str]) -> bool:
    """STT 결과가 확실히 다른 단어 하나일 때만 mismatch로 본다."""
    if len(transcript_tokens) != 1:
        return False
    similarity = _similarity(target_word, transcript_tokens[0])
    return similarity < _CONFIDENT_OTHER_WORD_THRESHOLD


def _transcribe_audio(audio_path: str, target_word: str | None = None) -> str:
    model = _load_whisper_model()
    kwargs: dict[str, str | int] = {
        "language": "en",
        "beam_size": 1,
    }
    if target_word:
        kwargs["initial_prompt"] = target_word
    segments, _ = model.transcribe(audio_path, **kwargs)
    return " ".join(segment.text for segment in segments).strip()


def _judge_transcript(transcript: str, normalized_target: str) -> bool | None:
    normalized_transcript = _normalize_text(transcript)
    if not normalized_transcript:
        return None

    transcript_tokens = normalized_transcript.split()
    if _has_exact_token_match(normalized_target, transcript_tokens):
        return True
    if _has_fuzzy_token_match(normalized_target, transcript_tokens):
        return True
    if _is_confident_other_word(normalized_target, transcript_tokens):
        return False
    return None


def _build_word_match_result(
    word_match: bool | None,
    stt_status: SttStatus,
    transcript: str | None,
    prompted_transcript: str | None,
) -> WordMatchResult:
    return {
        "word_match": word_match,
        "stt_status": stt_status,
        "transcript": transcript,
        "prompted_transcript": prompted_transcript,
    }


def check_word_match_detail(audio_path: str, target_word: str) -> WordMatchResult:
    """STT 2-pass 방식으로 목표 단어 일치 여부를 확인한다.

    1차는 prompt 없이 판단한다. 단, 1차 mismatch도 즉시 차단하지 않고
    target_word prompt로 한 번 더 확인해 정상 단어 false negative를 줄인다.
    """
    normalized_target = _normalize_text(target_word).replace(" ", "")
    if not normalized_target:
        return _build_word_match_result(None, "unavailable", None, None)

    try:
        transcript = _transcribe_audio(audio_path)
        first_pass_match = _judge_transcript(transcript, normalized_target)
        if first_pass_match is True:
            return _build_word_match_result(True, "matched", transcript, None)

        prompted_transcript = _transcribe_audio(audio_path, target_word=target_word)
        prompted_match = _judge_transcript(prompted_transcript, normalized_target)
        if first_pass_match is False and prompted_match is True:
            return _build_word_match_result(None, "uncertain", transcript, prompted_transcript)
        if first_pass_match is False:
            return _build_word_match_result(False, "mismatched", transcript, prompted_transcript)
        if prompted_match is True:
            return _build_word_match_result(True, "matched", transcript, prompted_transcript)

        return _build_word_match_result(None, "uncertain", transcript, prompted_transcript)
    except Exception:
        log.warning(
            "STT word match 확인 실패 (fail open): audio_path=%s, target_word=%s",
            audio_path,
            target_word,
            exc_info=True,
        )
        return _build_word_match_result(None, "unavailable", None, None)


def check_word_match(audio_path: str, target_word: str) -> bool | None:
    """기존 호출부 호환을 위해 word_match 값만 반환한다."""
    return check_word_match_detail(audio_path, target_word)["word_match"]


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


def _attach_quality_result_to_features(features: AudioFeatures, quality_result: RecordingQualityResult) -> None:
    features["recording_quality_status"] = quality_result["status"]
    features["issue_flags"] = quality_result["issue_flags"]
    features["word_match"] = quality_result["word_match"]
    features["stt_status"] = quality_result["stt_status"]
    features["transcript"] = quality_result["transcript"]
    features["prompted_transcript"] = quality_result["prompted_transcript"]


def evaluate_recording_quality(
    features: AudioFeatures,
    reference: ReferenceVector | None = None,
    audio_path: str | None = None,
    target_word: str | None = None,
) -> RecordingQualityResult:
    """녹음 품질 문제와 잘못된 단어 여부를 발음 점수와 분리해서 판단합니다.

    Args:
        audio_path: STT word match 확인에 사용할 녹음 경로. target_word와 함께 전달해야 동작한다.
        target_word: 사용자가 발음해야 할 단어. word_mismatch 감지에 사용한다.
    """
    duration_ms = _get_float_value(features, "duration_ms")
    rms_mean = _get_float_value(features, "rms_mean")
    zcr_mean = _get_float_value(features, "zcr_mean")
    reference_duration_ms = None if reference is None else _get_float_value(reference, "duration_ms")

    issue_flags: list[str] = []
    if duration_ms is not None:
        issue_flags.extend(_has_bad_duration(duration_ms, reference_duration_ms))
    issue_flags.extend(_has_bad_volume(rms_mean))
    issue_flags.extend(_has_bad_noise(zcr_mean, rms_mean))

    word_match_result: WordMatchResult = {
        "word_match": None,
        "stt_status": "unavailable",
        "transcript": None,
        "prompted_transcript": None,
    }
    if audio_path and target_word:
        word_match_result = check_word_match_detail(audio_path, target_word)
        if word_match_result["word_match"] is False:
            issue_flags.append("word_mismatch")

    status: QualityStatus = "bad" if issue_flags else "ok"
    quality_result: RecordingQualityResult = {
        "status": status,
        "issue_flags": issue_flags,
        "word_match": word_match_result["word_match"],
        "stt_status": word_match_result["stt_status"],
        "transcript": word_match_result["transcript"],
        "prompted_transcript": word_match_result["prompted_transcript"],
    }
    _attach_quality_result_to_features(features, quality_result)
    return quality_result
