from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.audio import load_trimmed_audio
from pipeline.features import VOWEL_CORE_PHONEMES, extract_features
from pipeline.liquid_features import extract_liquid_acoustic_features
from pipeline.quality import evaluate_recording_quality
from pipeline.reference import TargetWord
from pipeline.scorer import score_pronunciation
from pipeline.word_targets import attach_word_target_features, should_extract_onset
from scripts import evaluate_labeled_recordings as base_eval

LIQUID_FEATURE_FIELDS = (
    "liquid_analysis_window_ms",
    "liquid_energy_v_shape_score",
    "liquid_energy_roughness",
    "liquid_mel_f3_to_low_ratio",
    "liquid_mel_f3_to_mid_ratio",
    "liquid_transition_mfcc_distance",
    "liquid_transition_mfcc_slope",
    "liquid_transition_mfcc_delta_norm",
    "liquid_transition_mfcc_c0_delta",
    "liquid_transition_mfcc_c1_delta",
    "liquid_transition_mfcc_c2_delta",
)

DETAIL_FIELDS = tuple(
    dict.fromkeys(
        (
            *base_eval.DETAIL_FIELDS,
            "liquid_acoustic_penalty",
            "f_onset_penalty",
            "f_onset_zcr_ratio",
            "f_onset_crest_penalty",
            "f_onset_rms_crest_factor",
            "f_onset_mfcc_distance",
            "f_onset_mfcc_score",
            "f_onset_mfcc_penalty",
            "f_onset_centroid_ratio",
            "vowel_i_zcr_duration_penalty",
            "vowel_i_duration_ms",
            "vowel_i_zcr_mean",
            "vowel_core_peak_width_ms",
            "vowel_core_mfcc_delta_mean",
            "vowel_core_mfcc_std_mean",
            "vowel_core_mfcc_distance",
        )
    )
)


def score_audio_file(
    *,
    label: str,
    audio_path: Path,
    target: TargetWord,
    reference_vectors: dict[str, dict[str, Any]],
    ko_reference_vectors: dict[str, dict[str, Any]],
    word_targets: dict[str, Any],
    check_word_match: bool,
) -> dict[str, Any]:
    reference = reference_vectors.get(target.phoneme)
    if reference is None:
        raise KeyError(f"reference vector가 없습니다: /{target.phoneme}/")

    ko_reference = ko_reference_vectors.get(target.phoneme)
    liquid_alt_reference = ko_reference if target.phoneme in base_eval.LIQUID_REFERENCE_PHONEMES else None

    waveform, sample_rate = load_trimmed_audio(audio_path)
    include_onset = should_extract_onset(target.word, target.phoneme, word_targets)
    include_vowel_core = target.phoneme in VOWEL_CORE_PHONEMES
    features = extract_features(waveform, sample_rate, include_onset=include_onset, include_vowel_core=include_vowel_core)
    if target.phoneme in base_eval.LIQUID_REFERENCE_PHONEMES:
        features.update(extract_liquid_acoustic_features(waveform, sample_rate))
    attach_word_target_features(features, target.word, target.phoneme, word_targets)

    quality_result = evaluate_recording_quality(
        features=features,
        reference=reference,
        audio_path=str(audio_path) if check_word_match else None,
        target_word=target.word if check_word_match else None,
    )
    result = score_pronunciation(
        user_features=features,
        reference=reference,
        phoneme=target.phoneme,
        ko_reference=ko_reference,
        liquid_alt_reference=liquid_alt_reference,
        recording_quality_result=quality_result,
    )
    details = result.get("details", {})

    row: dict[str, Any] = {
        "label": label,
        "audio_path": audio_path.relative_to(base_eval.PROJECT_ROOT).as_posix()
        if audio_path.is_relative_to(base_eval.PROJECT_ROOT)
        else str(audio_path),
        "word": target.word,
        "phoneme": target.phoneme,
        "has_ko_reference": ko_reference is not None,
        "has_liquid_alt_reference": liquid_alt_reference is not None,
        "score": base_eval.safe_float(result.get("score")),
        "recording_quality_status": result.get("recording_quality_status"),
        "issue_flags": json.dumps(result.get("issue_flags") or [], ensure_ascii=False),
        "feedback": result.get("feedback"),
        "duration_ms": base_eval.safe_float(features.get("duration_ms")),
        "rms_mean": base_eval.safe_float(features.get("rms_mean")),
        "zcr_mean": base_eval.safe_float(features.get("zcr_mean")),
        "spectral_centroid_mean": base_eval.safe_float(features.get("spectral_centroid_mean")),
        "onset_rms_mean": base_eval.safe_float(features.get("onset_rms_mean")),
        "onset_rms_max": base_eval.safe_float(features.get("onset_rms_max")),
        "onset_window_ms": base_eval.safe_float(features.get("onset_window_ms")),
        "onset_spectral_centroid_mean": base_eval.safe_float(features.get("onset_spectral_centroid_mean")),
        "onset_zcr_mean": base_eval.safe_float(features.get("onset_zcr_mean")),
        "korean_pattern_status": details.get("korean_pattern_status"),
        "korean_pattern_penalty_policy": details.get("korean_pattern_penalty_policy"),
        "liquid_alt_status": details.get("liquid_alt_status"),
        "liquid_onset_status": details.get("liquid_onset_status"),
        "liquid_acoustic_status": details.get("liquid_acoustic_status"),
        "liquid_acoustic_penalty_applied": details.get("liquid_acoustic_penalty_applied"),
        "schwa_overstress_status": details.get("schwa_overstress_status"),
        "f_onset_penalty_status": details.get("f_onset_penalty_status"),
        "f_onset_rms_crest_status": details.get("f_onset_rms_crest_status"),
        "f_onset_mfcc_status": details.get("f_onset_mfcc_status"),
        "vowel_i_zcr_duration_status": details.get("vowel_i_zcr_duration_status"),
    }
    for key in LIQUID_FEATURE_FIELDS:
        row[key] = base_eval.safe_float(features.get(key))
    for key in DETAIL_FIELDS:
        row[key] = base_eval.safe_float(details.get(key))
    return row


def main() -> None:
    base_eval.DETAIL_FIELDS = DETAIL_FIELDS
    base_eval.score_audio_file = score_audio_file
    base_eval.main()


if __name__ == "__main__":
    main()
