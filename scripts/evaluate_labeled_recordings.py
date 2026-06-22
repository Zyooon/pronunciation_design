"""라벨링된 음성 파일을 현재 scorer로 평가하는 스크립트.

good/korean_like 라벨 폴더의 음성을 채점하고 점수 분포, 실패 사례, 오류를
JSON 요약과 CSV 리포트로 data/reports에 저장한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.audio import load_trimmed_audio
from pipeline.features import extract_features
from pipeline.quality import evaluate_recording_quality
from pipeline.reference import TargetWord, load_reference_vectors, load_target_words
from pipeline.scorer import score_pronunciation
import pipeline.scorer as scorer_module
from pipeline.word_targets import attach_word_target_features, load_word_targets, should_extract_onset

SOURCE_DIR = PROJECT_ROOT / "data" / "reference_ko" / "record"
KO_REFERENCE_AUDIO_DIR = PROJECT_ROOT / "data" / "reference_ko"
KO_REFERENCE_PATH = PROJECT_ROOT / "data" / "ko_reference_vectors.json"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
LABELS = ("good", "korean_like")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
SCORE_THRESHOLD = 75.0
KO_REFERENCE_PHONEMES = {"θ", "v", "æ", "f", "i"}
LIQUID_REFERENCE_PHONEMES = {"r", "l"}
ALT_REFERENCE_PHONEMES = KO_REFERENCE_PHONEMES | LIQUID_REFERENCE_PHONEMES
DETAIL_FIELDS = (
    "mfcc_score",
    "mfcc_no_c0_score",
    "mfcc_c0_score_gap",
    "duration_score",
    "rms_score",
    "zcr_score",
    "spectral_centroid_score",
    "base_score",
    "final_score",
    "pronunciation_penalty",
    "korean_like_penalty",
    "liquid_alt_penalty",
    "liquid_onset_penalty",
    "schwa_overstress_penalty",
    "total_penalty",
    "duration_ratio",
    "rms_mean",
    "zcr_mean",
    "spectral_centroid_mean",
    "en_distance",
    "ko_distance",
    "relative_distance_score",
    "liquid_target_distance",
    "liquid_alt_distance",
    "liquid_alt_relative_score",
    "liquid_onset_en_distance",
    "liquid_onset_ko_distance",
    "liquid_onset_relative_score",
    "schwa_onset_rms",
    "schwa_total_rms",
    "schwa_ref_onset_rms",
    "schwa_onset_rms_ratio",
    "schwa_onset_to_total_rms_ratio",
    "schwa_onset_rms_excess",
    "schwa_onset_to_total_excess",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "data/reference_ko/record/{good,korean_like,wrong_or_noisy} 실제 음성파일을 "
            "현재 scorer로 재채점하고 라벨별 분포를 저장합니다."
        ),
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--labels", nargs="+", default=list(LABELS), choices=LABELS)
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument(
        "--ko-reference-path",
        type=Path,
        default=KO_REFERENCE_PATH,
        help="한국어식 reference vector JSON 경로입니다. 없으면 --ko-reference-audio-dir에서 즉석 생성합니다.",
    )
    parser.add_argument(
        "--ko-reference-audio-dir",
        type=Path,
        default=KO_REFERENCE_AUDIO_DIR,
        help="한국어식 reference 음성 폴더입니다. record 하위 테스트셋은 reference 생성에서 제외합니다.",
    )
    parser.add_argument(
        "--disable-ko-reference",
        action="store_true",
        help="한국어식 reference penalty를 끄고 RMS/ZCR 기본 점수만 평가합니다.",
    )
    parser.add_argument(
        "--check-word-match",
        action="store_true",
        help="evaluate_recording_quality에 audio_path와 target_word를 넘겨 STT 단어 일치까지 확인합니다. 느릴 수 있습니다.",
    )
    parser.add_argument("--rms-floor", type=float, default=scorer_module._RMS_SCORE_FLOOR)
    parser.add_argument("--rms-tolerance", type=float, default=scorer_module._RMS_SCORE_TOLERANCE)
    parser.add_argument("--rms-steepness", type=float, default=scorer_module._RMS_SCORE_STEEPNESS)
    parser.add_argument("--zcr-floor", type=float, default=scorer_module._ZCR_SCORE_FLOOR)
    parser.add_argument("--zcr-tolerance", type=float, default=scorer_module._ZCR_SCORE_TOLERANCE)
    parser.add_argument("--zcr-steepness", type=float, default=scorer_module._ZCR_SCORE_STEEPNESS)
    parser.add_argument("--schwa-onset-rms-ratio-start", type=float, default=scorer_module._SCHWA_ONSET_RMS_RATIO_START)
    parser.add_argument("--schwa-onset-to-total-ratio-start", type=float, default=scorer_module._SCHWA_ONSET_TO_TOTAL_RATIO_START)
    parser.add_argument("--schwa-onset-rms-multiplier", type=float, default=scorer_module._SCHWA_ONSET_RMS_MULTIPLIER)
    parser.add_argument("--schwa-onset-to-total-multiplier", type=float, default=scorer_module._SCHWA_ONSET_TO_TOTAL_MULTIPLIER)
    parser.add_argument("--schwa-max-penalty", type=float, default=scorer_module._SCHWA_OVERSTRESS_MAX_PENALTY)
    parser.add_argument("--liquid-onset-score-start", type=float, default=scorer_module._LIQUID_ONSET_SCORE_START)
    parser.add_argument("--liquid-onset-score-strong", type=float, default=scorer_module._LIQUID_ONSET_SCORE_STRONG)
    parser.add_argument("--liquid-onset-multiplier", type=float, default=scorer_module._LIQUID_ONSET_PENALTY_MULTIPLIER)
    parser.add_argument("--liquid-onset-max-penalty", type=float, default=scorer_module._LIQUID_ONSET_PENALTY_MAX)
    return parser.parse_args()


def apply_score_parameters(args: argparse.Namespace) -> None:
    scorer_module._RMS_SCORE_FLOOR = args.rms_floor
    scorer_module._RMS_SCORE_TOLERANCE = args.rms_tolerance
    scorer_module._RMS_SCORE_STEEPNESS = args.rms_steepness
    scorer_module._ZCR_SCORE_FLOOR = args.zcr_floor
    scorer_module._ZCR_SCORE_TOLERANCE = args.zcr_tolerance
    scorer_module._ZCR_SCORE_STEEPNESS = args.zcr_steepness
    scorer_module._SCHWA_ONSET_RMS_RATIO_START = args.schwa_onset_rms_ratio_start
    scorer_module._SCHWA_ONSET_TO_TOTAL_RATIO_START = args.schwa_onset_to_total_ratio_start
    scorer_module._SCHWA_ONSET_RMS_MULTIPLIER = args.schwa_onset_rms_multiplier
    scorer_module._SCHWA_ONSET_TO_TOTAL_MULTIPLIER = args.schwa_onset_to_total_multiplier
    scorer_module._SCHWA_OVERSTRESS_MAX_PENALTY = args.schwa_max_penalty
    scorer_module._LIQUID_ONSET_SCORE_START = args.liquid_onset_score_start
    scorer_module._LIQUID_ONSET_SCORE_STRONG = args.liquid_onset_score_strong
    scorer_module._LIQUID_ONSET_PENALTY_MULTIPLIER = args.liquid_onset_multiplier
    scorer_module._LIQUID_ONSET_PENALTY_MAX = args.liquid_onset_max_penalty


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def audio_files_for_label(source_dir: Path, label: str) -> list[Path]:
    label_dir = source_dir / label
    if not label_dir.exists():
        return []
    return sorted(
        path
        for path in label_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def collect_audio_files(source_dir: Path, labels: list[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for label in labels:
        files.extend((label, path) for path in audio_files_for_label(source_dir, label))
    return files


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def filename_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in path.with_suffix("").parts:
        normalized = normalize_text(part)
        tokens.update(token for token in normalized.split() if token)
    return tokens


def build_word_index(target_words: list[TargetWord]) -> dict[str, list[TargetWord]]:
    index: dict[str, list[TargetWord]] = defaultdict(list)
    for target in target_words:
        index[target.word.lower()].append(target)
    return index


def find_target_by_filename(path: Path, word_index: dict[str, list[TargetWord]]) -> TargetWord:
    tokens = filename_tokens(path)
    candidates: list[TargetWord] = []
    for word, targets in word_index.items():
        if word in tokens:
            candidates.extend(targets)

    if not candidates:
        stem = normalize_text(path.stem)
        for word, targets in word_index.items():
            if stem == word or stem.startswith(f"{word} ") or f" {word} " in f" {stem} ":
                candidates.extend(targets)

    if not candidates:
        raise ValueError(f"파일명에서 target word를 찾을 수 없습니다: {path}")

    if len(candidates) == 1:
        return candidates[0]

    phoneme_tokens = tokens | set(normalize_text(path.stem).split())
    phoneme_matches = [target for target in candidates if normalize_text(target.phoneme) in phoneme_tokens]
    if len(phoneme_matches) == 1:
        return phoneme_matches[0]

    candidate_text = ", ".join(target.target_id for target in candidates)
    raise ValueError(f"파일명 target이 모호합니다: {path} -> {candidate_text}")


def _aggregate_float_feature(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [safe_float(sample.get(key)) for sample in samples]
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return round(float(mean(numeric_values)), 6)


def _aggregate_vector_feature(samples: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]] | None:
    values = [sample[key] for sample in samples if sample.get(key) is not None]
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return np.mean(arr, axis=0).round(6).tolist(), np.std(arr, axis=0).round(6).tolist()


def _aggregate_mfcc(samples: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    aggregated = _aggregate_vector_feature(samples, "mfcc_mean")
    if aggregated is None:
        raise ValueError("MFCC feature가 없는 한국어식 reference sample입니다.")
    return aggregated


def _aggregate_ko_reference(samples: list[dict[str, Any]], phoneme_type: str) -> dict[str, Any]:
    mfcc_mean, mfcc_std = _aggregate_mfcc(samples)
    vector: dict[str, Any] = {
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "phoneme_type": phoneme_type,
        "sample_count": len(samples),
    }
    for key in ("duration_ms", "rms_mean", "zcr_mean", "spectral_centroid_mean", "onset_rms_mean", "onset_window_ms", "onset_spectral_centroid_mean"):
        value = _aggregate_float_feature(samples, key)
        if value is not None:
            vector[key] = value
    for key in ("onset_mfcc_mean",):
        aggregated = _aggregate_vector_feature(samples, key)
        if aggregated is None:
            continue
        mean_values, std_values = aggregated
        vector[key] = mean_values
        vector[f"{key}_std"] = std_values
    return vector


def _is_inside_record_dir(path: Path, ko_reference_audio_dir: Path) -> bool:
    try:
        relative_parts = path.relative_to(ko_reference_audio_dir).parts
    except ValueError:
        return False
    return bool(relative_parts) and relative_parts[0].startswith("record")


def build_ko_reference_vectors_from_audio(
    ko_reference_audio_dir: Path,
    word_index: dict[str, list[TargetWord]],
    en_reference_vectors: dict[str, dict[str, Any]],
    word_targets: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not ko_reference_audio_dir.exists():
        return {}

    grouped_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for audio_path in sorted(ko_reference_audio_dir.rglob("*")):
        if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if _is_inside_record_dir(audio_path, ko_reference_audio_dir):
            continue
        try:
            target = find_target_by_filename(audio_path, word_index)
        except ValueError:
            continue
        if target.phoneme not in ALT_REFERENCE_PHONEMES:
            continue
        try:
            waveform, sample_rate = load_trimmed_audio(audio_path)
            include_onset = should_extract_onset(target.word, target.phoneme, word_targets)
            features = extract_features(waveform, sample_rate, include_onset=include_onset)
            grouped_samples[target.phoneme].append(features)
        except Exception:
            continue

    ko_reference_vectors: dict[str, dict[str, Any]] = {}
    for phoneme, samples in grouped_samples.items():
        if not samples:
            continue
        phoneme_type = str(en_reference_vectors.get(phoneme, {}).get("phoneme_type", "unknown"))
        ko_reference_vectors[phoneme] = _aggregate_ko_reference(samples, phoneme_type)
    return ko_reference_vectors


def load_ko_reference_vectors(
    ko_reference_path: Path,
    ko_reference_audio_dir: Path,
    word_index: dict[str, list[TargetWord]],
    en_reference_vectors: dict[str, dict[str, Any]],
    word_targets: dict[str, Any],
    disabled: bool,
) -> tuple[dict[str, dict[str, Any]], str]:
    if disabled:
        return {}, "disabled"

    built = build_ko_reference_vectors_from_audio(ko_reference_audio_dir, word_index, en_reference_vectors, word_targets)
    if ko_reference_path.exists():
        with ko_reference_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            merged = dict(loaded)
            for phoneme, vector in built.items():
                if phoneme not in merged or phoneme in LIQUID_REFERENCE_PHONEMES:
                    merged[phoneme] = vector
            return merged, f"{ko_reference_path}+built_liquid_from_audio:{ko_reference_audio_dir}"

    return built, f"built_from_audio:{ko_reference_audio_dir}"


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
    liquid_alt_reference = ko_reference if target.phoneme in LIQUID_REFERENCE_PHONEMES else None

    waveform, sample_rate = load_trimmed_audio(audio_path)
    include_onset = should_extract_onset(target.word, target.phoneme, word_targets)
    features = extract_features(waveform, sample_rate, include_onset=include_onset)
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
        "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix() if audio_path.is_relative_to(PROJECT_ROOT) else str(audio_path),
        "word": target.word,
        "phoneme": target.phoneme,
        "has_ko_reference": ko_reference is not None,
        "has_liquid_alt_reference": liquid_alt_reference is not None,
        "score": safe_float(result.get("score")),
        "recording_quality_status": result.get("recording_quality_status"),
        "issue_flags": json.dumps(result.get("issue_flags") or [], ensure_ascii=False),
        "feedback": result.get("feedback"),
        "duration_ms": safe_float(features.get("duration_ms")),
        "rms_mean": safe_float(features.get("rms_mean")),
        "zcr_mean": safe_float(features.get("zcr_mean")),
        "spectral_centroid_mean": safe_float(features.get("spectral_centroid_mean")),
        "onset_rms_mean": safe_float(features.get("onset_rms_mean")),
        "onset_window_ms": safe_float(features.get("onset_window_ms")),
        "onset_spectral_centroid_mean": safe_float(features.get("onset_spectral_centroid_mean")),
        "korean_pattern_status": details.get("korean_pattern_status"),
        "korean_pattern_penalty_policy": details.get("korean_pattern_penalty_policy"),
        "liquid_alt_status": details.get("liquid_alt_status"),
        "liquid_onset_status": details.get("liquid_onset_status"),
        "schwa_overstress_status": details.get("schwa_overstress_status"),
    }
    for key in DETAIL_FIELDS:
        row[key] = safe_float(details.get(key))
    return row


def score_values(rows: list[dict[str, Any]], key: str = "score") -> list[float]:
    values: list[float] = []
    for row in rows:
        value = safe_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "median": None, "min": None, "max": None}
    return {
        "avg": round(mean(values), 3),
        "median": round(median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def rate_at_or_above(rows: list[dict[str, Any]], threshold: float) -> float | None:
    scored_rows = [row for row in rows if safe_float(row.get("score")) is not None]
    if not scored_rows:
        return None
    passed = [row for row in scored_rows if float(row["score"]) >= threshold]
    return round(len(passed) / len(scored_rows), 3)


def quality_bad_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    bad_count = sum(1 for row in rows if row.get("recording_quality_status") == "bad")
    return round(bad_count / len(rows), 3)


def score_none_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    none_count = sum(1 for row in rows if safe_float(row.get("score")) is None)
    return round(none_count / len(rows), 3)


def ko_reference_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    count = sum(1 for row in rows if row.get("has_ko_reference"))
    return round(count / len(rows), 3)


def build_summary(results: list[dict[str, Any]], labels: list[str], threshold: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["label"])].append(row)

    summary: dict[str, Any] = {}
    for label in labels:
        rows = grouped.get(label, [])
        scores = score_values(rows)
        summary[label] = {
            "count": len(rows),
            "score": describe(scores),
            "score_none_rate": score_none_rate(rows),
            "quality_bad_rate": quality_bad_rate(rows),
            "ko_reference_rate": ko_reference_rate(rows),
            "at_or_above_threshold_rate": rate_at_or_above(rows, threshold),
            "at_or_above_threshold_count": sum(1 for row in rows if (safe_float(row.get("score")) or -1) >= threshold),
            "avg_mfcc_score": describe(score_values(rows, "mfcc_score"))["avg"],
            "avg_rms_score": describe(score_values(rows, "rms_score"))["avg"],
            "avg_zcr_score": describe(score_values(rows, "zcr_score"))["avg"],
            "avg_korean_like_penalty": describe(score_values(rows, "korean_like_penalty"))["avg"],
            "avg_liquid_alt_penalty": describe(score_values(rows, "liquid_alt_penalty"))["avg"],
            "avg_liquid_onset_penalty": describe(score_values(rows, "liquid_onset_penalty"))["avg"],
            "avg_relative_distance_score": describe(score_values(rows, "relative_distance_score"))["avg"],
            "avg_total_penalty": describe(score_values(rows, "total_penalty"))["avg"],
        }
    return summary


def is_failure_case(row: dict[str, Any], threshold: float) -> bool:
    label = str(row["label"])
    score = safe_float(row.get("score"))
    if label == "good":
        return score is None or score < threshold
    if label in ("korean_like", "wrong_or_noisy"):
        return score is not None and score >= threshold
    return False


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any], labels: list[str]) -> None:
    print("\nFolder score summary")
    print("label           count  avg     median  >=75_rate  none_rate  ko_ref  ko_pen  liq_pen  liq_on  rms_avg  zcr_avg")
    for label in labels:
        data = summary[label]
        score = data["score"]
        print(
            f"{label:<15} {data['count']:>5} "
            f"{score['avg'] if score['avg'] is not None else '-':>7} "
            f"{score['median'] if score['median'] is not None else '-':>7} "
            f"{data['at_or_above_threshold_rate'] if data['at_or_above_threshold_rate'] is not None else '-':>9} "
            f"{data['score_none_rate'] if data['score_none_rate'] is not None else '-':>9} "
            f"{data['ko_reference_rate'] if data['ko_reference_rate'] is not None else '-':>7} "
            f"{data['avg_korean_like_penalty'] if data['avg_korean_like_penalty'] is not None else '-':>7} "
            f"{data['avg_liquid_alt_penalty'] if data['avg_liquid_alt_penalty'] is not None else '-':>8} "
            f"{data['avg_liquid_onset_penalty'] if data['avg_liquid_onset_penalty'] is not None else '-':>7} "
            f"{data['avg_rms_score'] if data['avg_rms_score'] is not None else '-':>8} "
            f"{data['avg_zcr_score'] if data['avg_zcr_score'] is not None else '-':>8}"
        )


def main() -> None:
    args = parse_args()
    apply_score_parameters(args)

    if not args.source_dir.exists():
        raise FileNotFoundError(f"source dir를 찾을 수 없습니다: {args.source_dir}")

    target_words = load_target_words()
    word_index = build_word_index(target_words)
    reference_vectors = load_reference_vectors()
    word_targets = load_word_targets()
    ko_reference_vectors, ko_reference_source = load_ko_reference_vectors(
        args.ko_reference_path,
        args.ko_reference_audio_dir,
        word_index,
        reference_vectors,
        word_targets,
        args.disable_ko_reference,
    )
    audio_files = collect_audio_files(args.source_dir, args.labels)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for label, audio_path in audio_files:
        try:
            target = find_target_by_filename(audio_path, word_index)
            result = score_audio_file(
                label=label,
                audio_path=audio_path,
                target=target,
                reference_vectors=reference_vectors,
                ko_reference_vectors=ko_reference_vectors,
                word_targets=word_targets,
                check_word_match=args.check_word_match,
            )
            results.append(result)
            if is_failure_case(result, args.score_threshold):
                failures.append(result)
        except Exception as exc:
            errors.append(
                {
                    "label": label,
                    "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix() if audio_path.is_relative_to(PROJECT_ROOT) else str(audio_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = build_summary(results, args.labels, args.score_threshold)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(args.source_dir),
        "labels": args.labels,
        "score_threshold": args.score_threshold,
        "check_word_match": args.check_word_match,
        "ko_reference_source": ko_reference_source,
        "ko_reference_phonemes": sorted(ko_reference_vectors.keys()),
        "parameters": {
            "rms_floor": args.rms_floor,
            "rms_tolerance": args.rms_tolerance,
            "rms_steepness": args.rms_steepness,
            "zcr_floor": args.zcr_floor,
            "zcr_tolerance": args.zcr_tolerance,
            "zcr_steepness": args.zcr_steepness,
            "schwa_onset_rms_ratio_start": args.schwa_onset_rms_ratio_start,
            "schwa_onset_to_total_ratio_start": args.schwa_onset_to_total_ratio_start,
            "schwa_onset_rms_multiplier": args.schwa_onset_rms_multiplier,
            "schwa_onset_to_total_multiplier": args.schwa_onset_to_total_multiplier,
            "schwa_max_penalty": args.schwa_max_penalty,
            "liquid_onset_score_start": args.liquid_onset_score_start,
            "liquid_onset_score_strong": args.liquid_onset_score_strong,
            "liquid_onset_multiplier": args.liquid_onset_multiplier,
            "liquid_onset_max_penalty": args.liquid_onset_max_penalty,
        },
        "summary": summary,
        "audio_file_count": len(audio_files),
        "processed_count": len(results),
        "failure_count": len(failures),
        "error_count": len(errors),
    }

    summary_path = args.report_dir / f"folder_score_eval_summary_{timestamp}.json"
    details_path = args.report_dir / f"folder_score_eval_details_{timestamp}.csv"
    failures_path = args.report_dir / f"folder_score_eval_failures_{timestamp}.csv"
    errors_path = args.report_dir / f"folder_score_eval_errors_{timestamp}.csv"

    summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(details_path, results)
    write_csv(failures_path, failures)
    write_csv(errors_path, errors)

    print(f"ko_reference_source: {ko_reference_source}")
    print(f"ko_reference_phonemes: {', '.join(sorted(ko_reference_vectors.keys())) or '-'}")
    print_summary(summary, args.labels)
    print("\nSaved reports")
    print(f"summary : {summary_path}")
    print(f"details : {details_path}")
    print(f"failures: {failures_path}")
    if errors:
        print(f"errors  : {errors_path}")


if __name__ == "__main__":
    main()
