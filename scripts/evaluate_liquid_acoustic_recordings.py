from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.audio import load_trimmed_audio
from pipeline.features import extract_features
from pipeline.liquid_features import extract_liquid_acoustic_features
from pipeline.quality import evaluate_recording_quality
from pipeline.reference import TargetWord, load_reference_vectors, load_target_words
from pipeline.scorer import score_pronunciation
from pipeline.word_targets import attach_word_target_features, load_word_targets, should_extract_onset

SOURCE_DIR = PROJECT_ROOT / "data" / "reference_ko" / "record_selected_eval"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
LABELS = ("good", "korean_like")
LIQUID_PHONEMES = {"r", "l"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
SCORE_THRESHOLD = 75.0

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
DETAIL_FIELDS = (
    "base_score",
    "final_score",
    "mfcc_score",
    "duration_score",
    "zcr_score",
    "spectral_centroid_score",
    "pronunciation_penalty",
    "liquid_acoustic_penalty",
    "total_penalty",
    "liquid_acoustic_status",
    "liquid_acoustic_penalty_applied",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="/r/, /l/ acoustic feature penalty를 실제 scoring에 연결해 평가합니다.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--labels", nargs="+", default=list(LABELS))
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    raise ValueError(f"파일명 target이 모호합니다: {path}")


def collect_audio_files(source_dir: Path, labels: list[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for label in labels:
        label_dir = source_dir / label
        if not label_dir.exists():
            continue
        files.extend(
            (label, path)
            for path in sorted(label_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
    return files


def score_audio_file(
    label: str,
    audio_path: Path,
    target: TargetWord,
    reference_vectors: dict[str, dict[str, Any]],
    word_targets: dict[str, Any],
) -> dict[str, Any]:
    reference = reference_vectors.get(target.phoneme)
    if reference is None:
        raise KeyError(f"reference vector가 없습니다: /{target.phoneme}/")

    waveform, sample_rate = load_trimmed_audio(audio_path)
    include_onset = should_extract_onset(target.word, target.phoneme, word_targets)
    features = extract_features(waveform, sample_rate, include_onset=include_onset)
    features.update(extract_liquid_acoustic_features(waveform, sample_rate))
    attach_word_target_features(features, target.word, target.phoneme, word_targets)

    quality_result = evaluate_recording_quality(features=features, reference=reference)
    result = score_pronunciation(
        user_features=features,
        reference=reference,
        phoneme=target.phoneme,
        recording_quality_result=quality_result,
    )
    details = result.get("details", {})

    row: dict[str, Any] = {
        "label": label,
        "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix() if audio_path.is_relative_to(PROJECT_ROOT) else str(audio_path),
        "word": target.word,
        "phoneme": target.phoneme,
        "score": safe_float(result.get("score")),
        "recording_quality_status": result.get("recording_quality_status"),
    }
    for key in LIQUID_FEATURE_FIELDS:
        row[key] = safe_float(features.get(key))
    for key in DETAIL_FIELDS:
        value = details.get(key)
        row[key] = value if isinstance(value, str) else safe_float(value)
    return row


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "min": None, "max": None}
    return {"avg": round(mean(values), 3), "min": round(min(values), 3), "max": round(max(values), 3)}


def score_values(rows: list[dict[str, Any]], key: str = "score") -> list[float]:
    values: list[float] = []
    for row in rows:
        value = safe_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def build_summary(rows: list[dict[str, Any]], labels: list[str], threshold: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    word_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)
        word_grouped[f"{row['label']}:{row['word']}:{row['phoneme']}"].append(row)

    summary: dict[str, Any] = {"by_label": {}, "by_label_word": {}}
    for label in labels:
        label_rows = grouped.get(label, [])
        scores = score_values(label_rows)
        summary["by_label"][label] = {
            "count": len(label_rows),
            "score": describe(scores),
            "at_or_above_threshold_count": sum(1 for row in label_rows if (safe_float(row.get("score")) or -1) >= threshold),
            "avg_liquid_acoustic_penalty": describe(score_values(label_rows, "liquid_acoustic_penalty"))["avg"],
        }
    for key, group_rows in word_grouped.items():
        summary["by_label_word"][key] = {
            "count": len(group_rows),
            "score": describe(score_values(group_rows)),
            "avg_liquid_acoustic_penalty": describe(score_values(group_rows, "liquid_acoustic_penalty"))["avg"],
            "avg_mel_f3_to_low_ratio": describe(score_values(group_rows, "liquid_mel_f3_to_low_ratio"))["avg"],
            "avg_transition_mfcc_distance": describe(score_values(group_rows, "liquid_transition_mfcc_distance"))["avg"],
            "avg_transition_mfcc_c0_delta": describe(score_values(group_rows, "liquid_transition_mfcc_c0_delta"))["avg"],
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.source_dir.exists():
        raise FileNotFoundError(f"source dir를 찾을 수 없습니다: {args.source_dir}")

    target_words = load_target_words()
    word_index = build_word_index(target_words)
    reference_vectors = load_reference_vectors()
    word_targets = load_word_targets()
    audio_files = collect_audio_files(args.source_dir, args.labels)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for label, audio_path in audio_files:
        try:
            target = find_target_by_filename(audio_path, word_index)
            if target.phoneme not in LIQUID_PHONEMES:
                continue
            rows.append(score_audio_file(label, audio_path, target, reference_vectors, word_targets))
        except Exception as exc:
            errors.append({"label": label, "audio_path": str(audio_path), "error": f"{type(exc).__name__}: {exc}"})

    args.report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = args.report_dir / f"liquid_acoustic_score_eval_details_{timestamp}.csv"
    errors_path = args.report_dir / f"liquid_acoustic_score_eval_errors_{timestamp}.csv"
    summary_path = args.report_dir / f"liquid_acoustic_score_eval_summary_{timestamp}.json"

    write_csv(details_path, rows)
    write_csv(errors_path, errors)
    summary = build_summary(rows, args.labels, args.score_threshold)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Liquid acoustic score eval")
    print(f"processed: {len(rows)}")
    print(f"errors   : {len(errors)}")
    print(f"details  : {details_path}")
    print(f"summary  : {summary_path}")
    if errors:
        print(f"error csv: {errors_path}")


if __name__ == "__main__":
    main()
