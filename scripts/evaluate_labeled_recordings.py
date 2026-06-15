from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.audio import load_trimmed_audio
from pipeline.db import DEFAULT_DB_PATH
from pipeline.features import extract_features
from pipeline.quality import RecordingQualityResult, evaluate_recording_quality
from pipeline.reference import load_reference_vectors
from pipeline.scorer import score_pronunciation
import pipeline.scorer as scorer_module
from pipeline.word_targets import attach_word_target_features, load_word_targets, should_extract_onset

LABELS = ("good", "korean", "wrong", "exclude")
SCORE_THRESHOLD = 75.0
DETAIL_FIELDS = (
    "mfcc_score",
    "duration_score",
    "rms_score",
    "zcr_score",
    "spectral_centroid_score",
    "base_score",
    "final_score",
    "pronunciation_penalty",
    "korean_like_penalty",
    "liquid_alt_penalty",
    "total_penalty",
)
FEATURE_FIELDS = (
    "mfcc_mean",
    "duration_ms",
    "rms_mean",
    "zcr_mean",
    "spectral_centroid_mean",
    "target_id",
    "target_position",
    "target_phoneme",
    "onset_window_ms",
    "onset_mfcc_mean",
    "onset_zcr_mean",
    "onset_rms_mean",
    "onset_spectral_centroid_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="기존 라벨링된 user_recordings 음성파일을 현재 scorer로 재채점하고 기존 점수와 비교합니다.",
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--labels", nargs="+", default=list(LABELS), choices=LABELS)
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument(
        "--use-db-features",
        action="store_true",
        help="음성파일을 다시 읽지 않고 DB/details_json에 저장된 feature로만 빠르게 재채점합니다.",
    )
    parser.add_argument(
        "--recheck-quality",
        action="store_true",
        help="기존 quality 결과를 재사용하지 않고 evaluate_recording_quality를 다시 실행합니다. STT 때문에 느릴 수 있습니다.",
    )
    parser.add_argument("--rms-floor", type=float, default=scorer_module._RMS_SCORE_FLOOR)
    parser.add_argument("--rms-tolerance", type=float, default=scorer_module._RMS_SCORE_TOLERANCE)
    parser.add_argument("--rms-steepness", type=float, default=scorer_module._RMS_SCORE_STEEPNESS)
    parser.add_argument("--zcr-floor", type=float, default=scorer_module._ZCR_SCORE_FLOOR)
    parser.add_argument("--zcr-tolerance", type=float, default=scorer_module._ZCR_SCORE_TOLERANCE)
    parser.add_argument("--zcr-steepness", type=float, default=scorer_module._ZCR_SCORE_STEEPNESS)
    return parser.parse_args()


def apply_score_parameters(args: argparse.Namespace) -> None:
    scorer_module._RMS_SCORE_FLOOR = args.rms_floor
    scorer_module._RMS_SCORE_TOLERANCE = args.rms_tolerance
    scorer_module._RMS_SCORE_STEEPNESS = args.rms_steepness
    scorer_module._ZCR_SCORE_FLOOR = args.zcr_floor
    scorer_module._ZCR_SCORE_TOLERANCE = args.zcr_tolerance
    scorer_module._ZCR_SCORE_STEEPNESS = args.zcr_steepness


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_details(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_labeled_rows(db_path: Path, labels: list[str]) -> list[sqlite3.Row]:
    if not db_path.exists():
        raise FileNotFoundError(f"DB 파일을 찾을 수 없습니다: {db_path}")
    placeholders = ",".join("?" for _ in labels)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            f"""
            SELECT *
            FROM user_recordings
            WHERE test_label IN ({placeholders})
            ORDER BY id
            """,
            labels,
        ).fetchall()


def resolve_recording_path(recording_path: str | None) -> Path | None:
    if not recording_path:
        return None
    path = Path(recording_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def quality_from_details(details: dict[str, Any]) -> RecordingQualityResult:
    status = details.get("recording_quality_status")
    if status not in ("ok", "bad"):
        status = "ok"
    issue_flags = details.get("issue_flags")
    if not isinstance(issue_flags, list):
        issue_flags = []
    word_match = details.get("word_match")
    if not isinstance(word_match, bool):
        word_match = None
    stt_status = details.get("stt_status")
    if stt_status not in ("matched", "mismatched", "uncertain", "unavailable"):
        stt_status = "unavailable"
    return {
        "status": status,
        "issue_flags": [str(flag) for flag in issue_flags],
        "word_match": word_match,
        "stt_status": stt_status,
        "transcript": details.get("transcript") if isinstance(details.get("transcript"), str) else None,
        "prompted_transcript": details.get("prompted_transcript") if isinstance(details.get("prompted_transcript"), str) else None,
    }


def build_db_features(row: sqlite3.Row, details: dict[str, Any]) -> dict[str, Any]:
    features = {key: details[key] for key in FEATURE_FIELDS if key in details}
    for key in ("duration_ms", "rms_mean", "zcr_mean", "spectral_centroid_mean"):
        if key not in features:
            features[key] = row[key]
    return features


def extract_audio_features(row: sqlite3.Row, details: dict[str, Any], word_targets: dict[str, Any]) -> dict[str, Any]:
    recording_path = resolve_recording_path(row["recording_path"])
    if recording_path is None:
        raise FileNotFoundError("recording_path가 비어 있습니다.")
    if not recording_path.exists():
        raise FileNotFoundError(f"녹음 파일을 찾을 수 없습니다: {recording_path}")

    word = str(row["word"])
    phoneme = str(row["phoneme"])
    waveform, sample_rate = load_trimmed_audio(recording_path)
    include_onset = should_extract_onset(word, phoneme, word_targets)
    features = extract_features(waveform, sample_rate, include_onset=include_onset)
    attach_word_target_features(features, word, phoneme, word_targets)

    # 기존 분석 details에만 있던 메타 정보는 유지한다.
    for key in ("target_id", "target_position", "target_phoneme"):
        if key in details and key not in features:
            features[key] = details[key]
    return features


def row_to_result(
    row: sqlite3.Row,
    reference_vectors: dict[str, dict[str, Any]],
    word_targets: dict[str, Any],
    *,
    use_db_features: bool,
    recheck_quality: bool,
) -> dict[str, Any]:
    details = parse_details(row["details_json"])
    phoneme = str(row["phoneme"])
    reference = reference_vectors.get(phoneme)
    if reference is None:
        raise KeyError(f"reference vector가 없습니다: /{phoneme}/")

    if use_db_features:
        features = build_db_features(row, details)
    else:
        features = extract_audio_features(row, details, word_targets)

    quality_result = None
    if recheck_quality:
        recording_path = resolve_recording_path(row["recording_path"])
        quality_result = evaluate_recording_quality(
            features=features,
            reference=reference,
            audio_path=None if recording_path is None else str(recording_path),
            target_word=str(row["word"]),
        )
    else:
        quality_result = quality_from_details(details)

    new_result = score_pronunciation(
        user_features=features,
        reference=reference,
        phoneme=phoneme,
        recording_quality_result=quality_result,
    )
    new_details = new_result.get("details", {})
    return {
        "id": row["id"],
        "word": row["word"],
        "phoneme": phoneme,
        "label": row["test_label"],
        "recording_path": row["recording_path"],
        "old_score": safe_float(row["score"]),
        "new_score": safe_float(new_result.get("score")),
        "old_quality_status": details.get("recording_quality_status"),
        "new_quality_status": new_result.get("recording_quality_status"),
        "old_issue_flags": details.get("issue_flags") or [],
        "new_issue_flags": new_result.get("issue_flags") or [],
        **{f"old_{key}": safe_float(row[key]) if key in row.keys() else safe_float(details.get(key)) for key in DETAIL_FIELDS},
        **{f"new_{key}": safe_float(new_details.get(key)) for key in DETAIL_FIELDS},
    }


def score_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = safe_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def rate_at_or_above(rows: list[dict[str, Any]], key: str, threshold: float) -> float | None:
    scored_rows = [row for row in rows if safe_float(row.get(key)) is not None]
    if not scored_rows:
        return None
    passed = [row for row in scored_rows if float(row[key]) >= threshold]
    return len(passed) / len(scored_rows)


def describe(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "median": None, "min": None, "max": None}
    return {
        "avg": round(mean(values), 3),
        "median": round(median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def build_summary(results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["label"])].append(row)

    summary: dict[str, Any] = {}
    for label in LABELS:
        rows = grouped.get(label, [])
        old_scores = score_values(rows, "old_score")
        new_scores = score_values(rows, "new_score")
        old_avg = mean(old_scores) if old_scores else None
        new_avg = mean(new_scores) if new_scores else None
        summary[label] = {
            "count": len(rows),
            "old_score": describe(old_scores),
            "new_score": describe(new_scores),
            "avg_diff": None if old_avg is None or new_avg is None else round(new_avg - old_avg, 3),
            "old_score_none_rate": None if not rows else round(1 - len(old_scores) / len(rows), 3),
            "new_score_none_rate": None if not rows else round(1 - len(new_scores) / len(rows), 3),
            "old_at_or_above_threshold_rate": rate_at_or_above(rows, "old_score", threshold),
            "new_at_or_above_threshold_rate": rate_at_or_above(rows, "new_score", threshold),
            "old_at_or_above_threshold_count": sum(1 for row in rows if (safe_float(row.get("old_score")) or -1) >= threshold),
            "new_at_or_above_threshold_count": sum(1 for row in rows if (safe_float(row.get("new_score")) or -1) >= threshold),
        }
    return summary


def is_failure_case(row: dict[str, Any], threshold: float) -> bool:
    label = row["label"]
    old_score = safe_float(row.get("old_score"))
    new_score = safe_float(row.get("new_score"))
    if label == "good":
        return new_score is None or new_score < threshold
    if label in ("korean", "wrong"):
        return new_score is not None and new_score >= threshold
    if label == "exclude":
        return new_score is not None
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


def print_summary(summary: dict[str, Any]) -> None:
    print("\nLabel summary")
    print("label    count  old_avg  new_avg  diff  old>=75  new>=75  new_none")
    for label in LABELS:
        data = summary[label]
        old_avg = data["old_score"]["avg"]
        new_avg = data["new_score"]["avg"]
        old_rate = data["old_at_or_above_threshold_rate"]
        new_rate = data["new_at_or_above_threshold_rate"]
        print(
            f"{label:<8} {data['count']:>5} "
            f"{old_avg if old_avg is not None else '-':>7} "
            f"{new_avg if new_avg is not None else '-':>7} "
            f"{data['avg_diff'] if data['avg_diff'] is not None else '-':>5} "
            f"{old_rate if old_rate is not None else '-':>7} "
            f"{new_rate if new_rate is not None else '-':>7} "
            f"{data['new_score_none_rate'] if data['new_score_none_rate'] is not None else '-':>8}"
        )


def main() -> None:
    args = parse_args()
    apply_score_parameters(args)

    rows = load_labeled_rows(args.db_path, args.labels)
    reference_vectors = load_reference_vectors()
    word_targets = load_word_targets()

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for row in rows:
        try:
            result = row_to_result(
                row,
                reference_vectors,
                word_targets,
                use_db_features=args.use_db_features,
                recheck_quality=args.recheck_quality,
            )
            results.append(result)
            if is_failure_case(result, args.score_threshold):
                failures.append(result)
        except Exception as exc:
            error_row = {
                "id": row["id"],
                "word": row["word"],
                "phoneme": row["phoneme"],
                "label": row["test_label"],
                "recording_path": row["recording_path"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error_row)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = build_summary(results, args.score_threshold)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": str(args.db_path),
        "labels": args.labels,
        "score_threshold": args.score_threshold,
        "use_db_features": args.use_db_features,
        "recheck_quality": args.recheck_quality,
        "parameters": {
            "rms_floor": args.rms_floor,
            "rms_tolerance": args.rms_tolerance,
            "rms_steepness": args.rms_steepness,
            "zcr_floor": args.zcr_floor,
            "zcr_tolerance": args.zcr_tolerance,
            "zcr_steepness": args.zcr_steepness,
        },
        "summary": summary,
        "processed_count": len(results),
        "error_count": len(errors),
        "failure_count": len(failures),
    }

    summary_path = args.report_dir / f"scorer_eval_summary_{timestamp}.json"
    details_path = args.report_dir / f"scorer_eval_details_{timestamp}.csv"
    failures_path = args.report_dir / f"scorer_eval_failures_{timestamp}.csv"
    errors_path = args.report_dir / f"scorer_eval_errors_{timestamp}.csv"

    summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(details_path, results)
    write_csv(failures_path, failures)
    write_csv(errors_path, errors)

    print_summary(summary)
    print("\nSaved reports")
    print(f"summary : {summary_path}")
    print(f"details : {details_path}")
    print(f"failures: {failures_path}")
    print(f"errors  : {errors_path}")


if __name__ == "__main__":
    main()
