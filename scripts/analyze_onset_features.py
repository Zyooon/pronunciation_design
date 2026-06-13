import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from pipeline.db import DEFAULT_DB_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUTPUT_PATH = PROJECT_ROOT / "data" / "onset_analysis_report.json"
DEFAULT_CSV_OUTPUT_PATH = PROJECT_ROOT / "data" / "onset_analysis_report.csv"
TARGET_LABELS = ("good", "korean_like")
ONSET_SCALAR_KEYS = (
    "onset_window_ms",
    "onset_zcr_mean",
    "onset_rms_mean",
    "onset_spectral_centroid_mean",
)


@dataclass(frozen=True)
class OnsetGroupStats:
    count: int
    avg_score: float | None
    avg_onset_window_ms: float | None
    avg_onset_zcr_mean: float | None
    avg_onset_rms_mean: float | None
    avg_onset_spectral_centroid_mean: float | None
    avg_onset_mfcc_mean: list[float] | None


@dataclass(frozen=True)
class OnsetComparisonRow:
    word: str
    phoneme: str
    target_id: str | None
    target_position: str | None
    good_count: int
    korean_like_count: int
    good_score: float | None
    korean_like_score: float | None
    score_gap_good_minus_ko: float | None
    onset_zcr_gap_good_minus_ko: float | None
    onset_rms_gap_good_minus_ko: float | None
    onset_spectral_gap_good_minus_ko: float | None
    onset_mfcc_distance_good_ko: float | None
    suggested_action: str
    comment: str


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 6) if values else None


def _avg_vector(vectors: list[list[float]]) -> list[float] | None:
    if not vectors:
        return None
    try:
        return np.mean(np.array(vectors, dtype=float), axis=0).round(6).tolist()
    except Exception:
        return None


def _vector_distance(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    try:
        return round(float(np.linalg.norm(np.array(left) - np.array(right))), 6)
    except Exception:
        return None


def _gap(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 6)


def _parse_details_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _query_rows(db_path: Path, latest_only: bool) -> list[dict[str, Any]]:
    if not db_path.exists():
        raise FileNotFoundError(f"DB 파일이 없습니다: {db_path}")

    if latest_only:
        sql = """
        WITH latest AS (
            SELECT *
            FROM user_recordings u
            WHERE id = (
                SELECT MAX(id)
                FROM user_recordings
                WHERE word = u.word
                  AND COALESCE(phoneme, '') = COALESCE(u.phoneme, '')
                  AND COALESCE(test_label, '') = COALESCE(u.test_label, '')
            )
        )
        SELECT id, word, phoneme, test_label, score, details_json
        FROM latest
        WHERE test_label IN ('good', 'korean_like')
        ORDER BY word, phoneme, test_label, id
        """
        params: tuple[Any, ...] = ()
    else:
        sql = """
        SELECT id, word, phoneme, test_label, score, details_json
        FROM user_recordings
        WHERE test_label IN ('good', 'korean_like')
        ORDER BY word, phoneme, test_label, id
        """
        params = ()

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    return rows


def _has_onset_details(details: dict[str, Any]) -> bool:
    return any(key in details for key in ONSET_SCALAR_KEYS) or "onset_mfcc_mean" in details


def _build_group_stats(rows: list[dict[str, Any]]) -> OnsetGroupStats:
    scores: list[float] = []
    scalar_values: dict[str, list[float]] = {key: [] for key in ONSET_SCALAR_KEYS}
    mfcc_vectors: list[list[float]] = []

    for row in rows:
        score = _safe_float(row.get("score"))
        if score is not None:
            scores.append(score)

        details = row["_details"]
        for key in ONSET_SCALAR_KEYS:
            value = _safe_float(details.get(key))
            if value is not None:
                scalar_values[key].append(value)

        mfcc = _safe_float_list(details.get("onset_mfcc_mean"))
        if mfcc:
            mfcc_vectors.append(mfcc)

    return OnsetGroupStats(
        count=len(rows),
        avg_score=_avg(scores),
        avg_onset_window_ms=_avg(scalar_values["onset_window_ms"]),
        avg_onset_zcr_mean=_avg(scalar_values["onset_zcr_mean"]),
        avg_onset_rms_mean=_avg(scalar_values["onset_rms_mean"]),
        avg_onset_spectral_centroid_mean=_avg(scalar_values["onset_spectral_centroid_mean"]),
        avg_onset_mfcc_mean=_avg_vector(mfcc_vectors),
    )


def _suggest_action(row: OnsetComparisonRow) -> tuple[str, str]:
    if row.good_count == 0 or row.korean_like_count == 0:
        return "insufficient_data", "good 또는 korean_like 샘플이 부족합니다."

    if row.score_gap_good_minus_ko is not None and row.score_gap_good_minus_ko < 0:
        return "hold", "현재 점수에서 korean_like가 good보다 높아 바로 scorer 반영하면 위험합니다."

    strong_signal_count = 0
    if row.onset_mfcc_distance_good_ko is not None and row.onset_mfcc_distance_good_ko >= 80:
        strong_signal_count += 1
    if row.onset_zcr_gap_good_minus_ko is not None and abs(row.onset_zcr_gap_good_minus_ko) >= 0.02:
        strong_signal_count += 1
    if row.onset_spectral_gap_good_minus_ko is not None and abs(row.onset_spectral_gap_good_minus_ko) >= 100:
        strong_signal_count += 1

    if strong_signal_count >= 2:
        return "candidate", "onset feature 차이가 비교적 뚜렷합니다. 작은 penalty 후보로 검토하세요."
    if strong_signal_count == 1:
        return "watch", "일부 onset feature 신호가 있으나 추가 샘플 확인이 필요합니다."
    return "hold", "onset feature 차이가 아직 약합니다."


def build_onset_report(db_path: Path, latest_only: bool) -> dict[str, Any]:
    rows = _query_rows(db_path, latest_only=latest_only)
    parsed_rows: list[dict[str, Any]] = []
    skipped_without_onset = 0

    for row in rows:
        details = _parse_details_json(row.get("details_json"))
        if not _has_onset_details(details):
            skipped_without_onset += 1
            continue
        row["_details"] = details
        parsed_rows.append(row)

    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    metadata_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for row in parsed_rows:
        word = str(row.get("word") or "")
        phoneme = str(row.get("phoneme") or "")
        label = str(row.get("test_label") or "")
        key = (word, phoneme)
        grouped[key][label].append(row)
        details = row["_details"]
        metadata_by_key.setdefault(
            key,
            {
                "target_id": details.get("target_id"),
                "target_position": details.get("target_position"),
            },
        )

    comparison_rows: list[OnsetComparisonRow] = []
    group_stats_payload: dict[str, Any] = {}

    for (word, phoneme), label_groups in sorted(grouped.items()):
        good_stats = _build_group_stats(label_groups.get("good", []))
        ko_stats = _build_group_stats(label_groups.get("korean_like", []))
        metadata = metadata_by_key.get((word, phoneme), {})

        row = OnsetComparisonRow(
            word=word,
            phoneme=phoneme,
            target_id=metadata.get("target_id"),
            target_position=metadata.get("target_position"),
            good_count=good_stats.count,
            korean_like_count=ko_stats.count,
            good_score=good_stats.avg_score,
            korean_like_score=ko_stats.avg_score,
            score_gap_good_minus_ko=_gap(good_stats.avg_score, ko_stats.avg_score),
            onset_zcr_gap_good_minus_ko=_gap(good_stats.avg_onset_zcr_mean, ko_stats.avg_onset_zcr_mean),
            onset_rms_gap_good_minus_ko=_gap(good_stats.avg_onset_rms_mean, ko_stats.avg_onset_rms_mean),
            onset_spectral_gap_good_minus_ko=_gap(
                good_stats.avg_onset_spectral_centroid_mean,
                ko_stats.avg_onset_spectral_centroid_mean,
            ),
            onset_mfcc_distance_good_ko=_vector_distance(
                good_stats.avg_onset_mfcc_mean,
                ko_stats.avg_onset_mfcc_mean,
            ),
            suggested_action="",
            comment="",
        )
        action, comment = _suggest_action(row)
        row = OnsetComparisonRow(**{**asdict(row), "suggested_action": action, "comment": comment})
        comparison_rows.append(row)

        key = f"{word}|{phoneme}"
        group_stats_payload[key] = {
            "good": asdict(good_stats),
            "korean_like": asdict(ko_stats),
        }

    action_counts: dict[str, int] = defaultdict(int)
    for row in comparison_rows:
        action_counts[row.suggested_action] += 1

    return {
        "metadata": {
            "db_path": str(db_path),
            "latest_only": latest_only,
            "input_rows": len(rows),
            "onset_rows": len(parsed_rows),
            "skipped_without_onset": skipped_without_onset,
            "word_count": len(comparison_rows),
            "action_counts": dict(sorted(action_counts.items())),
        },
        "comparisons": [asdict(row) for row in comparison_rows],
        "group_stats": group_stats_payload,
    }


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def write_csv_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = report.get("comparisons", [])
    fieldnames = [
        "word",
        "phoneme",
        "target_id",
        "target_position",
        "good_count",
        "korean_like_count",
        "good_score",
        "korean_like_score",
        "score_gap_good_minus_ko",
        "onset_zcr_gap_good_minus_ko",
        "onset_rms_gap_good_minus_ko",
        "onset_spectral_gap_good_minus_ko",
        "onset_mfcc_distance_good_ko",
        "suggested_action",
        "comment",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def print_summary(report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    print("onset analysis report")
    print(f"- db_path: {metadata['db_path']}")
    print(f"- latest_only: {metadata['latest_only']}")
    print(f"- input_rows: {metadata['input_rows']}")
    print(f"- onset_rows: {metadata['onset_rows']}")
    print(f"- skipped_without_onset: {metadata['skipped_without_onset']}")
    print(f"- word_count: {metadata['word_count']}")
    print(f"- action_counts: {metadata['action_counts']}")

    candidates = [row for row in report["comparisons"] if row["suggested_action"] == "candidate"]
    if candidates:
        print("\ncandidate words:")
        for row in candidates:
            print(
                f"- {row['word']} /{row['phoneme']}/ "
                f"score_gap={row['score_gap_good_minus_ko']} "
                f"onset_mfcc={row['onset_mfcc_distance_good_ko']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DB details_json의 onset feature를 good/korean_like 기준으로 분석합니다.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="분석할 pronunciation.db 경로")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT_PATH, help="JSON 리포트 저장 경로")
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT_PATH, help="CSV 리포트 저장 경로")
    parser.add_argument("--all", action="store_true", help="word+phoneme+label 최신 row만 보지 않고 전체 row를 분석")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_onset_report(db_path=args.db, latest_only=not args.all)
    write_json_report(report, args.json_output)
    write_csv_report(report, args.csv_output)
    print_summary(report)
    print(f"\n저장 완료: {args.json_output}")
    print(f"저장 완료: {args.csv_output}")


if __name__ == "__main__":
    main()
