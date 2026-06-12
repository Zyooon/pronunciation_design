import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "comparison_results.json"
DEFAULT_JSON_OUTPUT_PATH = PROJECT_ROOT / "data" / "comparison_analysis_report.json"
DEFAULT_MARKDOWN_OUTPUT_PATH = PROJECT_ROOT / "data" / "comparison_analysis_report.md"

ANALYSIS_METRIC_NAMES = [
    "mfcc_distance",
    "mfcc_cosine_distance",
    "duration_diff_ms",
    "duration_relative_diff",
    "duration_ko_en_ratio",
    "zcr_diff",
    "zcr_relative_diff",
    "zcr_ko_en_ratio",
    "rms_diff",
    "rms_relative_diff",
    "rms_ko_en_ratio",
    "spectral_centroid_diff",
    "spectral_centroid_relative_diff",
    "spectral_centroid_ko_en_ratio",
]

DOMINANT_FEATURE_METRIC_NAMES = {
    "duration": "duration_relative_diff",
    "zcr": "zcr_relative_diff",
    "rms": "rms_relative_diff",
    "spectral_centroid": "spectral_centroid_relative_diff",
}

OUTLIER_METRIC_NAMES = [
    "mfcc_distance",
    "duration_relative_diff",
    "zcr_relative_diff",
    "rms_relative_diff",
    "spectral_centroid_relative_diff",
]


@dataclass(frozen=True)
class MetricSummary:
    name: str
    count: int
    average: float | None
    minimum: float | None
    maximum: float | None
    stdev: float | None

    @classmethod
    def from_values(cls, name: str, values: list[float]) -> "MetricSummary":
        if not values:
            return cls(
                name=name,
                count=0,
                average=None,
                minimum=None,
                maximum=None,
                stdev=None,
            )

        return cls(
            name=name,
            count=len(values),
            average=round(statistics.mean(values), 6),
            minimum=round(min(values), 6),
            maximum=round(max(values), 6),
            stdev=round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        )

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "name": self.name,
            "count": self.count,
            "average": self.average,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "stdev": self.stdev,
        }


@dataclass(frozen=True)
class OutlierCandidate:
    word: str
    phoneme: str
    metric_name: str
    value: float
    average: float
    stdev: float
    z_score: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "word": self.word,
            "phoneme": self.phoneme,
            "metric_name": self.metric_name,
            "value": round(self.value, 6),
            "average": round(self.average, 6),
            "stdev": round(self.stdev, 6),
            "z_score": round(self.z_score, 6),
        }


def load_comparison_results(input_path: Path) -> dict[str, Any]:
    """comparison_results.json을 읽어 반환한다."""
    if not input_path.exists():
        raise FileNotFoundError(f"comparison_results.json not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if "results" not in payload:
        raise ValueError("Invalid comparison_results.json: missing 'results' key.")

    if not isinstance(payload["results"], list):
        raise ValueError("Invalid comparison_results.json: 'results' must be a list.")

    return payload


def get_success_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """분석 가능한 성공 row만 반환한다."""
    return [
        row
        for row in rows
        if row.get("status", "ok") == "ok"
    ]


def get_error_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """실패 row만 반환한다."""
    return [
        row
        for row in rows
        if row.get("status") == "error"
    ]


def get_float_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_metric_values(rows: list[dict[str, Any]], metric_name: str) -> list[float]:
    values: list[float] = []

    for row in rows:
        value = get_float_value(row, metric_name)
        if value is None:
            continue

        values.append(value)

    return values


def group_rows_by_phoneme(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        phoneme = str(row.get("phoneme", "")).strip()
        if not phoneme:
            continue

        grouped_rows.setdefault(phoneme, []).append(row)

    return grouped_rows


def build_metric_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metric_summaries: dict[str, dict[str, Any]] = {}

    for metric_name in ANALYSIS_METRIC_NAMES:
        values = collect_metric_values(rows, metric_name)
        metric_summary = MetricSummary.from_values(metric_name, values)
        metric_summaries[metric_name] = metric_summary.to_dict()

    return metric_summaries


def get_metric_average(rows: list[dict[str, Any]], metric_name: str) -> float:
    values = collect_metric_values(rows, metric_name)
    if not values:
        return 0.0

    return statistics.mean(values)


def get_dominant_features(rows: list[dict[str, Any]], max_count: int = 3) -> list[str]:
    feature_scores = []

    for feature_name, metric_name in DOMINANT_FEATURE_METRIC_NAMES.items():
        average = get_metric_average(rows, metric_name)
        feature_scores.append((feature_name, average))

    feature_scores.sort(key=lambda item: item[1], reverse=True)

    return [
        feature_name
        for feature_name, average in feature_scores[:max_count]
        if average > 0
    ]


def find_outliers_for_metric(
    rows: list[dict[str, Any]],
    phoneme: str,
    metric_name: str,
    z_score_threshold: float,
) -> list[OutlierCandidate]:
    metric_values = collect_metric_values(rows, metric_name)

    if len(metric_values) < 3:
        return []

    average = statistics.mean(metric_values)
    stdev = statistics.pstdev(metric_values)

    if stdev == 0:
        return []

    outliers: list[OutlierCandidate] = []

    for row in rows:
        value = get_float_value(row, metric_name)
        if value is None:
            continue

        z_score = (value - average) / stdev
        if z_score < z_score_threshold:
            continue

        outliers.append(
            OutlierCandidate(
                word=str(row.get("word", "")),
                phoneme=phoneme,
                metric_name=metric_name,
                value=value,
                average=average,
                stdev=stdev,
                z_score=z_score,
            )
        )

    return outliers


def find_outliers(
    rows: list[dict[str, Any]],
    phoneme: str,
    z_score_threshold: float,
) -> list[dict[str, Any]]:
    outliers: list[OutlierCandidate] = []

    for metric_name in OUTLIER_METRIC_NAMES:
        outliers.extend(
            find_outliers_for_metric(
                rows=rows,
                phoneme=phoneme,
                metric_name=metric_name,
                z_score_threshold=z_score_threshold,
            )
        )

    outliers.sort(key=lambda outlier: outlier.z_score, reverse=True)

    return [outlier.to_dict() for outlier in outliers]


def build_phoneme_analysis(
    phoneme: str,
    rows: list[dict[str, Any]],
    z_score_threshold: float,
) -> dict[str, Any]:
    words = sorted({str(row.get("word", "")) for row in rows if row.get("word")})

    return {
        "phoneme": phoneme,
        "word_count": len(words),
        "words": words,
        "metrics": build_metric_summaries(rows),
        "dominant_features": get_dominant_features(rows),
        "outliers": find_outliers(
            rows=rows,
            phoneme=phoneme,
            z_score_threshold=z_score_threshold,
        ),
    }


def build_error_summary(error_rows: list[dict[str, Any]]) -> dict[str, Any]:
    error_items = []

    for row in error_rows:
        error_items.append(
            {
                "word": row.get("word"),
                "phoneme": row.get("phoneme"),
                "error_message": row.get("error_message"),
                "en_audio_exists": row.get("en_audio_exists"),
                "ko_audio_exists": row.get("ko_audio_exists"),
                "ko_match_strategy": row.get("ko_match_strategy"),
            }
        )

    return {
        "error_count": len(error_rows),
        "items": error_items,
    }


def build_report(
    payload: dict[str, Any],
    input_path: Path,
    z_score_threshold: float,
) -> dict[str, Any]:
    rows = payload["results"]
    success_rows = get_success_rows(rows)
    error_rows = get_error_rows(rows)
    grouped_rows = group_rows_by_phoneme(success_rows)

    phoneme_reports = {
        phoneme: build_phoneme_analysis(
            phoneme=phoneme,
            rows=phoneme_rows,
            z_score_threshold=z_score_threshold,
        )
        for phoneme, phoneme_rows in sorted(grouped_rows.items())
    }

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_path": str(input_path),
            "source_metadata": payload.get("metadata", {}),
            "total_rows": len(rows),
            "success_rows": len(success_rows),
            "error_rows": len(error_rows),
            "phoneme_count": len(phoneme_reports),
            "z_score_threshold": z_score_threshold,
        },
        "phonemes": phoneme_reports,
        "errors": build_error_summary(error_rows),
    }


def format_number(value: Any) -> str:
    if value is None:
        return "-"

    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")

    return str(value)


def format_metric_line(metrics: dict[str, Any], metric_name: str, label: str) -> str:
    metric = metrics.get(metric_name, {})
    average = format_number(metric.get("average"))
    stdev = format_number(metric.get("stdev"))

    return f"- {label}: 평균 {average}, 표준편차 {stdev}"


def build_markdown_report(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "# Comparison Analysis Report",
        "",
        "## 전체 요약",
        "",
        f"- 생성 시각: {metadata['generated_at']}",
        f"- 원본 파일: `{metadata['source_path']}`",
        f"- 전체 row 수: {metadata['total_rows']}",
        f"- 성공 row 수: {metadata['success_rows']}",
        f"- 실패 row 수: {metadata['error_rows']}",
        f"- 분석된 음소 수: {metadata['phoneme_count']}",
        "",
        "## 음소별 분석",
        "",
    ]

    for phoneme, phoneme_report in report["phonemes"].items():
        lines.extend(build_phoneme_markdown_section(phoneme, phoneme_report))

    lines.extend(build_error_markdown_section(report["errors"]))

    return "\n".join(lines) + "\n"


def build_phoneme_markdown_section(
    phoneme: str,
    phoneme_report: dict[str, Any],
) -> list[str]:
    metrics = phoneme_report["metrics"]
    dominant_features = ", ".join(phoneme_report["dominant_features"]) or "-"

    lines = [
        f"### /{phoneme}/",
        "",
        f"- 단어 수: {phoneme_report['word_count']}",
        f"- 주요 차이 feature 후보: {dominant_features}",
        "",
        "#### 주요 지표",
        "",
        format_metric_line(metrics, "mfcc_distance", "MFCC distance"),
        format_metric_line(metrics, "mfcc_cosine_distance", "MFCC cosine distance"),
        format_metric_line(metrics, "duration_ko_en_ratio", "Duration ko/en ratio"),
        format_metric_line(metrics, "zcr_ko_en_ratio", "ZCR ko/en ratio"),
        format_metric_line(metrics, "rms_ko_en_ratio", "RMS ko/en ratio"),
        format_metric_line(
            metrics,
            "spectral_centroid_ko_en_ratio",
            "Spectral centroid ko/en ratio",
        ),
        "",
    ]

    outliers = phoneme_report["outliers"]
    if not outliers:
        lines.extend(["#### Outlier 후보", "", "- 없음", ""])
        return lines

    lines.extend(["#### Outlier 후보", ""])

    for outlier in outliers[:10]:
        lines.append(
            "- "
            f"{outlier['word']} | "
            f"{outlier['metric_name']}={format_number(outlier['value'])}, "
            f"z={format_number(outlier['z_score'])}"
        )

    lines.append("")
    return lines


def build_error_markdown_section(error_summary: dict[str, Any]) -> list[str]:
    lines = [
        "## 실패 항목",
        "",
        f"- 실패 개수: {error_summary['error_count']}",
        "",
    ]

    if error_summary["error_count"] == 0:
        lines.append("- 없음")
        lines.append("")
        return lines

    for item in error_summary["items"]:
        lines.append(
            "- "
            f"{item.get('word')} /{item.get('phoneme')}/: "
            f"{item.get('error_message')}"
        )

    lines.append("")
    return lines


def write_json_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def write_markdown_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = build_markdown_report(report)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(markdown)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze comparison_results.json and generate developer reports."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to data/comparison_results.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT_PATH,
        help="Path to output JSON analysis report",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT_PATH,
        help="Path to output Markdown analysis report",
    )
    parser.add_argument(
        "--z-score-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for outlier detection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = load_comparison_results(args.input)
    report = build_report(
        payload=payload,
        input_path=args.input,
        z_score_threshold=args.z_score_threshold,
    )

    write_json_report(args.json_output, report)
    write_markdown_report(args.md_output, report)

    metadata = report["metadata"]
    print(f"Source: {args.input}")
    print(f"JSON report: {args.json_output}")
    print(f"Markdown report: {args.md_output}")
    print(f"Success rows: {metadata['success_rows']}")
    print(f"Error rows: {metadata['error_rows']}")
    print(f"Phonemes: {metadata['phoneme_count']}")


if __name__ == "__main__":
    main()