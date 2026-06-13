"""로컬 전용 발음 분석 뷰어.

배포용 main.py와 완전히 분리된 독립 서버다.
실행: uv run uvicorn analysis_viewer:app --host 127.0.0.1 --port 9000
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

REFERENCE_VECTORS_PATH  = DATA_DIR / "reference_vectors.json"
COMPARISON_RESULTS_PATH = DATA_DIR / "comparison_results.json"
ANALYSIS_REPORT_PATH    = DATA_DIR / "comparison_analysis_report.json"
DB_PATH                 = DATA_DIR / "pronunciation.db"

USER_RECORDINGS_TABLE   = "user_recordings"
USER_RECORDINGS_COLUMNS = [
    "id", "created_at", "word", "phoneme", "score", "grade",
    "feedback", "recording_path", "test_label",
    "duration_ms", "rms_mean", "zcr_mean", "spectral_centroid_mean", "mfcc_distance",
]
USER_RECORDINGS_LIMIT   = 200

app = FastAPI(title="Pronunciation Analysis Viewer", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")


# ── HTML 페이지 ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def analysis_viewer_page(request: Request) -> HTMLResponse:
    """분석 뷰어 메인 페이지를 반환한다."""
    file_status = _build_file_status()
    return templates.TemplateResponse(request, "analysis_viewer.html", {"file_status": file_status})


# ── JSON API ─────────────────────────────────────────────────────────────────

@app.get("/api/overview")
async def get_overview() -> Response:
    """전체 요약 JSON을 반환한다."""
    reference_vectors  = load_reference_vectors(REFERENCE_VECTORS_PATH)
    comparison_results = load_comparison_results(COMPARISON_RESULTS_PATH)
    analysis_report    = load_analysis_report(ANALYSIS_REPORT_PATH)

    summary = build_overview_summary(reference_vectors, comparison_results, analysis_report)
    return _json_response(summary)


@app.get("/api/reference-quality")
async def get_reference_quality() -> Response:
    """reference 음소별 품질 JSON을 반환한다."""
    reference_vectors = load_reference_vectors(REFERENCE_VECTORS_PATH)
    rows = build_reference_quality_rows(reference_vectors)
    return _json_response(rows)


@app.get("/api/phoneme-analysis")
async def get_phoneme_analysis() -> Response:
    """음소별 비교 분석 JSON을 반환한다."""
    comparison_results = load_comparison_results(COMPARISON_RESULTS_PATH)
    analysis_report    = load_analysis_report(ANALYSIS_REPORT_PATH)
    rows = build_phoneme_analysis_rows(comparison_results, analysis_report)
    return _json_response(rows)


@app.get("/api/word-results")
async def get_word_results() -> Response:
    """단어별 comparison 성공 결과 JSON을 반환한다."""
    comparison_results = load_comparison_results(COMPARISON_RESULTS_PATH)
    rows = build_word_result_rows(comparison_results)
    return _json_response(rows)


@app.get("/api/outliers")
async def get_outliers() -> Response:
    """outlier 후보 JSON을 반환한다."""
    analysis_report = load_analysis_report(ANALYSIS_REPORT_PATH)
    rows = build_outlier_rows(analysis_report)
    return _json_response(rows)


@app.get("/api/errors")
async def get_errors() -> Response:
    """comparison 실패 row JSON을 반환한다."""
    comparison_results = load_comparison_results(COMPARISON_RESULTS_PATH)
    rows = build_error_rows(comparison_results)
    return _json_response(rows)


@app.get("/api/user-results")
async def get_user_results(latest_only: bool = True) -> Response:
    """user_recordings 테이블의 최근 결과를 JSON으로 반환한다."""
    payload = load_user_results_from_db(DB_PATH, limit=USER_RECORDINGS_LIMIT, latest_only=latest_only)
    return _json_response(payload)


# ── 파일 로딩 ────────────────────────────────────────────────────────────────

def load_json_file(path: Path) -> dict[str, Any]:
    """JSON 파일을 읽어 dict로 반환한다. 파일이 없거나 파싱 실패 시 빈 dict."""
    if not path.exists():
        log.warning("파일 없음: %s", path)
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log.error("JSON 파싱 실패: path=%s, error=%s", path, e)
        return {}


def load_reference_vectors(path: Path) -> dict[str, Any]:
    """reference_vectors.json을 읽어 반환한다."""
    return load_json_file(path)


def load_comparison_results(path: Path) -> dict[str, Any]:
    """comparison_results.json을 읽어 반환한다."""
    return load_json_file(path)


def load_analysis_report(path: Path) -> dict[str, Any] | None:
    """comparison_analysis_report.json을 읽어 반환한다. 파일이 없으면 None."""
    if not path.exists():
        return None
    data = load_json_file(path)
    return data if data else None


def load_user_results_from_db(
    db_path: Path,
    limit: int = USER_RECORDINGS_LIMIT,
    latest_only: bool = True,
) -> dict[str, Any]:
    """user_recordings 테이블에서 결과를 조회해 반환한다.

    Args:
        latest_only: True면 word+test_label 조합당 MAX(id) row만 집계한다.
    """
    result: dict[str, Any] = {
        "exists": db_path.exists(),
        "path": str(db_path),
        "table": USER_RECORDINGS_TABLE,
        "row_count": 0,
        "columns": [],
        "rows": [],
        "label_summary": [],
        "phoneme_label_summary": [],
        "phoneme_label_breakdown": [],
        "label_feature_summary": [],
        "score_by_label": {},
        "latest_only": latest_only,
        "error": None,
    }

    if not db_path.exists():
        return result

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if USER_RECORDINGS_TABLE not in tables:
            result["error"] = f"{USER_RECORDINGS_TABLE} 테이블이 없습니다."
            return result

        if latest_only:
            _cte = (
                f"WITH latest AS ("
                f" SELECT * FROM {USER_RECORDINGS_TABLE} u"
                f" WHERE id = ("
                f"  SELECT MAX(id) FROM {USER_RECORDINGS_TABLE}"
                f"  WHERE word = u.word AND test_label = u.test_label"
                f" )"
                f") "
            )
            _src = "latest"
        else:
            _cte = ""
            _src = USER_RECORDINGS_TABLE

        col_list = ", ".join(USER_RECORDINGS_COLUMNS)
        rows = [
            dict(row)
            for row in conn.execute(
                f"{_cte}SELECT {col_list} FROM {_src} ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        ]

        total = conn.execute(
            f"SELECT COUNT(*) FROM {USER_RECORDINGS_TABLE}"
        ).fetchone()[0]

        label_summary = [
            dict(row)
            for row in conn.execute(
                f"{_cte}SELECT test_label, COUNT(*) AS count,"
                f" ROUND(AVG(score), 1) AS avg_score"
                f" FROM {_src} GROUP BY test_label ORDER BY test_label"
            )
        ]

        phoneme_label_summary = [
            dict(row)
            for row in conn.execute(
                f"{_cte}SELECT phoneme, COUNT(*) AS count,"
                f" ROUND(AVG(score), 1) AS avg_score"
                f" FROM {_src} GROUP BY phoneme ORDER BY phoneme"
            )
        ]

        phoneme_label_breakdown = [
            dict(row)
            for row in conn.execute(
                f"{_cte}SELECT phoneme, test_label, COUNT(*) AS count,"
                f" ROUND(AVG(score), 1) AS avg_score"
                f" FROM {_src} GROUP BY phoneme, test_label ORDER BY phoneme, test_label"
            )
        ]

        label_feature_summary = [
            dict(row)
            for row in conn.execute(
                f"{_cte}SELECT test_label,"
                f" ROUND(AVG(duration_ms), 1) AS avg_duration_ms,"
                f" ROUND(AVG(rms_mean), 6) AS avg_rms_mean,"
                f" ROUND(AVG(zcr_mean), 6) AS avg_zcr_mean,"
                f" ROUND(AVG(mfcc_distance), 3) AS avg_mfcc_distance"
                f" FROM {_src} WHERE duration_ms IS NOT NULL"
                f" GROUP BY test_label ORDER BY test_label"
            )
        ]

        score_by_label: dict[str, list[float]] = {}
        for srow in conn.execute(
            f"{_cte}SELECT test_label, score FROM {_src} WHERE score IS NOT NULL"
        ):
            key = srow["test_label"] or "NULL"
            score_by_label.setdefault(key, []).append(float(srow["score"]))

        result["row_count"] = total
        result["columns"] = USER_RECORDINGS_COLUMNS
        result["rows"] = rows
        result["label_summary"] = label_summary
        result["phoneme_label_summary"] = phoneme_label_summary
        result["phoneme_label_breakdown"] = phoneme_label_breakdown
        result["label_feature_summary"] = label_feature_summary
        result["score_by_label"] = score_by_label
        return result

    except sqlite3.Error as e:
        log.error("DB 읽기 실패: path=%s, error=%s", db_path, e)
        result["error"] = str(e)
        return result

    finally:
        if conn:
            conn.close()


# ── 데이터 빌더 ──────────────────────────────────────────────────────────────

def is_success_row(row: dict[str, Any]) -> bool:
    """comparison result row가 성공 상태인지 판별한다."""
    return row.get("status", "ok") in {"ok", "success"}


def build_overview_summary(
    reference_vectors: dict[str, Any],
    comparison_results: dict[str, Any],
    analysis_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """전체 요약 dict를 빌드한다."""
    results = comparison_results.get("results", [])
    successful_results = [r for r in results if is_success_row(r)]
    error_results      = [r for r in results if r.get("status") == "error"]

    phoneme_count_ref    = len(reference_vectors)
    total_sample_count   = sum(v.get("sample_count", 0) for v in reference_vectors.values())
    total_word_count_ref = sum(len(v.get("test_words", [])) for v in reference_vectors.values())
    analyzed_phonemes    = len({r.get("phoneme") for r in successful_results if r.get("phoneme")})
    outlier_count        = _count_outliers_from_report(analysis_report)

    return {
        "reference_phoneme_count":  phoneme_count_ref,
        "reference_sample_count":   total_sample_count,
        "reference_word_count":     total_word_count_ref,
        "comparison_total":         len(results),
        "comparison_success":       len(successful_results),
        "comparison_error":         len(error_results),
        "analyzed_phoneme_count":   analyzed_phonemes,
        "outlier_count":            outlier_count,
        "has_analysis_report":      analysis_report is not None,
        "metadata":                 comparison_results.get("metadata", {}),
    }


def build_reference_quality_rows(reference_vectors: dict[str, Any]) -> list[dict[str, Any]]:
    """reference 음소별 품질 행 목록을 빌드한다."""
    rows = []
    for phoneme, vector in reference_vectors.items():
        word_count      = len(vector.get("test_words", []))
        sample_count    = vector.get("sample_count", 0)
        duration_ms     = vector.get("duration_ms", 0.0)
        duration_std    = vector.get("duration_std", 0.0)
        rms_mean        = vector.get("rms_mean", 0.0)

        warnings = _compute_quality_warnings(sample_count, duration_ms, duration_std, rms_mean, word_count)

        rows.append({
            "phoneme":                   phoneme,
            "phoneme_type":              vector.get("phoneme_type", ""),
            "sample_count":              sample_count,
            "word_count":                word_count,
            "duration_ms":               round(duration_ms, 2),
            "duration_std":              round(duration_std, 2),
            "zcr_mean":                  round(vector.get("zcr_mean", 0.0), 6),
            "zcr_std":                   round(vector.get("zcr_std", 0.0), 6),
            "rms_mean":                  round(rms_mean, 6),
            "rms_std":                   round(vector.get("rms_std", 0.0), 6),
            "spectral_centroid_mean":    round(vector.get("spectral_centroid_mean", 0.0), 2),
            "spectral_centroid_std":     round(vector.get("spectral_centroid_std", 0.0), 2),
            "quality_warning":           ", ".join(warnings) if warnings else "",
            "test_words":                vector.get("test_words", []),
            "korean_pronunciations":     vector.get("korean_pronunciations", {}),
        })
    return sorted(rows, key=lambda r: r["phoneme"])


def build_phoneme_analysis_rows(
    comparison_results: dict[str, Any],
    analysis_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """음소별 비교 분석 행 목록을 빌드한다."""
    successful_results = [
        r for r in comparison_results.get("results", [])
        if is_success_row(r)
    ]

    if not successful_results:
        return []

    phoneme_groups: dict[str, list[dict[str, Any]]] = {}
    for row in successful_results:
        phoneme = row.get("phoneme", "")
        phoneme_groups.setdefault(phoneme, []).append(row)

    report_by_phoneme = _index_report_by_phoneme(analysis_report)

    rows = []
    for phoneme, group in sorted(phoneme_groups.items()):
        row = _build_single_phoneme_analysis(phoneme, group, report_by_phoneme.get(phoneme))
        rows.append(row)

    return rows


def build_word_result_rows(comparison_results: dict[str, Any]) -> list[dict[str, Any]]:
    """단어별 성공 comparison 행 목록을 빌드한다."""
    results = comparison_results.get("results", [])
    successful = [r for r in results if is_success_row(r)]

    return [
        {
            "word":                           r.get("word", ""),
            "korean_pronunciation":           r.get("korean_pronunciation", ""),
            "phoneme":                        r.get("phoneme", ""),
            "en_duration_ms":                 _round_or_none(r.get("en_duration_ms")),
            "ko_duration_ms":                 _round_or_none(r.get("ko_duration_ms")),
            "duration_ko_en_ratio":           _round_or_none(r.get("duration_ko_en_ratio")),
            "en_zcr_mean":                    _round_or_none(r.get("en_zcr_mean"), ndigits=6),
            "ko_zcr_mean":                    _round_or_none(r.get("ko_zcr_mean"), ndigits=6),
            "zcr_ko_en_ratio":                _round_or_none(r.get("zcr_ko_en_ratio")),
            "en_rms_mean":                    _round_or_none(r.get("en_rms_mean"), ndigits=6),
            "ko_rms_mean":                    _round_or_none(r.get("ko_rms_mean"), ndigits=6),
            "rms_ko_en_ratio":                _round_or_none(r.get("rms_ko_en_ratio")),
            "en_spectral_centroid_mean":      _round_or_none(r.get("en_spectral_centroid_mean")),
            "ko_spectral_centroid_mean":      _round_or_none(r.get("ko_spectral_centroid_mean")),
            "spectral_centroid_ko_en_ratio":  _round_or_none(r.get("spectral_centroid_ko_en_ratio")),
            "mfcc_distance":                  _round_or_none(r.get("mfcc_distance")),
            "mfcc_cosine_distance":           _round_or_none(r.get("mfcc_cosine_distance"), ndigits=4),
            "en_audio_path":                  r.get("en_audio_path", ""),
            "ko_audio_path":                  r.get("ko_audio_path", ""),
        }
        for r in successful
    ]


def build_outlier_rows(analysis_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """outlier 후보 행 목록을 빌드한다. analysis_report가 없으면 빈 리스트."""
    if not analysis_report:
        return []

    phoneme_reports = analysis_report.get("phonemes", {})
    if not isinstance(phoneme_reports, dict):
        return []

    rows = []
    for phoneme, report in phoneme_reports.items():
        if not isinstance(report, dict):
            continue
        for outlier in report.get("outliers", []):
            rows.append({
                "word":        outlier.get("word", ""),
                "phoneme":     outlier.get("phoneme", phoneme),
                "metric_name": outlier.get("metric_name", ""),
                "value":       _round_or_none(outlier.get("value")),
                "average":     _round_or_none(outlier.get("average")),
                "stdev":       _round_or_none(outlier.get("stdev")),
                "z_score":     _round_or_none(outlier.get("z_score"), ndigits=3),
            })

    return sorted(rows, key=lambda r: abs(r.get("z_score") or 0), reverse=True)


def build_error_rows(comparison_results: dict[str, Any]) -> list[dict[str, Any]]:
    """comparison 실패 행 목록을 빌드한다."""
    results = comparison_results.get("results", [])
    return [
        {
            "word":               r.get("word", ""),
            "korean_pronunciation": r.get("korean_pronunciation", ""),
            "phoneme":            r.get("phoneme", ""),
            "error_message":      r.get("error_message", ""),
            "en_audio_exists":    r.get("en_audio_exists", False),
            "ko_audio_exists":    r.get("ko_audio_exists", False),
        }
        for r in results
        if r.get("status") == "error"
    ]


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _compute_quality_warnings(
    sample_count: int,
    duration_ms: float,
    duration_std: float,
    rms_mean: float,
    word_count: int,
) -> list[str]:
    warnings: list[str] = []
    if sample_count < 10:
        warnings.append("sample_count 부족")
    if duration_ms > 0 and duration_std > duration_ms * 0.8:
        warnings.append("duration 편차 큼")
    if rms_mean < 0.02:
        warnings.append("RMS 낮음")
    if word_count == 0:
        warnings.append("test_words 없음")
    return warnings


def _count_outliers_from_report(analysis_report: dict[str, Any] | None) -> int:
    if not analysis_report:
        return 0
    phoneme_reports = analysis_report.get("phonemes", {})
    if not isinstance(phoneme_reports, dict):
        return 0
    return sum(
        len(report.get("outliers", []))
        for report in phoneme_reports.values()
        if isinstance(report, dict)
    )


def _index_report_by_phoneme(
    analysis_report: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not analysis_report:
        return {}

    phoneme_data = analysis_report.get("phonemes")
    if isinstance(phoneme_data, dict):
        return phoneme_data

    legacy_data = analysis_report.get("phoneme_analysis")
    if isinstance(legacy_data, dict):
        return legacy_data

    return {}


def _build_single_phoneme_analysis(
    phoneme: str,
    group: list[dict[str, Any]],
    report_data: dict[str, Any] | None,
) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [r[key] for r in group if r.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    dominant_features = report_data.get("dominant_features", []) if report_data else []
    outlier_count     = report_data.get("outlier_count", 0) if report_data else 0

    return {
        "phoneme":                          phoneme,
        "word_count":                       len(group),
        "dominant_features":                ", ".join(dominant_features) if dominant_features else "-",
        "avg_mfcc_distance":                avg("mfcc_distance"),
        "avg_mfcc_cosine_distance":         avg("mfcc_cosine_distance"),
        "avg_duration_ko_en_ratio":         avg("duration_ko_en_ratio"),
        "avg_zcr_ko_en_ratio":              avg("zcr_ko_en_ratio"),
        "avg_rms_ko_en_ratio":              avg("rms_ko_en_ratio"),
        "avg_spectral_centroid_ko_en_ratio": avg("spectral_centroid_ko_en_ratio"),
        "outlier_count":                    outlier_count,
    }


def _build_file_status() -> dict[str, bool]:
    return {
        "reference_vectors":  REFERENCE_VECTORS_PATH.exists(),
        "comparison_results": COMPARISON_RESULTS_PATH.exists(),
        "analysis_report":    ANALYSIS_REPORT_PATH.exists(),
        "db":                 DB_PATH.exists(),
    }


def _round_or_none(value: Any, ndigits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _json_response(data: Any) -> Response:
    body = json.dumps(data, ensure_ascii=False)
    return Response(content=body, media_type="application/json; charset=utf-8")


if __name__ == "__main__":
    uvicorn.run("analysis_viewer:app", host="127.0.0.1", port=9000, reload=True)
