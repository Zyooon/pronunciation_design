"""로컬 전용 발음 분석 뷰어.

배포용 main.py와 완전히 분리된 독립 서버다.
실행: uv run uvicorn analysis_viewer:app --host 127.0.0.1 --port 9000
"""

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

REFERENCE_VECTORS_PATH = DATA_DIR / "reference_vectors.json"
COMPARISON_RESULTS_PATH = DATA_DIR / "comparison_results.json"
ANALYSIS_REPORT_PATH = DATA_DIR / "comparison_analysis_report.json"
DB_PATH = DATA_DIR / "pronunciation.db"

USER_RECORDINGS_TABLE = "user_recordings"
USER_RECORDINGS_LIMIT = 200
LABEL_REVIEW_PAGE_SIZE = 50
ALLOWED_REVIEW_LABELS = {"good", "korean_like", "wrong_or_noisy", "unlabeled", "exclude", ""}

_TABLE_DISPLAY_COLS = [
    "id", "created_at", "word", "phoneme", "test_label",
    "score", "base_score", "final_score", "total_penalty", "recording_path",
]

_ALL_ROW_COLS = [
    "id", "created_at", "word", "phoneme", "test_label",
    "score", "grade", "feedback", "recording_path",
    "base_score", "final_score",
    "mfcc_score", "duration_score", "rms_score", "zcr_score", "spectral_centroid_score",
    "quality_penalty", "pronunciation_penalty", "total_penalty",
    "duration_penalty", "volume_penalty", "noise_penalty",
    "duration_ms", "rms_mean", "zcr_mean", "spectral_centroid_mean", "mfcc_distance",
    "duration_ratio", "details_json",
]

_WORD_COMPARE_METRIC_KEYS = [
    "duration_ratio",
    "mfcc_mean_dist", "mfcc_max_dist", "mfcc_std_dist",
    "zcr_mean_dist", "zcr_max_dist", "zcr_std_dist",
    "rms_mean_dist", "rms_max_dist", "rms_std_dist",
    "ko_mfcc_mean_dist", "ko_zcr_mean_dist", "ko_rms_mean_dist",
]

_PENALTY_COLS = [
    "quality_penalty", "pronunciation_penalty", "total_penalty",
    "duration_penalty", "volume_penalty", "noise_penalty",
]

_FEATURE_SCORE_COLS = [
    "mfcc_score", "duration_score", "rms_score", "zcr_score", "spectral_centroid_score",
]

app = FastAPI(title="Pronunciation Analysis Viewer", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")


@app.get("/", response_class=HTMLResponse)
async def analysis_viewer_page(request: Request) -> HTMLResponse:
    file_status = _build_file_status()
    return templates.TemplateResponse(request, "analysis_viewer.html", {"file_status": file_status})


@app.get("/api/overview")
async def get_overview() -> Response:
    reference_vectors = load_reference_vectors(REFERENCE_VECTORS_PATH)
    comparison_results = load_comparison_results(COMPARISON_RESULTS_PATH)
    analysis_report = load_analysis_report(ANALYSIS_REPORT_PATH)
    summary = build_overview_summary(reference_vectors, comparison_results, analysis_report)
    return _json_response(summary)


@app.get("/api/reference-quality")
async def get_reference_quality() -> Response:
    rows = build_reference_quality_rows(load_reference_vectors(REFERENCE_VECTORS_PATH))
    return _json_response(rows)


@app.get("/api/phoneme-analysis")
async def get_phoneme_analysis() -> Response:
    rows = build_phoneme_analysis_rows(load_comparison_results(COMPARISON_RESULTS_PATH), load_analysis_report(ANALYSIS_REPORT_PATH))
    return _json_response(rows)


@app.get("/api/word-results")
async def get_word_results() -> Response:
    rows = build_word_result_rows(load_comparison_results(COMPARISON_RESULTS_PATH))
    return _json_response(rows)


@app.get("/api/outliers")
async def get_outliers() -> Response:
    rows = build_outlier_rows(load_analysis_report(ANALYSIS_REPORT_PATH))
    return _json_response(rows)


@app.get("/api/errors")
async def get_errors() -> Response:
    rows = build_error_rows(load_comparison_results(COMPARISON_RESULTS_PATH))
    return _json_response(rows)


@app.get("/api/user-results")
async def get_user_results(latest_only: bool = True) -> Response:
    payload = load_user_results_from_db(DB_PATH, limit=USER_RECORDINGS_LIMIT, latest_only=latest_only)
    return _json_response(payload)


@app.get("/api/label-review-results")
async def get_label_review_results(
    label: str = "",
    page: int = 1,
    limit: int = LABEL_REVIEW_PAGE_SIZE,
) -> Response:
    payload = load_label_review_results(DB_PATH, label=label, page=page, limit=limit)
    return _json_response(payload)


@app.get("/api/recording/{recording_id}")
async def get_recording_audio(recording_id: int) -> FileResponse:
    recording_path = get_recording_path_by_id(DB_PATH, recording_id)
    if recording_path is None:
        raise HTTPException(status_code=404, detail="recording row가 없습니다.")
    file_path = resolve_recording_path(recording_path)
    if file_path is None or not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="recording 파일을 찾을 수 없습니다.")
    return FileResponse(file_path, media_type=guess_audio_media_type(file_path), filename=file_path.name)


@app.get("/api/word-compare-radar/{recording_id}")
async def get_word_compare_radar_chart(recording_id: int) -> Response:
    """Word Compare 탭 유저 방사형 차트 HTML을 Plotly로 생성해 반환한다."""
    row = load_recording_scores_by_id(DB_PATH, recording_id)
    if row is None:
        raise HTTPException(status_code=404, detail="녹음 row를 찾을 수 없습니다.")
    merged = _merge_details_json(dict(row))
    radar_html = build_word_compare_radar_html(merged, "사용자 발음")
    return _json_response({"radar_html": radar_html, "recording_id": recording_id})


@app.post("/api/user-label/{recording_id}")
async def update_user_label(recording_id: int, request: Request) -> Response:
    label = await _extract_review_label(request)
    if label not in ALLOWED_REVIEW_LABELS:
        raise HTTPException(status_code=400, detail=f"허용되지 않은 라벨입니다: {label}")
    stored_label = None if label in {"", "unlabeled"} else label
    try:
        updated = update_recording_label(DB_PATH, recording_id, stored_label)
    except sqlite3.Error as exc:
        log.error("라벨 업데이트 실패: id=%s, label=%s", recording_id, stored_label, exc_info=True)
        raise HTTPException(status_code=500, detail=f"DB 업데이트 실패: {exc}") from exc
    if not updated:
        raise HTTPException(status_code=404, detail="업데이트할 recording row가 없습니다.")
    return _json_response({"ok": True, "id": recording_id, "test_label": stored_label})


async def _extract_review_label(request: Request) -> str:
    query_label = request.query_params.get("test_label")
    if query_label is not None:
        return query_label.strip()
    try:
        body = await request.json()
        if isinstance(body, dict):
            return str(body.get("test_label", "")).strip()
    except Exception:
        pass
    try:
        form = await request.form()
        return str(form.get("test_label", "")).strip()
    except Exception:
        return ""


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        log.warning("파일 없음: %s", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        log.error("JSON 파싱 실패: path=%s, error=%s", path, exc)
        return {}


def load_reference_vectors(path: Path) -> dict[str, Any]:
    return load_json_file(path)


def load_comparison_results(path: Path) -> dict[str, Any]:
    return load_json_file(path)


def load_analysis_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = load_json_file(path)
    return data if data else None


def load_user_results_from_db(db_path: Path, limit: int = USER_RECORDINGS_LIMIT, latest_only: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": db_path.exists(),
        "path": str(db_path),
        "table": USER_RECORDINGS_TABLE,
        "row_count": 0,
        "columns": [],
        "rows": [],
        "label_summary": [],
        "phoneme_label_breakdown": [],
        "score_by_label": {},
        "summary_cards": None,
        "penalty_summary": None,
        "feature_score_summary": None,
        "score_comparison": [],
        "available_columns": [],
        "latest_only": latest_only,
        "error": None,
    }
    if not db_path.exists():
        return result
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if USER_RECORDINGS_TABLE not in tables:
            result["error"] = f"{USER_RECORDINGS_TABLE} 테이블이 없습니다."
            return result
        _ensure_user_recordings_label_column(conn)
        available_cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({USER_RECORDINGS_TABLE})")}
        result["available_columns"] = sorted(available_cols)
        if latest_only:
            _cte = (
                f"WITH latest AS ("
                f" SELECT * FROM {USER_RECORDINGS_TABLE} u"
                f" WHERE id = ("
                f"  SELECT MAX(id) FROM {USER_RECORDINGS_TABLE}"
                f"  WHERE word = u.word AND COALESCE(test_label, '') = COALESCE(u.test_label, '')"
                f" )"
                f") "
            )
            _src = "latest"
        else:
            _cte = ""
            _src = USER_RECORDINGS_TABLE
        row_cols = [c for c in _ALL_ROW_COLS if c in available_cols]
        rows = [_merge_details_json(dict(row)) for row in conn.execute(f"{_cte}SELECT {', '.join(row_cols)} FROM {_src} ORDER BY id DESC LIMIT ?", (limit,))]
        display_cols = [c for c in _TABLE_DISPLAY_COLS if c in available_cols]
        total = conn.execute(f"SELECT COUNT(*) FROM {USER_RECORDINGS_TABLE}").fetchone()[0]
        label_summary = [dict(row) for row in conn.execute(f"{_cte}SELECT test_label, COUNT(*) AS count, ROUND(AVG(score), 1) AS avg_score FROM {_src} GROUP BY test_label ORDER BY test_label")]
        phoneme_label_breakdown = [dict(row) for row in conn.execute(f"{_cte}SELECT phoneme, test_label, COUNT(*) AS count, ROUND(AVG(score), 1) AS avg_score FROM {_src} GROUP BY phoneme, test_label ORDER BY phoneme, test_label")]
        score_by_label: dict[str, list[float]] = {}
        for srow in conn.execute(f"{_cte}SELECT test_label, score FROM {_src} WHERE score IS NOT NULL"):
            key = srow["test_label"] or "NULL"
            score_by_label.setdefault(key, []).append(float(srow["score"]))
        avg_by_label = {(s["test_label"] or "NULL"): s["avg_score"] for s in label_summary}
        good_avg = avg_by_label.get("good")
        ko_avg = avg_by_label.get("korean_like")
        wrong_avg = avg_by_label.get("wrong_or_noisy")
        ordered = good_avg is not None and ko_avg is not None and wrong_avg is not None and good_avg > ko_avg > wrong_avg
        summary_cards = {"latest_count": len(rows), "avg_by_label": avg_by_label, "ordered_correctly": ordered}
        penalty_summary = _query_group_avg(conn, _cte, _src, [c for c in _PENALTY_COLS if c in available_cols])
        feature_score_summary = _query_group_avg(conn, _cte, _src, [c for c in _FEATURE_SCORE_COLS if c in available_cols])
        score_cmp_parts = ["ROUND(AVG(score), 1) AS avg_score"]
        if "base_score" in available_cols:
            score_cmp_parts.append("ROUND(AVG(base_score), 1) AS avg_base_score")
        if "final_score" in available_cols:
            score_cmp_parts.append("ROUND(AVG(final_score), 1) AS avg_final_score")
        score_comparison = [dict(row) for row in conn.execute(f"{_cte}SELECT test_label, {', '.join(score_cmp_parts)} FROM {_src} GROUP BY test_label ORDER BY test_label")]
        result.update({
            "row_count": total,
            "columns": display_cols,
            "rows": rows,
            "label_summary": label_summary,
            "phoneme_label_breakdown": phoneme_label_breakdown,
            "score_by_label": score_by_label,
            "summary_cards": summary_cards,
            "penalty_summary": penalty_summary,
            "feature_score_summary": feature_score_summary,
            "score_comparison": score_comparison,
        })
        return result
    except sqlite3.Error as exc:
        log.error("DB 읽기 실패: path=%s, error=%s", db_path, exc)
        result["error"] = str(exc)
        return result
    finally:
        if conn:
            conn.close()


def load_label_review_results(db_path: Path, label: str = "", page: int = 1, limit: int = LABEL_REVIEW_PAGE_SIZE) -> dict[str, Any]:
    page_size = max(1, min(limit, 50))
    current_page = max(1, page)
    offset = (current_page - 1) * page_size
    result = {
        "exists": db_path.exists(),
        "rows": [],
        "row_count": 0,
        "page": current_page,
        "page_size": page_size,
        "total_pages": 0,
        "error": None,
        "label": label,
    }
    if not db_path.exists():
        return result
    label = (label or "").strip()
    where = ""
    params: list[Any] = []
    if label == "unlabeled":
        where = "WHERE test_label IS NULL OR test_label = ''"
    elif label:
        where = "WHERE test_label = ?"
        params.append(label)
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if USER_RECORDINGS_TABLE not in tables:
                result["error"] = f"{USER_RECORDINGS_TABLE} 테이블이 없습니다."
                return result
            _ensure_user_recordings_label_column(conn)
            available_cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({USER_RECORDINGS_TABLE})")}
            count = conn.execute(f"SELECT COUNT(*) FROM {USER_RECORDINGS_TABLE} {where}", params).fetchone()[0]
            requested_cols = [
                "id", "created_at", "word", "phoneme", "test_label", "score", "grade",
                "recording_path", "duration_ms", "rms_mean", "zcr_mean",
                "spectral_centroid_mean", "mfcc_distance", "total_penalty",
                "base_score", "final_score", "duration_ratio", "details_json",
            ]
            select_cols = [c for c in requested_cols if c in available_cols]
            rows = [
                _merge_details_json(dict(row))
                for row in conn.execute(
                    f"""
                    SELECT {', '.join(select_cols)}
                    FROM {USER_RECORDINGS_TABLE}
                    {where}
                    ORDER BY id DESC
                    LIMIT ? OFFSET ?
                    """,
                    [*params, page_size, offset],
                )
            ]
            total_pages = (count + page_size - 1) // page_size if count else 0
            result.update({"rows": rows, "row_count": count, "total_pages": total_pages})
            return result
    except sqlite3.Error as exc:
        result["error"] = str(exc)
        return result


def _merge_details_json(row: dict[str, Any]) -> dict[str, Any]:
    raw_details = row.pop("details_json", None)
    if not raw_details:
        return row
    try:
        details = json.loads(raw_details)
    except (TypeError, json.JSONDecodeError):
        return row
    if not isinstance(details, dict):
        return row
    row["details"] = details
    for key in _WORD_COMPARE_METRIC_KEYS:
        if key not in row or row.get(key) is None:
            row[key] = details.get(key)
    for key in ("base_score", "final_score", "mfcc_score", "duration_score", "rms_score", "zcr_score", "spectral_centroid_score", "quality_penalty", "pronunciation_penalty", "total_penalty"):
        if key not in row or row.get(key) is None:
            row[key] = details.get(key)
    return row


def _ensure_user_recordings_label_column(conn: sqlite3.Connection) -> None:
    existing_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({USER_RECORDINGS_TABLE})")}
    if "test_label" not in existing_cols:
        conn.execute(f"ALTER TABLE {USER_RECORDINGS_TABLE} ADD COLUMN test_label TEXT")
        conn.commit()


def get_recording_path_by_id(db_path: Path, recording_id: int) -> str | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(f"SELECT recording_path FROM {USER_RECORDINGS_TABLE} WHERE id = ?", (recording_id,)).fetchone()
    return row[0] if row and row[0] else None


def update_recording_label(db_path: Path, recording_id: int, label: str | None) -> bool:
    if not db_path.exists():
        return False
    with sqlite3.connect(str(db_path)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if USER_RECORDINGS_TABLE not in tables:
            return False
        _ensure_user_recordings_label_column(conn)
        cur = conn.execute(f"UPDATE {USER_RECORDINGS_TABLE} SET test_label = ? WHERE id = ?", (label, recording_id))
        conn.commit()
        return cur.rowcount > 0


def resolve_recording_path(recording_path: str) -> Path | None:
    raw_path = Path(recording_path)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend([
            PROJECT_ROOT / raw_path,
            DATA_DIR / raw_path,
            PROJECT_ROOT / "recordings" / raw_path,
            DATA_DIR / "recordings" / raw_path,
        ])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def guess_audio_media_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".webm":
        return "audio/webm"
    if suffix == ".ogg":
        return "audio/ogg"
    if suffix in {".m4a", ".mp4"}:
        return "audio/mp4"
    return "audio/wav"


def _query_group_avg(conn: sqlite3.Connection, cte: str, src: str, cols: list[str]) -> list[dict[str, Any]] | None:
    if not cols:
        return None
    has_data_expr = " OR ".join(f"{c} IS NOT NULL" for c in cols)
    count = conn.execute(f"{cte}SELECT COUNT(*) FROM {src} WHERE {has_data_expr}").fetchone()[0]
    if count == 0:
        return None
    avg_exprs = ", ".join(f"ROUND(AVG({c}), 2) AS avg_{c}" for c in cols)
    return [dict(row) for row in conn.execute(f"{cte}SELECT test_label, {avg_exprs} FROM {src} GROUP BY test_label ORDER BY test_label")]


def is_success_row(row: dict[str, Any]) -> bool:
    return row.get("status", "ok") in {"ok", "success"}


def build_overview_summary(reference_vectors: dict[str, Any], comparison_results: dict[str, Any], analysis_report: dict[str, Any] | None) -> dict[str, Any]:
    results = comparison_results.get("results", [])
    successful_results = [r for r in results if is_success_row(r)]
    error_results = [r for r in results if r.get("status") == "error"]
    return {
        "reference_phoneme_count": len(reference_vectors),
        "reference_sample_count": sum(v.get("sample_count", 0) for v in reference_vectors.values()),
        "reference_word_count": sum(len(v.get("test_words", [])) for v in reference_vectors.values()),
        "comparison_total": len(results),
        "comparison_success": len(successful_results),
        "comparison_error": len(error_results),
        "analyzed_phoneme_count": len({r.get("phoneme") for r in successful_results if r.get("phoneme")}),
        "outlier_count": _count_outliers_from_report(analysis_report),
        "has_analysis_report": analysis_report is not None,
        "metadata": comparison_results.get("metadata", {}),
    }


def build_reference_quality_rows(reference_vectors: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for phoneme, vector in reference_vectors.items():
        word_count = len(vector.get("test_words", []))
        sample_count = vector.get("sample_count", 0)
        duration_ms = vector.get("duration_ms", 0.0)
        duration_std = vector.get("duration_std", 0.0)
        rms_mean = vector.get("rms_mean", 0.0)
        warnings = _compute_quality_warnings(sample_count, duration_ms, duration_std, rms_mean, word_count)
        rows.append({
            "phoneme": phoneme,
            "phoneme_type": vector.get("phoneme_type", ""),
            "sample_count": sample_count,
            "word_count": word_count,
            "duration_ms": round(duration_ms, 2),
            "duration_std": round(duration_std, 2),
            "zcr_mean": round(vector.get("zcr_mean", 0.0), 6),
            "zcr_std": round(vector.get("zcr_std", 0.0), 6),
            "rms_mean": round(rms_mean, 6),
            "rms_std": round(vector.get("rms_std", 0.0), 6),
            "spectral_centroid_mean": round(vector.get("spectral_centroid_mean", 0.0), 2),
            "spectral_centroid_std": round(vector.get("spectral_centroid_std", 0.0), 2),
            "quality_warning": ", ".join(warnings) if warnings else "",
            "test_words": vector.get("test_words", []),
            "korean_pronunciations": vector.get("korean_pronunciations", {}),
        })
    return sorted(rows, key=lambda r: r["phoneme"])


def build_phoneme_analysis_rows(comparison_results: dict[str, Any], analysis_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    successful_results = [r for r in comparison_results.get("results", []) if is_success_row(r)]
    if not successful_results:
        return []
    phoneme_groups: dict[str, list[dict[str, Any]]] = {}
    for row in successful_results:
        phoneme_groups.setdefault(row.get("phoneme", ""), []).append(row)
    report_by_phoneme = _index_report_by_phoneme(analysis_report)
    return [_build_single_phoneme_analysis(phoneme, group, report_by_phoneme.get(phoneme)) for phoneme, group in sorted(phoneme_groups.items())]


def build_word_result_rows(comparison_results: dict[str, Any]) -> list[dict[str, Any]]:
    successful = [r for r in comparison_results.get("results", []) if is_success_row(r)]
    return [
        {
            "word": r.get("word", ""),
            "korean_pronunciation": r.get("korean_pronunciation", ""),
            "phoneme": r.get("phoneme", ""),
            "en_duration_ms": _round_or_none(r.get("en_duration_ms")),
            "ko_duration_ms": _round_or_none(r.get("ko_duration_ms")),
            "duration_ko_en_ratio": _round_or_none(r.get("duration_ko_en_ratio")),
            "en_zcr_mean": _round_or_none(r.get("en_zcr_mean"), ndigits=6),
            "ko_zcr_mean": _round_or_none(r.get("ko_zcr_mean"), ndigits=6),
            "zcr_ko_en_ratio": _round_or_none(r.get("zcr_ko_en_ratio")),
            "en_rms_mean": _round_or_none(r.get("en_rms_mean"), ndigits=6),
            "ko_rms_mean": _round_or_none(r.get("ko_rms_mean"), ndigits=6),
            "rms_ko_en_ratio": _round_or_none(r.get("rms_ko_en_ratio")),
            "en_spectral_centroid_mean": _round_or_none(r.get("en_spectral_centroid_mean")),
            "ko_spectral_centroid_mean": _round_or_none(r.get("ko_spectral_centroid_mean")),
            "spectral_centroid_ko_en_ratio": _round_or_none(r.get("spectral_centroid_ko_en_ratio")),
            "mfcc_distance": _round_or_none(r.get("mfcc_distance")),
            "mfcc_cosine_distance": _round_or_none(r.get("mfcc_cosine_distance"), ndigits=4),
            "en_audio_path": r.get("en_audio_path", ""),
            "ko_audio_path": r.get("ko_audio_path", ""),
        }
        for r in successful
    ]


def build_outlier_rows(analysis_report: dict[str, Any] | None) -> list[dict[str, Any]]:
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
                "word": outlier.get("word", ""),
                "phoneme": outlier.get("phoneme", phoneme),
                "metric_name": outlier.get("metric_name", ""),
                "value": _round_or_none(outlier.get("value")),
                "average": _round_or_none(outlier.get("average")),
                "stdev": _round_or_none(outlier.get("stdev")),
                "z_score": _round_or_none(outlier.get("z_score"), ndigits=3),
            })
    return sorted(rows, key=lambda r: abs(r.get("z_score") or 0), reverse=True)


def build_error_rows(comparison_results: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "word": r.get("word", ""),
            "korean_pronunciation": r.get("korean_pronunciation", ""),
            "phoneme": r.get("phoneme", ""),
            "error_message": r.get("error_message", ""),
            "en_audio_exists": r.get("en_audio_exists", False),
            "ko_audio_exists": r.get("ko_audio_exists", False),
        }
        for r in comparison_results.get("results", [])
        if r.get("status") == "error"
    ]


def _compute_quality_warnings(sample_count: int, duration_ms: float, duration_std: float, rms_mean: float, word_count: int) -> list[str]:
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
    return sum(len(report.get("outliers", [])) for report in phoneme_reports.values() if isinstance(report, dict))


def _index_report_by_phoneme(analysis_report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not analysis_report:
        return {}
    phoneme_data = analysis_report.get("phonemes")
    if isinstance(phoneme_data, dict):
        return phoneme_data
    legacy_data = analysis_report.get("phoneme_analysis")
    if isinstance(legacy_data, dict):
        return legacy_data
    return {}


def _build_single_phoneme_analysis(phoneme: str, group: list[dict[str, Any]], report_data: dict[str, Any] | None) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        values = [r[key] for r in group if r.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None
    dominant_features = report_data.get("dominant_features", []) if report_data else []
    outlier_count = report_data.get("outlier_count", 0) if report_data else 0
    return {
        "phoneme": phoneme,
        "word_count": len(group),
        "dominant_features": ", ".join(dominant_features) if dominant_features else "-",
        "avg_mfcc_distance": avg("mfcc_distance"),
        "avg_mfcc_cosine_distance": avg("mfcc_cosine_distance"),
        "avg_duration_ko_en_ratio": avg("duration_ko_en_ratio"),
        "avg_zcr_ko_en_ratio": avg("zcr_ko_en_ratio"),
        "avg_rms_ko_en_ratio": avg("rms_ko_en_ratio"),
        "avg_spectral_centroid_ko_en_ratio": avg("spectral_centroid_ko_en_ratio"),
        "outlier_count": outlier_count,
    }


def _build_file_status() -> dict[str, bool]:
    return {
        "reference_vectors": REFERENCE_VECTORS_PATH.exists(),
        "comparison_results": COMPARISON_RESULTS_PATH.exists(),
        "analysis_report": ANALYSIS_REPORT_PATH.exists(),
        "db": DB_PATH.exists(),
    }


def _round_or_none(value: Any, ndigits: int = 2) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


_RADAR_SCORE_SPECS: tuple[tuple[str, str], ...] = (
    ("mfcc_score",             "음색 (MFCC)"),
    ("duration_score",          "박자 (Duration)"),
    ("rms_score",               "음량/강세 (RMS)"),
    ("zcr_score",               "자음 명확도 (ZCR)"),
    ("spectral_centroid_score", "맑기 (Spectral)"),
)


def _is_invalid_score(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _build_radar_scores_and_labels(
    scores: dict[str, Any],
) -> tuple[list[float], list[str]]:
    """5대 지표를 순회하며 None/NaN 점수를 100.0 더미로 치환하고 라벨에 '(제외)'를 붙인다."""
    values: list[float] = []
    labels: list[str] = []
    for key, base_label in _RADAR_SCORE_SPECS:
        raw = scores.get(key)
        if _is_invalid_score(raw):
            values.append(100.0)
            short_name = base_label.split(" (")[0]
            labels.append(f"{short_name} (제외)")
        else:
            values.append(float(raw))
            labels.append(base_label)
    return values, labels


def build_word_compare_radar_html(scores: dict[str, Any], title: str = "") -> str:
    """Word Compare 탭 유저 방사형 차트 HTML을 Plotly로 생성한다.

    None/NaN 점수는 100.0(Pass 더미)으로 치환하고 라벨에 '(제외)'를 붙인다.
    5개 지표 모두 유효하지 않으면 빈 문자열을 반환한다.
    """
    has_any_valid = any(not _is_invalid_score(scores.get(key)) for key, _ in _RADAR_SCORE_SPECS)
    if not has_any_valid:
        return ""
    values, labels = _build_radar_scores_and_labels(scores)
    theta = labels + [labels[0]]
    r = values + [values[0]]
    fig = go.Figure(go.Scatterpolar(
        r=r,
        theta=theta,
        fill="toself",
        fillcolor="rgba(59,130,246,0.18)",
        line={"color": "#3b82f6", "width": 2.5},
        name=title,
    ))
    fig.update_layout(
        polar={
            "radialaxis": {
                "range": [0, 100],
                "tickvals": [20, 40, 60, 80, 100],
                "tickfont": {"size": 10, "color": "#94a3b8"},
                "gridcolor": "rgba(148,163,184,0.3)",
            },
            "angularaxis": {"gridcolor": "rgba(148,163,184,0.4)"},
            "bgcolor": "rgba(0,0,0,0)",
        },
        showlegend=bool(title),
        margin={"l": 60, "r": 60, "t": 40, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        title={"text": title, "font": {"size": 13, "color": "#16324f"}} if title else None,
    )
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def load_recording_scores_by_id(db_path: Path, recording_id: int) -> dict[str, Any] | None:
    """recording_id로 score 컬럼과 details_json을 조회한다."""
    if not db_path.exists():
        return None
    score_cols = [
        "id", "word", "phoneme",
        "mfcc_score", "duration_score", "rms_score", "zcr_score",
        "spectral_centroid_score", "details_json",
    ]
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            available = {
                r["name"]
                for r in conn.execute(f"PRAGMA table_info({USER_RECORDINGS_TABLE})")
            }
            select_cols = [c for c in score_cols if c in available]
            row = conn.execute(
                f"SELECT {', '.join(select_cols)} FROM {USER_RECORDINGS_TABLE} WHERE id = ?",
                (recording_id,),
            ).fetchone()
        return dict(row) if row else None
    except sqlite3.Error as exc:
        log.warning("score 조회 실패: recording_id=%s, error=%s", recording_id, exc)
        return None


def _json_response(data: Any) -> Response:
    body = json.dumps(data, ensure_ascii=False)
    return Response(content=body, media_type="application/json; charset=utf-8")


if __name__ == "__main__":
    uvicorn.run("analysis_viewer:app", host="127.0.0.1", port=9000, reload=True)
