import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

log = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "pronunciation.db"


class ComparisonResultDict(TypedDict):
    created_at: str
    word: str
    korean_pronunciation: str
    phoneme: str
    en_audio_path: str
    ko_audio_path: str
    en_duration_ms: float
    ko_duration_ms: float
    duration_diff_ms: float
    duration_ratio: float
    en_zcr_mean: float
    ko_zcr_mean: float
    zcr_diff: float
    zcr_ratio: float
    en_rms_mean: float
    ko_rms_mean: float
    rms_diff: float
    rms_ratio: float
    en_spectral_centroid_mean: float
    ko_spectral_centroid_mean: float
    spectral_centroid_diff: float
    spectral_centroid_ratio: float
    mfcc_distance: float
    mfcc_cosine_distance: float
    en_mfcc_mean_json: list[float]
    ko_mfcc_mean_json: list[float]


INSERT_COMPARISON_SQL = """
INSERT INTO comparison_results (
    created_at,
    word,
    korean_pronunciation,
    phoneme,
    en_audio_path,
    ko_audio_path,
    en_duration_ms,
    ko_duration_ms,
    duration_diff_ms,
    duration_ratio,
    en_zcr_mean,
    ko_zcr_mean,
    zcr_diff,
    zcr_ratio,
    en_rms_mean,
    ko_rms_mean,
    rms_diff,
    rms_ratio,
    en_spectral_centroid_mean,
    ko_spectral_centroid_mean,
    spectral_centroid_diff,
    spectral_centroid_ratio,
    mfcc_distance,
    mfcc_cosine_distance,
    en_mfcc_mean_json,
    ko_mfcc_mean_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """SQLite 연결을 생성합니다."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """발음 비교 검증에 필요한 SQLite 테이블을 생성합니다."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS comparison_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                word TEXT NOT NULL,
                korean_pronunciation TEXT NOT NULL,
                phoneme TEXT NOT NULL,
                en_audio_path TEXT NOT NULL,
                ko_audio_path TEXT NOT NULL,
                en_duration_ms REAL NOT NULL,
                ko_duration_ms REAL NOT NULL,
                duration_diff_ms REAL NOT NULL,
                duration_ratio REAL NOT NULL,
                en_zcr_mean REAL NOT NULL,
                ko_zcr_mean REAL NOT NULL,
                zcr_diff REAL NOT NULL,
                zcr_ratio REAL NOT NULL,
                en_rms_mean REAL NOT NULL,
                ko_rms_mean REAL NOT NULL,
                rms_diff REAL NOT NULL,
                rms_ratio REAL NOT NULL,
                en_spectral_centroid_mean REAL NOT NULL,
                ko_spectral_centroid_mean REAL NOT NULL,
                spectral_centroid_diff REAL NOT NULL,
                spectral_centroid_ratio REAL NOT NULL,
                mfcc_distance REAL NOT NULL,
                mfcc_cosine_distance REAL NOT NULL,
                en_mfcc_mean_json TEXT NOT NULL,
                ko_mfcc_mean_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_comparison_results_word
            ON comparison_results(word)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_comparison_results_phoneme
            ON comparison_results(phoneme)
            """
        )
        conn.commit()


def _build_comparison_insert_values(comparison_result: ComparisonResultDict) -> tuple:
    return (
        comparison_result["created_at"],
        comparison_result["word"],
        comparison_result["korean_pronunciation"],
        comparison_result["phoneme"],
        comparison_result["en_audio_path"],
        comparison_result["ko_audio_path"],
        comparison_result["en_duration_ms"],
        comparison_result["ko_duration_ms"],
        comparison_result["duration_diff_ms"],
        comparison_result["duration_ratio"],
        comparison_result["en_zcr_mean"],
        comparison_result["ko_zcr_mean"],
        comparison_result["zcr_diff"],
        comparison_result["zcr_ratio"],
        comparison_result["en_rms_mean"],
        comparison_result["ko_rms_mean"],
        comparison_result["rms_diff"],
        comparison_result["rms_ratio"],
        comparison_result["en_spectral_centroid_mean"],
        comparison_result["ko_spectral_centroid_mean"],
        comparison_result["spectral_centroid_diff"],
        comparison_result["spectral_centroid_ratio"],
        comparison_result["mfcc_distance"],
        comparison_result["mfcc_cosine_distance"],
        json.dumps(comparison_result["en_mfcc_mean_json"], ensure_ascii=False),
        json.dumps(comparison_result["ko_mfcc_mean_json"], ensure_ascii=False),
    )


_CREATE_USER_RECORDINGS_SQL = """
CREATE TABLE IF NOT EXISTS user_recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    word TEXT NOT NULL,
    phoneme TEXT,
    score REAL,
    grade TEXT,
    feedback TEXT,
    recording_path TEXT,
    test_label TEXT,
    duration_ms REAL,
    rms_mean REAL,
    zcr_mean REAL,
    spectral_centroid_mean REAL,
    mfcc_distance REAL,
    details_json TEXT,
    base_score REAL,
    final_score REAL,
    mfcc_score REAL,
    duration_score REAL,
    rms_score REAL,
    zcr_score REAL,
    spectral_centroid_score REAL,
    quality_penalty REAL,
    pronunciation_penalty REAL,
    total_penalty REAL,
    duration_penalty REAL,
    volume_penalty REAL,
    noise_penalty REAL,
    duration_ratio REAL
)
"""

_USER_RECORDINGS_OPTIONAL_COLUMNS: list[tuple[str, str]] = [
    ("grade", "TEXT"),
    ("recording_path", "TEXT"),
    ("test_label", "TEXT"),
    ("duration_ms", "REAL"),
    ("rms_mean", "REAL"),
    ("zcr_mean", "REAL"),
    ("spectral_centroid_mean", "REAL"),
    ("mfcc_distance", "REAL"),
    ("details_json", "TEXT"),
    ("base_score", "REAL"),
    ("final_score", "REAL"),
    ("mfcc_score", "REAL"),
    ("duration_score", "REAL"),
    ("rms_score", "REAL"),
    ("zcr_score", "REAL"),
    ("spectral_centroid_score", "REAL"),
    ("quality_penalty", "REAL"),
    ("pronunciation_penalty", "REAL"),
    ("total_penalty", "REAL"),
    ("duration_penalty", "REAL"),
    ("volume_penalty", "REAL"),
    ("noise_penalty", "REAL"),
    ("duration_ratio", "REAL"),
]


def ensure_user_recordings_table(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """user_recordings 테이블을 생성하고 누락 컬럼을 안전하게 추가한다."""
    with get_connection(db_path) as conn:
        conn.execute(_CREATE_USER_RECORDINGS_SQL)
        existing_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(user_recordings)")
        }
        for col_name, col_type in _USER_RECORDINGS_OPTIONAL_COLUMNS:
            if col_name not in existing_cols:
                conn.execute(
                    f"ALTER TABLE user_recordings ADD COLUMN {col_name} {col_type}"
                )
        conn.commit()


def _metric_from_details(details: dict[str, Any] | None, key: str) -> float | None:
    if not details:
        return None
    value = details.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _details_to_json(details: dict[str, Any] | None) -> str | None:
    if not details:
        return None
    return json.dumps(details, ensure_ascii=False, sort_keys=True)


def save_user_recording_result(
    *,
    word: str,
    phoneme: str | None,
    score: float | None,
    grade: str | None,
    feedback: str | None,
    recording_path: str | None,
    test_label: str | None = None,
    duration_ms: float | None = None,
    rms_mean: float | None = None,
    zcr_mean: float | None = None,
    spectral_centroid_mean: float | None = None,
    mfcc_distance: float | None = None,
    details: dict[str, Any] | None = None,
    created_at: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """사용자 녹음 결과를 user_recordings 테이블에 저장한다.

    저장 실패 시 예외를 전파하지 않고 에러 로그만 남긴다.

    Args:
        test_label: 테스트 라벨. ENABLE_TEST_LABELS=false이면 None(NULL)으로 저장한다.
        details: scorer.py가 반환한 세부 점수와 penalty 정보.
        created_at: 명시적 생성 시각(ISO 문자열). None이면 현재 UTC 시간을 사용한다.
                    재채점 시 원본 녹음의 created_at을 유지하기 위해 사용한다.
    """
    try:
        ensure_user_recordings_table(db_path)
        created_at_value = created_at or datetime.now(timezone.utc).isoformat()
        final_score = _metric_from_details(details, "final_score")
        if final_score is None and score is not None:
            final_score = float(score)

        total_penalty = _metric_from_details(details, "total_penalty")
        if total_penalty is None:
            quality = _metric_from_details(details, "quality_penalty") or 0.0
            pronunciation = _metric_from_details(details, "pronunciation_penalty") or 0.0
            total_penalty = quality + pronunciation

        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO user_recordings (
                    created_at, word, phoneme, score, grade, feedback,
                    recording_path, test_label,
                    duration_ms, rms_mean, zcr_mean, spectral_centroid_mean, mfcc_distance,
                    details_json, base_score, final_score,
                    mfcc_score, duration_score, rms_score, zcr_score, spectral_centroid_score,
                    quality_penalty, pronunciation_penalty, total_penalty,
                    duration_penalty, volume_penalty, noise_penalty, duration_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at_value,
                    word,
                    phoneme,
                    score,
                    grade,
                    feedback,
                    recording_path,
                    test_label,
                    duration_ms,
                    rms_mean,
                    zcr_mean,
                    spectral_centroid_mean,
                    mfcc_distance,
                    _details_to_json(details),
                    _metric_from_details(details, "base_score"),
                    final_score,
                    _metric_from_details(details, "mfcc_score"),
                    _metric_from_details(details, "duration_score"),
                    _metric_from_details(details, "rms_score"),
                    _metric_from_details(details, "zcr_score"),
                    _metric_from_details(details, "spectral_centroid_score"),
                    _metric_from_details(details, "quality_penalty"),
                    _metric_from_details(details, "pronunciation_penalty"),
                    total_penalty,
                    _metric_from_details(details, "duration_penalty"),
                    _metric_from_details(details, "volume_penalty"),
                    _metric_from_details(details, "noise_penalty"),
                    _metric_from_details(details, "duration_ratio"),
                ),
            )
            conn.commit()

    except Exception:
        log.error(
            "user_recordings 저장 실패: word=%s, phoneme=%s",
            word, phoneme,
            exc_info=True,
        )


def insert_comparison_result(
    comparison_result: ComparisonResultDict,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """
    영어 발음과 한국어식 발음의 feature 비교 결과를 저장합니다.

    Args:
        comparison_result: 비교 결과 데이터
        db_path: SQLite DB 파일 경로

    Returns:
        삽입된 row의 id
    """
    initialize_database(db_path)

    insert_values = _build_comparison_insert_values(comparison_result)

    with get_connection(db_path) as conn:
        cursor = conn.execute(INSERT_COMPARISON_SQL, insert_values)
        conn.commit()
        return int(cursor.lastrowid)
