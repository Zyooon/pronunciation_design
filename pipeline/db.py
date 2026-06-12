import json
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "pronunciation.db"


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
    """
    SQLite 연결을 생성합니다.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """
    발음 비교 검증에 필요한 SQLite 테이블을 생성합니다.
    """
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


def insert_comparison_result(
    result: dict[str, Any],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """
    영어 발음과 한국어식 발음의 feature 비교 결과를 저장합니다.
    """
    initialize_database(db_path)

    values = (
        result["created_at"],
        result["word"],
        result["korean_pronunciation"],
        result["phoneme"],
        result["en_audio_path"],
        result["ko_audio_path"],
        result["en_duration_ms"],
        result["ko_duration_ms"],
        result["duration_diff_ms"],
        result["duration_ratio"],
        result["en_zcr_mean"],
        result["ko_zcr_mean"],
        result["zcr_diff"],
        result["zcr_ratio"],
        result["en_rms_mean"],
        result["ko_rms_mean"],
        result["rms_diff"],
        result["rms_ratio"],
        result["en_spectral_centroid_mean"],
        result["ko_spectral_centroid_mean"],
        result["spectral_centroid_diff"],
        result["spectral_centroid_ratio"],
        result["mfcc_distance"],
        result["mfcc_cosine_distance"],
        json.dumps(result["en_mfcc_mean_json"], ensure_ascii=False),
        json.dumps(result["ko_mfcc_mean_json"], ensure_ascii=False),
    )

    with get_connection(db_path) as conn:
        cursor = conn.execute(INSERT_COMPARISON_SQL, values)
        conn.commit()
        return int(cursor.lastrowid)
