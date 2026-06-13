import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.audio import load_trimmed_audio
from pipeline.db import DEFAULT_DB_PATH, save_user_recording_result
from pipeline.features import extract_features
from pipeline.quality import evaluate_recording_quality
from pipeline.reference import load_reference_vectors
from pipeline.scorer import score_pronunciation

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)
KO_REFERENCE_PATH = PROJECT_ROOT / "data" / "ko_reference_vectors.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", dest="labels", action="append")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def fetch_latest_rows(conn: sqlite3.Connection, labels: list[str] | None, limit: int | None) -> list[sqlite3.Row]:
    params: list[object] = []
    label_sql = ""
    if labels:
        placeholders = ", ".join("?" * len(labels))
        label_sql = f"AND test_label IN ({placeholders})"
        params.extend(labels)
    limit_sql = f"LIMIT {limit}" if limit else ""
    sql = f"""
        SELECT
            u.*,
            (
                SELECT MIN(created_at)
                FROM user_recordings
                WHERE recording_path = u.recording_path
                  AND created_at IS NOT NULL
            ) AS original_created_at
        FROM user_recordings u
        WHERE id = (
            SELECT MAX(id)
            FROM user_recordings
            WHERE word = u.word
              AND test_label IS u.test_label
        )
        {label_sql}
        ORDER BY id
        {limit_sql}
    """
    return conn.execute(sql, params).fetchall()


def derive_grade(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    return "Needs Practice"


def format_score(score: float | None) -> str:
    if score is None:
        return "None"
    return f"{score:.1f}"


def compute_mfcc_distance(user_mfcc: list[float] | None, ref_mfcc: list[float] | None) -> float | None:
    if user_mfcc is None or ref_mfcc is None:
        return None
    try:
        return float(np.linalg.norm(np.array(user_mfcc) - np.array(ref_mfcc)))
    except Exception:
        return None


def load_ko_vectors() -> dict:
    if not KO_REFERENCE_PATH.exists():
        log.warning("ko_reference_vectors.json 없음: 기존 reference만 사용합니다.")
        return {}
    with KO_REFERENCE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def rescore_row(row: sqlite3.Row, en_vectors: dict, ko_vectors: dict, dry_run: bool) -> str:
    word = row["word"]
    phoneme = row["phoneme"]
    recording_path = row["recording_path"]
    if not recording_path:
        return "skip_no_path"
    audio_path = Path(recording_path)
    if not audio_path.exists():
        return "skip_missing_file"
    if not phoneme or phoneme not in en_vectors:
        return "skip_no_reference"

    reference = en_vectors[phoneme]
    liquid_alt_reference = None
    if phoneme == "r":
        liquid_alt_reference = en_vectors.get("l")
    elif phoneme == "l":
        liquid_alt_reference = en_vectors.get("r")
    ko_reference = ko_vectors.get(phoneme)

    waveform, sr = load_trimmed_audio(audio_path)
    features = extract_features(waveform, sr)
    quality_result = evaluate_recording_quality(
        features=features,
        reference=reference,
        audio_path=str(audio_path),
        target_word=word,
    )
    score_result = score_pronunciation(
        features,
        reference,
        phoneme,
        ko_reference=ko_reference,
        liquid_alt_reference=liquid_alt_reference,
        recording_quality_result=quality_result,
    )
    score = score_result["score"]
    details = score_result.get("details", {})

    original_created_at = row["original_created_at"] or row["created_at"]

    if dry_run:
        log.info(
            "[dry-run] id=%s word=%s label=%s old=%s new=%s quality=%s issues=%s original_created_at=%s",
            row["id"],
            word,
            row["test_label"] or "NULL",
            row["score"],
            format_score(score),
            score_result.get("recording_quality_status"),
            score_result.get("issue_flags"),
            original_created_at,
        )
        return "dry_run"

    save_user_recording_result(
        word=word,
        phoneme=phoneme,
        score=score,
        grade=derive_grade(score),
        feedback=score_result["feedback"],
        recording_path=recording_path,
        test_label=row["test_label"],
        duration_ms=features.get("duration_ms"),
        rms_mean=features.get("rms_mean"),
        zcr_mean=features.get("zcr_mean"),
        spectral_centroid_mean=features.get("spectral_centroid_mean"),
        mfcc_distance=compute_mfcc_distance(features.get("mfcc_mean"), reference.get("mfcc_mean")),
        details=details,
        created_at=original_created_at,
    )
    log.info(
        "저장 완료: id=%s word=%s label=%s new=%s quality=%s issues=%s",
        row["id"],
        word,
        row["test_label"] or "NULL",
        format_score(score),
        score_result.get("recording_quality_status"),
        score_result.get("issue_flags"),
    )
    return "ok"


def main() -> None:
    args = parse_args()
    if not DEFAULT_DB_PATH.exists():
        log.error("DB 없음: %s", DEFAULT_DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(str(DEFAULT_DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        total_before = conn.execute("SELECT COUNT(*) FROM user_recordings").fetchone()[0]
        rows = fetch_latest_rows(conn, args.labels, args.limit)
    finally:
        conn.close()

    en_vectors = load_reference_vectors()
    ko_vectors = load_ko_vectors()
    counts: dict[str, int] = {}
    for row in rows:
        try:
            status = rescore_row(row, en_vectors, ko_vectors, args.dry_run)
        except Exception as exc:
            log.warning("스킵: id=%s detail=%s", row["id"], exc)
            status = "skip"
        counts[status] = counts.get(status, 0) + 1

    log.info("완료: %s", counts)
    if not args.dry_run:
        conn2 = sqlite3.connect(str(DEFAULT_DB_PATH))
        total_after = conn2.execute("SELECT COUNT(*) FROM user_recordings").fetchone()[0]
        conn2.close()
        log.info("DB row 수: %d -> %d (+%d)", total_before, total_after, total_after - total_before)


if __name__ == "__main__":
    main()
