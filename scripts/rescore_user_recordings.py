"""기존 user_recordings의 녹음 파일을 현재 scorer.py로 다시 채점해 새 row로 삽입한다.

실행:
    uv run python scripts/rescore_user_recordings.py
    uv run python scripts/rescore_user_recordings.py --label good --label korean_like
    uv run python scripts/rescore_user_recordings.py --limit 10 --dry-run
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.audio import load_trimmed_audio
from pipeline.db import DEFAULT_DB_PATH, save_user_recording_result
from pipeline.features import extract_features
from pipeline.reference import load_reference_vectors
from pipeline.scorer import score_pronunciation

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

ALLOWED_LABELS: frozenset[str] = frozenset(
    {"unlabeled", "good", "korean_like", "wrong_or_noisy"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="user_recordings 최신 row를 현재 scorer로 재채점해 새 row를 삽입합니다."
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        metavar="LABEL",
        help="재채점할 test_label 필터 (여러 번 사용 가능). 미지정 시 전체.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="처리할 최대 row 수.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="채점 결과를 출력만 하고 DB에 저장하지 않습니다.",
    )
    return parser.parse_args()


def fetch_latest_rows(
    conn: sqlite3.Connection,
    labels: list[str] | None,
    limit: int | None,
) -> list[sqlite3.Row]:
    """word + test_label 조합당 MAX(id) row만 조회한다."""
    and_clauses: list[str] = []
    params: list[object] = []

    if labels:
        placeholders = ", ".join("?" * len(labels))
        and_clauses.append(f"AND test_label IN ({placeholders})")
        params.extend(labels)

    and_sql = " ".join(and_clauses)
    limit_sql = f"LIMIT {limit}" if limit else ""

    sql = f"""
        SELECT *
        FROM user_recordings u
        WHERE id = (
            SELECT MAX(id)
            FROM user_recordings
            WHERE word = u.word
              AND test_label IS u.test_label
        )
        {and_sql}
        ORDER BY id
        {limit_sql}
    """
    return conn.execute(sql, params).fetchall()


def derive_grade(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    return "Needs Practice"


def compute_mfcc_distance(
    user_mfcc: list[float] | None,
    ref_mfcc: list[float] | None,
) -> float | None:
    if user_mfcc is None or ref_mfcc is None:
        return None
    try:
        return float(np.linalg.norm(np.array(user_mfcc) - np.array(ref_mfcc)))
    except Exception:
        return None


def rescore_row(
    row: sqlite3.Row,
    reference_vectors: dict,
    dry_run: bool,
) -> str:
    """단일 row를 재채점하고 결과를 저장한다. 결과 상태 문자열을 반환한다."""
    word = row["word"]
    phoneme = row["phoneme"]
    test_label = row["test_label"]
    recording_path = row["recording_path"]
    row_id = row["id"]

    if not recording_path:
        log.warning("recording_path 없음: id=%s, word=%s", row_id, word)
        return "skip_no_path"

    audio_path = Path(recording_path)
    if not audio_path.exists():
        log.warning("파일 없음: id=%s, path=%s", row_id, recording_path)
        return "skip_missing_file"

    if not phoneme or phoneme not in reference_vectors:
        log.warning("reference 없음: id=%s, phoneme=%s", row_id, phoneme)
        return "skip_no_reference"

    try:
        reference = reference_vectors[phoneme]
        waveform, sr = load_trimmed_audio(audio_path)
        features = extract_features(waveform, sr)
        score_result = score_pronunciation(
            user_features=features,
            reference=reference,
            phoneme=phoneme,
        )
    except Exception as e:
        log.warning("채점 실패: id=%s, word=%s, error=%s", row_id, word, e)
        return "skip_error"

    final_score = score_result["score"]
    mfcc_distance = compute_mfcc_distance(
        user_mfcc=features.get("mfcc_mean"),
        ref_mfcc=reference.get("mfcc_mean"),
    )

    if dry_run:
        log.info(
            "[dry-run] id=%s word=%-12s label=%-14s old_score=%-5s -> new_score=%.1f penalty=%.1f",
            row_id,
            word,
            test_label or "NULL",
            row["score"],
            final_score,
            score_result["details"].get("quality_penalty", 0.0),
        )
        return "dry_run"

    save_user_recording_result(
        word=word,
        phoneme=phoneme,
        score=final_score,
        grade=derive_grade(final_score),
        feedback=score_result["feedback"],
        recording_path=recording_path,
        test_label=test_label,
        duration_ms=features.get("duration_ms"),
        rms_mean=features.get("rms_mean"),
        zcr_mean=features.get("zcr_mean"),
        spectral_centroid_mean=features.get("spectral_centroid_mean"),
        mfcc_distance=mfcc_distance,
    )
    log.info(
        "저장 완료: id=%s word=%-12s label=%-14s old_score=%-5s -> new_score=%.1f",
        row_id,
        word,
        test_label or "NULL",
        row["score"],
        final_score,
    )
    return "ok"


def main() -> None:
    args = parse_args()

    labels: list[str] | None = None
    if args.labels:
        invalid = [l for l in args.labels if l not in ALLOWED_LABELS]
        if invalid:
            log.error("허용되지 않는 label: %s", invalid)
            sys.exit(1)
        labels = args.labels

    if not DEFAULT_DB_PATH.exists():
        log.error("DB 없음: %s", DEFAULT_DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(str(DEFAULT_DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        total_before = conn.execute("SELECT COUNT(*) FROM user_recordings").fetchone()[0]
        rows = fetch_latest_rows(conn, labels=labels, limit=args.limit)
    finally:
        conn.close()

    log.info(
        "재채점 대상: %d row (DB 전체: %d row, label=%s, limit=%s, dry-run=%s)",
        len(rows),
        total_before,
        labels or "전체",
        args.limit or "없음",
        args.dry_run,
    )

    try:
        reference_vectors = load_reference_vectors()
    except Exception as e:
        log.error("reference_vectors 로드 실패: %s", e)
        sys.exit(1)

    counts: dict[str, int] = {"ok": 0, "dry_run": 0, "skip_no_path": 0,
                               "skip_missing_file": 0, "skip_no_reference": 0,
                               "skip_error": 0}

    for row in rows:
        status = rescore_row(row, reference_vectors, dry_run=args.dry_run)
        counts[status] = counts.get(status, 0) + 1

    success = counts["ok"] + counts["dry_run"]
    skipped = len(rows) - success
    log.info(
        "완료 — 처리: %d / 성공: %d / 스킵: %d (no_path=%d, missing_file=%d, no_ref=%d, error=%d)",
        len(rows),
        success,
        skipped,
        counts["skip_no_path"],
        counts["skip_missing_file"],
        counts["skip_no_reference"],
        counts["skip_error"],
    )

    if not args.dry_run:
        conn2 = sqlite3.connect(str(DEFAULT_DB_PATH))
        total_after = conn2.execute("SELECT COUNT(*) FROM user_recordings").fetchone()[0]
        conn2.close()
        log.info("DB row 수: %d -> %d (+%d)", total_before, total_after, total_after - total_before)


if __name__ == "__main__":
    main()
