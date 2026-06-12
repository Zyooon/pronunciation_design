import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.audio import load_and_trim_audio
from pipeline.db import DEFAULT_DB_PATH, initialize_database, insert_comparison_result
from pipeline.features import extract_features
from pipeline.reference import DEFAULT_WORDS_PATH, load_target_words


DEFAULT_EN_AUDIO_DIR = PROJECT_ROOT / "data" / "reference_en" / "gtts"
DEFAULT_KO_AUDIO_DIR = PROJECT_ROOT / "data" / "reference_ko" / "gtts"
DEFAULT_JSON_OUTPUT_PATH = PROJECT_ROOT / "data" / "comparison_results.json"
SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".webm", ".flac"]
EPSILON = 1e-8


def find_audio_files(audio_dir: Path, word: str) -> list[Path]:
    """
    audio_dir 아래에서 특정 단어의 음성 파일을 찾습니다.

    지원 형태:
        data/reference_en/gtts/think.mp3
        data/reference_en/gtts/Bella/think.mp3
    """
    audio_files: list[Path] = []

    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        audio_files.extend(audio_dir.glob(f"{word}{ext}"))
        audio_files.extend(audio_dir.glob(f"*/{word}{ext}"))

    return sorted(set(audio_files))


def relative_path(path: Path) -> str:
    """
    출력, DB, JSON 저장을 위해 프로젝트 기준 상대 경로로 변환합니다.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_ratio(a: float, b: float) -> float:
    """
    두 값의 상대 차이 비율을 계산합니다.
    """
    return round(abs(a - b) / (abs(a) + EPSILON), 6)


def cosine_distance(a: list[float], b: list[float]) -> float:
    """
    두 MFCC 평균 벡터의 cosine distance를 계산합니다.
    """
    a_array = np.array(a, dtype=float)
    b_array = np.array(b, dtype=float)

    denominator = np.linalg.norm(a_array) * np.linalg.norm(b_array)

    if denominator < EPSILON:
        return 0.0

    cosine_similarity = float(np.dot(a_array, b_array) / denominator)
    cosine_similarity = float(np.clip(cosine_similarity, -1.0, 1.0))

    return round(1 - cosine_similarity, 6)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """
    두 MFCC 평균 벡터의 유클리드 거리를 계산합니다.
    """
    a_array = np.array(a, dtype=float)
    b_array = np.array(b, dtype=float)
    return round(float(np.linalg.norm(a_array - b_array)), 6)


def compare_feature_pair(
    word: str,
    korean_pronunciation: str,
    phoneme: str,
    en_audio_path: Path,
    ko_audio_path: Path,
) -> dict[str, Any]:
    """
    같은 단어의 영어 발음과 한국어식 발음을 feature 단위로 비교합니다.
    """
    en_y, en_sr = load_and_trim_audio(en_audio_path)
    ko_y, ko_sr = load_and_trim_audio(ko_audio_path)

    en_features = extract_features(en_y, en_sr)
    ko_features = extract_features(ko_y, ko_sr)

    en_duration = float(en_features["duration_ms"])
    ko_duration = float(ko_features["duration_ms"])
    en_zcr = float(en_features["zcr_mean"])
    ko_zcr = float(ko_features["zcr_mean"])
    en_rms = float(en_features["rms_mean"])
    ko_rms = float(ko_features["rms_mean"])
    en_centroid = float(en_features["spectral_centroid_mean"])
    ko_centroid = float(ko_features["spectral_centroid_mean"])

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "word": word,
        "korean_pronunciation": korean_pronunciation,
        "phoneme": phoneme,
        "en_audio_path": relative_path(en_audio_path),
        "ko_audio_path": relative_path(ko_audio_path),
        "en_duration_ms": round(en_duration, 3),
        "ko_duration_ms": round(ko_duration, 3),
        "duration_diff_ms": round(abs(en_duration - ko_duration), 3),
        "duration_ratio": safe_ratio(en_duration, ko_duration),
        "en_zcr_mean": round(en_zcr, 6),
        "ko_zcr_mean": round(ko_zcr, 6),
        "zcr_diff": round(abs(en_zcr - ko_zcr), 6),
        "zcr_ratio": safe_ratio(en_zcr, ko_zcr),
        "en_rms_mean": round(en_rms, 6),
        "ko_rms_mean": round(ko_rms, 6),
        "rms_diff": round(abs(en_rms - ko_rms), 6),
        "rms_ratio": safe_ratio(en_rms, ko_rms),
        "en_spectral_centroid_mean": round(en_centroid, 6),
        "ko_spectral_centroid_mean": round(ko_centroid, 6),
        "spectral_centroid_diff": round(abs(en_centroid - ko_centroid), 6),
        "spectral_centroid_ratio": safe_ratio(en_centroid, ko_centroid),
        "mfcc_distance": euclidean_distance(en_features["mfcc_mean"], ko_features["mfcc_mean"]),
        "mfcc_cosine_distance": cosine_distance(en_features["mfcc_mean"], ko_features["mfcc_mean"]),
        "en_mfcc_mean_json": en_features["mfcc_mean"],
        "ko_mfcc_mean_json": ko_features["mfcc_mean"],
    }


def write_json_results(
    output_path: Path,
    metadata: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    """
    이번 실행에서 성공한 비교 결과를 JSON 파일로 저장합니다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata,
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def compare_all_words(
    words_path: Path,
    en_audio_dir: Path,
    ko_audio_dir: Path,
    db_path: Path,
    json_output_path: Path,
    limit: int | None = None,
) -> None:
    """
    words.txt의 각 단어에 대해 영어 발음과 한국어식 발음을 비교하고 SQLite와 JSON에 저장합니다.
    """
    initialize_database(db_path)
    target_words = load_target_words(words_path)

    saved_count = 0
    skipped_count = 0
    results: list[dict[str, Any]] = []
    started_at = datetime.now().isoformat(timespec="seconds")

    for target in target_words:
        if limit is not None and saved_count >= limit:
            break

        en_audio_files = find_audio_files(en_audio_dir, target.word)

        # 한국어식 gTTS 파일명이 영어 단어 기준이면 target.word로 찾고,
        # 한글 발음 기준이면 target.korean_pronunciation으로도 한 번 더 찾습니다.
        ko_audio_files = find_audio_files(ko_audio_dir, target.word)
        if not ko_audio_files:
            ko_audio_files = find_audio_files(ko_audio_dir, target.korean_pronunciation)

        if not en_audio_files:
            skipped_count += 1
            print(f"[SKIP] English audio not found: {target.word}")
            continue

        if not ko_audio_files:
            skipped_count += 1
            print(f"[SKIP] Korean-style audio not found: {target.word}")
            continue

        en_audio_path = en_audio_files[0]
        ko_audio_path = ko_audio_files[0]

        try:
            result = compare_feature_pair(
                word=target.word,
                korean_pronunciation=target.korean_pronunciation,
                phoneme=target.phoneme,
                en_audio_path=en_audio_path,
                ko_audio_path=ko_audio_path,
            )
            row_id = insert_comparison_result(result, db_path=db_path)
            result["db_row_id"] = row_id
            results.append(result)
            saved_count += 1

            print(
                f"[OK] id={row_id} {target.word} /{target.phoneme}/ "
                f"mfcc={result['mfcc_distance']} zcr_diff={result['zcr_diff']}"
            )

        except Exception as e:
            skipped_count += 1
            print(f"[SKIP] Failed to compare {target.word}: {type(e).__name__}: {e}")

    finished_at = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "started_at": started_at,
        "finished_at": finished_at,
        "words_path": relative_path(words_path),
        "en_audio_dir": relative_path(en_audio_dir),
        "ko_audio_dir": relative_path(ko_audio_dir),
        "db_path": relative_path(db_path),
        "json_output_path": relative_path(json_output_path),
        "limit": limit,
        "saved_count": saved_count,
        "skipped_count": skipped_count,
    }
    write_json_results(json_output_path, metadata, results)

    print()
    print(f"Saved rows: {saved_count}")
    print(f"Skipped rows: {skipped_count}")
    print(f"Database: {db_path}")
    print(f"JSON: {json_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare English pronunciation audio with Korean-style pronunciation audio."
    )
    parser.add_argument(
        "--words",
        type=Path,
        default=DEFAULT_WORDS_PATH,
        help="Path to data/words.txt",
    )
    parser.add_argument(
        "--en-audio-dir",
        type=Path,
        default=DEFAULT_EN_AUDIO_DIR,
        help="Path to English reference audio directory",
    )
    parser.add_argument(
        "--ko-audio-dir",
        type=Path,
        default=DEFAULT_KO_AUDIO_DIR,
        help="Path to Korean-style reference audio directory",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT_PATH,
        help="Path to JSON output file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_all_words(
        words_path=args.words,
        en_audio_dir=args.en_audio_dir,
        ko_audio_dir=args.ko_audio_dir,
        db_path=args.db,
        json_output_path=args.json_output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
