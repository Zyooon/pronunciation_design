import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# scripts 폴더에서 직접 실행해도 프로젝트 루트의 pipeline 패키지를 찾을 수 있게 합니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.audio import load_and_trim_audio
from pipeline.features import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WORDS_PATH = PROJECT_ROOT / "data" / "words.txt"
DEFAULT_REFERENCE_AUDIO_DIR = PROJECT_ROOT / "data" / "reference_en"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "reference_vectors.json"

SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a"]


# MVP에서 우선 구현할 4개 음소입니다.
# 나중에 확장하고 싶으면 CLI에서 --include-all 옵션을 사용하면 됩니다.
MVP_PHONEMES = {"θ", "i", "iː", "æ", "ə"}


VOWEL_PHONEMES = {"i", "iː", "æ", "ə", "oʊ"}
CONSONANT_PHONEMES = {"θ", "f", "v", "r", "l"}


def load_word_entries(words_path: Path) -> list[dict]:
    """
    words.txt를 읽어서 단어 목록을 파싱합니다.

    words.txt 형식:
        영어단어,한국어발음,타겟음소

    주석(#)으로 시작하는 줄과 빈 줄은 무시합니다.
    """
    if not words_path.exists():
        raise FileNotFoundError(f"words.txt not found: {words_path}")

    entries = []

    with words_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            # 빈 줄 또는 주석은 무시합니다.
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid format at line {line_number}: {line}\n"
                    "Expected format: word,korean_pronunciation,phoneme"
                )

            word, korean_pronunciation, phoneme = parts

            entries.append(
                {
                    "word": word,
                    "korean_pronunciation": korean_pronunciation,
                    "phoneme": phoneme,
                }
            )

    return entries


def get_phoneme_type(phoneme: str) -> str:
    """
    음소가 모음인지 자음인지 구분합니다.

    MVP 채점에서 모음과 자음의 점수 계산 방식이 다르기 때문에
    reference vector에도 phoneme_type을 저장합니다.
    """
    if phoneme in VOWEL_PHONEMES:
        return "vowel"

    if phoneme in CONSONANT_PHONEMES:
        return "consonant"

    return "unknown"


def find_audio_files(reference_audio_dir: Path, word: str) -> list[Path]:
    """
    data/reference_en/* 폴더 아래에서 특정 단어의 음성 파일을 찾습니다.

    예:
        data/reference_en/Bella/think.mp3
        data/reference_en/Brian/think.wav
        data/reference_en/gtts/think.mp3
    """
    audio_files = []

    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        pattern = f"*/{word}{ext}"
        audio_files.extend(reference_audio_dir.glob(pattern))

    return sorted(audio_files)


def aggregate_reference_vectors(samples: list[dict]) -> dict:
    """
    여러 음성 샘플의 특징값을 평균/표준편차로 집계합니다.

    기준 벡터에는 mean과 std를 같이 저장합니다.
    이후 사용자 발음과 비교할 때 z-score 기반 점수화에 사용합니다.
    """
    mfcc_values = np.array([sample["mfcc_mean"] for sample in samples], dtype=float)

    zcr_values = np.array([sample["zcr_mean"] for sample in samples], dtype=float)
    duration_values = np.array([sample["duration_ms"] for sample in samples], dtype=float)
    rms_values = np.array([sample["rms_mean"] for sample in samples], dtype=float)
    centroid_values = np.array(
        [sample["spectral_centroid_mean"] for sample in samples],
        dtype=float,
    )

    return {
        "mfcc_mean": np.mean(mfcc_values, axis=0).round(6).tolist(),
        "mfcc_std": np.std(mfcc_values, axis=0).round(6).tolist(),
        "zcr_mean": round(float(np.mean(zcr_values)), 6),
        "zcr_std": round(float(np.std(zcr_values)), 6),
        "duration_ms": round(float(np.mean(duration_values)), 2),
        "duration_std": round(float(np.std(duration_values)), 2),
        "rms_mean": round(float(np.mean(rms_values)), 6),
        "rms_std": round(float(np.std(rms_values)), 6),
        "spectral_centroid_mean": round(float(np.mean(centroid_values)), 6),
        "spectral_centroid_std": round(float(np.std(centroid_values)), 6),
        "sample_count": len(samples),
    }


def build_reference_vectors(
    words_path: Path,
    reference_audio_dir: Path,
    output_path: Path,
    include_all: bool = False,
) -> dict:
    """
    words.txt와 reference_en 음성 파일들을 이용해 reference_vectors.json을 생성합니다.
    """
    entries = load_word_entries(words_path)

    if not include_all:
        entries = [entry for entry in entries if entry["phoneme"] in MVP_PHONEMES]

    grouped_samples: dict[str, list[dict]] = {}
    test_words_by_phoneme: dict[str, set[str]] = {}
    korean_by_word: dict[str, str] = {}

    missing_words = []

    for entry in entries:
        word = entry["word"]
        phoneme = entry["phoneme"]
        korean_pronunciation = entry["korean_pronunciation"]

        audio_files = find_audio_files(reference_audio_dir, word)

        if not audio_files:
            missing_words.append(word)
            continue

        grouped_samples.setdefault(phoneme, [])
        test_words_by_phoneme.setdefault(phoneme, set())
        korean_by_word[word] = korean_pronunciation

        for audio_path in audio_files:
            try:
                y, sr = load_and_trim_audio(audio_path)
                features = extract_features(y, sr)

                # 디버깅과 추적을 위해 어떤 파일에서 온 샘플인지 남겨둡니다.
                features["word"] = word
                features["source_file"] = str(audio_path.relative_to(PROJECT_ROOT))

                grouped_samples[phoneme].append(features)
                test_words_by_phoneme[phoneme].add(word)

                print(f"[OK] {word} ({phoneme}) <- {audio_path}")

            except Exception as e:
                print(f"[SKIP] Failed to process {audio_path}: {e}")

    reference_vectors = {}

    for phoneme, samples in grouped_samples.items():
        if not samples:
            continue

        vector = aggregate_reference_vectors(samples)

        vector["phoneme_type"] = get_phoneme_type(phoneme)
        vector["test_words"] = sorted(test_words_by_phoneme[phoneme])
        vector["korean_pronunciations"] = {
            word: korean_by_word[word] for word in sorted(test_words_by_phoneme[phoneme])
        }

        reference_vectors[phoneme] = vector

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(reference_vectors, f, ensure_ascii=False, indent=2)

    print()
    print(f"Saved reference vectors to: {output_path}")
    print(f"Phonemes: {', '.join(reference_vectors.keys())}")

    if missing_words:
        print()
        print("[WARN] Audio files not found for these words:")
        for word in sorted(set(missing_words)):
            print(f"  - {word}")

    return reference_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pronunciation reference vectors from reference audio files."
    )

    parser.add_argument(
        "--words",
        type=Path,
        default=DEFAULT_WORDS_PATH,
        help="Path to words.txt",
    )

    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_REFERENCE_AUDIO_DIR,
        help="Path to reference English audio directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output reference_vectors.json",
    )

    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all phonemes from words.txt, not only MVP phonemes.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_reference_vectors(
        words_path=args.words,
        reference_audio_dir=args.audio_dir,
        output_path=args.output,
        include_all=args.include_all,
    )


if __name__ == "__main__":
    main()