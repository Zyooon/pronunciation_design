import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.audio import load_trimmed_audio
from pipeline.features import VOWEL_CORE_PHONEMES, extract_features
from pipeline.word_targets import WordTarget, load_word_targets, should_extract_onset


DEFAULT_WORDS_PATH = PROJECT_ROOT / "data" / "words.txt"
DEFAULT_REFERENCE_AUDIO_DIR = PROJECT_ROOT / "data" / "reference_en"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "reference_vectors.json"

SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a"]
MVP_PHONEMES = {"θ", "i", "iː", "æ", "ə", "f", "v", "r", "l", "oʊ"}
VOWEL_PHONEMES = {"i", "iː", "æ", "ə", "oʊ"}
CONSONANT_PHONEMES = {"θ", "f", "v", "r", "l"}
SEQUENCE_FEATURE_KEYS = (
    "mfcc_start_mean",
    "mfcc_middle_mean",
    "mfcc_end_mean",
    "delta_mfcc_mean",
)
ONSET_SCALAR_FEATURE_KEYS = (
    "onset_rms_mean",
    "onset_window_ms",
    "onset_spectral_centroid_mean",
    "onset_zcr_mean",
)
ONSET_SEQUENCE_FEATURE_KEYS = (
    "onset_mfcc_mean",
)
VOWEL_CORE_SEQUENCE_FEATURE_KEYS = (
    "vowel_core_mfcc_mean",
)
VOWEL_CORE_SCALAR_FEATURE_KEYS = (
    "vowel_core_peak_width_ms",
    "vowel_core_mfcc_delta_mean",
    "vowel_core_mfcc_std_mean",
)


def load_word_entries(words_path: Path) -> list[dict]:
    if not words_path.exists():
        raise FileNotFoundError(f"words.txt not found: {words_path}")

    entries = []
    with words_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid format at line {line_number}: {line}\n"
                    "Expected format: word,korean_pronunciation,phoneme"
                )

            word, korean_pronunciation, phoneme = parts
            entries.append({"word": word, "korean_pronunciation": korean_pronunciation, "phoneme": phoneme})

    return entries


def get_phoneme_type(phoneme: str) -> str:
    if phoneme in VOWEL_PHONEMES:
        return "vowel"
    if phoneme in CONSONANT_PHONEMES:
        return "consonant"
    return "unknown"


def find_audio_files(reference_audio_dir: Path, word: str) -> list[Path]:
    audio_files = []
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        audio_files.extend(reference_audio_dir.glob(f"*/{word}{ext}"))
    return sorted(audio_files)


def _aggregate_feature(samples: list[dict], key: str) -> tuple[list[float], list[float]] | None:
    values = [sample[key] for sample in samples if key in sample and sample[key] is not None]
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return np.mean(arr, axis=0).round(6).tolist(), np.std(arr, axis=0).round(6).tolist()


def _aggregate_scalar_feature(samples: list[dict], key: str) -> tuple[float, float] | None:
    values = [float(sample[key]) for sample in samples if key in sample and sample[key] is not None]
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return round(float(np.mean(arr)), 6), round(float(np.std(arr)), 6)


def _collect_phonemes_needing_onset(word_targets: dict[str, WordTarget]) -> frozenset[str]:
    return frozenset(
        target.target_phoneme
        for target in word_targets.values()
        if target.position == "onset"
    )


def aggregate_reference_vectors(samples: list[dict]) -> dict:
    mfcc_values = np.array([sample["mfcc_mean"] for sample in samples], dtype=float)
    zcr_values = np.array([sample["zcr_mean"] for sample in samples], dtype=float)
    duration_values = np.array([sample["duration_ms"] for sample in samples], dtype=float)
    rms_values = np.array([sample["rms_mean"] for sample in samples], dtype=float)
    centroid_values = np.array([sample["spectral_centroid_mean"] for sample in samples], dtype=float)

    vector = {
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

    for key in SEQUENCE_FEATURE_KEYS + ONSET_SEQUENCE_FEATURE_KEYS + VOWEL_CORE_SEQUENCE_FEATURE_KEYS:
        aggregated = _aggregate_feature(samples, key)
        if aggregated is None:
            continue
        mean_values, std_values = aggregated
        vector[key] = mean_values
        vector[f"{key}_std"] = std_values

    for key in ONSET_SCALAR_FEATURE_KEYS + VOWEL_CORE_SCALAR_FEATURE_KEYS:
        aggregated = _aggregate_scalar_feature(samples, key)
        if aggregated is None:
            continue
        mean_value, std_value = aggregated
        vector[key] = mean_value
        vector[f"{key}_std"] = std_value

    return vector


def build_reference_vectors(words_path: Path, reference_audio_dir: Path, output_path: Path, include_all: bool = False) -> dict:
    entries = load_word_entries(words_path)
    if not include_all:
        entries = [entry for entry in entries if entry["phoneme"] in MVP_PHONEMES]

    word_targets = load_word_targets()
    onset_phonemes = _collect_phonemes_needing_onset(word_targets)

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

        include_onset = should_extract_onset(word, phoneme, word_targets)
        include_vowel_core = phoneme in VOWEL_CORE_PHONEMES
        for audio_path in audio_files:
            try:
                y, sr = load_trimmed_audio(audio_path)
                features = extract_features(y, sr, include_onset=include_onset, include_vowel_core=include_vowel_core)
                features["word"] = word
                features["source_file"] = str(audio_path.resolve().relative_to(PROJECT_ROOT.resolve()))
                grouped_samples[phoneme].append(features)
                test_words_by_phoneme[phoneme].add(word)
                onset_note = " + onset" if include_onset else ""
                vowel_core_note = " + vowel_core" if include_vowel_core else ""
                print(f"[OK] {word} ({phoneme}){onset_note}{vowel_core_note} <- {audio_path}")
            except Exception as exc:
                print(f"[SKIP] Failed to process {audio_path}: {exc}")

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
    if onset_phonemes:
        print(f"Onset phonemes configured: {', '.join(sorted(onset_phonemes))}")
    if missing_words:
        print()
        print("[WARN] Audio files not found for these words:")
        for word in sorted(set(missing_words)):
            print(f"  - {word}")
    return reference_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pronunciation reference vectors from reference audio files.")
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS_PATH, help="Path to words.txt")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_REFERENCE_AUDIO_DIR, help="Path to reference English audio directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to output reference_vectors.json")
    parser.add_argument("--include-all", action="store_true", help="Include all phonemes from words.txt, not only MVP phonemes.")
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
