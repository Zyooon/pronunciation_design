from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.audio import load_trimmed_audio
from pipeline.reference import TargetWord, load_target_words

SOURCE_DIR = PROJECT_ROOT / "data" / "reference_ko" / "record_selected_eval"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
LABELS = ("good", "korean_like")
LIQUID_PHONEMES = {"r", "l"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

ENERGY_FRAME_MS = 10.0
ENERGY_HOP_MS = 5.0
ANALYSIS_WINDOW_MS = 250.0
ONSET_MFCC_WINDOW_MS = 50.0
VOWEL_MFCC_START_MS = 100.0
VOWEL_MFCC_WINDOW_MS = 80.0

LOW_BAND_HZ = (0.0, 1000.0)
MID_BAND_HZ = (1000.0, 2000.0)
F3_BAND_HZ = (2000.0, 3000.0)
HIGH_BAND_HZ = (3000.0, 5000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="/r/, /l/ 발음 구분을 위한 energy, mel-band, transition feature를 CSV로 분석합니다.",
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--labels", nargs="+", default=list(LABELS))
    parser.add_argument("--analysis-window-ms", type=float, default=ANALYSIS_WINDOW_MS)
    parser.add_argument("--energy-frame-ms", type=float, default=ENERGY_FRAME_MS)
    parser.add_argument("--energy-hop-ms", type=float, default=ENERGY_HOP_MS)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-mfcc", type=int, default=13)
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def filename_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in path.with_suffix("").parts:
        normalized = normalize_text(part)
        tokens.update(token for token in normalized.split() if token)
    return tokens


def build_word_index(target_words: list[TargetWord]) -> dict[str, list[TargetWord]]:
    index: dict[str, list[TargetWord]] = defaultdict(list)
    for target in target_words:
        index[target.word.lower()].append(target)
    return index


def find_target_by_filename(path: Path, word_index: dict[str, list[TargetWord]]) -> TargetWord:
    tokens = filename_tokens(path)
    candidates: list[TargetWord] = []
    for word, targets in word_index.items():
        if word in tokens:
            candidates.extend(targets)

    if not candidates:
        stem = normalize_text(path.stem)
        for word, targets in word_index.items():
            if stem == word or stem.startswith(f"{word} ") or f" {word} " in f" {stem} ":
                candidates.extend(targets)

    if not candidates:
        raise ValueError(f"파일명에서 target word를 찾을 수 없습니다: {path}")

    if len(candidates) == 1:
        return candidates[0]

    phoneme_tokens = tokens | set(normalize_text(path.stem).split())
    phoneme_matches = [target for target in candidates if normalize_text(target.phoneme) in phoneme_tokens]
    if len(phoneme_matches) == 1:
        return phoneme_matches[0]

    candidate_text = ", ".join(target.target_id for target in candidates)
    raise ValueError(f"파일명 target이 모호합니다: {path} -> {candidate_text}")


def audio_files_for_label(source_dir: Path, label: str) -> list[Path]:
    label_dir = source_dir / label
    if not label_dir.exists():
        return []
    return sorted(
        path
        for path in label_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def collect_audio_files(source_dir: Path, labels: list[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for label in labels:
        files.extend((label, path) for path in audio_files_for_label(source_dir, label))
    return files


def slice_window(waveform: np.ndarray, sr: int, start_ms: float, window_ms: float) -> np.ndarray:
    start_sample = max(0, int(sr * start_ms / 1000.0))
    window_samples = max(1, int(sr * window_ms / 1000.0))
    end_sample = min(len(waveform), start_sample + window_samples)
    if end_sample <= start_sample:
        return np.array([], dtype=float)
    return waveform[start_sample:end_sample]


def slice_analysis_window(waveform: np.ndarray, sr: int, window_ms: float) -> np.ndarray:
    window_samples = max(1, int(sr * window_ms / 1000.0))
    return waveform[: min(len(waveform), window_samples)]


def compute_energy_features(waveform: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> dict[str, float | None]:
    frame_length = max(1, int(sr * frame_ms / 1000.0))
    hop_length = max(1, int(sr * hop_ms / 1000.0))
    if len(waveform) < frame_length:
        return {
            "energy_frame_count": 0.0,
            "energy_rms_mean": None,
            "energy_rms_min": None,
            "energy_rms_max": None,
            "energy_delta_min": None,
            "energy_delta_max": None,
            "energy_dip_depth": None,
            "energy_peak_after_dip": None,
            "energy_v_shape_score": None,
            "energy_roughness": None,
        }

    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length, center=False)[0]
    if len(rms) == 0:
        return {
            "energy_frame_count": 0.0,
            "energy_rms_mean": None,
            "energy_rms_min": None,
            "energy_rms_max": None,
            "energy_delta_min": None,
            "energy_delta_max": None,
            "energy_dip_depth": None,
            "energy_peak_after_dip": None,
            "energy_v_shape_score": None,
            "energy_roughness": None,
        }

    deltas = np.diff(rms) if len(rms) > 1 else np.array([0.0])
    min_idx = int(np.argmin(rms))
    pre_peak = float(np.max(rms[: min_idx + 1])) if min_idx >= 0 else float(rms[0])
    post_peak = float(np.max(rms[min_idx:])) if min_idx < len(rms) else float(rms[-1])
    dip = float(rms[min_idx])
    dip_depth = max(0.0, min(pre_peak, post_peak) - dip)
    peak_after_dip = max(0.0, post_peak - dip)
    v_shape_score = dip_depth / (float(np.mean(rms)) + 1e-8)
    roughness = float(np.mean(np.abs(deltas)) / (np.mean(rms) + 1e-8))

    return {
        "energy_frame_count": float(len(rms)),
        "energy_rms_mean": round(float(np.mean(rms)), 8),
        "energy_rms_min": round(float(np.min(rms)), 8),
        "energy_rms_max": round(float(np.max(rms)), 8),
        "energy_delta_min": round(float(np.min(deltas)), 8),
        "energy_delta_max": round(float(np.max(deltas)), 8),
        "energy_dip_depth": round(float(dip_depth), 8),
        "energy_peak_after_dip": round(float(peak_after_dip), 8),
        "energy_v_shape_score": round(float(v_shape_score), 6),
        "energy_roughness": round(float(roughness), 6),
    }


def band_energy(mel_power: np.ndarray, mel_frequencies: np.ndarray, band: tuple[float, float]) -> float | None:
    low, high = band
    mask = (mel_frequencies >= low) & (mel_frequencies < high)
    if not np.any(mask):
        return None
    return float(np.mean(mel_power[mask, :]))


def compute_mel_band_features(waveform: np.ndarray, sr: int, n_mels: int) -> dict[str, float | None]:
    if len(waveform) == 0:
        return {
            "mel_low_band_energy": None,
            "mel_mid_band_energy": None,
            "mel_f3_band_energy": None,
            "mel_high_band_energy": None,
            "mel_f3_to_low_ratio": None,
            "mel_f3_to_mid_ratio": None,
            "mel_high_to_low_ratio": None,
            "mel_f3_band_log_energy": None,
        }

    n_fft = min(512, max(64, 2 ** int(np.floor(np.log2(max(1, len(waveform)))))))
    hop_length = max(1, n_fft // 4)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=min(float(sr) / 2.0, HIGH_BAND_HZ[1]),
        power=2.0,
    )
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=min(float(sr) / 2.0, HIGH_BAND_HZ[1]))

    low = band_energy(mel, mel_frequencies, LOW_BAND_HZ)
    mid = band_energy(mel, mel_frequencies, MID_BAND_HZ)
    f3 = band_energy(mel, mel_frequencies, F3_BAND_HZ)
    high = band_energy(mel, mel_frequencies, HIGH_BAND_HZ)

    return {
        "mel_low_band_energy": round(low, 8) if low is not None else None,
        "mel_mid_band_energy": round(mid, 8) if mid is not None else None,
        "mel_f3_band_energy": round(f3, 8) if f3 is not None else None,
        "mel_high_band_energy": round(high, 8) if high is not None else None,
        "mel_f3_to_low_ratio": round(f3 / (low + 1e-10), 6) if low is not None and f3 is not None else None,
        "mel_f3_to_mid_ratio": round(f3 / (mid + 1e-10), 6) if mid is not None and f3 is not None else None,
        "mel_high_to_low_ratio": round(high / (low + 1e-10), 6) if low is not None and high is not None else None,
        "mel_f3_band_log_energy": round(float(np.log10((f3 or 0.0) + 1e-10)), 6) if f3 is not None else None,
    }


def mean_mfcc(waveform: np.ndarray, sr: int, n_mfcc: int) -> np.ndarray | None:
    if len(waveform) == 0:
        return None
    n_fft = min(512, max(64, 2 ** int(np.floor(np.log2(max(1, len(waveform)))))))
    hop_length = max(1, n_fft // 4)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    if mfcc.shape[1] == 0:
        return None
    return np.mean(mfcc, axis=1)


def compute_mfcc_transition_features(waveform: np.ndarray, sr: int, n_mfcc: int) -> dict[str, float | None]:
    onset = slice_window(waveform, sr, 0.0, ONSET_MFCC_WINDOW_MS)
    vowel = slice_window(waveform, sr, VOWEL_MFCC_START_MS, VOWEL_MFCC_WINDOW_MS)
    onset_mfcc = mean_mfcc(onset, sr, n_mfcc)
    vowel_mfcc = mean_mfcc(vowel, sr, n_mfcc)
    if onset_mfcc is None or vowel_mfcc is None:
        return {
            "transition_mfcc_distance": None,
            "transition_mfcc_slope": None,
            "transition_mfcc_delta_norm": None,
            "transition_mfcc_c0_delta": None,
            "transition_mfcc_c1_delta": None,
            "transition_mfcc_c2_delta": None,
        }

    delta = vowel_mfcc - onset_mfcc
    distance = float(np.linalg.norm(delta))
    slope = distance / max(1.0, VOWEL_MFCC_START_MS)
    return {
        "transition_mfcc_distance": round(distance, 6),
        "transition_mfcc_slope": round(float(slope), 6),
        "transition_mfcc_delta_norm": round(float(np.mean(np.abs(delta))), 6),
        "transition_mfcc_c0_delta": round(float(delta[0]), 6),
        "transition_mfcc_c1_delta": round(float(delta[1]), 6) if len(delta) > 1 else None,
        "transition_mfcc_c2_delta": round(float(delta[2]), 6) if len(delta) > 2 else None,
    }


def analyze_audio_file(label: str, audio_path: Path, target: TargetWord, args: argparse.Namespace) -> dict[str, Any]:
    waveform, sample_rate = load_trimmed_audio(audio_path)
    analysis_waveform = slice_analysis_window(waveform, sample_rate, args.analysis_window_ms)

    row: dict[str, Any] = {
        "label": label,
        "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix() if audio_path.is_relative_to(PROJECT_ROOT) else str(audio_path),
        "word": target.word,
        "phoneme": target.phoneme,
        "duration_ms": round(float(len(waveform) / sample_rate * 1000.0), 3),
        "analysis_window_ms": round(float(len(analysis_waveform) / sample_rate * 1000.0), 3),
    }
    row.update(compute_energy_features(analysis_waveform, sample_rate, args.energy_frame_ms, args.energy_hop_ms))
    row.update(compute_mel_band_features(analysis_waveform, sample_rate, args.n_mels))
    row.update(compute_mfcc_transition_features(analysis_waveform, sample_rate, args.n_mfcc))
    return row


def summarize_by_label_and_word(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    numeric_keys = [key for key in rows[0].keys() if key not in {"label", "audio_path", "word", "phoneme"}] if rows else []
    for row in rows:
        grouped[f"{row['label']}:{row['word']}:{row['phoneme']}"].append(row)

    summary: dict[str, Any] = {}
    for group_key, group_rows in grouped.items():
        summary[group_key] = {"count": len(group_rows)}
        for key in numeric_keys:
            values = [safe_float(row.get(key)) for row in group_rows]
            numeric_values = [value for value in values if value is not None]
            if numeric_values:
                summary[group_key][key] = round(float(mean(numeric_values)), 6)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.source_dir.exists():
        raise FileNotFoundError(f"source dir를 찾을 수 없습니다: {args.source_dir}")

    target_words = load_target_words()
    word_index = build_word_index(target_words)
    audio_files = collect_audio_files(args.source_dir, args.labels)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for label, audio_path in audio_files:
        try:
            target = find_target_by_filename(audio_path, word_index)
            if target.phoneme not in LIQUID_PHONEMES:
                continue
            rows.append(analyze_audio_file(label, audio_path, target, args))
        except Exception as exc:
            errors.append(
                {
                    "label": label,
                    "audio_path": audio_path.relative_to(PROJECT_ROOT).as_posix() if audio_path.is_relative_to(PROJECT_ROOT) else str(audio_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = args.report_dir / f"liquid_feature_analysis_{timestamp}.csv"
    summary_path = args.report_dir / f"liquid_feature_analysis_summary_{timestamp}.json"
    errors_path = args.report_dir / f"liquid_feature_analysis_errors_{timestamp}.csv"

    write_csv(details_path, rows)
    write_csv(errors_path, errors)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(args.source_dir),
        "labels": args.labels,
        "processed_count": len(rows),
        "error_count": len(errors),
        "summary": summarize_by_label_and_word(rows),
        "feature_notes": {
            "energy_v_shape_score": "초반 RMS dip 깊이를 평균 RMS로 정규화한 값입니다. 클수록 짧은 dip/회복이 강합니다.",
            "mel_f3_to_low_ratio": "2000~3000Hz 대역 에너지를 0~1000Hz 대역 에너지로 나눈 값입니다.",
            "transition_mfcc_distance": "0~50ms onset MFCC와 100~180ms vowel MFCC 사이의 거리입니다.",
        },
    }
    summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Liquid feature analysis")
    print(f"processed: {len(rows)}")
    print(f"errors   : {len(errors)}")
    print(f"details  : {details_path}")
    print(f"summary  : {summary_path}")
    if errors:
        print(f"error csv: {errors_path}")


if __name__ == "__main__":
    main()
