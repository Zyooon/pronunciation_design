import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.audio import load_trimmed_audio
from pipeline.features import extract_features
from pipeline.reference import load_reference_vectors
from pipeline.scorer import score_pronunciation
from webapp.schemas.pronunciation import AnalysisResultDto, WordDto

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioFeaturesSnapshot:
    """분석 pipeline에서 추출한 원시 feature 값 모음."""

    duration_ms: float | None
    rms_mean: float | None
    zcr_mean: float | None
    spectral_centroid_mean: float | None
    mfcc_distance: float | None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORDS_PATH = PROJECT_ROOT / "data" / "words.txt"
KO_REFERENCE_PATH = PROJECT_ROOT / "data" / "ko_reference_vectors.json"

# reference_vectors.json은 크기가 크므로 프로세스 당 한 번만 로드한다.
_reference_cache: dict | None = None
_ko_reference_cache: dict | None = None


def load_word_list() -> list[WordDto]:
    """words.txt를 읽어 WordDto 목록을 반환한다.

    빈 줄과 # 주석 줄은 무시한다.
    필드가 3개가 아닌 줄은 건너뛴다.
    """
    if not WORDS_PATH.exists():
        log.error("words.txt not found: %s", WORDS_PATH)
        return []

    words: list[WordDto] = []

    with WORDS_PATH.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                log.warning("잘못된 줄 건너뜀: line=%s, content=%s", line_number, line)
                continue

            word, korean_pronunciation, phoneme = parts
            words.append(WordDto(
                word=word,
                korean_pronunciation=korean_pronunciation,
                phoneme=phoneme,
            ))

    return words


def analyze_audio(word: str, phoneme: str, audio_path: Path) -> AnalysisResultDto:
    """음성 파일을 pipeline으로 분석하고 결과를 반환한다.

    Raises:
        KeyError: 해당 음소의 reference vector가 없을 때
        FileNotFoundError: reference_vectors.json 또는 오디오 파일이 없을 때
        ValueError: 오디오가 비어 있을 때
    """
    result, _ = analyze_audio_with_features(word, phoneme, audio_path)
    return result


def analyze_audio_with_features(
    word: str, phoneme: str, audio_path: Path
) -> tuple[AnalysisResultDto, AudioFeaturesSnapshot]:
    """음성 파일을 분석하고 채점 결과와 원시 feature 스냅샷을 함께 반환한다.

    Raises:
        KeyError: 해당 음소의 reference vector가 없을 때
        FileNotFoundError: reference_vectors.json 또는 오디오 파일이 없을 때
        ValueError: 오디오가 비어 있을 때
    """
    reference_vectors = _get_reference_vectors()
    ko_reference_vectors = _get_ko_reference_vectors()

    if phoneme not in reference_vectors:
        raise KeyError(
            f"'{phoneme}' 발음의 레퍼런스 데이터가 없습니다. "
            "scripts/build_reference.py를 먼저 실행해주세요."
        )

    reference = reference_vectors[phoneme]
    ko_reference = ko_reference_vectors.get(phoneme)
    waveform, sr = load_trimmed_audio(audio_path)
    features = extract_features(waveform, sr)
    score_result = score_pronunciation(
        user_features=features,
        reference=reference,
        phoneme=phoneme,
        ko_reference=ko_reference,
    )

    mfcc_distance = _compute_mfcc_distance(
        user_mfcc=features.get("mfcc_mean"),
        ref_mfcc=reference.get("mfcc_mean"),
    )
    features_snapshot = AudioFeaturesSnapshot(
        duration_ms=features.get("duration_ms"),
        rms_mean=features.get("rms_mean"),
        zcr_mean=features.get("zcr_mean"),
        spectral_centroid_mean=features.get("spectral_centroid_mean"),
        mfcc_distance=mfcc_distance,
    )

    result = AnalysisResultDto.of(word=word, phoneme=phoneme, score_result=score_result)
    return result, features_snapshot


def _compute_mfcc_distance(
    user_mfcc: list[float] | None,
    ref_mfcc: list[float] | None,
) -> float | None:
    """사용자 MFCC 벡터와 reference MFCC 평균 벡터 간의 L2 거리를 계산한다."""
    if user_mfcc is None or ref_mfcc is None:
        return None
    try:
        user_arr = np.array(user_mfcc, dtype=float)
        ref_arr = np.array(ref_mfcc, dtype=float)
        return float(np.linalg.norm(user_arr - ref_arr))
    except Exception:
        return None


def _get_reference_vectors() -> dict:
    """reference_vectors.json을 캐시해서 반환한다."""
    global _reference_cache
    if _reference_cache is None:
        _reference_cache = load_reference_vectors()
    return _reference_cache


def _get_ko_reference_vectors() -> dict:
    """ko_reference_vectors.json을 캐시해서 반환한다. 없으면 기존 scorer만 사용한다."""
    global _ko_reference_cache
    if _ko_reference_cache is None:
        if not KO_REFERENCE_PATH.exists():
            _ko_reference_cache = {}
        else:
            with KO_REFERENCE_PATH.open("r", encoding="utf-8") as f:
                _ko_reference_cache = json.load(f)
    return _ko_reference_cache
