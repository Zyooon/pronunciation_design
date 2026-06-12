import logging
from pathlib import Path

from pipeline.audio import load_trimmed_audio
from pipeline.features import extract_features
from pipeline.reference import load_reference_vectors
from pipeline.scorer import score_pronunciation
from webapp.schemas.pronunciation import AnalysisResultDto, WordDto

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORDS_PATH = PROJECT_ROOT / "data" / "words.txt"

# reference_vectors.json은 크기가 크므로 프로세스 당 한 번만 로드한다.
_reference_cache: dict | None = None


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

    Args:
        word: 타겟 영어 단어
        phoneme: 타겟 음소
        audio_path: 임시 저장된 오디오 파일 경로

    Returns:
        AnalysisResultDto

    Raises:
        KeyError: 해당 음소의 reference vector가 없을 때
        FileNotFoundError: reference_vectors.json 또는 오디오 파일이 없을 때
        ValueError: 오디오가 비어 있을 때
    """
    reference_vectors = _get_reference_vectors()

    if phoneme not in reference_vectors:
        raise KeyError(
            f"'{phoneme}' 발음의 레퍼런스 데이터가 없습니다. "
            "scripts/build_reference.py를 먼저 실행해주세요."
        )

    reference = reference_vectors[phoneme]
    waveform, sr = load_trimmed_audio(audio_path)
    features = extract_features(waveform, sr)
    score_result = score_pronunciation(
        user_features=features,
        reference=reference,
        phoneme=phoneme,
    )

    return AnalysisResultDto.of(
        word=word,
        phoneme=phoneme,
        score_result=score_result,
    )


def _get_reference_vectors() -> dict:
    """reference_vectors.json을 캐시해서 반환한다."""
    global _reference_cache
    if _reference_cache is None:
        _reference_cache = load_reference_vectors()
    return _reference_cache
