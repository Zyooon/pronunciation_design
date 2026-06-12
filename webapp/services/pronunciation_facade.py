import logging
from pathlib import Path

from webapp.schemas.pronunciation import WordDto

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORDS_PATH = PROJECT_ROOT / "data" / "words.txt"


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
