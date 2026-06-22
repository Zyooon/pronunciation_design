"""단어별 target phoneme 위치 메타데이터를 관리하는 모듈.

words.txt를 로드해 각 단어의 타겟 음소가 onset, nucleus, coda 중 어디에 해당하는지 조회한다.
조회한 위치 정보는 onset feature 추출 여부와 분석용 metadata 연결에 사용된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORDS_PATH = PROJECT_ROOT / "data" / "words.txt"

PhonemePosition = Literal["onset", "nucleus", "coda", "unknown"]

_DEFAULT_ONSET_PHONEMES: frozenset[str] = frozenset({"r", "l", "f", "v", "θ", "ə"})


@dataclass(frozen=True)
class WordTarget:
    word: str
    target_phoneme: str
    phoneme_position: PhonemePosition

    @property
    def target_id(self) -> str:
        return build_target_key(self.word, self.target_phoneme)


def build_target_key(word: str, phoneme: str) -> str:
    return f"{word.lower().strip()}|{phoneme.strip()}"


def infer_default_phoneme_position(phoneme: str) -> PhonemePosition:
    """4번째 컬럼이 없는 구버전 words.txt 줄에 대한 기본값 추론."""
    return "onset" if phoneme in _DEFAULT_ONSET_PHONEMES else "nucleus"


def _parse_phoneme_position(raw: str) -> PhonemePosition:
    value = raw.strip()
    if value in ("onset", "nucleus", "coda"):
        return value  # type: ignore[return-value]
    return "unknown"


def _parse_word_target_line(line: str) -> WordTarget | None:
    if not line or line.startswith("#"):
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return None
    word, phoneme = parts[0], parts[2]
    if not word or not phoneme:
        return None
    phoneme_position = _parse_phoneme_position(parts[3]) if len(parts) >= 4 else infer_default_phoneme_position(phoneme)
    return WordTarget(word=word, target_phoneme=phoneme, phoneme_position=phoneme_position)


def load_word_targets(path: str | Path = DEFAULT_WORDS_PATH) -> dict[str, WordTarget]:
    """words.txt를 읽어 단어별 음소 위치 메타데이터를 반환한다."""
    path = Path(path)
    if not path.exists():
        return {}
    targets: dict[str, WordTarget] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            target = _parse_word_target_line(line.strip())
            if target is None:
                continue
            targets[build_target_key(target.word, target.target_phoneme)] = target
    return targets


def get_word_target(
    word: str,
    phoneme: str,
    targets: dict[str, WordTarget] | None = None,
) -> WordTarget | None:
    targets = load_word_targets() if targets is None else targets
    return targets.get(build_target_key(word, phoneme))


def should_extract_onset(
    word: str,
    phoneme: str,
    targets: dict[str, WordTarget] | None = None,
) -> bool:
    target = get_word_target(word, phoneme, targets)
    return target is not None and target.phoneme_position == "onset"


def attach_word_target_features(
    features: dict[str, object],
    word: str,
    phoneme: str,
    targets: dict[str, WordTarget] | None = None,
) -> None:
    target = get_word_target(word, phoneme, targets)
    if target is None:
        return
    features["target_position"] = target.phoneme_position
    features["target_phoneme"] = target.target_phoneme
    features["target_id"] = target.target_id
