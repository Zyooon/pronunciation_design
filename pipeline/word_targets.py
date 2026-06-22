"""단어별 target phoneme 위치 메타데이터를 관리하는 모듈.

word_targets.json을 로드해 각 단어의 타겟 음소가 onset, nucleus, coda, full 중 어디에 해당하는지 조회한다.
조회한 위치 정보는 onset feature 추출 여부와 분석용 metadata 연결에 사용된다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORD_TARGETS_PATH = PROJECT_ROOT / "data" / "word_targets.json"
TargetPosition = Literal["onset", "nucleus", "coda", "full"]


@dataclass(frozen=True)
class WordTarget:
    word: str
    target_phoneme: str
    position: TargetPosition

    @property
    def target_id(self) -> str:
        return build_target_key(self.word, self.target_phoneme)


def build_target_key(word: str, phoneme: str) -> str:
    return f"{word.lower().strip()}|{phoneme.strip()}"


def _parse_word_target(key: str, payload: dict[str, str]) -> WordTarget:
    word = payload.get("word")
    target_phoneme = payload.get("target_phoneme")
    position = payload.get("position")
    if not word or not target_phoneme or position not in ("onset", "nucleus", "coda", "full"):
        raise ValueError(f"Invalid word target metadata: {key}")
    return WordTarget(
        word=word,
        target_phoneme=target_phoneme,
        position=position,  # type: ignore[arg-type]
    )


def load_word_targets(path: str | Path = DEFAULT_WORD_TARGETS_PATH) -> dict[str, WordTarget]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw_targets = json.load(f)
    return {key: _parse_word_target(key, payload) for key, payload in raw_targets.items()}


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
    return target is not None and target.position == "onset"


def attach_word_target_features(
    features: dict[str, object],
    word: str,
    phoneme: str,
    targets: dict[str, WordTarget] | None = None,
) -> None:
    target = get_word_target(word, phoneme, targets)
    if target is None:
        return
    features["target_position"] = target.position
    features["target_phoneme"] = target.target_phoneme
    features["target_id"] = target.target_id
