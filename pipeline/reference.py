import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WORDS_PATH = PROJECT_ROOT / "words.txt"
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "data" / "reference_vectors.json"


@dataclass(frozen=True)
class TargetWord:
    """
    사용자가 발음할 타겟 단어 정보입니다.

    같은 영어 단어가 여러 음소 테스트에 쓰일 수 있으므로
    word만 고유 key로 쓰지 않고 target_id를 따로 만듭니다.

    예:
        lid,i   -> target_id = "lid|i"
        lid,l   -> target_id = "lid|l"
    """

    word: str
    korean_pronunciation: str
    phoneme: str

    @property
    def target_id(self) -> str:
        return f"{self.word}|{self.phoneme}"

    @property
    def display_label(self) -> str:
        return f"{self.word} /{self.phoneme}/"


def load_reference_vectors(reference_path: str | Path = DEFAULT_REFERENCE_PATH) -> dict[str, dict]:
    """
    data/reference_vectors.json을 로드합니다.

    Returns:
        {
            "θ": {...},
            "i": {...},
            ...
        }
    """
    reference_path = Path(reference_path)

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference vector file not found: {reference_path}")

    with reference_path.open("r", encoding="utf-8") as f:
        reference_vectors = json.load(f)

    if not reference_vectors:
        raise ValueError(
            "Reference vector file is empty. "
            "Run: uv run python scripts\\build_reference.py --include-all"
        )

    return reference_vectors


def load_target_words(words_path: str | Path = DEFAULT_WORDS_PATH) -> list[TargetWord]:
    """
    words.txt를 읽어서 TargetWord 목록으로 변환합니다.

    words.txt 형식:
        영어단어,한국어발음,타겟음소

    주석(#)으로 시작하는 줄과 빈 줄은 무시합니다.
    """
    words_path = Path(words_path)

    if not words_path.exists():
        raise FileNotFoundError(f"words.txt not found: {words_path}")

    targets: list[TargetWord] = []

    with words_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid words.txt format at line {line_number}: {line}\n"
                    "Expected: word,korean_pronunciation,phoneme"
                )

            word, korean_pronunciation, phoneme = parts

            targets.append(
                TargetWord(
                    word=word,
                    korean_pronunciation=korean_pronunciation,
                    phoneme=phoneme,
                )
            )

    return targets


def build_target_index(target_words: list[TargetWord]) -> dict[str, TargetWord]:
    """
    target_id 기준으로 TargetWord를 빠르게 찾기 위한 dict를 만듭니다.

    Returns:
        {
            "think|θ": TargetWord(...),
            "ship|i": TargetWord(...),
            "sheep|iː": TargetWord(...),
        }
    """
    index: dict[str, TargetWord] = {}

    for target in target_words:
        if target.target_id in index:
            raise ValueError(f"Duplicated target_id found: {target.target_id}")

        index[target.target_id] = target

    return index


def get_available_targets(
    words_path: str | Path = DEFAULT_WORDS_PATH,
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
) -> list[TargetWord]:
    """
    reference vector가 존재하는 음소만 타겟 목록으로 반환합니다.

    예:
        reference_vectors.json에 θ, i, iː만 있으면
        words.txt 전체 중 θ, i, iː 단어만 반환합니다.
    """
    reference_vectors = load_reference_vectors(reference_path)
    target_words = load_target_words(words_path)

    available_phonemes = set(reference_vectors.keys())

    return [target for target in target_words if target.phoneme in available_phonemes]


def get_reference_for_target(
    target: TargetWord,
    reference_vectors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    특정 타겟 단어의 음소에 해당하는 reference vector를 가져옵니다.
    """
    if target.phoneme not in reference_vectors:
        raise KeyError(f"No reference vector found for phoneme: {target.phoneme}")

    return reference_vectors[target.phoneme]


def get_reference_for_target_id(
    target_id: str,
    target_index: dict[str, TargetWord],
    reference_vectors: dict[str, dict[str, Any]],
) -> tuple[TargetWord, dict[str, Any]]:
    """
    app.py에서 선택된 target_id를 받아 TargetWord와 reference vector를 함께 반환합니다.
    """
    if target_id not in target_index:
        raise KeyError(f"Unknown target_id: {target_id}")

    target = target_index[target_id]
    reference = get_reference_for_target(target, reference_vectors)

    return target, reference


def get_gradio_choices(target_words: list[TargetWord]) -> list[tuple[str, str]]:
    """
    Gradio Dropdown에서 사용할 choices를 만듭니다.

    Gradio는 보통:
        (화면에 보이는 label, 실제 value)
    형태의 tuple 목록을 받을 수 있습니다.

    예:
        ("think /θ/", "think|θ")
    """
    return [(target.display_label, target.target_id) for target in target_words]