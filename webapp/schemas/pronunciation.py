from dataclasses import dataclass


@dataclass(frozen=True)
class WordDto:
    """단어 목록 응답용 DTO."""

    word: str
    korean_pronunciation: str
    phoneme: str

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "korean_pronunciation": self.korean_pronunciation,
            "phoneme": self.phoneme,
        }
