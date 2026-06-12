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


@dataclass(frozen=True)
class AnalysisResultDto:
    """발음 분석 결과 응답용 DTO."""

    word: str
    phoneme: str
    score: float
    feedback: str
    details: dict

    @classmethod
    def of(
        cls,
        word: str,
        phoneme: str,
        score_result: dict,
    ) -> "AnalysisResultDto":
        return cls(
            word=word,
            phoneme=phoneme,
            score=score_result["score"],
            feedback=score_result["feedback"],
            details=score_result.get("details", {}),
        )

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "phoneme": self.phoneme,
            "score": self.score,
            "feedback": self.feedback,
            "details": self.details,
        }
