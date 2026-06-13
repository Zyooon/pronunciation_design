import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from pipeline.db import save_user_recording_result
from webapp.services.pronunciation_facade import analyze_audio, load_word_list

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

_RECORD_SAVE_DIR = Path("data/reference_ko/record")


@router.get("/words")
async def get_words() -> Response:
    """data/words.txt를 읽어 단어 목록을 JSON으로 반환한다."""
    words = load_word_list()
    body = json.dumps([w.to_dict() for w in words], ensure_ascii=False)
    return Response(content=body, media_type="application/json; charset=utf-8")


@router.post("/pronunciation/analyze")
async def analyze_pronunciation(
    word: str = Form(...),
    phoneme: str = Form(...),
    audio_file: UploadFile = File(...),
) -> Response:
    """업로드된 음성 파일을 분석하고 채점 결과를 반환한다."""
    word = word.strip()
    phoneme = phoneme.strip()

    if not word:
        return _error_response(400, "word 파라미터가 비어 있습니다.")
    if not phoneme:
        return _error_response(400, "phoneme 파라미터가 비어 있습니다.")

    audio_bytes = await audio_file.read()
    if not audio_bytes:
        return _error_response(400, "음성 파일이 비어 있습니다.")

    suffix = _safe_suffix(audio_file.filename)
    save_path = _save_record_audio(audio_bytes, word, suffix)

    try:
        result = analyze_audio(word=word, phoneme=phoneme, audio_path=save_path)

        save_user_recording_result(
            word=word,
            phoneme=phoneme,
            score=result.score,
            grade=_derive_grade(result.score),
            feedback=result.feedback,
            recording_path=str(save_path),
        )

        body = json.dumps(result.to_dict(), ensure_ascii=False)
        return Response(content=body, media_type="application/json; charset=utf-8")

    except FileNotFoundError as e:
        log.warning("reference 파일 없음: word=%s, phoneme=%s, error=%s", word, phoneme, e)
        return _error_response(
            503,
            "레퍼런스 데이터를 찾을 수 없습니다. 서버 설정을 확인해주세요.",
        )
    except KeyError as e:
        log.warning("reference 없음: word=%s, phoneme=%s, error=%s", word, phoneme, e)
        return _error_response(
            422,
            f"'{phoneme}' 발음의 레퍼런스 데이터가 없습니다. "
            "지원하는 발음인지 확인해주세요.",
        )
    except ValueError as e:
        log.warning("오디오 오류: word=%s, phoneme=%s, error=%s", word, phoneme, e)
        return _error_response(422, f"오디오 처리 중 오류가 발생했습니다: {e}")
    except Exception as e:
        log.error("분석 실패: word=%s, phoneme=%s", word, phoneme, exc_info=True)
        return _error_response(
            500,
            "분석 중 예상치 못한 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )


# ── 헬퍼 ────────────────────────────────────────────────────────────────────

def _derive_grade(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    return "Needs Practice"


def _save_record_audio(data: bytes, word: str, suffix: str) -> Path:
    """오디오 bytes를 data/reference_ko/record/{word}{suffix}로 저장하고 경로를 반환한다."""
    _RECORD_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _RECORD_SAVE_DIR / f"{word}{suffix}"
    save_path.write_bytes(data)
    return save_path


def _safe_suffix(filename: str | None) -> str:
    """업로드 파일명에서 확장자를 추출한다. 없으면 .webm을 반환한다."""
    if not filename:
        return ".webm"
    suffix = Path(filename).suffix.lower()
    allowed = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
    return suffix if suffix in allowed else ".webm"


def _error_response(status_code: int, message: str) -> Response:
    body = json.dumps({"detail": message}, ensure_ascii=False)
    return Response(
        content=body,
        status_code=status_code,
        media_type="application/json; charset=utf-8",
    )
