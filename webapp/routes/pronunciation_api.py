import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from pipeline.db import save_user_recording_result
from webapp.services.pronunciation_facade import analyze_audio, load_word_list

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

_RECORD_SAVE_DIR = Path("data/reference_ko/record")

ALLOWED_TEST_LABELS: frozenset[str] = frozenset(
    {"unlabeled", "good", "korean_like", "wrong_or_noisy"}
)


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
    test_label: str = Form("unlabeled"),
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
    normalized_label = normalize_test_label(test_label, enabled=_is_test_labels_enabled())
    save_path = _save_record_audio(audio_bytes, word, suffix, test_label=normalized_label)

    try:
        result = analyze_audio(word=word, phoneme=phoneme, audio_path=save_path)

        save_user_recording_result(
            word=word,
            phoneme=phoneme,
            score=result.score,
            grade=_derive_grade(result.score),
            feedback=result.feedback,
            recording_path=str(save_path),
            test_label=normalized_label,
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

def _is_test_labels_enabled() -> bool:
    return os.getenv("ENABLE_TEST_LABELS", "true").lower() == "true"


def normalize_test_label(value: str | None, *, enabled: bool) -> str | None:
    """test_label을 정규화한다.

    enabled=False이면 무조건 None을 반환한다.
    enabled=True이고 허용 값이면 그대로 반환, 아니면 'unlabeled'로 처리한다.
    """
    if not enabled:
        return None
    if value in ALLOWED_TEST_LABELS:
        return value
    return "unlabeled"


def _derive_grade(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    return "Needs Practice"


def _safe_word_for_filename(word: str) -> str:
    """단어에서 파일명에 안전한 문자열을 만든다."""
    return re.sub(r"[^\w-]", "_", word).strip("_") or "word"


def _save_record_audio(
    data: bytes,
    word: str,
    suffix: str,
    test_label: str | None = None,
) -> Path:
    """오디오 bytes를 라벨 기반 경로에 저장하고 경로를 반환한다.

    ENABLE_TEST_LABELS=false일 때도 unlabeled/ 하위 폴더에 저장해
    폴더 구조를 항상 일관되게 유지한다.
    """
    label = test_label if test_label in ALLOWED_TEST_LABELS else "unlabeled"
    save_dir = _RECORD_SAVE_DIR / label
    save_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_word = _safe_word_for_filename(word)
    filename = f"{ts}_{safe_word}_{label}{suffix}"

    save_path = save_dir / filename
    save_path.write_bytes(data)
    return save_path


def _safe_suffix(filename: str | None) -> str:
    """업로드 파일명에서 확장자를 추출한다. 없으면 .wav를 반환한다."""
    if not filename:
        return ".wav"
    suffix = Path(filename).suffix.lower()
    allowed = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
    return suffix if suffix in allowed else ".wav"


def _error_response(status_code: int, message: str) -> Response:
    body = json.dumps({"detail": message}, ensure_ascii=False)
    return Response(
        content=body,
        status_code=status_code,
        media_type="application/json; charset=utf-8",
    )
