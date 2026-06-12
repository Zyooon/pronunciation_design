import json

from fastapi import APIRouter
from fastapi.responses import Response

from webapp.services.pronunciation_facade import load_word_list

router = APIRouter(prefix="/api")


@router.get("/words")
async def get_words() -> Response:
    """data/words.txt를 읽어 단어 목록을 JSON으로 반환한다."""
    words = load_word_list()
    body = json.dumps([w.to_dict() for w in words], ensure_ascii=False)
    return Response(content=body, media_type="application/json; charset=utf-8")
