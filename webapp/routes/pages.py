import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parents[2]

router = APIRouter()
templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")


def _get_enable_test_labels() -> bool:
    value = os.getenv("ENABLE_TEST_LABELS", "true").strip().lower()
    return value not in {"false", "0", "no", "off"}


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """메인 발음 연습 화면을 렌더링한다."""
    enable_test_labels = _get_enable_test_labels()
    return templates.TemplateResponse(
        request, "index.html",
        {"enable_test_labels": enable_test_labels},
    )
