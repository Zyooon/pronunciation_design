import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.routes import pages, pronunciation_api

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(title="PronounceAI")

app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")

templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")

app.include_router(pages.router)
app.include_router(pronunciation_api.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
    """FastAPI 기본 422 오류를 프론트가 읽기 쉬운 형태로 변환한다."""
    missing = [e["loc"][-1] for e in exc.errors() if e.get("type") == "missing"]
    if missing:
        message = f"필수 항목이 누락됐습니다: {', '.join(str(f) for f in missing)}"
    else:
        message = "요청 형식이 올바르지 않습니다."
    body = json.dumps({"detail": message}, ensure_ascii=False)
    return Response(content=body, status_code=422, media_type="application/json; charset=utf-8")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
