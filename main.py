from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.routes import pages, pronunciation_api

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(title="PronounceAI")

app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")

templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")

app.include_router(pages.router)
app.include_router(pronunciation_api.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
