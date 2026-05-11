import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.rag import answer

# Questions log — one line per request, rotates at 5 MB, keeps 5 backups
_log_dir = Path(__file__).parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
_q_handler = RotatingFileHandler(
    _log_dir / "questions.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_q_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_q_logger = logging.getLogger("questions")
_q_logger.addHandler(_q_handler)
_q_logger.setLevel(logging.INFO)
_q_logger.propagate = False  # keep question log separate from uvicorn output

app = FastAPI(title="Ask Media Suite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

app.mount("/widget", StaticFiles(directory=Path(__file__).parent.parent / "widget"), name="widget")


class Message(BaseModel):
    role: str
    content: str


class Question(BaseModel):
    question: str
    history: list[Message] = []


@app.post("/ask")
def ask(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    return answer(body.question, history=[m.model_dump() for m in body.history])
