from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.rag import answer

app = FastAPI(title="Ask Media Suite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


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
