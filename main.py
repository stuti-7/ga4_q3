from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Chunk(BaseModel):
    chunk_id: str
    text: str

class Request(BaseModel):
    question: str
    chunks: list[Chunk]


def tokenize(s):
    return set(re.findall(r"\b[a-z0-9_]+\b", s.lower()))


@app.post("/grounded-answer")
def grounded_answer(req: Request):

    if not req.chunks:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }

    q = tokenize(req.question)

    best_chunk = None
    best_score = -1

    for chunk in req.chunks:

        words = tokenize(chunk.text)

        score = len(q & words)

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None or best_score == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }

    confidence = min(0.99, 0.5 + best_score / max(len(q), 1))

    return {
        "answer": best_chunk.text,
        "citations": [best_chunk.chunk_id],
        "confidence": round(confidence, 2),
        "answerable": True
    }