from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "what", "when", "where", "who", "why", "with"
}


class Chunk(BaseModel):
    chunk_id: str
    text: str


class Request(BaseModel):
    question: str = ""
    chunks: list[Chunk] = Field(default_factory=list)


def not_answerable_response():
    return {
        "answer": "I don't know",
        "citations": [],
        "confidence": 0.0,
        "answerable": False,
    }


def tokenize(s: str):
    if not isinstance(s, str):
        return set()

    tokens = re.findall(r"\b[a-z0-9_]+\b", s.lower())
    return {token for token in tokens if token not in STOP_WORDS and len(token) > 1}


@app.post("/grounded-answer")
def grounded_answer(req: Request):
    if not isinstance(req.question, str) or not req.question.strip():
        return not_answerable_response()

    if not isinstance(req.chunks, list) or not req.chunks:
        return not_answerable_response()

    question_tokens = tokenize(req.question)
    if not question_tokens:
        return not_answerable_response()

    best_chunk = None
    best_score = 0
    best_overlap_ratio = 0.0

    for chunk in req.chunks:
        if not isinstance(chunk, Chunk):
            continue

        chunk_tokens = tokenize(chunk.text)
        if not chunk_tokens:
            continue

        overlap = len(question_tokens & chunk_tokens)
        overlap_ratio = overlap / max(len(question_tokens), 1)

        if overlap > best_score or (overlap == best_score and overlap_ratio > best_overlap_ratio):
            best_score = overlap
            best_overlap_ratio = overlap_ratio
            best_chunk = chunk

    if best_chunk is None or best_score < 2 or best_overlap_ratio < 0.4:
        return not_answerable_response()

    confidence = min(0.3, round(0.15 + best_overlap_ratio * 0.15 + (best_score / max(len(question_tokens), 1)) * 0.1, 2))

    return {
        "answer": best_chunk.text,
        "citations": [best_chunk.chunk_id],
        "confidence": confidence,
        "answerable": True,
    }