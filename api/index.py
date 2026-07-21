import os
import json
import re
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.sanand.workers.dev/openai/v1/chat/completions"
MODEL = "gpt-4o-mini"


def fallback_response():
    return {
        "answer": "I don't know",
        "citations": [],
        "confidence": 0.2,
        "answerable": False,
    }


def build_prompt(question, chunks):
    context_text = "\n".join([f'[{c["chunk_id"]}]: {c["text"]}' for c in chunks])
    system_prompt = (
        "You are a strict grounded question-answering engine for medical/legal compliance.\n"
        "RULES (must follow exactly):\n"
        "1. Answer ONLY using facts explicitly present in the given context chunks.\n"
        "2. NEVER use outside knowledge, guessing, or inference beyond the text.\n"
        "3. If the answer is not clearly present in the chunks, you MUST set "
        "answerable=false, answer='I don't know', citations=[], confidence<=0.3.\n"
        "4. citations must ONLY contain chunk_id values that were actually provided.\n"
        "5. confidence is a float 0-1 representing how certain you are the answer is "
        "correct and fully supported by the cited chunks.\n"
        "6. Output ONLY valid JSON, no markdown, no explanation, in this exact schema:\n"
        '{"answer": "...", "citations": ["C1"], "confidence": 0.9, "answerable": true}\n'
    )
    user_prompt = f"Context chunks:\n{context_text}\n\nQuestion: {question}\n\nReturn only the JSON."
    return system_prompt, user_prompt


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def sanitize(result, valid_ids):
    if not isinstance(result, dict):
        return fallback_response()

    answerable = bool(result.get("answerable", False))
    answer = str(result.get("answer", "")).strip()
    citations = result.get("citations", [])
    confidence = result.get("confidence", 0.0)

    if not isinstance(citations, list):
        citations = []
    citations = [c for c in citations if c in valid_ids]

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if not answerable or answer.lower().strip() in ("", "i don't know", "i dont know"):
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": min(confidence, 0.3),
            "answerable": False,
        }

    if not citations:
        # answerable claimed but no valid citations -> force unanswerable (safety net)
        return fallback_response()

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "answerable": True,
    }


@app.post("/api/answer")
async def answer(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(fallback_response(), status_code=200)

    question = body.get("question")
    chunks = body.get("chunks")

    if not question or not isinstance(chunks, list) or len(chunks) == 0:
        return JSONResponse(fallback_response(), status_code=200)

    valid_ids = set()
    clean_chunks = []
    for c in chunks:
        if isinstance(c, dict) and "chunk_id" in c and "text" in c:
            valid_ids.add(c["chunk_id"])
            clean_chunks.append(c)

    if not clean_chunks:
        return JSONResponse(fallback_response(), status_code=200)

    system_prompt, user_prompt = build_prompt(question, clean_chunks)

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(
                AIPIPE_URL,
                headers={
                    "Authorization": f"Bearer {AIPIPE_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                },
            )
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        parsed = extract_json(raw_text)
        result = sanitize(parsed, valid_ids)
    except Exception:
        result = fallback_response()

    return JSONResponse(result, status_code=200)


@app.get("/api/answer")
async def health():
    return {"status": "ok"}