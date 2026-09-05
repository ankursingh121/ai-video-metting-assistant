"""Low-memory FastAPI adapter for the AI Video Meeting Assistant.

This service deliberately avoids importing Whisper, Torch, LangChain, ChromaDB,
or sentence-transformers at startup. Audio is downloaded/chunked with the
existing utility module, transcription is delegated to Sarvam, and meeting
analysis/Q&A are delegated to Mistral through server-side credentials.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from typing import Any
from uuid import uuid4

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

load_dotenv()

from utils.audio_processor import process_input  # noqa: E402

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
SARVAM_URL = "https://api.sarvam.ai/speech-to-text-translate"
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
SERVICE_KEY = os.getenv("SERVICE_API_KEY")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

app = FastAPI(title="AI Video Meeting Assistant API", version="2.0.0")
raw_origins = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    sourceUrl: HttpUrl
    language: str = Field(default="english", pattern="^(english|hinglish)$")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


def require_service_key(x_service_key: str | None = Header(default=None)) -> None:
    if SERVICE_KEY and not x_service_key:
        raise HTTPException(status_code=401, detail="Missing service key")
    if SERVICE_KEY and not secrets.compare_digest(x_service_key or "", SERVICE_KEY):
        raise HTTPException(status_code=401, detail="Invalid service key")


def purge_sessions() -> None:
    now = time.time()
    for meeting_id, session in list(_sessions.items()):
        if now - session["created_at"] > SESSION_TTL_SECONDS:
            _sessions.pop(meeting_id, None)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value


def _mistral(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    key = _require_env("MISTRAL_API_KEY")
    response = requests.post(
        MISTRAL_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": MISTRAL_MODEL,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=180,
    )
    if not response.ok:
        raise RuntimeError(f"Mistral request failed ({response.status_code})")
    body = response.json()
    return body["choices"][0]["message"]["content"].strip()


def _sarvam_piece(piece_path: str) -> str:
    key = _require_env("SARVAM_API_KEY")
    with open(piece_path, "rb") as audio_file:
        response = requests.post(
            SARVAM_URL,
            headers={"api-subscription-key": key},
            files={"file": (os.path.basename(piece_path), audio_file, "audio/wav")},
            data={"model": SARVAM_MODEL, "with_diarization": "false"},
            timeout=180,
        )
    if not response.ok:
        raise RuntimeError(f"Sarvam request failed ({response.status_code})")
    return response.json().get("transcript", "").strip()


def transcribe_chunks(chunks: list[str]) -> str:
    """Use Sarvam for all languages so the container never loads Whisper."""
    from pydub import AudioSegment

    piece_ms = 25 * 1000
    transcript_parts: list[str] = []
    for chunk_path in chunks:
        audio = AudioSegment.from_wav(chunk_path)
        for start in range(0, len(audio), piece_ms):
            piece = audio[start : start + piece_ms]
            piece_path = f"{chunk_path}_remote_{start}.wav"
            try:
                piece.export(piece_path, format="wav")
                text = _sarvam_piece(piece_path)
                if text:
                    transcript_parts.append(text)
            finally:
                if os.path.exists(piece_path):
                    os.remove(piece_path)
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
    return " ".join(transcript_parts).strip()


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(cleaned[start : end + 1])
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def analyze_transcript(transcript: str) -> dict[str, str]:
    bounded = transcript[:24000]
    analysis_text = _mistral(
        "You are a precise meeting analyst. Return valid JSON only.",
        """Analyze this meeting transcript. Return one JSON object with exactly these string keys:
summary, action_items, key_decisions, open_questions.
Use concise numbered or bulleted text inside each string. Include owners/deadlines when present. If a category is absent, write a short statement saying none were found.

TRANSCRIPT:
""" + bounded,
    )
    data = _parse_json(analysis_text)
    title = _mistral(
        "You create concise professional meeting titles. Return only the title, no punctuation or explanation.",
        "Create a title of at most 8 words for this meeting:\n" + bounded[:5000],
        temperature=0.1,
    )
    return {
        "title": title[:120] or "Untitled meeting",
        "summary": str(data.get("summary") or "No summary returned."),
        "action_items": str(data.get("action_items") or "No action items found."),
        "key_decisions": str(data.get("key_decisions") or "No key decisions found."),
        "open_questions": str(data.get("open_questions") or "No open questions found."),
    }


def relevant_context(transcript: str, question: str) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", transcript) if part.strip()]
    terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9']+", question) if len(term) > 2}
    ranked = sorted(sentences, key=lambda sentence: sum(term in sentence.lower() for term in terms), reverse=True)
    selected = ranked[:24]
    return "\n".join(selected)[:12000] or transcript[:12000]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "remote-lightweight"}


@app.post("/v1/meetings/analyze")
def analyze(request: AnalyzeRequest, _: None = Depends(require_service_key)) -> dict[str, Any]:
    purge_sessions()
    chunks = process_input(str(request.sourceUrl))
    transcript = transcribe_chunks(chunks)
    if not transcript:
        raise HTTPException(status_code=422, detail="No speech was detected in the source")
    analysis = analyze_transcript(transcript)
    meeting_id = uuid4().hex
    _sessions[meeting_id] = {"created_at": time.time(), "transcript": transcript}
    return {"meetingId": meeting_id, "transcript": transcript, **analysis}


@app.post("/v1/meetings/{meeting_id}/ask")
def ask(meeting_id: str, request: AskRequest, _: None = Depends(require_service_key)) -> dict[str, str]:
    purge_sessions()
    session = _sessions.get(meeting_id)
    if not session:
        raise HTTPException(status_code=404, detail="Meeting session not found or expired")
    context = relevant_context(session["transcript"], request.question)
    answer = _mistral(
        "You are a meeting assistant. Answer only from the supplied transcript context. If the answer is absent, say: I could not find this information in the meeting transcript. Be concise.",
        f"CONTEXT:\n{context}\n\nQUESTION:\n{request.question}",
        temperature=0.2,
    )
    return {"answer": answer}
