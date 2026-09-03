"""HTTPS adapter for the GitHub AI Video Meeting Assistant pipeline.

Copy this file into the root of the public Python repository together with
requirements-api.txt and run it with uvicorn. The existing core modules remain
the source of truth for ingestion, transcription, summarization, extraction,
and RAG.
"""

import asyncio
import os
import secrets
import time
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

load_dotenv()

from utils.audio_processor import process_input  # noqa: E402
from core.transcriber import transcribe_all  # noqa: E402
from core.summarizer import summarize, generate_title  # noqa: E402
from core.extractor import extract_action_items, extract_key_decisions, extract_questions  # noqa: E402
from core.rag_engine import build_rag_chain, ask_question  # noqa: E402


app = FastAPI(title="AI Video Meeting Assistant API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["content-type", "x-service-key"],
)

SERVICE_KEY = os.getenv("SERVICE_API_KEY")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
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
    expired = [key for key, value in _sessions.items() if now - value["created_at"] > SESSION_TTL_SECONDS]
    for key in expired:
        _sessions.pop(key, None)


def execute_pipeline(source: str, language: str) -> dict[str, Any]:
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language)
    rag_chain = build_rag_chain(transcript)
    return {
        "title": generate_title(transcript),
        "transcript": transcript,
        "summary": summarize(transcript),
        "action_items": extract_action_items(transcript),
        "key_decisions": extract_key_decisions(transcript),
        "open_questions": extract_questions(transcript),
        "rag_chain": rag_chain,
    }


@app.get("/health")
def health(_: None = Depends(require_service_key)) -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/meetings/analyze")
async def analyze(request: AnalyzeRequest, _: None = Depends(require_service_key)) -> dict[str, Any]:
    purge_sessions()
    result = await asyncio.to_thread(execute_pipeline, str(request.sourceUrl), request.language)
    meeting_id = uuid4().hex
    _sessions[meeting_id] = {"created_at": time.time(), "rag_chain": result.pop("rag_chain")}
    result["meetingId"] = meeting_id
    return result


@app.post("/v1/meetings/{meeting_id}/ask")
async def ask(meeting_id: str, request: AskRequest, _: None = Depends(require_service_key)) -> dict[str, str]:
    purge_sessions()
    session = _sessions.get(meeting_id)
    if not session:
        raise HTTPException(status_code=404, detail="Meeting session not found or expired")
    answer = await asyncio.to_thread(ask_question, session["rag_chain"], request.question)
    return {"answer": answer}
