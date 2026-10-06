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
import subprocess
import tempfile
import time
from typing import Any
from uuid import uuid4

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from yt_dlp.utils import DownloadError
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

from utils.audio_processor import chunk_audio, convert_to_wav, process_input  # noqa: E402

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
    sourceType: str = Field(default="youtube", pattern="^(youtube|video|audio)$")
    language: str = Field(default="english", pattern="^(english|hindi|hinglish)$")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    transcript: str | None = Field(default=None, max_length=2_000_000)


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
    payload = {"model": MISTRAL_MODEL, "temperature": temperature, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]}
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(MISTRAL_URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=180)
            if not response.ok:
                raise RuntimeError(f"Mistral request failed ({response.status_code})")
            body = response.json()
            return body["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Mistral request failed after retries: {last_error}") from last_error


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
    bounded = transcript[:14000]
    try:
        analysis_text = _mistral(
            "You are a precise meeting analyst. Return valid JSON only.",
            """Analyze this meeting transcript. Return one JSON object with exactly these string keys:
summary, action_items, key_decisions, open_questions.
Use concise numbered or bulleted text inside each string. Include owners/deadlines when present. If a category is absent, write a short statement saying none were found.

TRANSCRIPT:
""" + bounded,
        )
    except Exception as exc:
        print(f"[analysis] summary generation failed, returning transcript-safe fallback: {type(exc).__name__}")
        return {"title": "Meeting transcript", "summary": "Transcript is ready. AI summary is temporarily unavailable; please retry analysis to generate structured insights.", "action_items": "AI action-item extraction is temporarily unavailable.", "key_decisions": "AI decision extraction is temporarily unavailable.", "open_questions": "AI question extraction is temporarily unavailable."}
    data = _parse_json(analysis_text)
    try:
        title = _mistral("You create concise professional meeting titles. Return only the title, no punctuation or explanation.", "Create a title of at most 8 words for this meeting:\n" + bounded[:3500], temperature=0.1)
    except Exception as exc:
        print(f"[analysis] title generation failed, using fallback: {type(exc).__name__}")
        title = "Meeting analysis"
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


def process_uploaded_url(source: str, source_type: str) -> list[str]:
    """Download a signed storage URL and convert it with FFmpeg/pydub."""
    response = requests.get(source, stream=True, timeout=120)
    response.raise_for_status()
    suffix = ".mp4" if source_type == "video" else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as source_file:
        path = source_file.name
        total = 0
        for part in response.iter_content(chunk_size=1024 * 1024):
            total += len(part)
            if total > 45 * 1024 * 1024:
                raise RuntimeError("Uploaded source exceeds the 45 MB limit")
            source_file.write(part)
    try:
        wav_path = convert_to_wav(path)
        return chunk_audio(wav_path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def process_uploaded_bytes(data: bytes, source_type: str, filename: str) -> list[str]:
    suffix = ".mp4" if source_type == "video" else ".mp3"
    if "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()[:8]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as source_file:
        path = source_file.name
        source_file.write(data)
    wav_path = f"{path}.wav"
    try:
        command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav_path]
        converted = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if converted.returncode != 0 or not os.path.exists(wav_path):
            reason = (converted.stderr or "no audio stream was found").strip().splitlines()[-1][:240]
            raise RuntimeError(f"FFmpeg could not read this media file: {reason}")
        chunks = chunk_audio(wav_path)
        return chunks
    finally:
        if os.path.exists(path):
            os.remove(path)
        if os.path.exists(wav_path):
            os.remove(wav_path)


def youtube_video_id(source: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{6,})", source)
    if not match:
        raise ValueError("Could not extract a YouTube video id from the URL")
    return match.group(1)


def fetch_youtube_captions(source: str, language: str) -> str:
    languages = ["hi", "en"] if language in {"hindi", "hinglish"} else ["en", "hi"]
    fetched = YouTubeTranscriptApi().fetch(youtube_video_id(source), languages=languages)
    return " ".join(snippet.text for snippet in fetched).strip()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "remote-lightweight", "adapterVersion": "4d1dce7", "youtubeRuntime": "deno-ejs", "captionsFallback": "enabled", "multipartUpload": "enabled", "ffmpegDiagnostics": "enabled", "languages": "english-hindi"}


@app.post("/v1/meetings/analyze")
def analyze(request: AnalyzeRequest, _: None = Depends(require_service_key)) -> dict[str, Any]:
    purge_sessions()
    captions_fallback_failed = False
    try:
        transcript = ""
        if request.sourceType == "youtube":
            try:
                chunks = process_input(str(request.sourceUrl))
                transcript = transcribe_chunks(chunks)
            except DownloadError:
                print("[analysis] yt-dlp blocked; trying YouTube captions fallback")
                try:
                    transcript = fetch_youtube_captions(str(request.sourceUrl), request.language)
                except Exception:
                    captions_fallback_failed = True
                    raise
        else:
            chunks = process_uploaded_url(str(request.sourceUrl), request.sourceType)
            transcript = transcribe_chunks(chunks)
        if not transcript:
            raise HTTPException(status_code=422, detail="No speech was detected in the source")
        analysis = analyze_transcript(transcript)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[analysis] source processing failed: {type(exc).__name__}")
        detail = "Uploaded media could not be downloaded or converted. Please retry the upload and ensure it is a supported audio/video file under 45 MB."
        if request.sourceType == "youtube":
            if captions_fallback_failed:
                detail = "YouTube blocked both audio and captions from this server. Upload the video file, or configure a rotating proxy/API for YouTube access."
                raise HTTPException(status_code=502, detail=detail) from exc
            detail = "YouTube could not be downloaded. Confirm the video is public, not age-restricted, and use a direct youtube.com/watch or youtu.be link."
            if isinstance(exc, DownloadError):
                reason = str(exc).lower()
                if "sign in" in reason or "not a bot" in reason or "bot" in reason:
                    detail = "YouTube blocked this server request as automated traffic. Try a different public video or upload the video file instead."
                elif "private" in reason or "members-only" in reason or "login required" in reason:
                    detail = "This YouTube video is private, members-only, or requires login. Use a public video or upload the file instead."
                elif "age" in reason or "confirm your age" in reason:
                    detail = "This YouTube video is age-restricted. Use a non-age-restricted video or upload the file instead."
                elif "unavailable" in reason or "removed" in reason or "not found" in reason:
                    detail = "This YouTube video is unavailable, removed, or blocked in the server region. Try another public video."
                else:
                    detail = "YouTube audio is blocked on this server and captions were not available. Upload the video file instead or use a video with public captions."
        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from exc

    meeting_id = uuid4().hex
    _sessions[meeting_id] = {"created_at": time.time(), "transcript": transcript}
    return {"meetingId": meeting_id, "transcript": transcript, **analysis}


@app.post("/v1/meetings/analyze-upload")
async def analyze_upload(
    file: UploadFile = File(...),
    sourceType: str = Form(...),
    language: str = Form("english"),
    _: None = Depends(require_service_key),
) -> dict[str, Any]:
    if sourceType not in {"video", "audio"}:
        raise HTTPException(status_code=400, detail="sourceType must be video or audio")
    data = await file.read()
    if not data or len(data) > 45 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded media must be smaller than 45 MB")
    try:
        chunks = process_uploaded_bytes(data, sourceType, file.filename or "meeting-source")
    except Exception as exc:
        print(f"[analysis-upload] conversion failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=422, detail="Video conversion failed. Ensure the file contains an audio track and is a playable MP4, MOV, WebM, or MP3 file.") from exc
    try:
        transcript = transcribe_chunks(chunks)
        if not transcript:
            raise HTTPException(status_code=422, detail="No speech was detected in the uploaded media")
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[analysis-upload] transcription failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail="Video converted successfully, but transcription failed. Check the Sarvam service configuration or try a shorter file.") from exc
    try:
        analysis = analyze_transcript(transcript)
    except Exception as exc:
        print(f"[analysis-upload] analysis failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail="Transcription completed, but meeting analysis failed. Please retry this file.") from exc
    meeting_id = uuid4().hex
    _sessions[meeting_id] = {"created_at": time.time(), "transcript": transcript}
    return {"meetingId": meeting_id, "transcript": transcript, **analysis}


@app.post("/v1/meetings/{meeting_id}/ask")
def ask(meeting_id: str, request: AskRequest, _: None = Depends(require_service_key)) -> dict[str, str]:
    purge_sessions()
    session = _sessions.get(meeting_id)
    if not session:
        if not request.transcript:
            raise HTTPException(status_code=404, detail="Meeting session not found or expired")
        session = {"created_at": time.time(), "transcript": request.transcript}
        _sessions[meeting_id] = session
    context = relevant_context(session["transcript"], request.question)
    answer = _mistral(
        "You are a meeting assistant. Answer only from the supplied transcript context. If the answer is absent, say: I could not find this information in the meeting transcript. Be concise.",
        f"CONTEXT:\n{context}\n\nQUESTION:\n{request.question}",
        temperature=0.2,
    )
    return {"answer": answer}
