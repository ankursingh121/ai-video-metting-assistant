# Python analysis service

This adapter is the low-memory HTTP wrapper used by the hosted Signal Room app. It avoids importing Whisper, Torch, LangChain, ChromaDB, and sentence-transformers at startup. Audio is processed with yt-dlp/FFmpeg, transcription is delegated to Sarvam, and meeting analysis/Q&A are delegated to Mistral through server-side environment variables.

Start the service with:

```bash
uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
```

Required environment variables:

| Variable | Purpose |
|---|---|
| `MISTRAL_API_KEY` | Mistral analysis and Q&A |
| `SARVAM_API_KEY` | Speech-to-text transcription |
| `SERVICE_API_KEY` | Optional shared secret for the hosted Node server |
| `FRONTEND_ORIGIN` | Hosted site origin allowed by CORS |
| `SESSION_TTL_SECONDS` | Optional in-memory context lifetime; defaults to 3600 |

Routes:

- `GET /health`
- `POST /v1/meetings/analyze` with `{ "sourceUrl": "https://...", "language": "english" }`
- `POST /v1/meetings/{meetingId}/ask` with `{ "question": "...", "transcript": "..." }`

The `transcript` field is used only when the in-memory session has expired. The hosted Node server persists the transcript and sends it back to rebuild lightweight retrieval context, so Q&A remains available after the service TTL.

Never commit `.env` or provider keys. The Render source repository is `ankursingh121/ai-video-metting-assistant`; the corrected lightweight adapter is on commit `9ca0916`.
