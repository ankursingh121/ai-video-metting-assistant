# Python analysis service

This adapter reuses the existing GitHub modules for YouTube ingestion, Whisper/Sarvam transcription, Mistral summaries and extraction, Chroma retrieval, and meeting Q&A.

Copy `api.py` into the root of the GitHub repository. Install `requirements-api.txt` in addition to the repository's existing `Requirments.txt` dependencies. The runtime must also provide FFmpeg and enough memory for the configured Whisper model and HuggingFace embedding model.

Start the service with:

```bash
uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
```

Required environment variables:

| Variable | Purpose |
|---|---|
| `MISTRAL_API_KEY` | Mistral LLM calls for title, summary, extraction, and RAG answers |
| `WHISPER_MODEL` | Optional Whisper model name; defaults to `small` |
| `SARVAM_API_KEY` | Required only when `language=hinglish` |
| `SERVICE_API_KEY` | Optional shared secret for calls from the hosted Node server |
| `FRONTEND_ORIGIN` | Hosted site origin allowed by CORS |
| `SESSION_TTL_SECONDS` | Optional in-memory RAG session lifetime; defaults to 3600 |

The hosted web project expects these routes:

- `GET /health`
- `POST /v1/meetings/analyze` with `{ "sourceUrl": "https://...", "language": "english" }`
- `POST /v1/meetings/{meetingId}/ask` with `{ "question": "...", "transcript": "..." }`. The transcript is used only when the in-memory RAG session expired, allowing the Node server's durable meeting record to rebuild context.

The adapter intentionally keeps the RAG chain server-side and returns only JSON. The RAG session remains an in-memory performance cache, while the hosted Node server persists the transcript and can rehydrate it after the cache TTL.

The existing repository's `.env` was public in the repository audit. Revoke and regenerate any real credentials before deploying.
