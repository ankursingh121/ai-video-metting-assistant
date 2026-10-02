# Python analysis service

This adapter is the low-memory HTTP wrapper used by the hosted Signal Room app. It avoids importing Whisper, Torch, LangChain, ChromaDB, and sentence-transformers at startup. YouTube audio is downloaded with yt-dlp; uploaded video/audio is fetched from a temporary signed storage URL and converted with FFmpeg/pydub. Transcription is delegated to Sarvam, and meeting analysis/Q&A are delegated to Mistral through server-side environment variables.

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
- `POST /v1/meetings/analyze` with `{ "sourceUrl": "https://...", "sourceType": "youtube|video|audio", "language": "english" }`
- `POST /v1/meetings/{meetingId}/ask` with `{ "question": "...", "transcript": "..." }`

For uploaded media, the Node server stores the file securely and sends a short-lived signed URL with `sourceType=video` or `sourceType=audio`. The adapter downloads that URL, converts the media to WAV, chunks it, transcribes it with Sarvam, and returns the same analysis shape as YouTube. The `transcript` field is used only when the in-memory session has expired so Q&A remains available after the service TTL.

Never commit `.env` or provider keys. The Render source repository is `ankursingh121/ai-video-metting-assistant`; the current adapter fix is on GitHub commit `b6fef03`.
