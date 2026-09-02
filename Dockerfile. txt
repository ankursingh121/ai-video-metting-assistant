FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r Requirments.txt -r requirements-api.txt

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
