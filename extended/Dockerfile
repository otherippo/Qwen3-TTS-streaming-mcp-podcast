FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install ffmpeg for audio normalization and pydub dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=8883
ENV PYTHONPATH=/app

EXPOSE 8883

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8883/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8883"]
