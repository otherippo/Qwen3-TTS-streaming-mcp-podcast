# Backend Dockerfile for Qwen3-TTS GPU service
# Build context is project root
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    sox \
    libsox-fmt-all \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip / setuptools / wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support
RUN pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install build deps required for flash-attn compilation
RUN pip install numpy psutil packaging ninja

# Install flash-attn (try prebuilt wheel first; if unavailable it builds from source)
RUN pip install flash-attn==2.7.4.post1 --no-build-isolation || \
    pip install flash-attn --no-build-isolation

# Copy and install the qwen-tts package
COPY pyproject.toml MANIFEST.in /app/qwen-tts/
COPY qwen_tts/ /app/qwen-tts/
RUN pip install -e /app/qwen-tts/

# Copy backend source
COPY backend/ /app/backend/
WORKDIR /app/backend

# Install remaining Python dependencies
RUN pip install -r /app/backend/requirements.txt

# Environment defaults
ENV HOST=0.0.0.0
ENV PORT=8880
ENV TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV TTS_DEVICE=cuda:0
ENV TTS_DTYPE=float16
ENV TTS_WARMUP_ON_START=false
ENV TTS_CUSTOM_VOICES=/app/custom_voices
ENV TTS_VOICES_SCAN_DIR=/app/voices
ENV TTS_VOICE_PROMPT_CACHE_DIR=/app/cache/voice_prompts
ENV PYTHONPATH=/app/qwen-tts:/app/backend

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8880"]
