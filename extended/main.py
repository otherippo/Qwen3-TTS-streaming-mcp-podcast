# coding=utf-8
"""
Qwen3-TTS Extended Proxy Service.
Provides long-form TTS with chunking, job management, voice normalization, and Web UI.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from services.voice_manager import VoiceManager
from services.job_manager import JobManager
from services.chunking import chunk_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8883"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
STATIC_DIR = Path(__file__).parent / "static"

# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8880")

# Pydantic schemas
class PreviewRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to preview chunking for")


class CreateJobRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str = Field(..., min_length=1, description="Voice name")
    language: str = Field(default="German", description="Language code")
    model: str = Field(default="1.7B", description="Model size: 1.7B only")
    no_drift: bool = Field(default=True, description="Use cached voice prompt for consistent voice across chunks")


class PodcastTurn(BaseModel):
    speaker: str = Field(..., min_length=1, description="Voice name for this speaker")
    text: str = Field(..., min_length=1, description="Text spoken by this speaker")


class CreatePodcastRequest(BaseModel):
    turns: List[PodcastTurn] = Field(..., min_length=1, description="List of dialogue turns")
    language: str = Field(default="German", description="Language code")
    model: str = Field(default="1.7B", description="Model size: 1.7B only")
    no_drift: bool = Field(default=True, description="Use cached voice prompts for consistent voices")


class RetryChunkRequest(BaseModel):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: initialize voice manager and job manager."""
    logger.info("░" * 40)
    logger.info("  Qwen3-TTS Extended Proxy starting")
    logger.info(f"  Backend: {BACKEND_URL}")
    logger.info(f"  Port: {PORT}")
    logger.info("░" * 40)
    yield
    logger.info("Extended proxy shutting down...")


app = FastAPI(
    title="Qwen3-TTS Extended Proxy",
    description="Long-form TTS with intelligent chunking, job management, and custom voice support.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global managers
voice_manager = VoiceManager()
job_manager = JobManager(voice_manager)


def _get_backend_url(model: str) -> str:
    """Return backend URL for the selected model."""
    return BACKEND_URL


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(content="<h1>Qwen3-TTS Extended Proxy</h1><p>Static files missing.</p>")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "qwen3-tts-extended", "version": "1.1.0"}


# ---------------------------------------------------------------------------
# Backend Models / Health
# ---------------------------------------------------------------------------
@app.get("/api/backend/models")
async def list_backend_models():
    """List available backend models with their status."""
    import aiohttp
    models = []
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{BACKEND_URL}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models.append({
                        "id": "1.7B",
                        "url": BACKEND_URL,
                        "status": data.get("status", "unknown"),
                        "ready": data.get("backend", {}).get("ready", False),
                        "model_id": data.get("backend", {}).get("model_id", "unknown"),
                    })
                else:
                    models.append({"id": "1.7B", "url": BACKEND_URL, "status": "unreachable", "ready": False})
    except Exception:
        models.append({"id": "1.7B", "url": BACKEND_URL, "status": "offline", "ready": False})
    return {"models": models}


# ---------------------------------------------------------------------------
# Preview / Chunking
# ---------------------------------------------------------------------------
@app.post("/api/preview")
async def preview_chunks(request: PreviewRequest):
    """Return the exact chunks that would be generated for the given text."""
    chunks = chunk_text(request.text)
    return {
        "chunk_count": len(chunks),
        "chunks": [{"index": i, "text": t, "char_count": len(t)} for i, t in enumerate(chunks)],
    }


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------
@app.get("/api/voices")
async def list_voices():
    """List discovered voices with metadata."""
    voices = voice_manager.discover_voices()
    return {"voices": voices}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
@app.post("/api/jobs")
async def create_job(request: CreateJobRequest):
    """Create a new TTS job and start background processing."""
    # Validate voice exists
    if request.voice not in voice_manager.get_voice_names():
        raise HTTPException(status_code=404, detail=f"Voice '{request.voice}' not found")
    # Validate model
    if request.model not in ("1.7B",):
        raise HTTPException(status_code=400, detail="Model must be '1.7B'")
    backend_url = _get_backend_url(request.model)
    job = await job_manager.create_job(
        request.text,
        request.voice,
        request.language,
        request.no_drift,
        backend_url=backend_url,
    )
    return job.to_dict()


@app.post("/api/podcast")
async def create_podcast(request: CreatePodcastRequest):
    """Create a multi-speaker podcast job and start background processing."""
    # Validate all speakers exist
    available = voice_manager.get_voice_names()
    for turn in request.turns:
        if turn.speaker not in available:
            raise HTTPException(status_code=404, detail=f"Voice '{turn.speaker}' not found")
    if request.model not in ("1.7B",):
        raise HTTPException(status_code=400, detail="Model must be '1.7B'")
    backend_url = _get_backend_url(request.model)
    turns_data = [{"speaker": t.speaker, "text": t.text} for t in request.turns]
    job = await job_manager.create_podcast_job(
        turns_data,
        request.language,
        request.no_drift,
        backend_url=backend_url,
    )
    return job.to_dict()


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs, newest first."""
    jobs = await job_manager.list_jobs()
    return {"jobs": [j.to_dict() for j in jobs]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job state (polling fallback)."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running or pending job."""
    ok = await job_manager.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Job not found or already finished")
    return {"cancelled": True}


@app.post("/api/jobs/{job_id}/retry/{chunk_idx}")
async def retry_chunk(job_id: str, chunk_idx: int):
    """Retry a failed chunk."""
    job = await job_manager.retry_chunk(job_id, chunk_idx)
    if not job:
        raise HTTPException(status_code=404, detail="Job or chunk not found, or chunk not failed")
    return job.to_dict()


# ---------------------------------------------------------------------------
# SSE Streaming
# ---------------------------------------------------------------------------
@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """SSE stream for live job progress updates."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        async for msg in job_manager.subscribe(job_id):
            yield msg

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
@app.get("/api/jobs/{job_id}/download/final")
async def download_final(job_id: str):
    """Download the concatenated final MP3."""
    job = await job_manager.get_job(job_id)
    if not job or not job.final_mp3_path or not job.final_mp3_path.exists():
        raise HTTPException(status_code=404, detail="Final audio not ready")
    return FileResponse(
        path=job.final_mp3_path,
        media_type="audio/mpeg",
        filename=f"{job_id}_final.mp3",
    )


@app.get("/api/jobs/{job_id}/download/zip")
async def download_zip(job_id: str):
    """Download all chunks + final as ZIP."""
    job = await job_manager.get_job(job_id)
    if not job or not job.zip_path or not job.zip_path.exists():
        raise HTTPException(status_code=404, detail="ZIP archive not ready")
    return FileResponse(
        path=job.zip_path,
        media_type="application/zip",
        filename=f"{job_id}_chunks.zip",
    )


@app.get("/api/jobs/{job_id}/chunks/{chunk_idx}")
async def download_chunk(job_id: str, chunk_idx: int):
    """Download a single chunk MP3."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    chunk_mp3 = Path(os.getenv("CACHE_DIR", "/app/cache")) / "jobs" / job_id / "chunks" / f"chunk_{chunk_idx:03d}.mp3"
    if not chunk_mp3.exists():
        raise HTTPException(status_code=404, detail="Chunk not found")
    return FileResponse(
        path=chunk_mp3,
        media_type="audio/mpeg",
        filename=f"{job_id}_chunk_{chunk_idx:03d}.mp3",
    )


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
