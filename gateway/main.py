# coding=utf-8
"""
Unified OpenAI-compatible API Gateway.
Proxies chat completions to llama.cpp and audio/speech to Qwen3-TTS Extended Proxy.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aiohttp
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8885"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080")
TTS_URL = os.getenv("TTS_URL", "http://localhost:8883")
TTS_BACKEND_URL = os.getenv("TTS_BACKEND_URL", "http://localhost:8884")

TIMEOUT = aiohttp.ClientTimeout(total=600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("░" * 40)
    logger.info("  Unified API Gateway starting")
    logger.info(f"  LLM:   {LLAMA_URL}")
    logger.info(f"  TTS:   {TTS_URL}")
    logger.info(f"  TTS-BE:{TTS_BACKEND_URL}")
    logger.info(f"  Port:  {PORT}")
    logger.info("░" * 40)
    yield
    logger.info("Gateway shutting down...")


app = FastAPI(
    title="Unified AI Gateway",
    description="OpenAI-compatible gateway for llama.cpp (chat) + Qwen3-TTS (speech).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _proxy_request(
    target_url: str,
    method: str,
    path: str,
    request: Request,
    stream_response: bool = False,
) -> Response:
    """Proxy a request to a backend service."""
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    url = f"{target_url}{path}"
    logger.info(f"Proxy {method} {path} -> {target_url}")

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        async with session.request(
            method=method,
            url=url,
            headers=headers,
            data=body,
        ) as resp:
            if stream_response and resp.status == 200:
                async def stream_generator() -> AsyncGenerator[bytes, None]:
                    async for chunk in resp.content.iter_chunked(8192):
                        yield chunk
                return StreamingResponse(
                    stream_generator(),
                    status_code=resp.status,
                    headers={k: v for k, v in resp.headers.items() if k.lower() not in ("transfer-encoding", "content-length")},
                    media_type=resp.headers.get("content-type", "application/octet-stream"),
                )
            else:
                content = await resp.read()
                return Response(
                    content=content,
                    status_code=resp.status,
                    headers={k: v for k, v in resp.headers.items() if k.lower() not in ("transfer-encoding", "content-length")},
                    media_type=resp.headers.get("content-type", "application/json"),
                )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    llm_ok = False
    tts_ok = False
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{LLAMA_URL}/health") as r:
                llm_ok = r.status == 200
    except Exception:
        pass
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{TTS_URL}/health") as r:
                tts_ok = r.status == 200
    except Exception:
        pass
    return {
        "status": "healthy" if (llm_ok and tts_ok) else "degraded",
        "llm": {"url": LLAMA_URL, "ready": llm_ok},
        "tts": {"url": TTS_URL, "ready": tts_ok},
    }


# ---------------------------------------------------------------------------
# LLM proxies -> llama.cpp
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    is_stream = False
    try:
        body = await request.json()
        is_stream = body.get("stream", False)
    except Exception:
        pass
    return await _proxy_request(LLAMA_URL, "POST", "/v1/chat/completions", request, stream_response=is_stream)


@app.get("/v1/models")
async def list_llm_models(request: Request):
    return await _proxy_request(LLAMA_URL, "GET", "/v1/models", request)


# ---------------------------------------------------------------------------
# TTS proxies -> extended proxy / backend
# ---------------------------------------------------------------------------
@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    return await _proxy_request(TTS_BACKEND_URL, "POST", "/v1/audio/speech", request)


@app.get("/api/voices")
async def list_voices(request: Request):
    return await _proxy_request(TTS_URL, "GET", "/api/voices", request)


@app.post("/api/jobs")
async def create_job(request: Request):
    return await _proxy_request(TTS_URL, "POST", "/api/jobs", request)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    return await _proxy_request(TTS_URL, "GET", f"/api/jobs/{job_id}", request)


@app.get("/api/jobs/{job_id}/download/final")
async def download_final(job_id: str, request: Request):
    return await _proxy_request(TTS_URL, "GET", f"/api/jobs/{job_id}/download/final", request)


@app.post("/api/podcast")
async def create_podcast(request: Request):
    return await _proxy_request(TTS_URL, "POST", "/api/podcast", request)


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str, request: Request):
    return await _proxy_request(TTS_URL, "DELETE", f"/api/jobs/{job_id}", request)


@app.post("/api/jobs/{job_id}/retry/{chunk_idx}")
async def retry_chunk(job_id: str, chunk_idx: int, request: Request):
    return await _proxy_request(TTS_URL, "POST", f"/api/jobs/{job_id}/retry/{chunk_idx}", request)


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str, request: Request):
    return await _proxy_request(TTS_URL, "GET", f"/api/jobs/{job_id}/stream", request, stream_response=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
