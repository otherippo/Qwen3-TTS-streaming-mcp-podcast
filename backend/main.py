# coding=utf-8
"""
Qwen3-TTS Backend FastAPI Server.
Wraps the official qwen-tts library with an OpenAI-compatible API.
"""

import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import unicodedata

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DEVICE = os.getenv("TTS_DEVICE", "cuda:0")
DTYPE_STR = os.getenv("TTS_DTYPE", "float16").lower()
CUSTOM_VOICES_DIR = Path(os.getenv("TTS_CUSTOM_VOICES", "/app/custom_voices"))
WARMUP_ON_START = os.getenv("TTS_WARMUP_ON_START", "true").lower() == "true"
VOICES_SCAN_DIR = Path(os.getenv("TTS_VOICES_SCAN_DIR", "/app/voices"))
VOICE_PROMPT_CACHE_DIR = Path(os.getenv("TTS_VOICE_PROMPT_CACHE_DIR", "/app/cache/voice_prompts"))

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
tts_model: Optional[Qwen3TTSModel] = None
model_ready = False
executor = ThreadPoolExecutor(max_workers=1)
custom_voices_cache: Dict[str, List[VoiceClonePromptItem]] = {}

# Max audio samples cap (~208 seconds at 24kHz)
MAX_AUDIO_SAMPLES = int(os.getenv("TTS_MAX_AUDIO_SAMPLES", "5000000"))
# Max new tokens for LLM generation (lower than 2048 default for safety)
MAX_NEW_TOKENS = int(os.getenv("TTS_MAX_NEW_TOKENS", "1500"))


def _normalize_text(text: str) -> str:
    """Normalize problematic Unicode characters that confuse the tokenizer."""
    # Smart quotes → straight quotes
    replacements = {
        '\u2018': "'",   # LEFT SINGLE QUOTATION MARK
        '\u2019': "'",   # RIGHT SINGLE QUOTATION MARK
        '\u201c': '"',   # LEFT DOUBLE QUOTATION MARK
        '\u201d': '"',   # RIGHT DOUBLE QUOTATION MARK
        '\u201a': ",",   # SINGLE LOW-9 QUOTATION MARK
        '\u201e': '"',   # DOUBLE LOW-9 QUOTATION MARK
        '\u2013': "-",   # EN DASH
        '\u2014': "-",   # EM DASH
        '\u2026': "...", # HORIZONTAL ELLIPSIS
        '\u00a0': " ",   # NON-BREAKING SPACE
        '\u202f': " ",   # NARROW NO-BREAK SPACE
        '\u200b': "",    # ZERO WIDTH SPACE
        '\u200c': "",    # ZERO WIDTH NON-JOINER
        '\u200d': "",    # ZERO WIDTH JOINER
        '\ufeff': "",    # ZERO WIDTH NO-BREAK SPACE (BOM)
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Normalize remaining combining characters
    text = unicodedata.normalize("NFKC", text)
    return text


def _resolve_dtype() -> torch.dtype:
    if DTYPE_STR in ("float16", "fp16", "half"):
        return torch.float16
    if DTYPE_STR in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def _wav_to_mp3(wav: np.ndarray, sr: int) -> bytes:
    """Convert float32 wav to MP3 bytes using ffmpeg."""
    wav_bytes = io.BytesIO()
    # Normalize to int16 for ffmpeg
    sf.write(wav_bytes, wav, sr, format="WAV", subtype="PCM_16")
    wav_bytes.seek(0)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "mp3", "-q:a", "2", "pipe:1"],
        input=wav_bytes.read(),
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")[:500]
        raise RuntimeError(f"ffmpeg mp3 encoding failed: {err}")
    return proc.stdout


def _load_model() -> None:
    global tts_model
    logger.info(f"Loading model {MODEL_NAME} on {DEVICE} with dtype={DTYPE_STR} ...")
    dtype = _resolve_dtype()
    tts_model = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    logger.info("Model loaded.")
    # torch.compile causes SIGSEGV (exit 139) on this GPU; skip optimizations.
    # tts_model.enable_streaming_optimizations(...)
    logger.info("Streaming optimizations SKIPPED due to GPU incompatibility with torch.compile.")


def _warmup() -> None:
    if not WARMUP_ON_START or tts_model is None:
        return
    logger.info("Warming up model with a dummy generation...")
    try:
        tts_model.generate_voice_clone(
            text="Hallo Welt.",
            language="German",
            ref_audio=None,
            voice_clone_prompt=list(custom_voices_cache.values())[0]
            if custom_voices_cache else None,
        )
        logger.info("Warmup complete.")
    except Exception as exc:
        logger.warning(f"Warmup failed (non-critical): {exc}")


def _save_voice_prompt(items: List[VoiceClonePromptItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"items": [asdict(it) for it in items]}
    torch.save(payload, path)


def _load_voice_prompt(path: Path) -> List[VoiceClonePromptItem]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    items_raw = payload["items"]
    if not isinstance(items_raw, list):
        raise ValueError("Invalid prompt file format")
    items: List[VoiceClonePromptItem] = []
    for d in items_raw:
        if not isinstance(d, dict):
            raise ValueError("Invalid item in prompt file")
        ref_code = d.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        ref_spk = d.get("ref_spk_embedding", None)
        if ref_spk is None:
            raise ValueError("Missing ref_spk_embedding in prompt file")
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                ref_text=d.get("ref_text", None),
            )
        )
    return items


def _discover_voices() -> Dict[str, Path]:
    """Discover .wav files in VOICES_SCAN_DIR and return {voice_name: wav_path}."""
    voices: Dict[str, Path] = {}
    if not VOICES_SCAN_DIR.exists():
        return voices
    for f in VOICES_SCAN_DIR.iterdir():
        if f.suffix.lower() == ".wav":
            voices[f.stem] = f
    return voices


def _preprocess_voice_for_prompt(input_path: Path, output_path: Path, max_duration: Optional[int] = None) -> None:
    """Normalize audio for voice-clone prompt creation.

    No truncation is applied by default so the full reference audio + transcript
    can be used for ICL mode. The model's context window is 32,768 tokens;
    at 12 Hz that accommodates several minutes of reference audio before
    hitting the limit.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "24000",
        "-ac", "1",
        "-af",
        "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=24000",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    if max_duration is not None:
        cmd.insert(4, "-t")
        cmd.insert(5, str(max_duration))
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")[-500:]
        raise RuntimeError(f"ffmpeg failed: {err}")


def _load_custom_voices() -> Dict[str, str]:
    """Scan voices dir, build or load cached prompts. Returns errors."""
    global custom_voices_cache
    errors: Dict[str, str] = {}
    voices = _discover_voices()
    VOICE_PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for name, wav_path in voices.items():
        txt_path = wav_path.with_suffix(".txt")
        transcript = None
        if txt_path.exists():
            try:
                transcript = txt_path.read_text(encoding="utf-8").strip()
            except Exception as exc:
                logger.warning(f"Could not read transcript for {name}: {exc}")

        # Use v2 cache filename to force rebuild after preprocessing fix
        cache_path = VOICE_PROMPT_CACHE_DIR / f"{name}.v2.cached_prompt.pt"
        processed_wav = VOICE_PROMPT_CACHE_DIR / f"{name}_processed.wav"
        # Rebuild cache if missing or source files are newer
        rebuild = True
        if cache_path.exists() and processed_wav.exists():
            cache_mtime = cache_path.stat().st_mtime
            wav_mtime = wav_path.stat().st_mtime
            txt_mtime = txt_path.stat().st_mtime if txt_path.exists() else 0
            if cache_mtime >= wav_mtime and cache_mtime >= txt_mtime:
                rebuild = False

        if rebuild:
            try:
                logger.info(f"Building voice prompt for '{name}' (transcript={'yes' if transcript else 'no'})...")
                xvec_only = not bool(transcript)
                # Full audio is used for both ICL and x-vector modes. The model's
                # 32k context window can accommodate several minutes of 12Hz audio.
                processed_wav = VOICE_PROMPT_CACHE_DIR / f"{name}_processed.wav"
                _preprocess_voice_for_prompt(wav_path, processed_wav, max_duration=None)
                items = tts_model.create_voice_clone_prompt(
                    ref_audio=str(processed_wav),
                    ref_text=transcript if transcript else None,
                    x_vector_only_mode=xvec_only,
                )
                _save_voice_prompt(items, cache_path)
                logger.info(f"Cached voice prompt for '{name}' -> {cache_path}")
            except Exception as exc:
                errors[name] = str(exc)
                logger.error(f"Failed to build voice prompt for '{name}': {exc}")
                continue

        # Load from cache
        try:
            custom_voices_cache[name] = _load_voice_prompt(cache_path)
        except Exception as exc:
            errors[name] = str(exc)
            logger.error(f"Failed to load cached prompt for '{name}': {exc}")

    # Remove stale cache entries for deleted voices
    stale = [k for k in custom_voices_cache if k not in voices]
    for k in stale:
        del custom_voices_cache[k]
        for stale_cache in VOICE_PROMPT_CACHE_DIR.glob(f"{k}.*cached_prompt.pt"):
            stale_cache.unlink()
        stale_processed = VOICE_PROMPT_CACHE_DIR / f"{k}_processed.wav"
        if stale_processed.exists():
            stale_processed.unlink()

    return errors


async def _load_model_async() -> None:
    global model_ready
    model_ready = False
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, _load_model)
    # Load custom voices after model is ready
    await loop.run_in_executor(executor, _load_custom_voices)
    await loop.run_in_executor(executor, _warmup)
    model_ready = True
    logger.info("Backend ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_load_model_async())
    yield
    executor.shutdown(wait=False)
    logger.info("Backend shutting down...")


app = FastAPI(
    title="Qwen3-TTS Backend",
    description="OpenAI-compatible API for Qwen3-TTS Base model with voice cloning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    model: str = Field(default="qwen3-tts")
    input: str = Field(..., min_length=1)
    voice: Optional[str] = None
    language: str = Field(default="German")
    response_format: str = Field(default="mp3")
    speed: float = Field(default=1.0)


class VoiceCloneRequest(BaseModel):
    input: str = Field(..., min_length=1)
    ref_audio: str = Field(..., min_length=1)  # base64
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = Field(default=False)
    language: str = Field(default="German")
    response_format: str = Field(default="mp3")
    speed: float = Field(default=1.0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_ready else "initializing",
        "backend": {
            "ready": model_ready,
            "model_id": MODEL_NAME,
            "custom_voices_loaded": list(custom_voices_cache.keys()),
        },
    }


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    if not model_ready or tts_model is None:
        raise HTTPException(status_code=503, detail="Model not ready yet")
    if not request.voice:
        raise HTTPException(status_code=400, detail="voice is required")

    voice_name = request.voice
    if voice_name not in custom_voices_cache:
        # Try to rebuild just this voice on-demand
        wav_path = VOICES_SCAN_DIR / f"{voice_name}.wav"
        if not wav_path.exists():
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
        loop = asyncio.get_event_loop()
        errors = await loop.run_in_executor(executor, _load_custom_voices)
        if voice_name not in custom_voices_cache:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load voice '{voice_name}': {errors.get(voice_name, 'unknown error')}"
            )

    prompt_items = custom_voices_cache[voice_name]

    def _gen():
        text = _normalize_text(request.input.strip())
        t0 = time.perf_counter()
        wavs, sr = tts_model.generate_voice_clone(
            text=text,
            language=request.language,
            voice_clone_prompt=prompt_items,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0
        wav = wavs[0]
        # Cap to max audio samples
        original_len = wav.shape[0]
        if original_len > MAX_AUDIO_SAMPLES:
            logger.warning(
                f"Generated audio exceeds cap: {original_len} > {MAX_AUDIO_SAMPLES} samples "
                f"({original_len/sr:.1f}s > {MAX_AUDIO_SAMPLES/sr:.1f}s). Truncating."
            )
            wav = wav[:MAX_AUDIO_SAMPLES]
        cps = len(text) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Generated speech: shape={wav.shape}, sr={sr}, range=[{wav.min():.4f}, {wav.max():.4f}], "
            f"mean={wav.mean():.4f}, elapsed={elapsed:.2f}s, chars={len(text)}, cps={cps:.1f}"
        )
        if not np.isfinite(wav).all():
            raise RuntimeError("Generated audio contains NaN or Inf")
        if np.max(np.abs(wav)) < 1e-6:
            raise RuntimeError("Generated audio is silent")
        return wav, sr

    loop = asyncio.get_event_loop()
    wav, sr = await loop.run_in_executor(executor, _gen)
    mp3_bytes = await loop.run_in_executor(executor, _wav_to_mp3, wav, sr)

    return Response(content=mp3_bytes, media_type="audio/mpeg")


@app.post("/v1/audio/voice-clone")
async def voice_clone(request: VoiceCloneRequest):
    if not model_ready or tts_model is None:
        raise HTTPException(status_code=503, detail="Model not ready yet")

    # Decode base64 audio
    try:
        audio_b64 = request.ref_audio
        if "," in audio_b64 and audio_b64.strip().startswith("data:"):
            audio_b64 = audio_b64.split(",", 1)[1]
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 ref_audio")

    # Write to temp wav for the model (it accepts paths or numpy arrays)
    with io.BytesIO(audio_bytes) as f:
        ref_wav, ref_sr = sf.read(f, dtype="float32", always_2d=False)

    if ref_wav.ndim > 1:
        ref_wav = np.mean(ref_wav, axis=-1)

    def _gen():
        text = _normalize_text(request.input.strip())
        t0 = time.perf_counter()
        wavs, sr = tts_model.generate_voice_clone(
            text=text,
            language=request.language,
            ref_audio=(ref_wav, ref_sr),
            ref_text=request.ref_text.strip() if request.ref_text else None,
            x_vector_only_mode=request.x_vector_only_mode,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0
        wav = wavs[0]
        original_len = wav.shape[0]
        if original_len > MAX_AUDIO_SAMPLES:
            logger.warning(
                f"Generated audio exceeds cap: {original_len} > {MAX_AUDIO_SAMPLES} samples "
                f"({original_len/sr:.1f}s > {MAX_AUDIO_SAMPLES/sr:.1f}s). Truncating."
            )
            wav = wav[:MAX_AUDIO_SAMPLES]
        cps = len(text) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Generated voice-clone speech: shape={wav.shape}, sr={sr}, range=[{wav.min():.4f}, {wav.max():.4f}], "
            f"mean={wav.mean():.4f}, elapsed={elapsed:.2f}s, chars={len(text)}, cps={cps:.1f}"
        )
        if not np.isfinite(wav).all():
            raise RuntimeError("Generated audio contains NaN or Inf")
        if np.max(np.abs(wav)) < 1e-6:
            raise RuntimeError("Generated audio is silent")
        return wav, sr

    loop = asyncio.get_event_loop()
    wav, sr = await loop.run_in_executor(executor, _gen)
    mp3_bytes = await loop.run_in_executor(executor, _wav_to_mp3, wav, sr)

    return Response(content=mp3_bytes, media_type="audio/mpeg")


@app.post("/v1/reload-custom-voices")
async def reload_custom_voices():
    if not model_ready or tts_model is None:
        raise HTTPException(status_code=503, detail="Model not ready yet")
    loop = asyncio.get_event_loop()
    errors = await loop.run_in_executor(executor, _load_custom_voices)
    loaded = list(custom_voices_cache.keys())
    return {"loaded": loaded, "errors": errors}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "qwen"}
        ],
    }
