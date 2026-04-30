# coding=utf-8
"""
Background async job management with sequential chunk processing,
progress tracking, SSE streaming, cancellation, and per-chunk retry.
"""

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from .chunking import chunk_text
from .voice_manager import VoiceManager
from .audio_processor import concatenate_mp3s, create_zip_archive

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))
_CACHE_DIR = Path(os.getenv("CACHE_DIR", "/app/cache")) / "jobs"


def _ensure_dirs() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ChunkState:
    index: int
    text: str
    voice: str = ""  # Per-chunk voice (for podcast mode); empty = use job.voice
    status: JobStatus = JobStatus.PENDING
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None


@dataclass
class Job:
    id: str
    text: str
    voice: str
    language: str
    status: JobStatus = JobStatus.PENDING
    total_chunks: int = 0
    chunks: List[ChunkState] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    final_mp3_path: Optional[Path] = None
    zip_path: Optional[Path] = None
    eta_seconds: Optional[float] = None
    chars_per_second: Optional[float] = None
    backend_url: str = field(default="")
    no_drift: bool = field(default=False)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    _task: Optional[asyncio.Task] = None

    @property
    def completed_chunks(self) -> int:
        return sum(1 for c in self.chunks if c.status in (JobStatus.DONE, JobStatus.FAILED))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "voice": self.voice,
            "language": self.language,
            "total_chunks": self.total_chunks,
            "completed_chunks": self.completed_chunks,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "eta_seconds": self.eta_seconds,
            "chars_per_second": self.chars_per_second,
            "final_mp3": str(self.final_mp3_path) if self.final_mp3_path else None,
            "zip": str(self.zip_path) if self.zip_path else None,
            "chunks": [
                {
                    "index": c.index,
                    "text": c.text,
                    "voice": c.voice or self.voice,
                    "status": c.status.value,
                    "elapsed_seconds": c.elapsed_seconds,
                    "error": c.error,
                }
                for c in self.chunks
            ],
        }


class JobManager:
    def __init__(self, voice_manager: VoiceManager):
        _ensure_dirs()
        self.voice_manager = voice_manager
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._listeners: Dict[str, List[asyncio.Queue]] = {}
        self._backend_custom_voices_loaded: Dict[str, set] = {}

    async def create_job(self, text: str, voice: str, language: str = "German", no_drift: bool = False, backend_url: str = "") -> Job:
        chunks = chunk_text(text)
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            text=text,
            voice=voice,
            language=language,
            total_chunks=len(chunks),
            chunks=[ChunkState(index=i, text=t) for i, t in enumerate(chunks)],
            backend_url=backend_url,
            no_drift=no_drift,
        )
        # Initial ETA estimate: ~19 chars/sec for German TTS (empirical: 760c≈40s, 1030c≈53s)
        total_chars = len(text)
        job.eta_seconds = total_chars / 19.0

        async with self._lock:
            self._jobs[job_id] = job
            self._listeners[job_id] = []

        # Start background processing
        job._task = asyncio.create_task(self._run_job(job, no_drift=no_drift))
        logger.info(f"Created job {job_id} with {len(chunks)} chunks (no_drift={no_drift}, backend={backend_url})")
        return job

    async def create_podcast_job(
        self,
        turns: List[dict],
        language: str = "German",
        no_drift: bool = False,
        backend_url: str = "",
    ) -> Job:
        """Create a podcast job: multiple speakers, each turn chunked independently.

        turns: list of {"speaker": str, "text": str}
        """
        all_chunks: List[ChunkState] = []
        total_text = ""
        for turn in turns:
            speaker = turn["speaker"]
            text = turn["text"]
            total_text += text + " "
            turn_chunks = chunk_text(text)
            for t in turn_chunks:
                all_chunks.append(ChunkState(index=len(all_chunks), text=t, voice=speaker))

        job_id = str(uuid.uuid4())[:8]
        # Use first speaker as job-level voice (for backward compat)
        first_voice = turns[0]["speaker"] if turns else ""
        job = Job(
            id=job_id,
            text=total_text.strip(),
            voice=first_voice,
            language=language,
            total_chunks=len(all_chunks),
            chunks=all_chunks,
            backend_url=backend_url,
            no_drift=no_drift,
        )
        total_chars = len(total_text)
        job.eta_seconds = total_chars / 19.0

        async with self._lock:
            self._jobs[job_id] = job
            self._listeners[job_id] = []

        job._task = asyncio.create_task(self._run_job(job, no_drift=no_drift))
        logger.info(f"Created podcast job {job_id} with {len(all_chunks)} chunks across {len(turns)} turns")
        return job

    async def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def list_jobs(self) -> List[Job]:
        """Return jobs ordered newest-first."""
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    async def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        job.status = JobStatus.CANCELLED
        job._cancel_event.set()
        if job._task and not job._task.done():
            job._task.cancel()
        await self._emit(job, {"event": "cancelled"})
        logger.info(f"Job {job_id} cancelled")
        return True

    async def retry_chunk(self, job_id: str, chunk_idx: int) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        if chunk_idx < 0 or chunk_idx >= len(job.chunks):
            return None
        chunk = job.chunks[chunk_idx]
        if chunk.status != JobStatus.FAILED:
            return None
        chunk.status = JobStatus.PENDING
        chunk.error = None
        chunk.elapsed_seconds = None
        # If job is done/failed, set back to running for the remaining failed chunks
        if job.status in (JobStatus.DONE, JobStatus.FAILED):
            job.status = JobStatus.RUNNING
            job.completed_at = None
        # Re-start processing task if not running
        if job._task is None or job._task.done():
            job._task = asyncio.create_task(self._run_job(job, no_drift=job.no_drift))
        logger.info(f"Retrying chunk {chunk_idx} for job {job_id}")
        return job

    async def subscribe(self, job_id: str) -> AsyncGenerator[str, None]:
        """SSE subscription for live job updates."""
        job = self._jobs.get(job_id)
        if not job:
            return

        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            if job_id in self._listeners:
                self._listeners[job_id].append(queue)

        try:
            # Send current state immediately
            yield f"data: {json.dumps({'event': 'state', 'data': job.to_dict()})}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("event") in ("done", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"
        finally:
            async with self._lock:
                if job_id in self._listeners and queue in self._listeners[job_id]:
                    self._listeners[job_id].remove(queue)

    async def _emit(self, job: Job, message: dict) -> None:
        listeners = self._listeners.get(job.id, [])
        for q in listeners:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass

    def _chunk_timeout(self, text: str) -> float:
        """Calculate per-chunk timeout: 3x expected time based on ~19 chars/sec, min 90s."""
        expected = len(text) / 19.0
        return max(90.0, expected * 3.0)

    async def _check_backend_health(self, backend_url: str) -> bool:
        """Quick health check to detect stuck backend before sending a chunk."""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{backend_url}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("status") in ("healthy", "initializing")
        except Exception:
            pass
        return False

    async def _generate_chunk_voice_clone(
        self,
        session: aiohttp.ClientSession,
        text: str,
        voice: str,
        language: str,
        ref_audio_path: Path,
        ref_text: Optional[str],
        backend_url: str,
    ) -> bytes:
        """Call backend /v1/audio/voice-clone for a single chunk (default mode with drift)."""
        with open(ref_audio_path, "rb") as f:
            ref_audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "input": text,
            "ref_audio": ref_audio_b64,
            "ref_text": ref_text,
            "x_vector_only_mode": ref_text is None,
            "language": language,
            "response_format": "mp3",
            "speed": 1.0,
        }

        async with session.post(
            f"{backend_url}/v1/audio/voice-clone",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Backend returned {resp.status}: {body[:500]}")
            return await resp.read()

    async def _generate_chunk_custom_voice(
        self,
        session: aiohttp.ClientSession,
        text: str,
        voice: str,
        language: str,
        backend_url: str,
    ) -> bytes:
        """Call backend /v1/audio/speech with a cached custom voice (no-drift mode)."""
        payload = {
            "model": "qwen3-tts",
            "input": text,
            "voice": voice,
            "language": language,
            "response_format": "mp3",
            "speed": 1.0,
        }

        async with session.post(
            f"{backend_url}/v1/audio/speech",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Backend returned {resp.status}: {body[:500]}")
            return await resp.read()

    async def _ensure_backend_custom_voice(self, voice: str, backend_url: str) -> None:
        """Ensure backend has loaded the custom voice by calling reload endpoint."""
        loaded_key = f"{backend_url}:{voice}"
        if loaded_key in self._backend_custom_voices_loaded.get(backend_url, set()):
            return

        # Call backend reload endpoint — backend scans its own voices dir directly
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{backend_url}/v1/reload-custom-voices",
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"Backend reloaded custom voices: {data.get('loaded', [])}")
                    if voice in data.get("loaded", []):
                        if backend_url not in self._backend_custom_voices_loaded:
                            self._backend_custom_voices_loaded[backend_url] = set()
                        self._backend_custom_voices_loaded[backend_url].add(voice)
                else:
                    body = await resp.text()
                    logger.warning(f"Backend reload returned {resp.status}: {body[:500]}")

    async def _run_job(self, job: Job, no_drift: bool = False) -> None:
        """Main job runner: sequential chunk processing."""
        job_dir = _CACHE_DIR / job.id
        chunks_dir = job_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        try:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            await self._emit(job, {"event": "started", "data": job.to_dict()})

            # Lazy voice preprocessing cache (supports podcast multi-voice)
            voice_cache: Dict[str, tuple] = {}
            async def _get_voice(voice_name: str) -> tuple:
                if voice_name not in voice_cache:
                    voice_cache[voice_name] = await self.voice_manager.get_preprocessed_voice(voice_name)
                return voice_cache[voice_name]

            # No-drift mode: ensure backend has loaded all unique custom voice prompts
            if no_drift:
                unique_voices = {c.voice or job.voice for c in job.chunks}
                for v in unique_voices:
                    try:
                        await self._ensure_backend_custom_voice(v, job.backend_url)
                    except Exception as exc:
                        logger.warning(f"No-drift setup failed for '{v}', falling back to voice-clone: {exc}")
                        no_drift = False
                        break

            # Determine processing order: for multi-voice jobs (podcasts), batch by
            # voice to avoid expensive ICL re-alignment on every chunk.
            distinct_voices = {c.voice or job.voice for c in job.chunks}
            is_podcast = len(distinct_voices) > 1

            if is_podcast:
                # Group chunks by voice while preserving their original indices
                voice_batches: Dict[str, List[ChunkState]] = {}
                for chunk in job.chunks:
                    v = chunk.voice or job.voice
                    voice_batches.setdefault(v, []).append(chunk)
                logger.info(
                    f"Job {job.id} is a podcast with {len(distinct_voices)} voices; "
                    f"processing {len(voice_batches)} voice batches"
                )
                chunks_to_process: List[ChunkState] = []
                for v in sorted(voice_batches.keys()):
                    chunks_to_process.extend(voice_batches[v])
            else:
                chunks_to_process = job.chunks

            timeout = aiohttp.ClientTimeout(total=300)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for chunk in chunks_to_process:
                    if job._cancel_event.is_set():
                        chunk.status = JobStatus.CANCELLED
                        await self._emit(job, {"event": "cancelled", "data": job.to_dict()})
                        return

                    if chunk.status == JobStatus.DONE:
                        continue

                    # Health check before each chunk to catch stuck backend early
                    if not await self._check_backend_health(job.backend_url):
                        logger.warning(f"Backend health check failed before chunk {chunk.index}, proceeding anyway")

                    chunk.status = JobStatus.RUNNING
                    await self._emit(job, {"event": "chunk_start", "chunk_index": chunk.index, "data": job.to_dict()})

                    chunk_start = time.time()
                    chunk_timeout = self._chunk_timeout(chunk.text)
                    voice = chunk.voice or job.voice
                    try:
                        if no_drift:
                            coro = self._generate_chunk_custom_voice(
                                session,
                                chunk.text,
                                voice,
                                job.language,
                                job.backend_url,
                            )
                        else:
                            ref_audio_path, ref_text = await _get_voice(voice)
                            coro = self._generate_chunk_voice_clone(
                                session,
                                chunk.text,
                                voice,
                                job.language,
                                ref_audio_path,
                                ref_text,
                                job.backend_url,
                            )
                        audio_bytes = await asyncio.wait_for(coro, timeout=chunk_timeout)
                        chunk_mp3 = chunks_dir / f"chunk_{chunk.index:03d}.mp3"
                        with open(chunk_mp3, "wb") as f:
                            f.write(audio_bytes)

                        chunk.elapsed_seconds = time.time() - chunk_start
                        chunk.status = JobStatus.DONE

                        # Update ETA based on actual timing across ALL remaining chunks
                        chars_done = len(chunk.text)
                        if chunk.elapsed_seconds > 0:
                            cps = chars_done / chunk.elapsed_seconds
                            # Use a running average of speed
                            if job.chars_per_second:
                                job.chars_per_second = (job.chars_per_second + cps) / 2
                            else:
                                job.chars_per_second = cps
                        remaining_chars = sum(
                            len(c.text) for c in job.chunks if c.status not in (JobStatus.DONE, JobStatus.FAILED)
                        )
                        if job.chars_per_second and job.chars_per_second > 0:
                            job.eta_seconds = remaining_chars / job.chars_per_second
                        else:
                            job.eta_seconds = remaining_chars / 19.0

                        await self._emit(job, {"event": "chunk_done", "chunk_index": chunk.index, "data": job.to_dict()})

                    except asyncio.TimeoutError:
                        chunk.status = JobStatus.FAILED
                        chunk.error = f"Timed out after {chunk_timeout:.0f}s (expected ~{len(chunk.text)/19.0:.0f}s)"
                        chunk.elapsed_seconds = time.time() - chunk_start
                        logger.error(f"Job {job.id} chunk {chunk.index} timed out after {chunk_timeout:.0f}s")
                        await self._emit(job, {"event": "chunk_failed", "chunk_index": chunk.index, "error": chunk.error, "data": job.to_dict()})

                    except asyncio.CancelledError:
                        chunk.status = JobStatus.CANCELLED
                        raise
                    except Exception as exc:
                        chunk.status = JobStatus.FAILED
                        chunk.error = str(exc)
                        chunk.elapsed_seconds = time.time() - chunk_start
                        logger.error(f"Job {job.id} chunk {chunk.index} failed: {exc}")
                        await self._emit(job, {"event": "chunk_failed", "chunk_index": chunk.index, "error": str(exc), "data": job.to_dict()})

            # After loop: check if any chunk succeeded
            successful = [c for c in job.chunks if c.status == JobStatus.DONE]
            if not successful:
                job.status = JobStatus.FAILED
                await self._emit(job, {"event": "failed", "error": "All chunks failed", "data": job.to_dict()})
                return

            # Concatenate all successful chunks in order
            mp3_paths = []
            for c in job.chunks:
                mp3 = chunks_dir / f"chunk_{c.index:03d}.mp3"
                if mp3.exists():
                    mp3_paths.append(mp3)

            final_mp3 = _OUTPUT_DIR / job.id / "final.mp3"
            final_mp3.parent.mkdir(parents=True, exist_ok=True)
            concatenate_mp3s(mp3_paths, final_mp3)
            job.final_mp3_path = final_mp3

            # Create ZIP
            zip_path = _OUTPUT_DIR / job.id / "chunks.zip"
            create_zip_archive(mp3_paths, final_mp3, zip_path)
            job.zip_path = zip_path

            job.status = JobStatus.DONE
            job.completed_at = time.time()
            job.eta_seconds = 0
            await self._emit(job, {"event": "done", "data": job.to_dict()})
            logger.info(f"Job {job.id} completed in {job.completed_at - job.started_at:.1f}s")

        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            await self._emit(job, {"event": "cancelled", "data": job.to_dict()})
            logger.info(f"Job {job.id} was cancelled")
        except Exception as exc:
            job.status = JobStatus.FAILED
            logger.exception(f"Job {job.id} failed unexpectedly")
            await self._emit(job, {"event": "failed", "error": str(exc), "data": job.to_dict()})
