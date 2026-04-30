# coding=utf-8
"""
Qwen3-TTS MCP Server — exposes TTS capabilities as MCP tools for OpenWebUI / Claude / etc.
Transport: streamable-http (default) or stdio
Auth: optional Bearer token via MCP_AUTH_TOKEN env var

Design notes for LLM callers:
- generate_speech creates a job and tries to finish it within ~15s.
  If the text is long it returns a job_id; the LLM MUST call
  check_job_status(job_id='...') to poll until done.
- The job continues running server-side even if the tool call returns early.
- retry_failed_chunks follows the same pattern.
- get_audio_file is for audio-capable models that want base64 on demand.
"""

import asyncio
import base64
import json
import logging
import os
import time
from typing import List, Optional

import aiohttp
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import AudioContent, ResourceLink, TextContent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public URL resolution (manual override only)
# ---------------------------------------------------------------------------
def _resolve_public_url() -> str:
    """Return the externally-reachable gateway URL.

    Priority:
    1. PUBLIC_GATEWAY_URL env var (full URL, e.g. http://192.168.1.50:8885)
    2. SERVER_HOSTNAME env var (hostname or IP, e.g. 192.168.1.50)
    3. Fallback to localhost (same-machine usage only)

    NOTE: Auto-detecting the container IP via socket would return a Docker
    internal address (172.x.x.x) which is NOT reachable from other machines.
    For remote access you MUST set SERVER_HOSTNAME or PUBLIC_GATEWAY_URL.
    """
    full_url = os.getenv("PUBLIC_GATEWAY_URL")
    if full_url:
        return full_url

    hostname = os.getenv("SERVER_HOSTNAME")
    if hostname:
        return f"http://{hostname}:8885"

    return "http://localhost:8885"


PUBLIC_GATEWAY_URL = _resolve_public_url()
logger.info(f"Resolved public gateway URL: {PUBLIC_GATEWAY_URL}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN")
TTS_URL = os.getenv("TTS_URL", "http://qwen3-tts-extended:8883")
TTS_BACKEND_URL = os.getenv("TTS_BACKEND_URL", "http://qwen3-tts-gpu:8880")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "Gerd")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "German")
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "900"))
SHORT_POLL_SECONDS = 15.0
CHARS_PER_SECOND_ESTIMATE = 19.0

# German language markers
_GERMAN_CHARS = set("äöüßÄÖÜ")
_GERMAN_WORDS = {
    "die", "der", "das", "und", "ist", "zu", "mit", "auf", "für", "von",
    "ein", "eine", "nicht", "ich", "du", "er", "sie", "es", "wir", "ihr",
    "sie", "den", "dem", "des", "im", "an", "als", "auch", "wie", "bei",
    "oder", "aus", "nach", "aber", "war", "wird", "hat", "um", "man", "noch",
    "sein", "wurde", "durch", "kann", "sich", "nur", "vor", "zur", "gegen",
}

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Qwen3-TTS",
    instructions=(
        "Qwen3-TTS MCP Server provides text-to-speech generation tools.\n"
        "generate_speech creates a job and either returns the finished audio "
        "(short texts) or a job_id for polling. If you receive a job_id, "
        "call check_job_status(job_id='...') repeatedly until status is 'done'.\n"
        "retry_failed_chunks retries failed chunks and follows the same pattern.\n"
        "list_voices shows available voices.\n"
        "get_audio_file returns base64 audio for audio-capable models."
    ),
)


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _check_auth(ctx: Optional[Context]) -> bool:
    """Return True if request is authorized (or no token configured)."""
    if not MCP_AUTH_TOKEN:
        return True
    if ctx is None:
        return False
    try:
        request = ctx.request_context.request
        auth = request.headers.get("authorization", "")
        return auth.startswith("Bearer ") and auth[7:] == MCP_AUTH_TOKEN
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
def _detect_language(text: str) -> str:
    """Heuristic language detection: German vs English fallback."""
    if not text:
        return DEFAULT_LANGUAGE
    text_lower = text.lower()
    if any(c in text for c in _GERMAN_CHARS):
        return "German"
    words = set(text_lower.split())
    german_score = len(words & _GERMAN_WORDS)
    if german_score >= 2:
        return "German"
    english_words = {"the", "and", "is", "to", "of", "a", "in", "that", "it", "for"}
    english_score = len(words & english_words)
    if english_score >= 2:
        return "English"
    return DEFAULT_LANGUAGE


# ---------------------------------------------------------------------------
# HTTP helpers (all timeouts bumped to 15 min)
# ---------------------------------------------------------------------------
async def _http_post(path: str, payload: dict, timeout: int = 900) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{TTS_URL}{path}", json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            body = await resp.json() if resp.status == 200 else await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"TTS returned {resp.status}: {body[:500]}")
            return body


async def _http_get(path: str, timeout: int = 900) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{TTS_URL}{path}", timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            body = await resp.json() if resp.status == 200 else await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"TTS returned {resp.status}: {body[:500]}")
            return body


async def _download_mp3(job_id: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{TTS_URL}/api/jobs/{job_id}/download/final",
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Download returned {resp.status}")
            return await resp.read()


async def _http_delete(path: str, timeout: int = 60) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.delete(
            f"{TTS_URL}{path}", timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            body = await resp.json() if resp.status == 200 else await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"TTS returned {resp.status}: {body[:500]}")
            return body


# ---------------------------------------------------------------------------
# Audio response text builder
# ---------------------------------------------------------------------------
def _make_audio_response_text(download_url: str) -> str:
    """Build a text block with multiple audio embed attempts for OpenWebUI."""
    lines = [
        download_url,
        "",
        f"![audio]({download_url})",
        "",
        f"[Download MP3]({download_url})",
        "",
        f'<audio controls src="{download_url}"></audio>',
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Poll helpers
# ---------------------------------------------------------------------------
async def _poll_job_short(
    job_id: str,
    ctx: Optional[Context],
    max_wait: float = SHORT_POLL_SECONDS,
) -> dict:
    """Poll a TTS job for up to max_wait seconds. Return latest job dict."""
    start = time.time()
    last_progress = 0.0

    while time.time() - start < max_wait:
        job = await _http_get(f"/api/jobs/{job_id}")
        status = job.get("status", "unknown")

        if status in ("done", "failed", "cancelled"):
            return job

        total = job.get("total_chunks", 1)
        completed = job.get("completed_chunks", 0)
        progress = completed / total if total > 0 else 0.0

        if ctx and progress > last_progress:
            try:
                await ctx.report_progress(
                    progress=progress,
                    total=1.0,
                    message=f"Chunk {completed}/{total} complete",
                )
            except Exception:
                pass
            last_progress = progress

        await asyncio.sleep(2.5)

    # Return current status after timeout
    return await _http_get(f"/api/jobs/{job_id}")


async def _poll_job_full(
    job_id: str,
    ctx: Optional[Context],
    timeout: int = JOB_TIMEOUT_SECONDS,
) -> dict:
    """Poll a TTS job until completion, failure, or timeout."""
    start = time.time()
    last_progress = 0.0

    while time.time() - start < timeout:
        job = await _http_get(f"/api/jobs/{job_id}")
        status = job.get("status", "unknown")

        if status in ("done", "failed", "cancelled"):
            return job

        total = job.get("total_chunks", 1)
        completed = job.get("completed_chunks", 0)
        progress = completed / total if total > 0 else 0.0

        if ctx and progress > last_progress:
            try:
                await ctx.report_progress(
                    progress=progress,
                    total=1.0,
                    message=f"Chunk {completed}/{total} complete",
                )
            except Exception:
                pass
            last_progress = progress

        await asyncio.sleep(2.5)

    return await _http_get(f"/api/jobs/{job_id}")


def _format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    return f"{minutes:.1f}min"


def _build_running_response(
    job_id: str,
    job: dict,
    tool_name: str,
) -> List[TextContent]:
    """Build a response for a job that is still running."""
    total_chunks = job.get("total_chunks", 1)
    completed = job.get("completed_chunks", 0)
    status = job.get("status", "unknown")
    total_eta = job.get("eta_seconds", 0)
    chunks = job.get("chunks", [])

    lines = [
        f"Job {job_id} is running (status: {status}).",
        f"Chunks: {total_chunks} total | {completed} complete",
    ]

    if total_eta > 0:
        lines.append(f"Estimated total time: {_format_eta(total_eta)}")

    lines.append("")
    lines.append("Per-chunk progress:")
    for c in chunks:
        chunk_text = c.get("text", "")
        chunk_eta = len(chunk_text) / CHARS_PER_SECOND_ESTIMATE if chunk_text else 0
        elapsed = c.get("elapsed_seconds")
        eta_str = f" (~{_format_eta(chunk_eta)} est.)" if chunk_eta > 0 and c.get("status") != "done" else ""
        elapsed_str = f" | took {elapsed:.1f}s" if elapsed else ""
        lines.append(f"  Chunk {c['index']}: {c['status']}{eta_str}{elapsed_str}")

    lines.append("")
    lines.append(f"Call check_job_status(job_id='{job_id}') to poll for completion.")
    lines.append(f"You can also retry failed chunks with: retry_failed_chunks(job_id='{job_id}')")

    return [TextContent(type="text", text="\n".join(lines))]


def _build_done_response(
    job_id: str,
    job: dict,
    voice: str,
    language: str,
    text_len: int,
) -> List:
    """Build a response for a completed job with audio links."""
    total_time = job.get("completed_at", 0) - job.get("started_at", 0)
    chunks_info = job.get("chunks", [])
    failed = [c for c in chunks_info if c.get("status") == "failed"]
    cps = job.get("chars_per_second")

    # Detect podcast (multiple distinct voices across chunks)
    voices = {c.get("voice", voice) for c in chunks_info}
    is_podcast = len(voices) > 1

    if is_podcast:
        summary_lines = [
            f"Podcast generation complete!",
            f"Job ID: {job_id}",
            f"Speakers: {', '.join(sorted(voices))} | Language: {language}",
            f"Chunks: {len(chunks_info)} | Characters: {text_len}",
        ]
    else:
        summary_lines = [
            f"Speech generation complete!",
            f"Job ID: {job_id}",
            f"Voice: {voice} | Language: {language}",
            f"Chunks: {len(chunks_info)} | Characters: {text_len}",
        ]
    if cps:
        summary_lines.append(f"Speed: {cps:.1f} chars/sec")
    if total_time > 0:
        summary_lines.append(f"Total time: {total_time:.1f}s")
    if failed:
        summary_lines.append(f"⚠️ {len(failed)} chunk(s) failed — call retry_failed_chunks(job_id='{job_id}') to retry.")

    summary_lines.append("")
    summary_lines.append("Per-chunk breakdown:")
    for c in chunks_info:
        elapsed = c.get("elapsed_seconds")
        elapsed_str = f" ({elapsed:.1f}s)" if elapsed else ""
        chunk_voice = c.get("voice", voice)
        if is_podcast:
            summary_lines.append(f"  Chunk {c['index']}: {chunk_voice} — {c['status']}{elapsed_str}")
        else:
            summary_lines.append(f"  Chunk {c['index']}: {c['status']}{elapsed_str}")

    download_url = f"{PUBLIC_GATEWAY_URL}/api/jobs/{job_id}/download/final"
    summary_lines.append("")
    summary_lines.append(_make_audio_response_text(download_url))

    return [
        TextContent(type="text", text="\n".join(summary_lines)),
        ResourceLink(
            type="resource_link",
            uri=download_url,
            name=f"{job_id}_speech.mp3",
            description="Download the generated speech MP3",
            mimeType="audio/mpeg",
        ),
    ]


def _build_failed_response(
    job_id: str,
    job: dict,
) -> List[TextContent]:
    """Build a response for a failed job — always includes job_id."""
    chunks_info = job.get("chunks", [])
    failed = [c for c in chunks_info if c.get("status") == "failed"]
    errors = "\n".join(
        f"Chunk {c['index']}: {c.get('error', 'unknown error')}" for c in failed
    )

    lines = [
        f"TTS generation failed for job {job_id}.",
        f"Status: {job.get('status', 'unknown')}",
        "",
        "Failed chunks:",
        errors if errors else "  (no specific chunk errors reported)",
        "",
        f"You can retry with: retry_failed_chunks(job_id='{job_id}')",
        f"Or check status again with: check_job_status(job_id='{job_id}')",
    ]

    return [TextContent(type="text", text="\n".join(lines))]


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------
@mcp.tool()
async def list_voices(ctx: Context) -> List:
    """List all available custom voices for TTS generation.

    Returns voice names, whether they have transcripts (ICL mode),
    and a short description of each voice.
    """
    if not _check_auth(ctx):
        raise RuntimeError("Unauthorized: invalid or missing Bearer token")

    try:
        data = await _http_get("/api/voices")
        voices = data.get("voices", [])
        lines = [f"Available voices ({len(voices)}):"]
        for v in voices:
            mode = "ICL (transcript-based)" if v.get("has_transcript") else "X-Vector (speaker embedding only)"
            transcript_preview = ""
            if v.get("transcript"):
                preview = v["transcript"][:60].replace("\n", " ")
                transcript_preview = f' | Sample: "{preview}..."'
            lines.append(f"- {v['name']} ({mode}){transcript_preview}")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as exc:
        raise RuntimeError(f"Failed to list voices: {exc}") from exc


@mcp.tool()
async def generate_speech(
    text: str,
    voice: str = "Gerd",
    language: str = "auto",
    no_drift: bool = True,
    ctx: Context = None,
) -> List:
    """Generate spoken audio from text using Qwen3-TTS.

    This tool creates a TTS job and tries to finish it within ~15 seconds.
    For short texts it returns the finished audio immediately.
    For longer texts it returns a job_id — you MUST call
    check_job_status(job_id='...') repeatedly until status is 'done'.
    The job continues running server-side even if this call returns early.

    Args:
        text: The text to speak. Can be very long (automatically chunked).
        voice: Voice name (e.g. "Gerd", "Anke", "Nina", "MargritS").
               Use list_voices to discover available voices.
        language: Language code — "German", "English", "Chinese", etc.
                  Set to "auto" to let the server detect the language.
        no_drift: If True (default), uses cached voice prompts for consistent
                  voice across chunks. Slightly slower first chunk but no drift.
    """
    if not _check_auth(ctx):
        raise RuntimeError("Unauthorized: invalid or missing Bearer token")

    if not text or not text.strip():
        raise RuntimeError("Error: text cannot be empty")

    detected_lang = _detect_language(text) if language == "auto" else language
    text = text.strip()

    try:
        if ctx:
            try:
                await ctx.report_progress(progress=0.0, total=1.0, message="Creating TTS job...")
            except Exception:
                pass

        # Create job
        payload = {
            "text": text,
            "voice": voice,
            "language": detected_lang,
            "model": "1.7B",
            "no_drift": no_drift,
        }
        job = await _http_post("/api/jobs", payload)
        job_id = job["id"]
        total_chunks = job.get("total_chunks", 1)
        eta_seconds = job.get("eta_seconds", 0)

        logger.info(f"generate_speech: created job {job_id} ({total_chunks} chunks, ~{eta_seconds:.1f}s est.)")

        # Short-poll: try to finish within ~15s
        final_job = await _poll_job_short(job_id, ctx, max_wait=SHORT_POLL_SECONDS)
        status = final_job.get("status", "unknown")

        if status == "done":
            return _build_done_response(job_id, final_job, voice, detected_lang, len(text))

        if status == "failed":
            return _build_failed_response(job_id, final_job)

        if status == "cancelled":
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Job {job_id} was cancelled.\n"
                        f"Check status with: check_job_status(job_id='{job_id}')"
                    ),
                )
            ]

        # Still running — return job_id with ETAs for polling
        return _build_running_response(job_id, final_job, "generate_speech")

    except Exception as exc:
        logger.exception("generate_speech failed")
        raise RuntimeError(f"Error generating speech: {exc}") from exc


@mcp.tool()
async def check_job_status(job_id: str, ctx: Context = None) -> List:
    """Check the status of a previously created TTS job and get results when done.

    Use this after generate_speech or retry_failed_chunks returns a running job_id.
    When status is 'done', this response includes the download link and audio.
    """
    if not _check_auth(ctx):
        raise RuntimeError("Unauthorized: invalid or missing Bearer token")

    try:
        job = await _http_get(f"/api/jobs/{job_id}")
        status = job.get("status", "unknown")

        if status == "done":
            # Try to infer voice/language from job metadata if available
            voice = job.get("voice", "unknown")
            language = job.get("language", "unknown")
            text_len = sum(len(c.get("text", "")) for c in job.get("chunks", []))
            return _build_done_response(job_id, job, voice, language, text_len)

        if status == "failed":
            return _build_failed_response(job_id, job)

        if status == "cancelled":
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Job {job_id} was cancelled.\n"
                        f"You can retry failed chunks with: retry_failed_chunks(job_id='{job_id}')"
                    ),
                )
            ]

        # Still running
        return _build_running_response(job_id, job, "check_job_status")

    except Exception as exc:
        raise RuntimeError(f"Error checking job: {exc}") from exc


@mcp.tool()
async def retry_failed_chunks(job_id: str, ctx: Context = None) -> List:
    """Retry any failed chunks in a previously created TTS job.

    This tool retries failed chunks and then polls for up to ~15 seconds.
    If the job is not done yet, it returns a job_id — call
    check_job_status(job_id='...') repeatedly until status is 'done'.
    The retry continues server-side even if this call returns early.

    Args:
        job_id: The job ID to retry failed chunks for.
    """
    if not _check_auth(ctx):
        raise RuntimeError("Unauthorized: invalid or missing Bearer token")

    try:
        # First check which chunks are failed
        job = await _http_get(f"/api/jobs/{job_id}")
        chunks = job.get("chunks", [])
        failed_indices = [c["index"] for c in chunks if c.get("status") == "failed"]

        if not failed_indices:
            # Job might already be done — check and return appropriately
            status = job.get("status", "unknown")
            if status == "done":
                voice = job.get("voice", "unknown")
                language = job.get("language", "unknown")
                text_len = sum(len(c.get("text", "")) for c in chunks)
                return _build_done_response(job_id, job, voice, language, text_len)
            return [
                TextContent(
                    type="text",
                    text=(
                        f"No failed chunks found in job {job_id}.\n"
                        f"Current status: {status}.\n"
                        f"Check again with: check_job_status(job_id='{job_id}')"
                    ),
                )
            ]

        # Retry each failed chunk
        retried = []
        for idx in failed_indices:
            await _http_post(f"/api/jobs/{job_id}/retry/{idx}", {})
            retried.append(idx)

        logger.info(f"retry_failed_chunks: retried chunks {retried} for job {job_id}")

        # Short-poll: try to finish within ~15s
        final_job = await _poll_job_short(job_id, ctx, max_wait=SHORT_POLL_SECONDS)
        status = final_job.get("status", "unknown")

        if status == "done":
            voice = final_job.get("voice", "unknown")
            language = final_job.get("language", "unknown")
            text_len = sum(len(c.get("text", "")) for c in final_job.get("chunks", []))
            return _build_done_response(job_id, final_job, voice, language, text_len)

        if status == "failed":
            return _build_failed_response(job_id, final_job)

        if status == "cancelled":
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Job {job_id} was cancelled after retrying chunks {retried}.\n"
                        f"Check status with: check_job_status(job_id='{job_id}')"
                    ),
                )
            ]

        # Still running
        running_resp = _build_running_response(job_id, final_job, "retry_failed_chunks")
        # Prepend retry info
        if running_resp and hasattr(running_resp[0], "text"):
            running_resp[0].text = (
                f"Retried chunks: {retried}.\n\n" + running_resp[0].text
            )
        return running_resp

    except Exception as exc:
        logger.exception("retry_failed_chunks failed")
        raise RuntimeError(f"Error retrying chunks: {exc}") from exc


@mcp.tool()
async def cancel_job(job_id: str, ctx: Context = None) -> List:
    """Cancel a running or pending TTS job.

    Args:
        job_id: The job ID to cancel.
    """
    if not _check_auth(ctx):
        raise RuntimeError("Unauthorized: invalid or missing Bearer token")

    try:
        result = await _http_delete(f"/api/jobs/{job_id}")
        if result.get("cancelled"):
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Job {job_id} cancelled successfully.\n"
                        f"If you need the audio, you must create a new job."
                    ),
                )
            ]
        return [
            TextContent(
                type="text",
                text=(
                    f"Job {job_id} could not be cancelled.\n"
                    f"It may already be finished or not found.\n"
                    f"Check status with: check_job_status(job_id='{job_id}')"
                ),
            )
        ]
    except Exception as exc:
        raise RuntimeError(f"Error cancelling job: {exc}") from exc


@mcp.tool()
async def create_podcast(
    turns: list,
    language: str = "auto",
    no_drift: bool = True,
    ctx: Context = None,
) -> List:
    """Generate a multi-speaker podcast-style audio from dialogue turns.

    Each turn is spoken by a different voice. Long turns are automatically
    chunked while keeping the same voice. All chunks are combined into a
    single audio file that plays like a podcast.

    This tool creates a job and tries to finish within ~15 seconds.
    For longer dialogues it returns a job_id — you MUST call
    check_job_status(job_id='...') repeatedly until status is 'done'.

    Args:
        turns: List of turns, each as {"speaker": "VoiceName", "text": "..."}.
               Use list_voices to discover available voices.
        language: Language code — "German", "English", etc.
                  Set to "auto" to detect from the first turn's text.
        no_drift: If True (default), uses cached voice prompts for consistent
                  voices across chunks. Slightly slower first chunk but no drift.
    """
    if not _check_auth(ctx):
        raise RuntimeError("Unauthorized: invalid or missing Bearer token")

    if not turns or not isinstance(turns, list):
        raise RuntimeError("Error: turns must be a non-empty list")

    # Validate turns structure
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise RuntimeError(f"Error: turn {i} must be an object with 'speaker' and 'text'")
        if "speaker" not in turn or "text" not in turn:
            raise RuntimeError(f"Error: turn {i} missing 'speaker' or 'text'")
        if not turn["text"].strip():
            raise RuntimeError(f"Error: turn {i} text cannot be empty")

    # Detect language from first turn if auto
    first_text = turns[0]["text"]
    detected_lang = _detect_language(first_text) if language == "auto" else language

    # Calculate total text length for stats
    total_text_len = sum(len(t["text"]) for t in turns)

    try:
        if ctx:
            try:
                await ctx.report_progress(progress=0.0, total=1.0, message="Creating podcast job...")
            except Exception:
                pass

        payload = {
            "turns": turns,
            "language": detected_lang,
            "model": "1.7B",
            "no_drift": no_drift,
        }
        job = await _http_post("/api/podcast", payload)
        job_id = job["id"]
        total_chunks = job.get("total_chunks", 1)
        eta_seconds = job.get("eta_seconds", 0)

        logger.info(f"create_podcast: created job {job_id} ({total_chunks} chunks, ~{eta_seconds:.1f}s est.)")

        # Short-poll: try to finish within ~15s
        final_job = await _poll_job_short(job_id, ctx, max_wait=SHORT_POLL_SECONDS)
        status = final_job.get("status", "unknown")

        if status == "done":
            return _build_done_response(job_id, final_job, "multi-speaker", detected_lang, total_text_len)

        if status == "failed":
            return _build_failed_response(job_id, final_job)

        if status == "cancelled":
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Job {job_id} was cancelled.\n"
                        f"Check status with: check_job_status(job_id='{job_id}')"
                    ),
                )
            ]

        # Still running — return job_id with ETAs for polling
        return _build_running_response(job_id, final_job, "create_podcast")

    except Exception as exc:
        logger.exception("create_podcast failed")
        raise RuntimeError(f"Error creating podcast: {exc}") from exc


@mcp.tool()
async def get_audio_file(job_id: str, ctx: Context = None) -> List:
    """Download the final audio file for a completed TTS job as base64.

    This tool is intended for audio-capable LLM models that can process
    raw audio data. For regular use, the download link from check_job_status
    or generate_speech is sufficient.

    Args:
        job_id: The completed job ID.
    """
    if not _check_auth(ctx):
        raise RuntimeError("Unauthorized: invalid or missing Bearer token")

    try:
        mp3_bytes = await _download_mp3(job_id)
        b64_audio = base64.b64encode(mp3_bytes).decode("utf-8")
        download_url = f"{PUBLIC_GATEWAY_URL}/api/jobs/{job_id}/download/final"
        text_lines = [
            f"Audio for job {job_id}:",
            "",
            _make_audio_response_text(download_url),
        ]
        return [
            TextContent(type="text", text="\n".join(text_lines)),
            AudioContent(type="audio", data=b64_audio, mimeType="audio/mpeg"),
            ResourceLink(
                type="resource_link",
                uri=download_url,
                name=f"{job_id}_speech.mp3",
                description="Download the generated speech MP3",
                mimeType="audio/mpeg",
            ),
        ]
    except Exception as exc:
        raise RuntimeError(f"Error downloading audio: {exc}") from exc


# ---------------------------------------------------------------------------
# Auth middleware + entry point
# ---------------------------------------------------------------------------
class _AuthMiddleware:
    """Lightweight ASGI auth middleware."""

    def __init__(self, app, token: Optional[str] = None):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.token:
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode("utf-8", errors="ignore")
        if not auth.startswith("Bearer ") or auth[7:] != self.token:
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        [b"content-type", b"application/json"],
                        [b"www-authenticate", b"Bearer"],
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps({"error": "Unauthorized"}).encode(),
                }
            )
            return

        await self.app(scope, receive, send)


if __name__ == "__main__":
    import uvicorn

    mcp.settings.host = os.getenv("HOST", "0.0.0.0")
    mcp.settings.port = int(os.getenv("PORT", "8886"))

    logger.info(f"Starting Qwen3-TTS MCP Server (tts_url={TTS_URL}, gateway={PUBLIC_GATEWAY_URL})")
    if MCP_AUTH_TOKEN:
        logger.info("Bearer token authentication is ENABLED")
    else:
        logger.info("Bearer token authentication is DISABLED")

    # Disable DNS rebinding protection so the server accepts requests from
    # Docker network aliases (e.g. qwen3-tts-mcp:8886) and LAN IPs.
    mcp.settings.transport_security.enable_dns_rebinding_protection = False

    # Build the MCP streamable HTTP ASGI app
    mcp_app = mcp.streamable_http_app()

    # Wrap with optional auth
    app = _AuthMiddleware(mcp_app, token=MCP_AUTH_TOKEN)

    uvicorn.run(
        app,
        host=mcp.settings.host,
        port=mcp.settings.port,
        log_level="info",
    )
