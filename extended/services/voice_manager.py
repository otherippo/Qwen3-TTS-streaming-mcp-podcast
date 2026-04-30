# coding=utf-8
"""
Voice management: runtime discovery, ffmpeg normalization, on-demand caching.
"""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles

logger = logging.getLogger(__name__)

_VOICES_DIR = Path(os.getenv("VOICES_DIR", "/app/voices"))
_CACHE_DIR = Path(os.getenv("CACHE_DIR", "/app/cache")) / "voices"


def _ensure_dirs() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _voice_hash(wav_path: Path) -> str:
    """Simple hash based on file mtime and size for cache invalidation."""
    stat = wav_path.stat()
    data = f"{wav_path.name}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _discover_wav_files() -> List[Path]:
    """Return all .wav files in voices directory."""
    if not _VOICES_DIR.exists():
        return []
    return sorted([f for f in _VOICES_DIR.iterdir() if f.suffix.lower() == ".wav"])


def _find_transcript(wav_path: Path) -> Optional[str]:
    """Look for a matching .txt file next to the .wav."""
    txt_path = wav_path.with_suffix(".txt")
    if txt_path.exists():
        try:
            content = txt_path.read_text(encoding="utf-8").strip()
            return content if content else None
        except Exception as exc:
            logger.warning(f"Could not read transcript {txt_path}: {exc}")
    return None


async def _run_ffmpeg(input_path: Path, output_path: Path) -> None:
    """Normalize audio with ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-t", "10",
        "-ar", "24000",
        "-ac", "1",
        "-af",
        "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=24000",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="ignore")[-500:]
        raise RuntimeError(f"ffmpeg failed: {err}")


class VoiceManager:
    """Handles runtime voice discovery and on-demand preprocessing."""

    def __init__(self):
        _ensure_dirs()
        self._cache: Dict[str, dict] = {}

    def _cache_entry_path(self, voice_name: str, voice_hash: str) -> Path:
        return _CACHE_DIR / f"{voice_name}_{voice_hash}"

    def discover_voices(self) -> List[dict]:
        """Return list of discovered voices with metadata."""
        voices = []
        for wav_path in _discover_wav_files():
            name = wav_path.stem
            transcript = _find_transcript(wav_path)
            vhash = _voice_hash(wav_path)
            cache_path = self._cache_entry_path(name, vhash)
            cached = (cache_path / "preprocessed.wav").exists()
            voices.append({
                "name": name,
                "source": str(wav_path),
                "transcript": transcript,
                "has_transcript": transcript is not None,
                "cached": cached,
                "hash": vhash,
            })
        return voices

    async def get_preprocessed_voice(self, voice_name: str) -> Tuple[Path, Optional[str]]:
        """
        Return path to preprocessed wav and optional transcript.
        Preprocesses on first use if not cached.
        """
        wav_path = _VOICES_DIR / f"{voice_name}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Voice '{voice_name}' not found at {wav_path}")

        transcript = _find_transcript(wav_path)
        vhash = _voice_hash(wav_path)
        cache_entry = self._cache_entry_path(voice_name, vhash)
        processed_wav = cache_entry / "preprocessed.wav"
        meta_path = cache_entry / "meta.json"

        if processed_wav.exists() and meta_path.exists():
            # Verify hash match
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("hash") == vhash:
                    return processed_wav, transcript
            except Exception:
                pass

        # Need to preprocess
        logger.info(f"Preprocessing voice '{voice_name}'...")
        cache_entry.mkdir(parents=True, exist_ok=True)
        await _run_ffmpeg(wav_path, processed_wav)
        meta = {"name": voice_name, "hash": vhash, "source": str(wav_path)}
        async with aiofiles.open(meta_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(meta, indent=2))
        logger.info(f"Voice '{voice_name}' cached to {processed_wav}")
        return processed_wav, transcript

    def get_voice_names(self) -> List[str]:
        return [p.stem for p in _discover_wav_files()]

    def is_backend_custom_voice_ready(self, voice_name: str, target_dir: Path) -> bool:
        """Check if a custom voice directory already exists for the backend."""
        voice_dir = target_dir / voice_name
        ref_wav = voice_dir / "reference.wav"
        return ref_wav.exists()

    async def prepare_backend_custom_voice(self, voice_name: str, target_dir: Path) -> Path:
        """
        Create a backend-compatible custom voice directory.
        The backend's load_custom_voices() expects:
          <voice_name>/reference.wav
          <voice_name>/reference.txt
        Returns the directory path.
        """
        if self.is_backend_custom_voice_ready(voice_name, target_dir):
            return target_dir / voice_name

        # Get preprocessed voice and transcript
        processed_wav, transcript = await self.get_preprocessed_voice(voice_name)

        voice_dir = target_dir / voice_name
        voice_dir.mkdir(parents=True, exist_ok=True)
        voice_dir.chmod(0o777)

        ref_wav = voice_dir / "reference.wav"
        ref_txt = voice_dir / "reference.txt"

        # Copy preprocessed audio (don't symlink — backend may need stable path)
        import shutil
        shutil.copy2(str(processed_wav), str(ref_wav))
        ref_wav.chmod(0o666)

        if transcript:
            async with aiofiles.open(ref_txt, "w", encoding="utf-8") as f:
                await f.write(transcript)
            ref_txt.chmod(0o666)
        else:
            # Create empty reference.txt so backend knows it's there
            async with aiofiles.open(ref_txt, "w", encoding="utf-8") as f:
                await f.write("")
            ref_txt.chmod(0o666)

        logger.info(f"Prepared backend custom voice '{voice_name}' at {voice_dir}")
        return voice_dir
