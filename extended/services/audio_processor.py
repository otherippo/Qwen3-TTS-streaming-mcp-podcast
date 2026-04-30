# coding=utf-8
"""
Audio processing: MP3 concatenation, ZIP packaging, per-chunk serving.
"""

import io
import logging
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not available — MP3 concatenation and ZIP will be limited")


def concatenate_mp3s(mp3_paths: List[Path], output_path: Path) -> None:
    """Concatenate multiple MP3 files into one using pydub."""
    if not PYDUB_AVAILABLE:
        raise RuntimeError("pydub is required for MP3 concatenation")
    if not mp3_paths:
        raise ValueError("No MP3 files to concatenate")

    segments: List[AudioSegment] = []
    for p in mp3_paths:
        seg = AudioSegment.from_mp3(str(p))
        segments.append(seg)

    combined = segments[0]
    for seg in segments[1:]:
        combined += seg

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_path), format="mp3", bitrate="192k")
    logger.info(f"Concatenated {len(mp3_paths)} chunks to {output_path}")


def create_zip_archive(mp3_paths: List[Path], final_path: Optional[Path], output_path: Path) -> None:
    """Create a ZIP with all chunk MP3s and optionally the final concatenated MP3."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, p in enumerate(mp3_paths, start=1):
            arcname = f"chunks/chunk_{idx:03d}.mp3"
            zf.write(p, arcname)
        if final_path and final_path.exists():
            zf.write(final_path, "final.mp3")
    logger.info(f"ZIP archive created: {output_path}")
