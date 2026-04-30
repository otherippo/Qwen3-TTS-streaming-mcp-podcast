#!/usr/bin/env python3
"""Generate comparison samples from 1.7B and 0.6B backends."""
import os
import requests

TEXT = "Guten Tag! Willkommen bei der deutschen Sprachausgabe. Dies ist ein Test zur Qualitätsbewertung verschiedener Modelle."
VOICE = "Anke"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for name, port in [("1.7B", 8880), ("0.6B", 8882)]:
    url = f"http://localhost:{port}/v1/audio/speech"
    resp = requests.post(url, json={
        "model": "qwen3-tts",
        "input": TEXT,
        "voice": VOICE,
        "language": "German",
        "response_format": "mp3",
    }, timeout=120)
    resp.raise_for_status()
    fname = os.path.join(OUTPUT_DIR, f"sample_{name}.mp3")
    with open(fname, "wb") as f:
        f.write(resp.content)
    print(f"Saved {name} sample: {fname} ({len(resp.content)} bytes)")
