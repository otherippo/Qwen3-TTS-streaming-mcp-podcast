# Qwen3-TTS Streaming

Streaming inference implementation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) that the official repo doesn't provide, plus a full Docker-based serving stack with MCP integration.

The official team mentions "Extreme Low-Latency Streaming Generation" in their paper and marketing, but the actual streaming code was never released - they point users to vLLM-Omni, which still doesn't support online serving.

This fork adds real streaming generation directly to the `qwen-tts` package, along with an **~6x inference speedup** vs upstream qwen-tts - both for non-streaming and streaming mode.

## What's Added

### Streaming
- `stream_generate_pcm()` - real-time PCM audio streaming
- `stream_generate_voice_clone()` - streaming with voice cloning

### Serving Stack (Docker)
- **Backend** — GPU-accelerated TTS API server (OpenAI-compatible `/v1/audio/speech`)
- **Extended Proxy** — Long-form TTS with intelligent chunking, job management, and Web UI
- **Gateway** — (Optional) Unified OpenAI-compatible endpoint for LLM + TTS
- **MCP Server** — Exposes TTS as MCP tools for OpenWebUI, Claude, and other MCP clients
- **Podcast generation** — Multi-speaker dialogue with per-chunk voice consistency

### MCP Tools
The MCP server provides these tools for LLM integrations:
- `generate_speech` — Generate speech from text (auto-chunks long texts)
- `check_job_status` — Poll job progress and get download links
- `retry_failed_chunks` — Retry failed chunks
- `cancel_job` — Cancel a running job
- `create_podcast` — Multi-speaker podcast generation
- `list_voices` — Discover available voices
- `get_audio_file` — Get base64 audio for audio-capable models

## Voice Samples

> **Important:** Voice quality depends heavily on the reference sample.

Place voice files in the `voices/` directory. Each voice consists of:
- `<name>.wav` — The reference audio sample
- `<name>.txt` — An exact transcript of what is spoken in the audio

### Requirements
- **Format:** WAV (any sample rate — will be normalized to 24kHz mono)
- **Duration:** 10–30 seconds recommended
- **Quality:** Clean recording, minimal background noise, clear speech

Stereo files are accepted and automatically converted to mono during preprocessing.

### ICL Mode (with transcript) — Recommended
When a `.txt` transcript is provided, the model uses in-context learning (ICL). This produces:
- Highly consistent voice across chunks and generations
- Accurate prosody and intonation matching the reference

### X-Vector Mode (without transcript)
If no `.txt` file is present, the model falls back to x-vector speaker embedding mode. This:
- Does not require a transcript
- Works with any audio containing speech
- **But:** Voice characteristics will vary more between generations — less consistent, more fluctuation

> **Recommendation:** Always provide an accurate transcript for production use. X-vector mode is useful for quick experiments or when transcripts are unavailable.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Gateway (:8885)         [Optional] OpenAI-compatible    │
│  ├─ /v1/chat/completions → llama.cpp                     │
│  ├─ /v1/audio/speech     → Backend                       │
│  └─ /api/*               → Extended Proxy                │
├──────────────────────────────────────────────────────────┤
│  Extended Proxy (:8883)  Chunking + Job Management + UI  │
│  ├─ /api/jobs            Create, poll, retry, cancel     │
│  ├─ /api/podcast         Multi-speaker podcast jobs      │
│  └─ /api/voices          Voice discovery                 │
├──────────────────────────────────────────────────────────┤
│  Backend (:8880)         GPU TTS Engine                  │
│  └─ /v1/audio/speech     Qwen3-TTS 1.7B Base            │
├──────────────────────────────────────────────────────────┤
│  MCP Server (:8886)      MCP Tools for LLMs              │
│  └─ Tools: generate_speech, check_job_status,            │
│            create_podcast, list_voices, ...               │
└──────────────────────────────────────────────────────────┘
```

The **Gateway** is optional. It combines LLM and TTS into a single OpenAI-compatible endpoint, useful if you want one URL for both chat and speech. Without it, you can use the Extended Proxy (Web UI + jobs), MCP Server (LLM tools), and Backend (direct API) independently.

## Docker Setup

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (~6.5 GB VRAM for the 1.7B model)

### 1. Clone the repository
```bash
git clone https://github.com/otherippo/Qwen3-TTS-streaming-mcp-podcast.git
cd Qwen3-TTS-streaming-mcp-podcast
```

### 2. Configure environment
```bash
cp .env.example .env
```

Edit `.env` and set:
- `SERVER_HOSTNAME` — Your machine's LAN IP (for access from other devices like OpenWebUI)
- `HF_CACHE_HOST_PATH` — Path to HuggingFace cache directory on the host
- `MCP_AUTH_TOKEN` — (Optional) Bearer token for MCP server authentication

### 3. Add voice samples
Place your voice files in `voices/`:
```
voices/
├── Gerd.wav
├── Gerd.txt        ← exact transcript of Gerd.wav
├── Anke.wav
├── Anke.txt        ← exact transcript of Anke.wav
└── ...
```

### 4. Build and start
```bash
docker compose up -d --build
```

The first build downloads the model (~3.5GB) and takes several minutes. Subsequent starts are fast.

### 5. Verify
```bash
curl http://localhost:8880/health    # Backend (GPU)
curl http://localhost:8883/health    # Extended Proxy
curl http://localhost:8885/health    # Gateway (optional)
```

### Services and Ports

| Service | Port | Description |
|---------|------|-------------|
| Backend | 8880 (mapped to 8884) | GPU TTS engine |
| Extended Proxy | 8883 | Chunking, jobs, Web UI |
| Gateway | 8885 | (Optional) Unified OpenAI-compatible API |
| MCP Server | 8886 | MCP tools endpoint |

### GPU Selection
The backend reserves GPU device `0` by default. Edit `docker-compose.yml` to change:
```yaml
device_ids: ['0']  # Change to '1' or other GPU index
```

### Connecting to llama.cpp
The Gateway optionally proxies LLM requests to a llama.cpp server. By default it connects to `llama-cpp-server-cuda-qwen-only:8080` on the `llama-cpp-network` Docker network. Adjust `LLAMA_URL` and network settings in `docker-compose.yml` if using a different setup. The Gateway can be disabled entirely if you don't need LLM+TTS unified routing.

## MCP Integration

### OpenWebUI
Add the MCP server in OpenWebUI settings:
- URL: `http://<your-server-ip>:8886/mcp`
- If auth is enabled, set the Bearer token

### Claude Desktop / Other MCP Clients
Use the streamable-HTTP transport:
```json
{
  "mcpServers": {
    "qwen3-tts": {
      "url": "http://<your-server-ip>:8886/mcp",
      "headers": {
        "Authorization": "Bearer <your-token>"
      }
    }
  }
}
```

## API Usage

### Generate speech (OpenAI-compatible)
```bash
curl http://localhost:8884/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts",
    "input": "Hallo, wie geht es Ihnen?",
    "voice": "Gerd",
    "language": "German"
  }' \
  --output speech.mp3
```

### Create a job (long text, via Extended Proxy)
```bash
curl -X POST http://localhost:8883/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text here...",
    "voice": "Anke",
    "language": "German"
  }'
```

### Create a podcast
```bash
curl -X POST http://localhost:8883/api/podcast \
  -H "Content-Type: application/json" \
  -d '{
    "turns": [
      {"speaker": "Anke", "text": "Welcome to our podcast!"},
      {"speaker": "Gerd", "text": "Thanks for having me."}
    ],
    "language": "German"
  }'
```

## Benchmark (RTX 5090)

### Non-streaming (full inference)

<img width="602" height="145" alt="image" src="https://github.com/user-attachments/assets/0cbfcc71-e854-46e2-81bc-ec3955ff3ff0" />

### Streaming

<img width="766" height="183" alt="image" src="https://github.com/user-attachments/assets/f5df9a38-e091-47ae-a08f-ef364f8710ea" />

## Python Usage (without Docker)

> **Note:** This only sets up the GPU backend for direct Python usage. It does **not** include the Extended Proxy (chunking, jobs, Web UI), MCP Server, or Gateway. For the full feature set, use the Docker setup above.

See `examples/` for streaming and non-streaming usage:
- [test_streaming_optimized.py](examples/test_streaming_optimized.py)
- [test_optimized_no_streaming.py](examples/test_optimized_no_streaming.py)

### Installation (Python 3.12)

> Note: torch versions differ between Linux/Windows due to available flash_attn prebuilt wheels.

**1. Install SOX**

Linux:
```bash
sudo apt install sox libsox-fmt-all
```

Windows: Download from https://sourceforge.net/projects/sox/ and add to PATH.

**2. Create environment**
```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

**3. Install dependencies**

Linux:
```bash
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.8/flash_attn-2.8.3%2Bcu130torch2.9-cp312-cp312-linux_x86_64.whl
```

Windows:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-win_amd64.whl
pip install -U "triton-windows<3.7"
```

**4. Install package**
```bash
pip install -e .
```

## Streaming Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 4 | Emit audio every N frames (~0.33s at 12Hz) |
| `decode_window_frames` | 80 | Decoder context window |

## Why This Exists

From official Qwen3-TTS README:
> Now only offline inference is supported. Online serving will be supported later.

This fork provides streaming now, without waiting for vLLM-Omni updates.

---

Based on [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache-2.0 License
