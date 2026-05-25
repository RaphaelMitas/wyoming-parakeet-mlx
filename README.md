# wyoming-parakeet-mlx

A [Wyoming protocol](https://github.com/OHF-Voice/wyoming) speech-to-text
server that runs NVIDIA's [Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
ASR model on Apple Silicon via
[`parakeet-mlx`](https://github.com/senstella/parakeet-mlx). Drops straight
into Home Assistant Assist, Wyoming satellites, or any other Wyoming
client as a local STT engine.

The default model is `mlx-community/parakeet-tdt-0.6b-v3` — a 0.6B-parameter
transducer covering 25 European languages.

> **Apple Silicon only.** MLX is macOS/arm64 only and cannot run inside
> a typical Linux Docker container. If you want Parakeet on Linux or
> NVIDIA GPUs, use an [NeMo](https://github.com/NVIDIA/NeMo)-based engine
> instead. See the FAQ at the bottom.

## Requirements

- macOS 14 (Sonoma) or newer on Apple Silicon (M1, M2, M3, M4)
- Python 3.11+
- `ffmpeg` on PATH (`brew install ffmpeg`) — used internally by
  `parakeet-mlx` to decode audio
- 16 GB unified memory recommended (8 GB minimum)

## Install

Recommended via [`uv`](https://github.com/astral-sh/uv):

```sh
brew install ffmpeg
uv tool install git+https://github.com/raphaelmitas/wyoming-parakeet-mlx
```

Or with `pipx`:

```sh
brew install ffmpeg
pipx install git+https://github.com/raphaelmitas/wyoming-parakeet-mlx
```

For local development:

```sh
git clone https://github.com/raphaelmitas/wyoming-parakeet-mlx
cd wyoming-parakeet-mlx
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Run

```sh
wyoming-parakeet-mlx --uri tcp://0.0.0.0:10300
```

First start downloads the model from Hugging Face (~1.2 GB) and runs a
warmup pass. Subsequent starts use the cached weights and are ready in
a few seconds.

### Configuration

All flags also read from `WYOMING_PARAKEET_*` environment variables.

| Flag | Env var | Default |
|------|---------|---------|
| `--uri` | `WYOMING_PARAKEET_URI` | `tcp://0.0.0.0:10300` |
| `--model` | `WYOMING_PARAKEET_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` |
| `--language` (repeatable) | `WYOMING_PARAKEET_LANGUAGE` (comma-separated) | full v3 language set |
| `--cache-dir` | `WYOMING_PARAKEET_CACHE_DIR` | `~/.cache/huggingface` |
| `--max-audio-seconds` | `WYOMING_PARAKEET_MAX_AUDIO_SECONDS` | `300` |
| `--no-warmup` | `WYOMING_PARAKEET_NO_WARMUP` | unset |
| `--log-level` | `WYOMING_PARAKEET_LOG_LEVEL` | `INFO` |

## Run as a persistent macOS service

A launchd template lives at `scripts/com.github.wyoming-parakeet-mlx.plist`.
Edit the `ProgramArguments` path to match `which wyoming-parakeet-mlx`,
then:

```sh
cp scripts/com.github.wyoming-parakeet-mlx.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.github.wyoming-parakeet-mlx.plist
```

Restart after changing env vars:

```sh
launchctl kickstart -k gui/$(id -u)/com.github.wyoming-parakeet-mlx
```

Logs go to `/tmp/wyoming-parakeet-mlx.log`.

## Connect from Home Assistant

1. Settings → Devices & Services → Add Integration → "Wyoming Protocol"
2. Host: the IP of your Mac. Port: `10300`.
3. The integration appears as an STT provider; pick it in your Assist
   pipeline (Settings → Voice assistants → Assist).

## Probe / debug

A small Wyoming client lives at `scripts/probe.py`:

```sh
python scripts/probe.py describe   tcp://127.0.0.1:10300
python scripts/probe.py transcribe tcp://127.0.0.1:10300 ./my-audio.wav
```

## Architecture

```
Wyoming client                       wyoming-parakeet-mlx
  Describe          ──────────▶       AsrHandler.handle_event
                    ◀──────────       Info(asr=[AsrProgram(...)])
  Transcribe        ──────────▶       store language hint
  AudioStart        ──────────▶       open temp WAV
  AudioChunk(s)     ──────────▶       AudioChunkConverter → WAV writer
  AudioStop         ──────────▶       to_thread(model.transcribe) under lock
                    ◀──────────       Transcript(text=...)
```

One Parakeet model loaded at startup, shared across connections. An
`asyncio.Lock` serialises inference calls (MLX doesn't parallelise
cleanly on one model instance). Each connection handles one utterance,
matching the canonical Wyoming server pattern.

## Tests

```sh
pytest -m "not requires_mlx"   # unit tests, no MLX needed
pytest                          # full suite incl. real-model E2E
```

## FAQ

**Why no Docker image?** `parakeet-mlx` requires Apple's MLX framework
which only runs on Apple Silicon. It cannot run inside a typical Linux
Docker container. This project intentionally stays native-macOS-only.

**I want Parakeet on Linux with my NVIDIA GPU.** Use an NeMo-based
Wyoming server instead — this project is the wrong tool. The Parakeet
model itself is the same; the inference backend differs.

**Can I use other Parakeet variants?** Yes — pass any
`mlx-community/parakeet-*` model id with `--model`. You may also need
to adjust the advertised `--language` list to match.

**Does it support streaming partial transcripts?**
Not yet. Wyoming supports them and `parakeet-mlx` has
`transcribe_stream`; wiring them up is a planned enhancement.

## License

[Apache 2.0](./LICENSE).

Built on the patterns from
[`rhasspy/wyoming-faster-whisper`](https://github.com/rhasspy/wyoming-faster-whisper).
Parakeet model © NVIDIA; MLX port © [senstella](https://github.com/senstella/parakeet-mlx).
