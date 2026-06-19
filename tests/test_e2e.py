"""End-to-end test: real server subprocess + real Wyoming client + real MLX.

Skipped unless parakeet-mlx is importable (i.e. Apple Silicon macOS).
"""

from __future__ import annotations

import asyncio
import math
import os
import struct
import sys
import wave
from pathlib import Path

import pytest

mlx_available = False
try:
    import parakeet_mlx  # noqa: F401

    mlx_available = True
except ImportError:
    pass

pytestmark = pytest.mark.requires_mlx
if not mlx_available:
    pytest.skip("parakeet-mlx not installed", allow_module_level=True)


def _write_silent_wav(path: Path, seconds: float = 1.0) -> None:
    rate = 16000
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for _ in range(n):
            wf.writeframes(struct.pack("<h", 0))


def _write_tone_wav(path: Path, seconds: float = 1.0, freq: float = 440.0) -> None:
    rate = 16000
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for i in range(n):
            wf.writeframes(
                struct.pack("<h", int(32767 * 0.2 * math.sin(2 * math.pi * freq * i / rate)))
            )


@pytest.mark.asyncio
async def test_end_to_end_describe_and_transcribe(tmp_path: Path) -> None:
    """Spawn the server, send a WAV, assert we get a Transcript back."""
    from wyoming.asr import Transcript
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.client import AsyncTcpClient
    from wyoming.info import Describe, Info

    port = 17300  # arbitrary unused port
    wav_path = tmp_path / "tone.wav"
    _write_tone_wav(wav_path, seconds=1.0)

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_parakeet_mlx",
        "--uri",
        f"tcp://127.0.0.1:{port}",
        "--no-warmup",  # speed up CI; not strictly necessary
        "--log-level",
        "DEBUG",
        env={**os.environ},
    )
    try:
        # Wait for the server to come up by polling for a successful connect.
        client: AsyncTcpClient | None = None
        for _ in range(60):
            await asyncio.sleep(1.0)
            try:
                client = AsyncTcpClient("127.0.0.1", port)
                await client.connect()
                break
            except (ConnectionRefusedError, OSError):
                client = None
                continue
        assert client is not None, "server never came up"

        try:
            # Describe -> Info
            await client.write_event(Describe().event())
            info_event = await asyncio.wait_for(client.read_event(), timeout=10)
            assert info_event is not None
            info = Info.from_event(info_event)
            assert info.asr[0].name == "parakeet-mlx"
        finally:
            await client.disconnect()

        # New connection for the transcription, matching the
        # one-utterance-per-connection pattern.
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())

            with wave.open(str(wav_path), "rb") as wf:
                while True:
                    frames = wf.readframes(1024)
                    if not frames:
                        break
                    await client.write_event(
                        AudioChunk(rate=16000, width=2, channels=1, audio=frames).event()
                    )

            await client.write_event(AudioStop().event())
            transcript_event = await asyncio.wait_for(client.read_event(), timeout=60)
            assert transcript_event is not None
            transcript = Transcript.from_event(transcript_event)
            # We don't assert on content (a 440 Hz tone won't decode to words);
            # just that we got a well-formed Transcript event back.
            assert isinstance(transcript.text, str)
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            proc.kill()
