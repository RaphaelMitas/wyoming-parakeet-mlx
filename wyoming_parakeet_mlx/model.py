"""Parakeet model wrapper: loads once, serialises inference."""

from __future__ import annotations

import asyncio
import logging
import wave
from pathlib import Path
from typing import Protocol

from .const import CHANNELS, SAMPLE_RATE, WIDTH

_LOGGER = logging.getLogger(__name__)


class Transcriber(Protocol):
    """Minimal interface used by the handler and tests."""

    async def transcribe(self, wav_path: str | Path) -> str: ...


class ParakeetModel:
    """Thread-safe wrapper around a parakeet-mlx model.

    Loads the model eagerly. All transcribe calls run on a worker thread
    under an asyncio.Lock so concurrent Wyoming connections don't race on
    the same MLX model state.
    """

    def __init__(self, model_id: str, cache_dir: str | Path | None = None) -> None:
        # Import here so non-Apple-Silicon environments (CI on Linux) can
        # import this module for type-checking without MLX present.
        from parakeet_mlx import from_pretrained  # type: ignore[import-not-found]

        _LOGGER.info("Loading model %s", model_id)
        self._model = from_pretrained(model_id, cache_dir=cache_dir)
        self._lock = asyncio.Lock()
        self._model_id = model_id
        _LOGGER.info("Model loaded")

    @property
    def model_id(self) -> str:
        return self._model_id

    async def transcribe(self, wav_path: str | Path) -> str:
        """Run inference on a WAV file. Returns the recognized text."""
        async with self._lock:
            result = await asyncio.to_thread(self._model.transcribe, str(wav_path))
        return (result.text or "").strip()

    async def warmup(self) -> None:
        """Run inference on a short silent clip to trigger kernel compilation."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "warmup.wav"
            _write_silent_wav(wav_path, seconds=1.0)
            _LOGGER.info("Running warmup pass")
            text = await self.transcribe(wav_path)
            _LOGGER.info("Warmup complete (warmup transcript: %r)", text)


def _write_silent_wav(path: Path, seconds: float = 1.0) -> None:
    """Write a silent 16 kHz / 16-bit / mono WAV file."""
    frames = int(seconds * SAMPLE_RATE)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00\x00" * frames)
