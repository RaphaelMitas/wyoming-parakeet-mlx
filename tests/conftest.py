"""Shared pytest fixtures."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest


@pytest.fixture
def sample_wav_path(tmp_path: Path) -> Path:
    """A 0.5s, 16 kHz, 16-bit mono WAV containing a 440 Hz sine tone.

    Used for handler-level tests. The audio content doesn't matter for the
    fake-model unit tests — only that the buffered bytes look like valid PCM.
    """
    path = tmp_path / "sample_16k_mono.wav"
    rate = 16000
    seconds = 0.5
    n = int(rate * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for i in range(n):
            sample = int(32767 * 0.2 * math.sin(2 * math.pi * 440 * i / rate))
            wf.writeframes(struct.pack("<h", sample))
    return path
