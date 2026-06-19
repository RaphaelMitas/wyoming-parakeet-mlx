"""Handler-level tests with a fake Transcriber. No MLX required."""

from __future__ import annotations

import asyncio
import io
import wave

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import (
    AsrModel,
    AsrProgram,
    Attribution,
    Describe,
    Info,
)

from wyoming_parakeet_mlx.handler import AsrHandler


class _FakeReader:
    """Stub StreamReader — never delivers data; we drive handle_event directly."""

    async def read(self, n: int = -1) -> bytes:
        await asyncio.sleep(0)
        return b""


class _FakeWriter:
    """Captures every event written by the handler."""

    def __init__(self) -> None:
        self.closed = False
        self.buffer = bytearray()

    # Methods AsyncEventHandler / async_write_event may call
    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    def writelines(self, parts) -> None:
        for p in parts:
            self.buffer.extend(p)

    async def drain(self) -> None:
        return

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return

    def get_extra_info(self, name: str, default=None):
        return default


class _FakeTranscriber:
    def __init__(self, return_text: str = "hello world") -> None:
        self.return_text = return_text
        self.calls: list[str] = []

    async def transcribe(self, wav_path) -> str:
        self.calls.append(str(wav_path))
        # Confirm the buffered WAV is a valid RIFF file with audio frames.
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2
            assert wf.getnchannels() == 1
            assert wf.getnframes() > 0
        return self.return_text


def _wyoming_info() -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="parakeet-mlx",
                description="test",
                attribution=Attribution(name="test", url="https://example.com"),
                installed=True,
                version="0.0.0",
                models=[
                    AsrModel(
                        name="test-model",
                        description="test",
                        attribution=Attribution(name="test", url="https://example.com"),
                        installed=True,
                        languages=["en"],
                        version="test",
                    )
                ],
            )
        ]
    )


def _wav_bytes(seconds: float = 0.5) -> bytes:
    """Raw 16-bit PCM frames (no RIFF header) — what AudioChunk carries."""
    n = int(16000 * seconds)
    buf = io.BytesIO()
    # Just emit alternating samples — content doesn't matter for the test.
    for i in range(n):
        buf.write((1000 if i % 2 else -1000).to_bytes(2, "little", signed=True))
    return buf.getvalue()


async def _make_handler(
    model: _FakeTranscriber,
    max_audio_seconds: float = 300.0,
) -> tuple[AsrHandler, _FakeWriter]:
    writer = _FakeWriter()
    handler = AsrHandler(
        _wyoming_info(),
        model,
        max_audio_seconds,
        _FakeReader(),
        writer,  # type: ignore[arg-type]
    )
    return handler, writer


async def _decode_written_events(writer: _FakeWriter) -> list[Event]:
    """Parse the bytes the handler wrote back into Event objects."""
    from wyoming.event import async_read_event

    reader = asyncio.StreamReader()
    reader.feed_data(bytes(writer.buffer))
    reader.feed_eof()
    events: list[Event] = []
    while True:
        ev = await async_read_event(reader)
        if ev is None:
            break
        events.append(ev)
    return events


@pytest.mark.asyncio
async def test_describe_returns_info() -> None:
    model = _FakeTranscriber()
    handler, writer = await _make_handler(model)
    keep_going = await handler.handle_event(Describe().event())
    assert keep_going is True
    events = await _decode_written_events(writer)
    assert len(events) == 1
    info = Info.from_event(events[0])
    assert info.asr[0].name == "parakeet-mlx"
    assert info.asr[0].models[0].name == "test-model"
    assert model.calls == []  # describe must not touch the model


@pytest.mark.asyncio
async def test_full_transcription_flow() -> None:
    model = _FakeTranscriber(return_text="hello world")
    handler, writer = await _make_handler(model)

    # Transcribe (language hint)
    assert await handler.handle_event(Transcribe(language="en").event()) is True

    # AudioStart -> just acks
    assert await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event()) is True

    # Three chunks of audio
    pcm = _wav_bytes(0.3)
    third = len(pcm) // 3
    for start in (0, third, 2 * third):
        chunk = AudioChunk(rate=16000, width=2, channels=1, audio=pcm[start : start + third])
        assert await handler.handle_event(chunk.event()) is True

    # AudioStop -> triggers inference -> connection ends (returns False)
    assert await handler.handle_event(AudioStop().event()) is False

    assert len(model.calls) == 1
    events = await _decode_written_events(writer)
    assert len(events) == 1
    transcript = Transcript.from_event(events[0])
    assert transcript.text == "hello world"


@pytest.mark.asyncio
async def test_empty_audio_returns_empty_transcript() -> None:
    model = _FakeTranscriber(return_text="should-not-be-used")
    handler, writer = await _make_handler(model)
    # AudioStop without any AudioChunk
    assert await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event()) is True
    assert await handler.handle_event(AudioStop().event()) is False

    # Model should never have been called.
    assert model.calls == []
    events = await _decode_written_events(writer)
    assert len(events) == 1
    transcript = Transcript.from_event(events[0])
    assert transcript.text == ""


@pytest.mark.asyncio
async def test_oversized_audio_is_truncated_not_rejected() -> None:
    model = _FakeTranscriber(return_text="ok")
    # Cap at 0.1s — the chunk we push is bigger than the cap.
    handler, writer = await _make_handler(model, max_audio_seconds=0.1)

    chunk = AudioChunk(rate=16000, width=2, channels=1, audio=_wav_bytes(1.0))
    assert await handler.handle_event(chunk.event()) is True
    # Pushing more should be silently dropped.
    assert await handler.handle_event(chunk.event()) is True
    assert await handler.handle_event(AudioStop().event()) is False

    # Inference still ran on the truncated audio.
    assert len(model.calls) == 1
    events = await _decode_written_events(writer)
    assert Transcript.from_event(events[0]).text == "ok"


@pytest.mark.asyncio
async def test_inference_exception_returns_empty_transcript() -> None:
    class _BadModel:
        async def transcribe(self, wav_path) -> str:
            raise RuntimeError("inference exploded")

    model = _BadModel()
    handler, writer = await _make_handler(model)  # type: ignore[arg-type]
    assert await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event()) is True
    assert (
        await handler.handle_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=_wav_bytes(0.1)).event()
        )
        is True
    )
    assert await handler.handle_event(AudioStop().event()) is False

    events = await _decode_written_events(writer)
    assert Transcript.from_event(events[0]).text == ""
