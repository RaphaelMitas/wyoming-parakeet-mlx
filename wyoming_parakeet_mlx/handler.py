"""Wyoming event handler for a single client connection."""

from __future__ import annotations

import logging
import os
import tempfile
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .const import CHANNELS, SAMPLE_RATE, WIDTH
from .model import Transcriber

_LOGGER = logging.getLogger(__name__)


class AsrHandler(AsyncEventHandler):
    """Buffers an utterance, transcribes it, returns one Transcript.

    Mirrors the pattern in rhasspy/wyoming-faster-whisper's
    DispatchEventHandler: one transcription per connection, audio buffered
    to a temp WAV, model invoked under a shared lock from
    `model.transcribe(...)`.
    """

    def __init__(
        self,
        wyoming_info: Info,
        model: Transcriber,
        max_audio_seconds: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._wyoming_info_event = wyoming_info.event()
        self._model = model
        self._max_audio_seconds = max_audio_seconds

        self._language: str | None = None
        self._audio_converter = AudioChunkConverter(
            rate=SAMPLE_RATE, width=WIDTH, channels=CHANNELS
        )

        self._wav_dir: tempfile.TemporaryDirectory | None = None
        self._wav_path: str | None = None
        self._wav_file: wave.Wave_write | None = None
        self._wav_frames_written = 0
        self._overflow_logged = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language
            _LOGGER.debug("Transcribe (language=%s)", self._language)
            return True

        if AudioChunk.is_type(event.type):
            chunk = self._audio_converter.convert(AudioChunk.from_event(event))

            if self._wav_file is None:
                self._wav_dir = tempfile.TemporaryDirectory()
                self._wav_path = os.path.join(self._wav_dir.name, "utterance.wav")
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            max_frames = int(self._max_audio_seconds * chunk.rate)
            if self._wav_frames_written >= max_frames:
                if not self._overflow_logged:
                    _LOGGER.warning(
                        "Audio exceeds --max-audio-seconds (%s); dropping further chunks",
                        self._max_audio_seconds,
                    )
                    self._overflow_logged = True
                return True

            # chunk.audio is raw PCM (width-byte frames).
            frames_in_chunk = len(chunk.audio) // (chunk.width * chunk.channels)
            allowed_frames = max_frames - self._wav_frames_written
            if frames_in_chunk > allowed_frames:
                bytes_allowed = allowed_frames * chunk.width * chunk.channels
                self._wav_file.writeframes(chunk.audio[:bytes_allowed])
                self._wav_frames_written += allowed_frames
            else:
                self._wav_file.writeframes(chunk.audio)
                self._wav_frames_written += frames_in_chunk

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped (%d frames buffered)", self._wav_frames_written)

            text = ""
            if self._wav_file is not None and self._wav_path is not None:
                self._wav_file.close()
                self._wav_file = None

                try:
                    text = await self._model.transcribe(self._wav_path)
                except Exception:  # noqa: BLE001 - report any inference failure as empty
                    _LOGGER.exception("Transcription failed")
                    text = ""

            self._cleanup_wav()

            _LOGGER.info("Transcript: %r", text)
            await self.write_event(Transcript(text=text).event())

            # Reset session state. Returning False ends the connection,
            # matching the wyoming-faster-whisper pattern: HA opens a fresh
            # connection per utterance.
            self._language = None
            self._wav_frames_written = 0
            self._overflow_logged = False
            return False

        return True

    async def disconnect(self) -> None:
        """Best-effort cleanup if the client drops mid-stream."""
        self._cleanup_wav()

    def _cleanup_wav(self) -> None:
        if self._wav_file is not None:
            try:
                self._wav_file.close()
            except Exception:  # noqa: BLE001
                pass
            self._wav_file = None
        if self._wav_dir is not None:
            try:
                self._wav_dir.cleanup()
            except Exception:  # noqa: BLE001
                pass
            self._wav_dir = None
        self._wav_path = None
