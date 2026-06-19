#!/usr/bin/env python3
"""Hand-runnable Wyoming client for diagnosing a wyoming-parakeet-mlx server.

Usage:
  python scripts/probe.py describe   tcp://127.0.0.1:10300
  python scripts/probe.py transcribe tcp://127.0.0.1:10300 /path/to/audio.wav
"""

from __future__ import annotations

import asyncio
import json
import sys
import wave
from dataclasses import asdict
from urllib.parse import urlparse

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info


def _parse_uri(uri: str) -> tuple[str, int]:
    parsed = urlparse(uri)
    if parsed.scheme != "tcp" or not parsed.hostname or not parsed.port:
        raise SystemExit(f"Expected tcp://host:port, got {uri!r}")
    return parsed.hostname, parsed.port


async def cmd_describe(uri: str) -> int:
    host, port = _parse_uri(uri)
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Describe().event())
        event = await asyncio.wait_for(client.read_event(), timeout=10)
        if event is None:
            print("No Info event received", file=sys.stderr)
            return 1
        info = Info.from_event(event)
        print(json.dumps(asdict(info), indent=2, default=str))
    return 0


async def cmd_transcribe(uri: str, wav_path: str) -> int:
    host, port = _parse_uri(uri)
    with wave.open(wav_path, "rb") as wf:
        rate = wf.getframerate()
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        async with AsyncTcpClient(host, port) as client:
            await client.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
            while True:
                frames = wf.readframes(1024)
                if not frames:
                    break
                await client.write_event(
                    AudioChunk(rate=rate, width=width, channels=channels, audio=frames).event()
                )
            await client.write_event(AudioStop().event())
            event = await asyncio.wait_for(client.read_event(), timeout=120)
            if event is None:
                print("No Transcript event received", file=sys.stderr)
                return 1
            transcript = Transcript.from_event(event)
            print(transcript.text)
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, uri = sys.argv[1], sys.argv[2]
    if cmd == "describe":
        return asyncio.run(cmd_describe(uri))
    if cmd == "transcribe":
        if len(sys.argv) < 4:
            print(__doc__, file=sys.stderr)
            return 2
        return asyncio.run(cmd_transcribe(uri, sys.argv[3]))
    print(f"Unknown command: {cmd}", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
