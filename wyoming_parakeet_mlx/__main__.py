"""Entrypoint: parse args, build Info, load the model, run the Wyoming server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from functools import partial

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .const import (
    DEFAULT_MAX_AUDIO_SECONDS,
    DEFAULT_MODEL,
    DEFAULT_URI,
    V3_LANGUAGES,
)
from .handler import AsrHandler
from .model import ParakeetModel

_LOGGER = logging.getLogger(__name__)


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"WYOMING_PARAKEET_{name}", default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wyoming-parakeet-mlx",
        description="Wyoming STT server backed by NVIDIA Parakeet (parakeet-mlx).",
    )
    parser.add_argument(
        "--uri",
        default=_env("URI", DEFAULT_URI),
        help=f"Wyoming server URI (default: {DEFAULT_URI})",
    )
    parser.add_argument(
        "--model",
        default=_env("MODEL", DEFAULT_MODEL),
        help=f"Hugging Face model id or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--language",
        action="append",
        default=None,
        help=(
            "Language to advertise (repeatable). Defaults to the full set "
            "supported by parakeet-tdt-0.6b-v3."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=_env("CACHE_DIR"),
        help="Hugging Face cache directory (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=float(_env("MAX_AUDIO_SECONDS", str(DEFAULT_MAX_AUDIO_SECONDS))),
        help=f"Max length of a single utterance (default: {DEFAULT_MAX_AUDIO_SECONDS}s)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        default=bool(_env("NO_WARMUP")),
        help="Skip warmup pass at startup",
    )
    parser.add_argument(
        "--log-level",
        default=_env("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    args = parser.parse_args()
    if not args.language:
        # Default to env var (comma-separated) or the full V3 set.
        env_langs = _env("LANGUAGE")
        if env_langs:
            args.language = [s.strip() for s in env_langs.split(",") if s.strip()]
        else:
            args.language = list(V3_LANGUAGES)
    return args


def _build_info(model_id: str, languages: list[str]) -> Info:
    return Info(
        asr=[
            AsrProgram(
                name="parakeet-mlx",
                description="NVIDIA Parakeet TDT via parakeet-mlx on Apple Silicon",
                attribution=Attribution(
                    name="senstella / parakeet-mlx",
                    url="https://github.com/senstella/parakeet-mlx",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_id,
                        description=f"NVIDIA Parakeet TDT ({model_id})",
                        attribution=Attribution(
                            name="NVIDIA NeMo",
                            url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
                        ),
                        installed=True,
                        languages=languages,
                        version="0.6b-v3",
                    )
                ],
            )
        ],
    )


async def _amain() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _LOGGER.debug("Parsed args: %s", args)

    wyoming_info = _build_info(args.model, args.language)

    model = ParakeetModel(args.model, cache_dir=args.cache_dir)
    if not args.no_warmup:
        await model.warmup()

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready on %s", args.uri)

    await server.run(
        partial(
            AsrHandler,
            wyoming_info,
            model,
            args.max_audio_seconds,
        )
    )


def run() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
