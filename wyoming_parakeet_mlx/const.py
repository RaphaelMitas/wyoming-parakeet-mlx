"""Constants and defaults."""

from __future__ import annotations

DEFAULT_URI = "tcp://0.0.0.0:10300"
DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

SAMPLE_RATE = 16000
WIDTH = 2  # bytes per sample (16-bit)
CHANNELS = 1

# Languages supported by parakeet-tdt-0.6b-v3. Source: the model card at
# https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3. Reconcile if NVIDIA
# updates the list.
V3_LANGUAGES: list[str] = [
    "bg",  # Bulgarian
    "hr",  # Croatian
    "cs",  # Czech
    "da",  # Danish
    "nl",  # Dutch
    "en",  # English
    "et",  # Estonian
    "fi",  # Finnish
    "fr",  # French
    "de",  # German
    "el",  # Greek
    "hu",  # Hungarian
    "it",  # Italian
    "lv",  # Latvian
    "lt",  # Lithuanian
    "mt",  # Maltese
    "pl",  # Polish
    "pt",  # Portuguese
    "ro",  # Romanian
    "sk",  # Slovak
    "sl",  # Slovenian
    "es",  # Spanish
    "sv",  # Swedish
    "ru",  # Russian
    "uk",  # Ukrainian
]

# Hard cap on a single utterance — protects the inference lock from a
# runaway client. Configurable at runtime.
DEFAULT_MAX_AUDIO_SECONDS = 300
