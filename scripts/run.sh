#!/usr/bin/env bash
# Development launcher.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m wyoming_parakeet_mlx --uri "${WYOMING_PARAKEET_URI:-tcp://0.0.0.0:10300}" "$@"
