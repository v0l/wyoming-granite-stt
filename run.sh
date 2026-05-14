#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"

if [ ! -f "$PYTHON" ]; then
	echo "Error: venv not found. Run: uv venv && uv pip install -r requirements.txt" >&2
	exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

exec "$PYTHON" "$SCRIPT_DIR/wyoming_granite_stt.py" "$@"
