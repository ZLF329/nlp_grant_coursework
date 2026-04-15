#!/usr/bin/env bash
# Local launcher for xsx — uses the existing ai_venv instead of creating a new one.
# Assumes Ollama is already running locally (ollama serve) with the 5090.

set -euo pipefail

cd "$(dirname "$0")"

AI_VENV="${AI_VENV:-/d/MSc_AI/ai_venv}"
PORT="${PORT:-8000}"

# 1. activate existing venv ----------------------------------------------------
if [ ! -f "$AI_VENV/Scripts/activate" ]; then
  echo "[start_local_xsx] ERROR: ai_venv not found at $AI_VENV"
  echo "  Set AI_VENV=/path/to/your/venv and re-run."
  exit 1
fi
source "$AI_VENV/Scripts/activate"
echo "[start_local_xsx] using venv: $AI_VENV ($(python --version))"

# 2. data dirs -----------------------------------------------------------------
mkdir -p data/uploads/json_data data/results

# 3. run -----------------------------------------------------------------------
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"
export PORT
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

echo "[start_local_xsx] OLLAMA_HOST = $OLLAMA_HOST"
echo "[start_local_xsx] launching server on http://0.0.0.0:${PORT}"
exec python web/server.py
