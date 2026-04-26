#!/usr/bin/env bash
set -euo pipefail

# 1. Start Ollama in the background
ollama serve >/var/log/ollama.log 2>&1 &
OLLAMA_PID=$!

# 2. Wait for Ollama to be reachable
echo "[entrypoint] waiting for Ollama..."
for i in $(seq 1 60); do
    if curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "[entrypoint] Ollama is up"
        break
    fi
    sleep 1
done

# 3. Pull the model if not already present (no-op if BAKE_MODEL=1 at build)
if ! ollama list | awk 'NR>1 {print $1}' | grep -qx "$OLLAMA_MODEL"; then
    echo "[entrypoint] pulling $OLLAMA_MODEL (first run, may take a while)..."
    ollama pull "$OLLAMA_MODEL"
fi

# 4. Forward shutdown signals to Ollama too
trap 'kill -TERM $OLLAMA_PID 2>/dev/null || true' TERM INT

# 5. Launch the web server in the foreground
echo "[entrypoint] launching server on http://0.0.0.0:${PORT}"
exec python web/server.py
