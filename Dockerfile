FROM python:3.11-slim

ARG OLLAMA_MODEL=qwen3.5:27b
ARG BAKE_MODEL=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    OLLAMA_HOST=http://127.0.0.1:11434 \
    OLLAMA_MODEL=${OLLAMA_MODEL}

# System deps: poppler (pdf2image), tesseract (pytesseract), curl (ollama installer)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates zstd \
        poppler-utils tesseract-ocr \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (single static binary + service script)
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Python deps (cached layer)
COPY requirements.txt ./
RUN pip install --upgrade pip wheel setuptools \
    && pip install -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Project source
COPY . .

# Pre-pull the LLM weights into the image so the container is fully self-contained.
# Set BAKE_MODEL=0 at build time to skip and pull on first container start instead.
RUN if [ "$BAKE_MODEL" = "1" ]; then \
        ollama serve & \
        SERVE_PID=$!; \
        until curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do sleep 1; done; \
        ollama pull "$OLLAMA_MODEL"; \
        kill $SERVE_PID; wait $SERVE_PID 2>/dev/null || true; \
    fi

RUN mkdir -p data/uploads/json_data data/results
ENV PYTHONPATH=/app:/app/src

EXPOSE 8000
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
