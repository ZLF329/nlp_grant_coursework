"""
Qwen3 Ollama grant scorer.

Section scoring runs on configurable Ollama models with rule-based retrieval.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests

from src.scoring.pipeline import score_application_base

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:30b-a3b")
OLLAMA_MODEL_A = os.environ.get("OLLAMA_MODEL_A", OLLAMA_MODEL)
OLLAMA_MODEL_B = os.environ.get("OLLAMA_MODEL_B", OLLAMA_MODEL)
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "1200"))


class _Scorer:
    def __init__(self, model_name: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.host = host.rstrip("/")
        print(f"[qwen3_ollama] using {self.host} model={self.model_name}", flush=True)

    def generate_json(self, messages: list[dict[str, str]], *, schema: dict[str, Any], max_tokens: int) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": max_tokens,
            },
            "think": False,
        }
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.host}. Start the Ollama server or set OLLAMA_HOST to the correct "
                "endpoint before running scoring."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                "Timed out waiting for Ollama at "
                f"{self.host}. Increase OLLAMA_TIMEOUT or check whether the model server is healthy."
            ) from exc
        response.raise_for_status()
        body = response.json()
        content = ((body.get("message") or {}).get("content") or "").strip()
        # Some backends may still inline think tags even when reasoning is also surfaced separately.
        return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()


def score_application(
    application: dict[str, Any],
    criteria_path: str | Path,
    *,
    doc_id: str | None = None,
    scorer: _Scorer | None = None,
    scorer_client_a: _Scorer | None = None,
    scorer_client_b: _Scorer | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    scorer_client_a = scorer_client_a or scorer or _Scorer(model_name=OLLAMA_MODEL_A, host=OLLAMA_HOST)
    scorer_client_b = scorer_client_b or (
        scorer_client_a
        if OLLAMA_MODEL_B == getattr(scorer_client_a, "model_name", None)
        else _Scorer(model_name=OLLAMA_MODEL_B, host=OLLAMA_HOST)
    )
    return score_application_base(
        application=application,
        criteria_path=criteria_path,
        doc_id=doc_id,
        scorer_client_a=scorer_client_a,
        scorer_client_b=scorer_client_b,
        artifacts_dir=artifacts_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("application_json")
    parser.add_argument("--criteria", default=str(Path(__file__).parent / "criteria_points.json"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    in_path = Path(args.application_json)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_scored.json")
    application = json.loads(in_path.read_text(encoding="utf-8"))
    result = score_application(
        application,
        args.criteria,
        doc_id=in_path.stem,
        artifacts_dir=out_path.parent,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[qwen3_ollama] wrote {out_path}")
    sys.exit(0)
