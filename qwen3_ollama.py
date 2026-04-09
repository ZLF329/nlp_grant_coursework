"""
Qwen3 Ollama grant scorer.

Drop-in replacement for qwen3_vllm that talks to a local Ollama server
instead of loading vLLM in-process. Reuses all helpers (schema building,
prompt construction, aggregation, score_application) from qwen3_vllm and
only swaps the _Scorer class.

Env vars:
    OLLAMA_HOST   default http://127.0.0.1:11434
    OLLAMA_MODEL  default qwen3:30b-a3b-instruct-2507-q4_K_M
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

from qwen3_vllm import (
    _build_section_messages,
    _empty_criterion_result,
    _section_schema,
    score_application as _score_application_base,
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:27b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "600"))


class _Scorer:
    def __init__(self, model_name: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.host = host.rstrip("/")
        print(f"[qwen3_ollama] using {self.host} model={self.model_name}", flush=True)
        # Sanity check the server is reachable; don't crash if not — just warn.
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            print(f"[qwen3_ollama] WARNING: cannot reach Ollama at {self.host}: {e}",
                  flush=True)

    def _chat(self, messages: list[dict], schema: dict, max_tokens: int) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": schema,                 # JSON-schema guided decoding
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": max_tokens,
            },
            "think": False,
        }
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "")

    def score_sections(self,
                       jobs: list[tuple[str, str, list[dict]]],
                       always_text: str,
                       evidence_map: dict[str, str]) -> dict[str, list[dict]]:
        results: dict[str, list[dict]] = {}
        for (crit_key, name, sub_items) in jobs:
            messages = _build_section_messages(
                name, sub_items, always_text, evidence_map.get(crit_key, ""))
            schema = _section_schema(len(sub_items))
            max_tokens = 512 + 256 * len(sub_items)

            try:
                text = self._chat(messages, schema, max_tokens).strip()
                if text.startswith("```"):
                    text = text.strip("`")
                    if "\n" in text:
                        text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text[:-3]
                parsed = json.loads(text)
                arr = parsed.get("sub_items", [])
                if not isinstance(arr, list) or len(arr) != len(sub_items):
                    raise ValueError(
                        f"expected {len(sub_items)} sub_items, got "
                        f"{len(arr) if isinstance(arr, list) else 'non-list'}")
                scored: list[dict] = []
                for it, sub in zip(arr, sub_items):
                    if not isinstance(it, dict):
                        scored.append(_empty_criterion_result(
                            sub["name"], "ollama returned non-dict sub_item"))
                        continue
                    it["name"] = sub["name"]
                    it.setdefault("exists", "no")
                    it.setdefault("quality", "missing")
                    it.setdefault("quality_score_0to10", 0)
                    it.setdefault("rubric_subscores_0to2",
                                  {"coverage": 0, "specificity": 0, "strength": 0})
                    it.setdefault("evidence", [])
                    it.setdefault("evidence_ids", [])
                    it.setdefault("rationale", "")
                    scored.append(it)
            except Exception as e:
                scored = [_empty_criterion_result(s["name"], f"ollama error: {e}")
                          for s in sub_items]
            results[crit_key] = scored
        return results


def score_application(application: dict, criteria_path, doc_id: str | None = None,
                      scorer: _Scorer | None = None) -> dict:
    if scorer is None:
        scorer = _Scorer()
    return _score_application_base(application, criteria_path,
                                   doc_id=doc_id, scorer=scorer)


if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("application_json")
    p.add_argument("--criteria", default=str(Path(__file__).parent / "criteria_points.json"))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    in_path = Path(args.application_json)
    application = json.loads(in_path.read_text(encoding="utf-8"))
    result = score_application(application, args.criteria, doc_id=in_path.stem)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_scored.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[qwen3_ollama] wrote {out_path}")
    sys.exit(0)
