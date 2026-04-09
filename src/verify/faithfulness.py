from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

JUDGE_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
JUDGE_MODEL = os.environ.get("OLLAMA_JUDGE_MODEL", "gemma4:e4b")
JUDGE_TIMEOUT = float(os.environ.get("OLLAMA_JUDGE_TIMEOUT", "600"))
JUDGE_PARALLEL = int(os.environ.get("OLLAMA_JUDGE_PARALLEL", "4"))


def build_judge_messages(section_key: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are auditing whether cited evidence chunk IDs actually support grant-scoring "
        "judgments. Be lenient. Only assign low plausibility when the cited evidence clearly "
        "fails to support the judgment. Score each signal from 0 to 5. Do not rescore the "
        "application itself. Reply with JSON only."
    )
    evidence_context = payload.get("evidence_context", "")
    audit_package = {
        "section_key": payload.get("section_key", section_key),
        "sub_criteria": payload.get("sub_criteria", []),
    }
    user = (
        f"Section key: {section_key}\n\n"
        "Evidence context in original order. `...` means omitted gap between cited chunks. "
        "Each parser section is wrapped by matching `[Section Name]` markers.\n\n"
        f"{evidence_context}\n\n"
        "Audit package:\n"
        f"{json.dumps(audit_package, ensure_ascii=False, indent=2)}\n\n"
        "Return one judgment per signal. `plausibility` meanings: 5=clear support, "
        "4=mostly supported, 3=some jump but acceptable, 2=weakly supported, "
        "1=barely supported, 0=not supported or contradicted."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def judge_schema(expected_sids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "judgments": {
                "type": "array",
                "minItems": len(expected_sids),
                "maxItems": len(expected_sids),
                "items": {
                    "type": "object",
                    "properties": {
                        "sid": {"type": "string", "enum": expected_sids},
                        "plausibility": {"type": "integer", "minimum": 0, "maximum": 5},
                        "note": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["sid", "plausibility"],
                },
            }
        },
        "required": ["judgments"],
    }


class OllamaFaithfulnessJudge:
    def __init__(self, host: str = JUDGE_HOST, model_name: str = JUDGE_MODEL):
        self.host = host.rstrip("/")
        self.model_name = model_name

    def _chat(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 2048,
            },
            "think": False,
        }
        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=JUDGE_TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
        content = ((body.get("message") or {}).get("content") or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if "\n" in content:
                content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        return json.loads(content)

    def judge_section(self, section_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        sids = [
            signal["sid"]
            for sub in payload.get("sub_criteria", [])
            for signal in sub.get("signals", [])
        ]
        if not sids:
            return {"judgments": []}
        return self._chat(build_judge_messages(section_key, payload), judge_schema(sids))


class FallbackJudge:
    def judge_section(self, section_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        del section_key
        judgments = []
        for sub in payload.get("sub_criteria", []):
            for signal in sub.get("signals", []):
                judgments.append({
                    "sid": signal["sid"],
                    "plausibility": 3,
                    "note": "judge_unavailable",
                })
        return {"judgments": judgments}


def run_faithfulness_audit(
    sections_payload: dict[str, dict[str, Any]],
    judge: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    if not sections_payload:
        return {}

    results: dict[str, dict[str, dict[str, Any]]] = {}

    def run_one(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, dict[str, Any]]]:
        section_key, payload = item
        try:
            raw = judge.judge_section(section_key, payload)
            judgments = raw.get("judgments", []) if isinstance(raw, dict) else []
            normalized: dict[str, dict[str, Any]] = {}
            for entry in judgments:
                if not isinstance(entry, dict) or "sid" not in entry:
                    continue
                normalized[entry["sid"]] = {
                    "plausibility": int(entry.get("plausibility", 3)),
                    "note": entry.get("note"),
                }
            return section_key, normalized
        except Exception as exc:
            fallback: dict[str, dict[str, Any]] = {}
            for sub in payload.get("sub_criteria", []):
                for signal in sub.get("signals", []):
                    fallback[signal["sid"]] = {
                        "plausibility": 3,
                        "note": f"judge_error: {exc}",
                    }
            return section_key, fallback

    workers = max(1, min(JUDGE_PARALLEL, len(sections_payload)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for section_key, normalized in executor.map(run_one, sections_payload.items()):
            results[section_key] = normalized
    return results
