"""
Qwen3 vLLM grant scorer.

Loads a parsed grant application JSON (produced by all_type_parser) and the
checklist in criteria_points.json, scores every criterion with Qwen3 via vLLM,
and writes a structured JSON file consumable by web/public/result.html.

Usage:
    python qwen3_vllm.py <parsed.json> [--out scored.json]

Output schema (matches what result.html expects):
{
  "doc_id": "...",
  "run_info": {"ran_at_utc": "...", "model": "Qwen/Qwen3-..."},
  "features": {
    "<section_key>": {
      "overall": {final_score_0to100, coverage_score_0to100, quality_score_0to100,
                  evidence_score_0to100, quality_score_avg_0to10, total_items,
                  good_items, positive_items, expected_items, evidence_count,
                  target_evidence_per_item},
      "criteria": [
        {name, exists, quality, quality_score_0to10, evidence: [...],
         rubric_subscores_0to2: {coverage, specificity, strength}}
      ]
    },
    ...
  },
  "overall": {... aggregated ...}
}
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-30B-A3B")
TARGET_EVIDENCE_PER_ITEM = 2

# Maps the human-readable section names in criteria_points.json to the short
# section keys that web/public/result.html knows how to render.
SECTION_KEY_MAP = {
    "General": "general",
    "Proposed research": "proposed_research",
    "Training and development": "training_development",
    "Sites and support": "sites_support",
    "Working with people and communities": "wpcc",
    "Application Form": "application_form",
}

# JSON schema used for vLLM guided decoding — one criterion at a time.
CRITERION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "exists": {"type": "string", "enum": ["yes", "no", "partial"]},
        "quality": {"type": "string", "enum": ["good", "mixed", "weak", "missing"]},
        "quality_score_0to10": {"type": "integer", "minimum": 0, "maximum": 10},
        "rubric_subscores_0to2": {
            "type": "object",
            "properties": {
                "coverage":    {"type": "integer", "minimum": 0, "maximum": 2},
                "specificity": {"type": "integer", "minimum": 0, "maximum": 2},
                "strength":    {"type": "integer", "minimum": 0, "maximum": 2},
            },
            "required": ["coverage", "specificity", "strength"],
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 4,
        },
        "rationale": {"type": "string"},
    },
    "required": [
        "exists", "quality", "quality_score_0to10",
        "rubric_subscores_0to2", "evidence", "rationale",
    ],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _flatten_section(section_name: str, section_value: Any) -> list[dict]:
    """
    Convert a section block from criteria_points.json into a flat list of
    criterion dicts with keys: name, definition, signals.
    """
    items: list[dict] = []

    if isinstance(section_value, list):
        for it in section_value:
            if isinstance(it, dict) and "name" in it:
                items.append({
                    "name": it["name"],
                    "definition": it.get("definition", ""),
                    "signals": it.get("signals", []),
                })
        return items

    if isinstance(section_value, dict):
        # The "General" section has nested groups (checklists + Applicant list).
        for sub_key, sub_val in section_value.items():
            if isinstance(sub_val, list):
                for it in sub_val:
                    if isinstance(it, dict) and "name" in it:
                        items.append({
                            "name": it["name"],
                            "definition": it.get("definition", ""),
                            "signals": it.get("signals", []),
                        })
                    elif isinstance(it, str):
                        items.append({
                            "name": f"{sub_key.replace('_', ' ').title()}: {it[:80]}",
                            "definition": it,
                            "signals": [],
                        })
    return items


def _truncate_application(app_text: str, max_chars: int = 60000) -> str:
    if len(app_text) <= max_chars:
        return app_text
    return app_text[:max_chars] + "\n\n[…truncated…]"


def _build_messages(application_json: dict, section_name: str, criterion: dict) -> list[dict]:
    sys_prompt = (
        "You are a meticulous reviewer of UK NIHR grant applications. "
        "You will receive ONE scoring criterion plus the parsed application. "
        "Score it, extract verbatim evidence, and reply with a single JSON "
        "object that matches the supplied schema. Do not include any prose "
        "outside the JSON."
    )
    user_prompt = (
        f"Section: {section_name}\n"
        f"Criterion: {criterion['name']}\n"
        f"Definition: {criterion['definition']}\n"
        f"Signals to look for: {json.dumps(criterion['signals'], ensure_ascii=False)}\n\n"
        f"Application JSON:\n{_truncate_application(json.dumps(application_json, ensure_ascii=False, indent=2))}\n\n"
        "Score `quality_score_0to10` 0=missing, 5=present but weak, 8=clear and "
        "well-supported, 10=excellent and comprehensive. Each rubric sub-score "
        "is 0/1/2 (absent/partial/strong). Quote evidence verbatim from the "
        "application; leave the array empty if nothing supports the criterion. "
        "Return ONLY the JSON object."
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_prompt},
    ]


def _empty_criterion_result(name: str, reason: str) -> dict:
    return {
        "name": name,
        "exists": "no",
        "quality": "missing",
        "quality_score_0to10": 0,
        "rubric_subscores_0to2": {"coverage": 0, "specificity": 0, "strength": 0},
        "evidence": [],
        "evidence_ids": [],
        "rationale": reason,
    }


def _aggregate_section(criteria_results: list[dict]) -> dict:
    if not criteria_results:
        return {
            "final_score_0to100": 0, "coverage_score_0to100": 0,
            "quality_score_0to100": 0, "evidence_score_0to100": 0,
            "quality_score_avg_0to10": 0, "total_items": 0, "good_items": 0,
            "positive_items": 0, "expected_items": 0, "evidence_count": 0,
            "target_evidence_per_item": TARGET_EVIDENCE_PER_ITEM,
        }
    n = len(criteria_results)
    quality_avg = sum(c["quality_score_0to10"] for c in criteria_results) / n
    good_items = sum(1 for c in criteria_results if c["quality"] == "good")
    positive_items = sum(1 for c in criteria_results if c["exists"] in ("yes", "partial"))
    evidence_count = sum(len(c.get("evidence", [])) for c in criteria_results)

    coverage = (positive_items / n) * 100
    quality_pct = quality_avg * 10
    target_total = n * TARGET_EVIDENCE_PER_ITEM
    evidence_pct = min(100.0, (evidence_count / target_total) * 100) if target_total else 0
    final_score = round(0.5 * quality_pct + 0.3 * coverage + 0.2 * evidence_pct, 2)

    return {
        "final_score_0to100": final_score,
        "coverage_score_0to100": round(coverage, 2),
        "quality_score_0to100": round(quality_pct, 2),
        "evidence_score_0to100": round(evidence_pct, 2),
        "quality_score_avg_0to10": round(quality_avg, 2),
        "total_items": n,
        "good_items": good_items,
        "positive_items": positive_items,
        "expected_items": n,
        "evidence_count": evidence_count,
        "target_evidence_per_item": TARGET_EVIDENCE_PER_ITEM,
    }


def _aggregate_overall(features: dict) -> dict:
    sections = list(features.values())
    totals = {
        "total_items": sum(s["overall"]["total_items"] for s in sections),
        "good_items":  sum(s["overall"]["good_items"]  for s in sections),
        "positive_items": sum(s["overall"]["positive_items"] for s in sections),
        "expected_items": sum(s["overall"]["expected_items"] for s in sections),
        "evidence_count": sum(s["overall"]["evidence_count"] for s in sections),
    }
    n_items = totals["total_items"] or 1
    def w(field):
        return sum(s["overall"][field] * s["overall"]["total_items"] for s in sections) / n_items
    return {
        "final_score_0to100":   round(w("final_score_0to100"), 2),
        "coverage_score_0to100": round(w("coverage_score_0to100"), 2),
        "quality_score_0to100":  round(w("quality_score_0to100"), 2),
        "evidence_score_0to100": round(w("evidence_score_0to100"), 2),
        "quality_score_avg_0to10": round(w("quality_score_avg_0to10"), 2),
        **totals,
        "target_evidence_per_item": TARGET_EVIDENCE_PER_ITEM,
    }


# ── vLLM driver ───────────────────────────────────────────────────────────────

class _Scorer:
    def __init__(self, model_name: str = MODEL_NAME):
        from vllm import LLM, SamplingParams  # imported lazily
        try:
            from vllm.sampling_params import GuidedDecodingParams
        except ImportError:  # older vllm
            GuidedDecodingParams = None  # type: ignore

        self._SamplingParams = SamplingParams
        self._GuidedDecodingParams = GuidedDecodingParams
        self.model_name = model_name

        print(f"[qwen3_vllm] loading {model_name} via vLLM…", flush=True)
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=float(os.environ.get("QWEN3_GPU_UTIL", "0.9")),
            max_model_len=int(os.environ.get("QWEN3_MAX_LEN", "16384")),
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _sampling_params(self):
        kwargs = dict(temperature=0.2, top_p=0.9, max_tokens=1024)
        if self._GuidedDecodingParams is not None:
            kwargs["guided_decoding"] = self._GuidedDecodingParams(json=CRITERION_SCHEMA)
        return self._SamplingParams(**kwargs)

    def _format_prompt(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def score_batch(self, jobs: list[tuple[str, str, dict]], application: dict) -> dict:
        """
        jobs: list of (section_key, section_name, criterion)
        Returns: {section_key: [criterion_result, ...]} preserving job order.
        """
        prompts = [
            self._format_prompt(_build_messages(application, sn, crit))
            for (_, sn, crit) in jobs
        ]
        sp = self._sampling_params()
        outputs = self.llm.generate(prompts, sp)

        results: dict[str, list[dict]] = {}
        for (sec_key, _sec_name, crit), out in zip(jobs, outputs):
            text = out.outputs[0].text.strip()
            try:
                # Strip any leading/trailing fences just in case.
                if text.startswith("```"):
                    text = text.strip("`")
                    if "\n" in text:
                        text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text[:-3]
                parsed = json.loads(text)
                parsed["name"] = crit["name"]
                parsed.setdefault("evidence_ids", [])
            except Exception as e:
                parsed = _empty_criterion_result(crit["name"], f"parse error: {e}")
            results.setdefault(sec_key, []).append(parsed)
        return results


# ── public API ────────────────────────────────────────────────────────────────

def score_application(application: dict, criteria_path: str | Path,
                      doc_id: str | None = None,
                      scorer: _Scorer | None = None) -> dict:
    rubric = json.loads(Path(criteria_path).read_text(encoding="utf-8"))

    jobs: list[tuple[str, str, dict]] = []
    for section_name, section_value in rubric.items():
        if section_name not in SECTION_KEY_MAP:
            continue
        section_key = SECTION_KEY_MAP[section_name]
        for crit in _flatten_section(section_name, section_value):
            jobs.append((section_key, section_name, crit))

    own_scorer = scorer is None
    if own_scorer:
        scorer = _Scorer()
    section_results = scorer.score_batch(jobs, application)

    features: dict = {}
    for section_key in SECTION_KEY_MAP.values():
        crits = section_results.get(section_key, [])
        if not crits:
            continue
        features[section_key] = {
            "criteria": crits,
            "overall": _aggregate_section(crits),
        }

    return {
        "doc_id": doc_id or "unknown",
        "run_info": {
            "ran_at_utc": _dt.datetime.utcnow().isoformat() + "Z",
            "model": scorer.model_name,
        },
        "features": features,
        "overall": _aggregate_overall(features),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("application_json", type=str)
    p.add_argument("--criteria", default=str(Path(__file__).parent / "criteria_points.json"))
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    in_path = Path(args.application_json)
    application = json.loads(in_path.read_text(encoding="utf-8"))
    result = score_application(application, args.criteria, doc_id=in_path.stem)

    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_scored.json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[qwen3_vllm] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
