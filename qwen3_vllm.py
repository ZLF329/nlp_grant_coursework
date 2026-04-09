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
MODEL_NAME = os.environ.get("QWEN3_MODEL", "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")
QUANTIZATION = os.environ.get("QWEN3_QUANTIZATION", "none")  # auto-detect by default; override with awq_marlin / gptq_marlin / compressed-tensors
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

# JSON schema used for vLLM guided decoding — one sub-item at a time.
SUBITEM_SCHEMA: dict = {
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


def _section_schema(n_subitems: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "sub_items": {
                "type": "array",
                "minItems": n_subitems,
                "maxItems": n_subitems,
                "items": SUBITEM_SCHEMA,
            }
        },
        "required": ["sub_items"],
    }


def _build_section_messages(criterion_name: str, sub_items: list[dict],
                            always_text: str, evidence_text: str) -> list[dict]:
    sys_prompt = (
        "You are a meticulous reviewer of UK NIHR grant applications. "
        "You will receive ONE rubric criterion (with multiple sub-items) plus "
        "context drawn from the application. Score every sub-item, extract "
        "verbatim evidence from the supplied excerpts, and reply with a single "
        "JSON object whose `sub_items` array has exactly N entries, in the same "
        "order as listed. Do not include any prose outside the JSON."
    )
    items_block = "\n".join(
        f"{i+1}. {it['name']}\n   Definition: {it.get('definition','')}\n"
        f"   Signals: {json.dumps(it.get('signals', []), ensure_ascii=False)}"
        for i, it in enumerate(sub_items)
    )
    user_prompt = (
        f"Criterion: {criterion_name}\n\n"
        f"Sub-items to score (return scores in this exact order):\n{items_block}\n\n"
        f"Application context (always-included):\n{always_text or '(none)'}\n\n"
        f"Retrieved evidence:\n{evidence_text}\n\n"
        "Score `quality_score_0to10` 0=missing, 5=present but weak, 8=clear "
        "and well-supported, 10=excellent and comprehensive. Each rubric "
        "sub-score is 0/1/2 (absent/partial/strong). Quote evidence verbatim "
        "from the supplied excerpts; leave the array empty if nothing supports "
        "the sub-item. Return ONLY the JSON object with the `sub_items` array."
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
        llm_kwargs = dict(
            model=model_name,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=float(os.environ.get("QWEN3_GPU_UTIL", "0.9")),
            max_model_len=int(os.environ.get("QWEN3_MAX_LEN", "16384")),
        )
        if QUANTIZATION and QUANTIZATION.lower() != "none":
            llm_kwargs["quantization"] = QUANTIZATION
            print(f"[qwen3_vllm] using quantization={QUANTIZATION}", flush=True)
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _sampling_params(self, schema: dict | None = None, max_tokens: int = 2048):
        kwargs = dict(temperature=0.2, top_p=0.9, max_tokens=max_tokens)
        if schema is not None and self._GuidedDecodingParams is not None:
            kwargs["guided_decoding"] = self._GuidedDecodingParams(json=schema)
        return self._SamplingParams(**kwargs)

    def _format_prompt(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def score_sections(self,
                       jobs: list[tuple[str, str, list[dict]]],
                       always_text: str,
                       evidence_map: dict[str, str]) -> dict[str, list[dict]]:
        """
        jobs: list of (criterion_key, criterion_name, sub_items).
        Returns: {criterion_key: [scored_sub_item, ...]} in sub_item order.
        """
        prompts: list[str] = []
        sps = []
        for (_key, name, sub_items) in jobs:
            messages = _build_section_messages(
                name, sub_items, always_text, evidence_map.get(_key, ""))
            prompts.append(self._format_prompt(messages))
            schema = _section_schema(len(sub_items))
            sps.append(self._sampling_params(schema=schema,
                                             max_tokens=512 + 256 * len(sub_items)))

        # vLLM accepts a single SamplingParams or a list aligned with prompts.
        outputs = self.llm.generate(prompts, sps)

        results: dict[str, list[dict]] = {}
        for (crit_key, _name, sub_items), out in zip(jobs, outputs):
            text = out.outputs[0].text.strip()
            scored: list[dict]
            try:
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
                scored = []
                for it, sub in zip(arr, sub_items):
                    it["name"] = sub["name"]
                    it.setdefault("evidence_ids", [])
                    scored.append(it)
            except Exception as e:
                scored = [_empty_criterion_result(s["name"], f"parse error: {e}")
                          for s in sub_items]
            results[crit_key] = scored
        return results


# ── public API ────────────────────────────────────────────────────────────────

def score_application(application: dict, criteria_path: str | Path,
                      doc_id: str | None = None,
                      scorer: _Scorer | None = None) -> dict:
    rubric = json.loads(Path(criteria_path).read_text(encoding="utf-8"))

    # Build per-criterion jobs (one LLM call per top-level rubric criterion).
    jobs: list[tuple[str, str, list[dict]]] = []
    for crit_name, crit_value in rubric.items():
        if crit_name not in SECTION_KEY_MAP:
            continue
        sub_items = _flatten_section(crit_name, crit_value)
        if sub_items:
            jobs.append((SECTION_KEY_MAP[crit_name], crit_name, sub_items))

    # ── RAG: per-criterion evidence retrieval ─────────────────────────────────
    from src.rag.retriever import retrieve_for_application
    from src.rag.stitcher import stitch_chunks

    rag_criteria = [
        {"key": k, "name": n, "sub_items": s} for (k, n, s) in jobs
    ]
    total_budget = int(os.environ.get("RAG_TOTAL_BUDGET", "40000"))
    always_text, chunk_map = retrieve_for_application(
        application, rag_criteria, total_budget=total_budget)
    evidence_map = {k: stitch_chunks(chunks) for k, chunks in chunk_map.items()}

    # ── LLM scoring (one call per criterion, batched through vLLM) ────────────
    own_scorer = scorer is None
    if own_scorer:
        scorer = _Scorer()
    section_results = scorer.score_sections(jobs, always_text, evidence_map)

    features: dict = {}
    for (crit_key, _name, _sub) in jobs:
        crits = section_results.get(crit_key, [])
        if not crits:
            continue
        features[crit_key] = {
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
