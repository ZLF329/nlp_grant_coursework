from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from src.pool.build_pool import build_chunk_pool, write_pool_artifacts
from src.verify.faithfulness import FallbackJudge, run_faithfulness_audit

SECTION_KEY_MAP = {
    "General": "general",
    "Proposed research": "proposed_research",
    "Training and development": "training_development",
    "Sites and support": "sites_support",
    "Working with people and communities": "wpcc",
    "Application Form": "application_form",
}
SECTION_NAME_MAP = {value: key for key, value in SECTION_KEY_MAP.items()}
ID_PATTERN = re.compile(r"^sec[a-z0-9]+__\d{3}(_[a-z])?$")

PLAUSIBILITY_TO_MULTIPLIER = {
    5: (1.0, None),
    4: (1.0, None),
    3: (1.0, "low_confidence"),
    2: (0.7, "low_confidence"),
    1: (0.4, "weak_evidence"),
    0: (0.0, "hallucination"),
}


def load_rubric(criteria_path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(criteria_path).read_text(encoding="utf-8"))
    sections: list[dict[str, Any]] = []
    for human_name, payload in raw.items():
        section_key = SECTION_KEY_MAP.get(human_name)
        if not section_key:
            continue
        sub_criteria = []
        for sub in payload.get("sub_criteria", []):
            signals = []
            for signal in sub.get("signals", []):
                signals.append({
                    "sid": signal["sid"],
                    "text": signal["text"],
                    "weight": float(signal.get("weight", 1.0)),
                })
            sub_criteria.append({
                "sub_id": sub["sub_id"],
                "name": sub["name"],
                "definition": sub.get("definition", ""),
                "weight": float(sub.get("weight", 1.0)),
                "signals": signals,
            })
        sections.append({
            "human_name": human_name,
            "section_key": section_key,
            "weight": float(payload.get("weight", 1.0)),
            "sub_criteria": sub_criteria,
        })
    return sections


def build_stage1_messages(
    *,
    application: dict[str, Any],
    rubric_sections: list[dict[str, Any]],
    pool_index_text: str,
) -> list[dict[str, str]]:
    rubric_payload = []
    for section in rubric_sections:
        rubric_payload.append({
            "section_key": section["section_key"],
            "section_name": section["human_name"],
            "sub_criteria": [
                {
                    "sub_id": sub["sub_id"],
                    "name": sub["name"],
                    "definition": sub["definition"],
                    "signals": [
                        {"sid": signal["sid"], "text": signal["text"]}
                        for signal in sub["signals"]
                    ],
                }
                for sub in section["sub_criteria"]
            ],
        })

    system = (
        "You are scoring a UK NIHR grant application against a rubric. "
        "Return JSON only. Evidence must be chosen strictly from the provided chunk IDs. "
        "For each sub-criterion, return exactly five evidence slots in order of support "
        "strength. If there are fewer than five valid chunks, fill remaining slots with null. "
        "Never quote evidence text. Score each signal as 0, 1, or 2. Keep each rationale short."
    )
    user = (
        "Application JSON:\n"
        f"{json.dumps(application, ensure_ascii=False, indent=2)}\n\n"
        "Rubric:\n"
        f"{json.dumps(rubric_payload, ensure_ascii=False, indent=2)}\n\n"
        "Chunk pool index:\n"
        f"{pool_index_text}\n\n"
        "Rules:\n"
        "1. Use only chunk IDs that appear in the chunk pool index.\n"
        "2. `evidence_top5` must always have exactly 5 slots.\n"
        "3. Use null for unsupported slots instead of inventing evidence.\n"
        "4. Return sub-criteria in the same order as the rubric.\n"
        "5. Return signals in the same order as the rubric."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_stage1_schema(rubric_sections: list[dict[str, Any]]) -> dict[str, Any]:
    sub_ids = [sub["sub_id"] for section in rubric_sections for sub in section["sub_criteria"]]
    signal_ids = [
        signal["sid"]
        for section in rubric_sections
        for sub in section["sub_criteria"]
        for signal in sub["signals"]
    ]
    return {
        "type": "object",
        "properties": {
            "sub_criteria": {
                "type": "array",
                "minItems": len(sub_ids),
                "maxItems": len(sub_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "sub_id": {"type": "string", "enum": sub_ids},
                        "evidence_top5": {
                            "type": "array",
                            "minItems": 5,
                            "maxItems": 5,
                            "items": {
                                "anyOf": [
                                    {"type": "string", "pattern": ID_PATTERN.pattern},
                                    {"type": "null"},
                                ]
                            },
                        },
                        "signals": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sid": {"type": "string", "enum": signal_ids},
                                    "score": {"type": "integer", "minimum": 0, "maximum": 2},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["sid", "score", "rationale"],
                            },
                        },
                    },
                    "required": ["sub_id", "evidence_top5", "signals"],
                },
            }
        },
        "required": ["sub_criteria"],
    }


def _safe_json_loads(text: str) -> dict[str, Any]:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = clean.strip("`")
        if "\n" in clean:
            clean = clean.split("\n", 1)[1]
        if clean.endswith("```"):
            clean = clean[:-3]
    return json.loads(clean)


def _normalize_stage1_output(
    parsed: dict[str, Any],
    rubric_sections: list[dict[str, Any]],
    pool_lookup: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    raw_subs = parsed.get("sub_criteria", []) if isinstance(parsed, dict) else []
    by_sub_id = {}
    for entry in raw_subs:
        if isinstance(entry, dict) and isinstance(entry.get("sub_id"), str):
            by_sub_id.setdefault(entry["sub_id"], entry)

    normalized_sections: list[dict[str, Any]] = []
    for section in rubric_sections:
        section_subs: list[dict[str, Any]] = []
        for expected_sub in section["sub_criteria"]:
            raw_sub = by_sub_id.get(expected_sub["sub_id"], {})
            raw_signals = raw_sub.get("signals", []) if isinstance(raw_sub, dict) else []
            raw_signal_map = {}
            for item in raw_signals:
                if isinstance(item, dict) and isinstance(item.get("sid"), str):
                    raw_signal_map.setdefault(item["sid"], item)

            evidence_slots = raw_sub.get("evidence_top5", []) if isinstance(raw_sub, dict) else []
            evidence_slots = list(evidence_slots)[:5]
            while len(evidence_slots) < 5:
                evidence_slots.append(None)

            seen_ids: set[str] = set()
            normalized_slots: list[str | None] = []
            evidence_ids: list[str] = []
            for slot in evidence_slots:
                if not isinstance(slot, str) or not ID_PATTERN.match(slot) or slot not in pool_lookup:
                    normalized_slots.append(None)
                    continue
                if slot in seen_ids:
                    normalized_slots.append(None)
                    continue
                seen_ids.add(slot)
                normalized_slots.append(slot)
                evidence_ids.append(slot)

            signals = []
            has_positive = False
            for expected_signal in expected_sub["signals"]:
                raw_signal = raw_signal_map.get(expected_signal["sid"], {})
                raw_score = int(raw_signal.get("score", 0)) if isinstance(raw_signal.get("score", 0), int) else 0
                raw_score = max(0, min(raw_score, 2))
                rationale = raw_signal.get("rationale", "") if isinstance(raw_signal.get("rationale"), str) else ""
                signals.append({
                    "sid": expected_signal["sid"],
                    "signal_text": expected_signal["text"],
                    "weight": expected_signal["weight"],
                    "score_0to2_raw": raw_score,
                    "rationale": rationale.strip(),
                })
                has_positive = has_positive or raw_score > 0

            empty_zero_evidence = len(evidence_ids) == 0 and not has_positive
            evidence_status = "ok"
            if len(evidence_ids) == 0 and has_positive:
                evidence_status = "invalid_evidence"
            elif 0 < len(evidence_ids) <= 2:
                evidence_status = "sparse_evidence"

            if evidence_status == "invalid_evidence" and has_positive:
                for signal in signals:
                    signal["score_0to2_raw"] = 0

            section_subs.append({
                "sub_id": expected_sub["sub_id"],
                "name": expected_sub["name"],
                "definition": expected_sub["definition"],
                "weight": expected_sub["weight"],
                "evidence_top5": normalized_slots,
                "evidence_ids": evidence_ids,
                "evidence_count": len(evidence_ids),
                "evidence_status": evidence_status,
                "empty_zero_evidence": empty_zero_evidence,
                "signals": signals,
            })

        normalized_sections.append({
            "human_name": section["human_name"],
            "section_key": section["section_key"],
            "weight": section["weight"],
            "sub_criteria": section_subs,
        })
    return normalized_sections


def _section_audit_payload(section: dict[str, Any], pool_lookup: dict[str, dict[str, str]]) -> dict[str, Any]:
    referenced_ids: set[str] = set()
    for sub in section["sub_criteria"]:
        for chunk_id in sub["evidence_ids"]:
            referenced_ids.add(chunk_id)

    section_blocks: list[str] = []
    parser_section_order: list[str] = []
    seen_sections: set[str] = set()
    for chunk_id, meta in pool_lookup.items():
        if chunk_id not in referenced_ids:
            continue
        parser_section = meta["parser_section"]
        if parser_section not in seen_sections:
            seen_sections.add(parser_section)
            parser_section_order.append(parser_section)

    for parser_section in parser_section_order:
        parts: list[str] = [f"[{parser_section}]"]
        seen_selected = False
        in_gap = False
        for chunk_id, meta in pool_lookup.items():
            if meta["parser_section"] != parser_section:
                continue
            if chunk_id in referenced_ids:
                if seen_selected and in_gap:
                    parts.append("...")
                parts.append(meta["text"])
                seen_selected = True
                in_gap = False
            elif seen_selected:
                in_gap = True
        parts.append(f"[{parser_section}]")
        section_blocks.append("\n".join(parts))

    return {
        "evidence_context": "\n\n".join(section_blocks),
        "section_key": section["section_key"],
        "sub_criteria": [
            {
                "sub_id": sub["sub_id"],
                "sub_criterion": sub["name"],
                "evidence_status": sub["evidence_status"],
                "signals": [
                    {
                        "sid": signal["sid"],
                        "signal_text": signal["signal_text"],
                        "stage1_score": signal["score_0to2_raw"],
                        "stage1_rationale": signal["rationale"],
                    }
                    for signal in sub["signals"]
                ],
            }
            for sub in section["sub_criteria"]
            if sub["evidence_status"] != "invalid_evidence" and not sub.get("empty_zero_evidence")
        ],
    }


def _apply_plausibility(section: dict[str, Any], audited: dict[str, dict[str, Any]]) -> None:
    for sub in section["sub_criteria"]:
        for signal in sub["signals"]:
            if sub["evidence_status"] == "invalid_evidence":
                plausibility = 0
                note = "invalid_evidence"
            elif sub.get("empty_zero_evidence"):
                plausibility = 5
                note = "no_evidence_for_zero_score"
            else:
                result = audited.get(signal["sid"], {})
                plausibility = int(result.get("plausibility", 3))
                plausibility = max(0, min(plausibility, 5))
                note = result.get("note")
            multiplier, flag = PLAUSIBILITY_TO_MULTIPLIER[plausibility]
            signal["plausibility_0to5"] = plausibility
            signal["multiplier"] = multiplier
            signal["flag"] = flag
            signal["note"] = note
            signal["score_0to2_weighted"] = round(signal["score_0to2_raw"] * multiplier, 4)


def _aggregate_sub_criterion(sub: dict[str, Any], pool_lookup: dict[str, dict[str, str]]) -> dict[str, Any]:
    total_weight = sum(signal["weight"] for signal in sub["signals"]) or 1.0
    weighted_sum = sum(signal["score_0to2_weighted"] * signal["weight"] for signal in sub["signals"])
    score_10 = round((weighted_sum / (2 * total_weight)) * 10, 2)
    avg_plausibility = round(
        sum(signal["plausibility_0to5"] for signal in sub["signals"]) / max(1, len(sub["signals"])),
        2,
    )
    weak_signal_count = sum(1 for signal in sub["signals"] if signal["plausibility_0to5"] <= 1)
    evidence = [
        {
            "id": chunk_id,
            "text": pool_lookup[chunk_id]["text"],
            "section": pool_lookup[chunk_id]["parser_section"],
            "source_path": pool_lookup[chunk_id]["source_path"],
        }
        for chunk_id in sub["evidence_ids"]
        if chunk_id in pool_lookup
    ]
    return {
        **sub,
        "score_10": score_10,
        "avg_plausibility_0to5": avg_plausibility,
        "weak_signal_count": weak_signal_count,
        "evidence": evidence,
        "quality_score_0to10": score_10,
        "quality": (
            "good" if score_10 >= 7.5 else
            "mixed" if score_10 >= 4 else
            "weak"
        ),
        "exists": "yes" if score_10 > 0 else "no",
    }


def _aggregate_section(section: dict[str, Any], pool_lookup: dict[str, dict[str, str]]) -> dict[str, Any]:
    sub_criteria = [
        _aggregate_sub_criterion(sub, pool_lookup) for sub in section["sub_criteria"]
    ]
    total_weight = sum(sub["weight"] for sub in sub_criteria) or 1.0
    weighted_score = sum(sub["score_10"] * sub["weight"] for sub in sub_criteria)
    score_10 = round(weighted_score / total_weight, 2)
    all_signals = [signal for sub in sub_criteria for signal in sub["signals"]]
    avg_plausibility = round(
        sum(signal["plausibility_0to5"] for signal in all_signals) / max(1, len(all_signals)),
        2,
    )
    weak_signal_count = sum(1 for signal in all_signals if signal["plausibility_0to5"] <= 1)
    evidence_count = sum(sub["evidence_count"] for sub in sub_criteria)
    total_items = len(sub_criteria)
    good_items = sum(1 for sub in sub_criteria if sub["score_10"] >= 7.5)
    positive_items = sum(1 for sub in sub_criteria if sub["score_10"] > 0)

    overall = {
        "score_10": score_10,
        "final_score_0to100": round(score_10 * 10, 2),
        "coverage_score_0to100": round((positive_items / max(1, total_items)) * 100, 2),
        "quality_score_0to100": round(score_10 * 10, 2),
        "evidence_score_0to100": round((avg_plausibility / 5) * 100, 2),
        "quality_score_avg_0to10": score_10,
        "avg_plausibility_0to5": avg_plausibility,
        "weak_signal_count": weak_signal_count,
        "total_items": total_items,
        "signal_count": len(all_signals),
        "good_items": good_items,
        "positive_items": positive_items,
        "expected_items": total_items,
        "evidence_count": evidence_count,
        "target_evidence_per_item": 5,
    }
    return {
        "score_10": score_10,
        "avg_plausibility_0to5": avg_plausibility,
        "weak_signal_count": weak_signal_count,
        "sub_criteria": sub_criteria,
        "criteria": sub_criteria,
        "overall": overall,
    }


def _aggregate_overall(features: dict[str, dict[str, Any]], section_weights: dict[str, float]) -> dict[str, Any]:
    if not features:
        return {
            "score_10": 0,
            "final_score_0to100": 0,
            "quality_score_0to100": 0,
            "coverage_score_0to100": 0,
            "evidence_score_0to100": 0,
            "quality_score_avg_0to10": 0,
            "avg_plausibility_0to5": 0,
            "weak_signal_count": 0,
            "total_items": 0,
            "signal_count": 0,
            "evidence_count": 0,
            "target_evidence_per_item": 5,
        }

    total_weight = sum(section_weights.get(key, 1.0) for key in features) or 1.0
    weighted_score = sum(
        features[key]["score_10"] * section_weights.get(key, 1.0) for key in features
    )
    score_10 = round(weighted_score / total_weight, 2)
    totals = {
        "total_items": sum(section["overall"]["total_items"] for section in features.values()),
        "signal_count": sum(section["overall"]["signal_count"] for section in features.values()),
        "weak_signal_count": sum(section["overall"]["weak_signal_count"] for section in features.values()),
        "evidence_count": sum(section["overall"]["evidence_count"] for section in features.values()),
        "good_items": sum(section["overall"]["good_items"] for section in features.values()),
        "positive_items": sum(section["overall"]["positive_items"] for section in features.values()),
        "expected_items": sum(section["overall"]["expected_items"] for section in features.values()),
    }
    avg_plausibility = round(
        sum(
            section["overall"]["avg_plausibility_0to5"] * section["overall"]["signal_count"]
            for section in features.values()
        ) / max(1, totals["signal_count"]),
        2,
    )
    return {
        "score_10": score_10,
        "final_score_0to100": round(score_10 * 10, 2),
        "quality_score_0to100": round(score_10 * 10, 2),
        "coverage_score_0to100": round(
            (totals["positive_items"] / max(1, totals["total_items"])) * 100,
            2,
        ),
        "evidence_score_0to100": round((avg_plausibility / 5) * 100, 2),
        "quality_score_avg_0to10": score_10,
        "avg_plausibility_0to5": avg_plausibility,
        **totals,
        "target_evidence_per_item": 5,
    }


def score_application_base(
    *,
    application: dict[str, Any],
    criteria_path: str | Path,
    doc_id: str | None,
    stage1_client: Any,
    judge: Any | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    rubric_sections = load_rubric(criteria_path)
    pool = build_chunk_pool(application)
    if artifacts_dir is not None:
        write_pool_artifacts(
            pool_lookup=pool["pool_lookup"],
            pool_index_text=pool["pool_index_text"],
            artifacts_dir=artifacts_dir,
            doc_id=doc_id or "unknown",
        )

    messages = build_stage1_messages(
        application=application,
        rubric_sections=rubric_sections,
        pool_index_text=pool["pool_index_text"],
    )
    schema = build_stage1_schema(rubric_sections)
    raw_response = stage1_client.generate_json(messages, schema=schema, max_tokens=8192)
    try:
        parsed = _safe_json_loads(raw_response)
    except Exception:
        parsed = {}

    sections = _normalize_stage1_output(parsed, rubric_sections, pool["pool_lookup"])

    judge = judge or FallbackJudge()
    audit_payloads = {
        section["section_key"]: _section_audit_payload(section, pool["pool_lookup"])
        for section in sections
    }
    audited = run_faithfulness_audit(audit_payloads, judge)
    for section in sections:
        _apply_plausibility(section, audited.get(section["section_key"], {}))

    features = {
        section["section_key"]: _aggregate_section(section, pool["pool_lookup"])
        for section in sections
    }
    section_weights = {section["section_key"]: section["weight"] for section in sections}

    return {
        "doc_id": doc_id or "unknown",
        "run_info": {
            "ran_at_utc": dt.datetime.utcnow().isoformat() + "Z",
            "model": getattr(stage1_client, "model_name", "unknown"),
            "judge_model": getattr(judge, "model_name", getattr(judge, "__class__", type(judge)).__name__),
        },
        "pool_size": len(pool["pool_lookup"]),
        "pool_lookup": pool["pool_lookup"],
        "section_chunk_ids": pool["section_chunk_ids"],
        "features": features,
        "overall": _aggregate_overall(features, section_weights),
    }
