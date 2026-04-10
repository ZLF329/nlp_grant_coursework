from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from src.pool.build_pool import MAX_CHARS, build_chunk_pool, write_pool_artifacts

SECTION_KEY_MAP = {
    "General": "general",
    "Proposed research": "proposed_research",
    "Training and development": "training_development",
    "Sites and support": "sites_support",
    "Working with people and communities": "wpcc",
    "Application Form": "application_form",
}
SECTION_ID_PREFIX = {
    "general": "g",
    "proposed_research": "pr",
    "training_development": "td",
    "sites_support": "ss",
    "wpcc": "wp",
    "application_form": "af",
}
SCORER_ALLOWED_SCORES = (0, 1, 2)
RETRIEVAL_MAX_CHUNKS = 8
USED_CHUNK_MAX = 5
CONFIDENCE_TO_SCORE = {
    "low_confidence": 0,
    "medium_confidence": 1,
    "high_confidence": 2,
}


def load_rubric(criteria_path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(criteria_path).read_text(encoding="utf-8"))
    sections: list[dict[str, Any]] = []

    def build_signal_objects(prefix: str, signal_texts: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "sid": f"{prefix}.{chr(97 + idx)}",
                "text": signal_text,
                "weight": 1.0,
            }
            for idx, signal_text in enumerate(signal_texts)
        ]

    def build_sub(
        *,
        sub_id: str,
        name: str,
        definition: str,
        signal_texts: list[str],
        group_name: str | None = None,
        weight: float = 1.0,
    ) -> dict[str, Any]:
        sub = {
            "sub_id": sub_id,
            "name": name,
            "definition": definition,
            "weight": float(weight),
            "signals": build_signal_objects(sub_id, signal_texts),
        }
        if group_name:
            sub["group_name"] = group_name
        return sub

    for human_name, payload in raw.items():
        if human_name == "meta":
            continue
        section_key = SECTION_KEY_MAP.get(human_name)
        if not section_key:
            continue
        sub_criteria = []
        if isinstance(payload, dict) and isinstance(payload.get("sub_criteria"), list):
            for sub in payload.get("sub_criteria", []):
                signals = []
                for signal in sub.get("signals", []):
                    signals.append({
                        "sid": signal["sid"],
                        "text": signal["text"],
                        "weight": float(signal.get("weight", 1.0)),
                    })
                item = {
                    "sub_id": sub["sub_id"],
                    "name": sub["name"],
                    "definition": sub.get("definition", ""),
                    "weight": float(sub.get("weight", 1.0)),
                    "signals": signals,
                }
                if sub.get("group_name"):
                    item["group_name"] = sub["group_name"]
                sub_criteria.append(item)
        elif human_name == "General" and isinstance(payload, dict):
            idx = 1
            for signal_text in payload.get("common_characteristics_of_good_applications", []):
                sub_criteria.append(build_sub(
                    sub_id=f"g.{idx}",
                    name=f"Common Characteristics Of Good Applications: {signal_text}",
                    definition=signal_text,
                    signal_texts=[signal_text],
                    group_name="Common Characteristics Of Good Applications",
                ))
                idx += 1
            for signal_text in payload.get("tell_us_why_you_need_this_award", []):
                sub_criteria.append(build_sub(
                    sub_id=f"g.{idx}",
                    name=f"Tell Us Why You Need This Award: {signal_text}",
                    definition=signal_text,
                    signal_texts=[signal_text],
                    group_name="Tell Us Why You Need This Award",
                ))
                idx += 1
            for applicant_item in payload.get("Applicant", []):
                sub_criteria.append(build_sub(
                    sub_id=f"g.{idx}",
                    name=applicant_item["name"],
                    definition=applicant_item.get("definition", ""),
                    signal_texts=list(applicant_item.get("signals", [])),
                    group_name="Applicant",
                ))
                idx += 1
        elif isinstance(payload, list):
            prefix = SECTION_ID_PREFIX[section_key]
            for idx, sub in enumerate(payload, start=1):
                sub_criteria.append(build_sub(
                    sub_id=f"{prefix}.{idx}",
                    name=sub["name"],
                    definition=sub.get("definition", ""),
                    signal_texts=list(sub.get("signals", [])),
                ))
        else:
            continue
        sections.append({
            "human_name": human_name,
            "section_key": section_key,
            "weight": float(payload.get("weight", 1.0)) if isinstance(payload, dict) else 1.0,
            "sub_criteria": sub_criteria,
        })
    return sections


def load_raw_criteria(criteria_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(criteria_path).read_text(encoding="utf-8"))


def _safe_json_loads(text: str) -> dict[str, Any]:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = clean.strip("`")
        if "\n" in clean:
            clean = clean.split("\n", 1)[1]
        if clean.endswith("```"):
            clean = clean[:-3]
    return json.loads(clean)


def _response_preview(text: str, limit: int = 500) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _chunk_order_map(pool_lookup: dict[str, dict[str, str]]) -> dict[str, int]:
    return {chunk_id: idx for idx, chunk_id in enumerate(pool_lookup)}


def _sort_chunk_ids(chunk_ids: list[str], chunk_order: dict[str, int]) -> list[str]:
    return sorted(chunk_ids, key=lambda chunk_id: chunk_order.get(chunk_id, 10**9))


def _confidence_label(avg_gap: float) -> str:
    if avg_gap < 0.5:
        return "high_confidence"
    if avg_gap < 1.5:
        return "medium_confidence"
    return "low_confidence"


def _compat_plausibility(avg_confidence_0to2: float) -> float:
    return round((avg_confidence_0to2 / 2) * 5, 2)


def _compat_evidence_score(avg_confidence_0to2: float) -> float:
    return round((avg_confidence_0to2 / 2) * 100, 2)


def build_retrieval_messages(
    *,
    criteria_payload: dict[str, Any],
    pool_index_text: str,
) -> list[dict[str, str]]:
    system = (
        "You are selecting relevant evidence chunks for grant scoring.\n\n"
        "Return JSON only.\n"
        f"For each output section key, select as many potentially relevant chunk IDs as needed, up to {RETRIEVAL_MAX_CHUNKS}.\n"
        "Bias toward recall over precision: if a chunk may help score any sub-criterion or signal in that rubric section, include it.\n"
        "It is better to include borderline relevant chunks than to miss useful evidence.\n"
        "Use the criteria JSON exactly as provided.\n"
        "For each rubric section, select only chunk IDs from the provided pool index.\n"
        "Do not output explanations, prose, or markdown.\n"
        f"Return at most {RETRIEVAL_MAX_CHUNKS} chunk IDs per rubric section."
    )
    user = (
        "Criteria points JSON:\n"
        f"{json.dumps(criteria_payload, ensure_ascii=False, indent=2)}\n\n"
        "Output section key mapping:\n"
        f"{json.dumps(SECTION_KEY_MAP, ensure_ascii=False, indent=2)}\n\n"
        "Pool index:\n"
        f"{pool_index_text}\n\n"
        "Return format:\n"
        "{\n"
        '  "general": ["chunk_id_1", "chunk_id_2"],\n'
        '  "proposed_research": ["chunk_id_3"]\n'
        "}\n\n"
        "Return one top-level property per output section key from the mapping.\n"
        "Interpret each human-readable criteria section using the output section key mapping.\n"
        "Choose a high-recall set of chunk IDs for each rubric section.\n"
        "Include chunks that may support any sub-criterion or signal in that section.\n"
        "Do not be overly selective."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_retrieval_schema(rubric_sections: list[dict[str, Any]]) -> dict[str, Any]:
    properties = {
        section["section_key"]: {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": RETRIEVAL_MAX_CHUNKS,
        }
        for section in rubric_sections
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _normalize_retrieval_output(
    parsed: dict[str, Any],
    rubric_sections: list[dict[str, Any]],
    pool_lookup: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    valid_ids = set(pool_lookup)
    for section in rubric_sections:
        raw_ids = parsed.get(section["section_key"], []) if isinstance(parsed, dict) else []
        if not isinstance(raw_ids, list):
            raw_ids = []
        cleaned = [
            chunk_id
            for chunk_id in raw_ids
            if isinstance(chunk_id, str) and chunk_id in valid_ids
        ]
        normalized[section["section_key"]] = _dedupe_preserve_order(cleaned)[:RETRIEVAL_MAX_CHUNKS]
    return normalized


def build_evidence_text(
    chunk_ids: list[str],
    pool_lookup: dict[str, dict[str, str]],
    chunk_order: dict[str, int],
) -> str:
    if not chunk_ids:
        return "(no retrieved chunks)"

    ordered_ids = _sort_chunk_ids(_dedupe_preserve_order(chunk_ids), chunk_order)
    lines: list[str] = []
    last_position: int | None = None
    last_section: str | None = None

    for chunk_id in ordered_ids:
        position = chunk_order.get(chunk_id, 10**9)
        meta = pool_lookup[chunk_id]
        current_section = meta["parser_section"]

        if last_section != current_section:
            if lines:
                lines.append("")
            lines.append(f"===== {current_section} =====")
            lines.append("")

        if last_position is not None and position - last_position > 1:
            lines.extend(["", "...", ""])
        lines.extend([
            f"<{chunk_id}>",
            meta["text"],
            f"<{chunk_id}>",
            "",
        ])
        last_position = position
        last_section = current_section

    return "\n".join(lines).strip()


def _single_section_payload(section: dict[str, Any]) -> dict[str, Any]:
    return {
        "section_key": section["section_key"],
        "section_name": section["human_name"],
        "sub_criteria": [
            {
                "sub_id": sub["sub_id"],
                "name": sub["name"],
                "definition": sub["definition"],
                "group_name": sub.get("group_name"),
                "signals": [
                    {"sid": signal["sid"], "text": signal["text"]}
                    for signal in sub["signals"]
                ],
            }
            for sub in section["sub_criteria"]
        ],
    }


def build_scoring_messages(
    *,
    rubric_section: dict[str, Any],
    retrieved_chunk_ids: list[str],
    evidence_text: str,
) -> list[dict[str, str]]:
    system = (
        "You are scoring one rubric section of a grant application.\n\n"
        "Return JSON only.\n"
        "Each signal score must be exactly one of: 0, 1, 2.\n"
        "Each `used_chunk_ids` entry must be chosen only from the provided retrieved chunk IDs.\n"
        "Do not output explanations, prose, markdown, or extra keys."
    )
    user = (
        "rubric_section:\n"
        f"{json.dumps(_single_section_payload(rubric_section), ensure_ascii=False, indent=2)}\n\n"
        "retrieved_chunk_ids:\n"
        f"{json.dumps(retrieved_chunk_ids, ensure_ascii=False, indent=2)}\n\n"
        "evidence_text:\n"
        f"{evidence_text}\n\n"
        "return_format_rules:\n"
        "- Top-level keys must be the sub_id values for this rubric section.\n"
        "- Each sub_id object must contain `signals` and `used_chunk_ids`.\n"
        "- Each signal score must be 0, 1, or 2.\n"
        "- `used_chunk_ids` must contain only IDs from `retrieved_chunk_ids`.\n"
        "- If no supporting evidence is shown, score 0 and use an empty `used_chunk_ids` array.\n"
        "- You may only rely on chunk text shown in `evidence_text`; do not infer omitted content from `...`.\n"
        "- Return JSON only.\n\n"
        "example_output:\n"
        "{\n"
        '  "pr.1": {\n'
        '    "signals": {\n'
        '      "pr.1.a": 2,\n'
        '      "pr.1.b": 1\n'
        "    },\n"
        '    "used_chunk_ids": ["secadr__001", "secadr__004"]\n'
        "  },\n"
        '  "pr.2": {\n'
        '    "signals": {\n'
        '      "pr.2.a": 0,\n'
        '      "pr.2.b": 1\n'
        "    },\n"
        '    "used_chunk_ids": ["secadr__004"]\n'
        "  }\n"
        "}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_scoring_schema(rubric_section: dict[str, Any], retrieved_chunk_ids: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for sub in rubric_section["sub_criteria"]:
        signal_properties = {
            signal["sid"]: {
                "type": "integer",
                "enum": list(SCORER_ALLOWED_SCORES),
            }
            for signal in sub["signals"]
        }
        properties[sub["sub_id"]] = {
            "type": "object",
            "properties": {
                "signals": {
                    "type": "object",
                    "properties": signal_properties,
                    "additionalProperties": False,
                },
                "used_chunk_ids": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": retrieved_chunk_ids,
                    },
                    "maxItems": USED_CHUNK_MAX,
                },
            },
            "required": ["signals", "used_chunk_ids"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _normalize_score(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    normalized = int(value)
    return normalized if normalized in SCORER_ALLOWED_SCORES and normalized == value else 0


def _normalize_model_section_output(
    parsed: dict[str, Any],
    rubric_section: dict[str, Any],
    retrieved_chunk_ids: list[str],
) -> dict[str, dict[str, Any]]:
    allowed_ids = set(retrieved_chunk_ids)
    normalized: dict[str, dict[str, Any]] = {}

    for sub in rubric_section["sub_criteria"]:
        raw_sub = parsed.get(sub["sub_id"], {}) if isinstance(parsed, dict) else {}
        if not isinstance(raw_sub, dict):
            raw_sub = {}
        raw_signals = raw_sub.get("signals", {})
        if not isinstance(raw_signals, dict):
            raw_signals = {}
        raw_used_ids = raw_sub.get("used_chunk_ids", [])
        if not isinstance(raw_used_ids, list):
            raw_used_ids = []

        used_chunk_ids = [
            chunk_id
            for chunk_id in raw_used_ids
            if isinstance(chunk_id, str) and chunk_id in allowed_ids
        ]
        used_chunk_ids = _dedupe_preserve_order(used_chunk_ids)[:USED_CHUNK_MAX]

        signals: dict[str, int] = {}
        has_positive = False
        for signal in sub["signals"]:
            score = _normalize_score(raw_signals.get(signal["sid"], 0))
            signals[signal["sid"]] = score
            has_positive = has_positive or score > 0

        evidence_status = "ok"
        if has_positive and not used_chunk_ids:
            evidence_status = "invalid_evidence"
            signals = {sid: 0 for sid in signals}
        elif len(used_chunk_ids) == 1:
            evidence_status = "sparse_evidence"

        normalized[sub["sub_id"]] = {
            "signals": signals,
            "used_chunk_ids": used_chunk_ids,
            "evidence_status": evidence_status,
        }
    return normalized


def _build_evidence(
    used_chunk_ids: list[str],
    pool_lookup: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    evidence = []
    for chunk_id in used_chunk_ids:
        meta = pool_lookup.get(chunk_id)
        if not meta:
            continue
        evidence.append({
            "id": chunk_id,
            "section_id": chunk_id,
            "text": meta["text"],
            "section": meta["parser_section"],
            "source_path": meta["source_path"],
        })
    return evidence


def _aggregate_sub_criterion(
    sub: dict[str, Any],
    pool_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    total_weight = sum(signal["weight"] for signal in sub["signals"]) or 1.0
    weighted_sum = sum(signal["score_0to2_weighted"] * signal["weight"] for signal in sub["signals"])
    score_10 = round((weighted_sum / (2 * total_weight)) * 10, 2)
    confidence_score = CONFIDENCE_TO_SCORE[sub["confidence_label"]]
    avg_plausibility = _compat_plausibility(confidence_score)
    evidence = _build_evidence(sub["used_chunk_ids"], pool_lookup)
    return {
        **sub,
        "score_10": score_10,
        "quality_score_0to10": score_10,
        "avg_confidence_0to2": float(confidence_score),
        "avg_plausibility_0to5": avg_plausibility,
        "weak_signal_count": 1 if sub["confidence_label"] == "low_confidence" else 0,
        "evidence": evidence,
        "quality": (
            "good" if score_10 >= 7.5 else
            "mixed" if score_10 >= 4 else
            "weak"
        ),
        "exists": "yes" if score_10 > 0 else "no",
    }


def _aggregate_section(section: dict[str, Any], pool_lookup: dict[str, dict[str, str]]) -> dict[str, Any]:
    sub_criteria = [_aggregate_sub_criterion(sub, pool_lookup) for sub in section["sub_criteria"]]
    total_weight = sum(sub["weight"] for sub in sub_criteria) or 1.0
    weighted_score = sum(sub["score_10"] * sub["weight"] for sub in sub_criteria)
    score_10 = round(weighted_score / total_weight, 2)
    total_items = len(sub_criteria)
    signal_count = sum(len(sub["signals"]) for sub in sub_criteria)
    evidence_count = sum(sub["evidence_count"] for sub in sub_criteria)
    good_items = sum(1 for sub in sub_criteria if sub["score_10"] >= 7.5)
    positive_items = sum(1 for sub in sub_criteria if sub["score_10"] > 0)
    high_confidence_count = sum(1 for sub in sub_criteria if sub["confidence_label"] == "high_confidence")
    medium_confidence_count = sum(1 for sub in sub_criteria if sub["confidence_label"] == "medium_confidence")
    low_confidence_count = sum(1 for sub in sub_criteria if sub["confidence_label"] == "low_confidence")
    avg_confidence_0to2 = round(
        sum(CONFIDENCE_TO_SCORE[sub["confidence_label"]] for sub in sub_criteria) / max(1, total_items),
        2,
    )
    avg_plausibility = _compat_plausibility(avg_confidence_0to2)
    overall = {
        "score_10": score_10,
        "final_score_0to100": round(score_10 * 10, 2),
        "coverage_score_0to100": round((positive_items / max(1, total_items)) * 100, 2),
        "quality_score_0to100": round(score_10 * 10, 2),
        "evidence_score_0to100": _compat_evidence_score(avg_confidence_0to2),
        "quality_score_avg_0to10": score_10,
        "avg_confidence_0to2": avg_confidence_0to2,
        "avg_plausibility_0to5": avg_plausibility,
        "high_confidence_subcriterion_count": high_confidence_count,
        "medium_confidence_subcriterion_count": medium_confidence_count,
        "low_confidence_subcriterion_count": low_confidence_count,
        "weak_signal_count": low_confidence_count,
        "total_items": total_items,
        "signal_count": signal_count,
        "good_items": good_items,
        "positive_items": positive_items,
        "expected_items": total_items,
        "evidence_count": evidence_count,
        "target_evidence_per_item": USED_CHUNK_MAX,
    }
    return {
        "score_10": score_10,
        "avg_confidence_0to2": avg_confidence_0to2,
        "avg_plausibility_0to5": avg_plausibility,
        "high_confidence_subcriterion_count": high_confidence_count,
        "medium_confidence_subcriterion_count": medium_confidence_count,
        "low_confidence_subcriterion_count": low_confidence_count,
        "weak_signal_count": low_confidence_count,
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
            "avg_confidence_0to2": 0,
            "avg_plausibility_0to5": 0,
            "high_confidence_subcriterion_count": 0,
            "medium_confidence_subcriterion_count": 0,
            "low_confidence_subcriterion_count": 0,
            "weak_signal_count": 0,
            "total_items": 0,
            "signal_count": 0,
            "evidence_count": 0,
            "target_evidence_per_item": USED_CHUNK_MAX,
        }

    total_weight = sum(section_weights.get(key, 1.0) for key in features) or 1.0
    weighted_score = sum(features[key]["score_10"] * section_weights.get(key, 1.0) for key in features)
    score_10 = round(weighted_score / total_weight, 2)
    total_items = sum(section["overall"]["total_items"] for section in features.values())
    signal_count = sum(section["overall"]["signal_count"] for section in features.values())
    evidence_count = sum(section["overall"]["evidence_count"] for section in features.values())
    good_items = sum(section["overall"]["good_items"] for section in features.values())
    positive_items = sum(section["overall"]["positive_items"] for section in features.values())
    high_confidence_count = sum(section["overall"]["high_confidence_subcriterion_count"] for section in features.values())
    medium_confidence_count = sum(section["overall"]["medium_confidence_subcriterion_count"] for section in features.values())
    low_confidence_count = sum(section["overall"]["low_confidence_subcriterion_count"] for section in features.values())
    avg_confidence_0to2 = round(
        sum(section["overall"]["avg_confidence_0to2"] * section["overall"]["total_items"] for section in features.values())
        / max(1, total_items),
        2,
    )
    return {
        "score_10": score_10,
        "final_score_0to100": round(score_10 * 10, 2),
        "quality_score_0to100": round(score_10 * 10, 2),
        "coverage_score_0to100": round((positive_items / max(1, total_items)) * 100, 2),
        "evidence_score_0to100": _compat_evidence_score(avg_confidence_0to2),
        "quality_score_avg_0to10": score_10,
        "avg_confidence_0to2": avg_confidence_0to2,
        "avg_plausibility_0to5": _compat_plausibility(avg_confidence_0to2),
        "high_confidence_subcriterion_count": high_confidence_count,
        "medium_confidence_subcriterion_count": medium_confidence_count,
        "low_confidence_subcriterion_count": low_confidence_count,
        "weak_signal_count": low_confidence_count,
        "total_items": total_items,
        "signal_count": signal_count,
        "evidence_count": evidence_count,
        "good_items": good_items,
        "positive_items": positive_items,
        "expected_items": total_items,
        "target_evidence_per_item": USED_CHUNK_MAX,
    }


def _ensemble_section(
    rubric_section: dict[str, Any],
    model_a_section: dict[str, dict[str, Any]],
    model_b_section: dict[str, dict[str, Any]],
    pool_lookup: dict[str, dict[str, str]],
    chunk_order: dict[str, int],
) -> dict[str, Any]:
    section_subs: list[dict[str, Any]] = []
    for sub in rubric_section["sub_criteria"]:
        model_a_sub = model_a_section.get(sub["sub_id"], {})
        model_b_sub = model_b_section.get(sub["sub_id"], {})
        union_ids = _dedupe_preserve_order([
            *model_a_sub.get("used_chunk_ids", []),
            *model_b_sub.get("used_chunk_ids", []),
        ])
        union_ids = _sort_chunk_ids(union_ids, chunk_order)[:USED_CHUNK_MAX]
        evidence_status = "ok"
        if len(union_ids) == 1:
            evidence_status = "sparse_evidence"

        signals: list[dict[str, Any]] = []
        gaps: list[int] = []
        for signal in sub["signals"]:
            score_a = int(model_a_sub.get("signals", {}).get(signal["sid"], 0))
            score_b = int(model_b_sub.get("signals", {}).get(signal["sid"], 0))
            gap = abs(score_a - score_b)
            avg_score = round((score_a + score_b) / 2, 4)
            signals.append({
                "sid": signal["sid"],
                "signal_text": signal["text"],
                "weight": signal["weight"],
                "model_a_score": score_a,
                "model_b_score": score_b,
                "score_0to2_raw": avg_score,
                "score_0to2_weighted": avg_score,
            })
            gaps.append(gap)

        confidence_gap = round(sum(gaps) / max(1, len(gaps)), 2)
        confidence_label = _confidence_label(confidence_gap)
        section_subs.append({
            "sub_id": sub["sub_id"],
            "name": sub["name"],
            "definition": sub["definition"],
            "group_name": sub.get("group_name"),
            "weight": sub["weight"],
            "used_chunk_ids": union_ids,
            "evidence_count": len(union_ids),
            "evidence_status": evidence_status,
            "confidence_gap": confidence_gap,
            "confidence_label": confidence_label,
            "signals": signals,
        })

    return {
        "human_name": rubric_section["human_name"],
        "section_key": rubric_section["section_key"],
        "weight": rubric_section["weight"],
        "sub_criteria": section_subs,
    }


def _write_artifacts(
    *,
    artifacts_dir: str | Path | None,
    doc_id: str,
    pool_lookup: dict[str, dict[str, str]],
    pool_index_text: str,
    retrieval_raw: str,
    retrieval_parsed: dict[str, Any],
    model_a_raw_by_section: dict[str, str],
    model_b_raw_by_section: dict[str, str],
    normalized_sections: list[dict[str, Any]],
) -> dict[str, str]:
    if artifacts_dir is None:
        return {}

    artifacts = write_pool_artifacts(
        pool_lookup=pool_lookup,
        pool_index_text=pool_index_text,
        artifacts_dir=artifacts_dir,
        doc_id=doc_id,
    )
    artifacts_path = Path(artifacts_dir)
    retrieval_raw_path = artifacts_path / f"{doc_id}_retrieval_raw.txt"
    retrieval_parsed_path = artifacts_path / f"{doc_id}_retrieval_parsed.json"
    model_a_path = artifacts_path / f"{doc_id}_model_a_raw.json"
    model_b_path = artifacts_path / f"{doc_id}_model_b_raw.json"
    ensemble_path = artifacts_path / f"{doc_id}_ensemble_normalized.json"

    retrieval_raw_path.write_text(retrieval_raw, encoding="utf-8")
    retrieval_parsed_path.write_text(json.dumps(retrieval_parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    model_a_path.write_text(json.dumps(model_a_raw_by_section, ensure_ascii=False, indent=2), encoding="utf-8")
    model_b_path.write_text(json.dumps(model_b_raw_by_section, ensure_ascii=False, indent=2), encoding="utf-8")
    ensemble_path.write_text(json.dumps(normalized_sections, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        **artifacts,
        "retrieval_raw_response": str(retrieval_raw_path),
        "retrieval_parsed_response": str(retrieval_parsed_path),
        "model_a_raw_response": str(model_a_path),
        "model_b_raw_response": str(model_b_path),
        "ensemble_normalized_response": str(ensemble_path),
    }


def score_application_base(
    *,
    application: dict[str, Any],
    criteria_path: str | Path,
    doc_id: str | None,
    retrieval_client: Any,
    scorer_client_a: Any,
    scorer_client_b: Any,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    criteria_payload = load_raw_criteria(criteria_path)
    rubric_sections = load_rubric(criteria_path)
    pool_data = build_chunk_pool(application, max_chars=MAX_CHARS)
    pool_lookup = pool_data["pool_lookup"]
    pool_index_text = pool_data["pool_index_text"]
    chunk_order = _chunk_order_map(pool_lookup)

    retrieval_messages = build_retrieval_messages(
        criteria_payload=criteria_payload,
        pool_index_text=pool_index_text,
    )
    retrieval_schema = build_retrieval_schema(rubric_sections)
    retrieval_raw = retrieval_client.generate_json(retrieval_messages, schema=retrieval_schema, max_tokens=4096)
    try:
        retrieval_parsed = _safe_json_loads(retrieval_raw)
    except Exception as exc:
        raise ValueError(
            "Retrieval model did not return valid JSON. "
            f"Response preview: {_response_preview(retrieval_raw)!r}"
        ) from exc
    retrieved_chunks = _normalize_retrieval_output(retrieval_parsed, rubric_sections, pool_lookup)

    model_a_raw_by_section: dict[str, str] = {}
    model_b_raw_by_section: dict[str, str] = {}
    sections: list[dict[str, Any]] = []

    for rubric_section in rubric_sections:
        section_key = rubric_section["section_key"]
        section_chunk_ids = retrieved_chunks.get(section_key, [])
        evidence_text = build_evidence_text(section_chunk_ids, pool_lookup, chunk_order)
        messages = build_scoring_messages(
            rubric_section=rubric_section,
            retrieved_chunk_ids=section_chunk_ids,
            evidence_text=evidence_text,
        )
        schema = build_scoring_schema(rubric_section, section_chunk_ids)

        raw_a = scorer_client_a.generate_json(messages, schema=schema, max_tokens=4096)
        model_a_raw_by_section[section_key] = raw_a
        raw_b = scorer_client_b.generate_json(messages, schema=schema, max_tokens=4096)
        model_b_raw_by_section[section_key] = raw_b

        try:
            parsed_a = _safe_json_loads(raw_a)
        except Exception as exc:
            raise ValueError(
                f"Scorer A returned invalid JSON for section {section_key}. "
                f"Response preview: {_response_preview(raw_a)!r}"
            ) from exc
        try:
            parsed_b = _safe_json_loads(raw_b)
        except Exception as exc:
            raise ValueError(
                f"Scorer B returned invalid JSON for section {section_key}. "
                f"Response preview: {_response_preview(raw_b)!r}"
            ) from exc

        normalized_a = _normalize_model_section_output(parsed_a, rubric_section, section_chunk_ids)
        normalized_b = _normalize_model_section_output(parsed_b, rubric_section, section_chunk_ids)
        sections.append(_ensemble_section(
            rubric_section,
            normalized_a,
            normalized_b,
            pool_lookup,
            chunk_order,
        ))

    artifact_paths = _write_artifacts(
        artifacts_dir=artifacts_dir,
        doc_id=doc_id or "unknown",
        pool_lookup=pool_lookup,
        pool_index_text=pool_index_text,
        retrieval_raw=retrieval_raw,
        retrieval_parsed=retrieval_parsed,
        model_a_raw_by_section=model_a_raw_by_section,
        model_b_raw_by_section=model_b_raw_by_section,
        normalized_sections=sections,
    )

    features = {
        section["section_key"]: _aggregate_section(section, pool_lookup)
        for section in sections
    }
    section_weights = {section["section_key"]: section["weight"] for section in sections}

    return {
        "doc_id": doc_id or "unknown",
        "run_info": {
            "ran_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
            "retrieval_model": getattr(retrieval_client, "model_name", "unknown"),
            "scorer_model_a": getattr(scorer_client_a, "model_name", "unknown"),
            "scorer_model_b": getattr(scorer_client_b, "model_name", "unknown"),
        },
        "pool_size": len(pool_lookup),
        "pool_lookup": pool_lookup,
        "section_chunk_ids": pool_data["section_chunk_ids"],
        "features": features,
        "overall": _aggregate_overall(features, section_weights),
        "debug": {
            "scoring_contract_version": "chunk_retrieval_dual_model_v1",
            "retrieved_chunks": retrieved_chunks,
            "artifacts": artifact_paths,
        },
    }
