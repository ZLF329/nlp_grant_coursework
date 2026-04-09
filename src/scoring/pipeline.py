from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

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


def _stringify_leaf(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, indent=2).strip()


def _child_path(path: list[str], key: Any) -> list[str]:
    if isinstance(key, int):
        return [*path, f"[{key}]"]
    return [*path, str(key)]


def _iter_leaves(value: Any, path: list[str]) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for key, child in value.items():
            out.extend(_iter_leaves(child, _child_path(path, key)))
        return out
    if isinstance(value, list):
        out: list[tuple[str, str]] = []
        for idx, child in enumerate(value):
            out.extend(_iter_leaves(child, _child_path(path, idx)))
        return out

    text = _stringify_leaf(value)
    return [(text, " > ".join(path))] if text else []


def build_section_index(application: dict[str, Any]) -> dict[str, Any]:
    ordered_sections: list[str] = []
    section_text_parts: dict[str, list[str]] = {}
    section_source_paths: dict[str, list[str]] = {}

    def register_section(parser_section: str) -> None:
        if parser_section not in section_text_parts:
            ordered_sections.append(parser_section)
            section_text_parts[parser_section] = []
            section_source_paths[parser_section] = []

    for root_key, root_value in application.items():
        if isinstance(root_value, dict):
            for child_key, child_value in root_value.items():
                parser_section = str(child_key)
                register_section(parser_section)
                for leaf_text, source_path in _iter_leaves(
                    child_value,
                    [str(root_key), str(child_key)],
                ):
                    section_text_parts[parser_section].append(leaf_text)
                    section_source_paths[parser_section].append(source_path)
        else:
            parser_section = str(root_key)
            register_section(parser_section)
            for leaf_text, source_path in _iter_leaves(root_value, [str(root_key)]):
                section_text_parts[parser_section].append(leaf_text)
                section_source_paths[parser_section].append(source_path)

    section_index: dict[str, str] = {}
    section_name_to_id: dict[str, str] = {}
    section_text_lookup: dict[str, dict[str, Any]] = {}

    for idx, section_name in enumerate(ordered_sections, start=1):
        section_id = f"S{idx:02d}"
        section_index[section_id] = section_name
        section_name_to_id[section_name] = section_id
        dedup_source_paths = list(dict.fromkeys(section_source_paths.get(section_name, [])))
        section_text_lookup[section_id] = {
            "section": section_name,
            "text": "\n\n".join(section_text_parts.get(section_name, [])),
            "source_paths": dedup_source_paths,
        }

    return {
        "section_index": section_index,
        "section_name_to_id": section_name_to_id,
        "section_text_lookup": section_text_lookup,
    }


def build_stage1_messages(
    *,
    application: dict[str, Any],
    rubric_sections: list[dict[str, Any]],
    section_index: dict[str, str],
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
        "You are scoring a UK NIHR grant application against a rubric.\n\n"
        "Return JSON only.\n\n"
        "Use only section IDs from the provided section index.\n"
        "Do not output rationale, explanations, prose, markdown, section names, or chunk IDs.\n"
        "For each signal, return a score of 0, 1, or 2.\n\n"
        "Important:\n"
        "Inside each rubric section, the sub-criteria must be stored as object properties keyed by sub_id.\n"
        "Do not use arrays for sub-criteria."
    )
    user = (
        "Application JSON:\n"
        f"{json.dumps(application, ensure_ascii=False, indent=2)}\n\n"
        "Section index:\n"
        f"{json.dumps(section_index, ensure_ascii=False, indent=2)}\n\n"
        "Rubric:\n"
        f"{json.dumps(rubric_payload, ensure_ascii=False, indent=2)}\n\n"
        "Return format rules:\n"
        "1. Top-level keys must be rubric section keys.\n"
        "2. Inside each rubric section, use `sub_id` as the property name.\n"
        "3. Each sub-criterion object must contain:\n"
        "   - `signals`\n"
        "   - `needed_section_ids`\n"
        "4. `signals` maps signal IDs to scores 0, 1, or 2.\n"
        "5. `needed_section_ids` must contain only IDs from the section index.\n"
        "6. If unsupported, use score 0 and an empty `needed_section_ids` array.\n"
        "7. Return JSON only.\n\n"
        "Example:\n"
        "{\n"
        '  "general": {\n'
        '    "g.1": {\n'
        '      "signals": {\n'
        '        "g.1.a": 2\n'
        "      },\n"
        '      "needed_section_ids": ["S09", "S12"]\n'
        "    },\n"
        '    "g.2": {\n'
        '      "signals": {\n'
        '        "g.2.a": 1\n'
        "      },\n"
        '      "needed_section_ids": ["S10"]\n'
        "    }\n"
        "  },\n"
        '  "proposed_research": {\n'
        '    "pr.1": {\n'
        '      "signals": {\n'
        '        "pr.1.a": 1,\n'
        '        "pr.1.b": 2,\n'
        '        "pr.1.c": 2\n'
        "      },\n"
        '      "needed_section_ids": ["S08", "S10"]\n'
        "    }\n"
        "  }\n"
        "}\n\n"
        "Use exactly the same object structure as the example."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_stage1_schema(rubric_sections: list[dict[str, Any]]) -> dict[str, Any]:
    section_properties: dict[str, Any] = {}
    required_sections: list[str] = []
    for section in rubric_sections:
        required_sections.append(section["section_key"])
        sub_properties: dict[str, Any] = {}
        required_sub_ids: list[str] = []
        for sub in section["sub_criteria"]:
            required_sub_ids.append(sub["sub_id"])
            signal_properties = {
                signal["sid"]: {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 2,
                }
                for signal in sub["signals"]
            }
            sub_properties[sub["sub_id"]] = {
                "type": "object",
                "properties": {
                    "signals": {
                        "type": "object",
                        "properties": signal_properties,
                        "additionalProperties": False,
                    },
                    "needed_section_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": r"^S\d{2}$",
                        },
                        "maxItems": 5,
                    },
                },
                "required": ["signals", "needed_section_ids"],
                "additionalProperties": False,
            }
        section_properties[section["section_key"]] = {
            "type": "object",
            "properties": sub_properties,
            "required": required_sub_ids,
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": section_properties,
        "required": required_sections,
        "additionalProperties": False,
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


def _response_preview(text: str, limit: int = 500) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def _has_valid_stage1_shape(parsed: dict[str, Any], rubric_sections: list[dict[str, Any]]) -> bool:
    if isinstance(parsed.get("sub_criteria"), list):
        return True
    return any(isinstance(parsed.get(section["section_key"]), dict) for section in rubric_sections)


def _normalize_stage1_output(
    parsed: dict[str, Any],
    rubric_sections: list[dict[str, Any]],
    section_data: dict[str, Any],
) -> list[dict[str, Any]]:
    section_index = section_data["section_index"]
    raw_subs = parsed.get("sub_criteria", []) if isinstance(parsed, dict) else []
    legacy_by_sub_id: dict[str, dict[str, Any]] = {}
    for entry in raw_subs:
        if isinstance(entry, dict) and isinstance(entry.get("sub_id"), str):
            legacy_by_sub_id.setdefault(entry["sub_id"], entry)

    normalized_sections: list[dict[str, Any]] = []
    for section in rubric_sections:
        raw_section = parsed.get(section["section_key"], {}) if isinstance(parsed, dict) else {}
        if not isinstance(raw_section, dict):
            raw_section = {}

        section_subs: list[dict[str, Any]] = []
        for expected_sub in section["sub_criteria"]:
            raw_sub = raw_section.get(expected_sub["sub_id"], {})
            if not isinstance(raw_sub, dict):
                raw_sub = {}
            if not raw_sub and expected_sub["sub_id"] in legacy_by_sub_id:
                raw_sub = legacy_by_sub_id[expected_sub["sub_id"]]

            raw_signals = raw_sub.get("signals", {})
            raw_signal_map: dict[str, Any] = {}
            if isinstance(raw_signals, dict):
                raw_signal_map = raw_signals
            elif isinstance(raw_signals, list):
                for item in raw_signals:
                    if isinstance(item, dict) and isinstance(item.get("sid"), str):
                        raw_signal_map[item["sid"]] = item.get("score", 0)

            scalar_score = raw_sub.get("score")
            scalar_score = scalar_score if isinstance(scalar_score, int) else None

            raw_needed_ids = raw_sub.get("needed_section_ids", [])
            if not isinstance(raw_needed_ids, list):
                raw_needed_ids = []

            seen_section_ids: set[str] = set()
            needed_section_ids: list[str] = []
            for section_id in raw_needed_ids:
                if not isinstance(section_id, str) or section_id not in section_index:
                    continue
                if section_id in seen_section_ids:
                    continue
                seen_section_ids.add(section_id)
                needed_section_ids.append(section_id)

            signals = []
            has_positive = False
            for expected_signal in expected_sub["signals"]:
                raw_score = raw_signal_map.get(expected_signal["sid"], scalar_score if scalar_score is not None else 0)
                raw_score = raw_score if isinstance(raw_score, int) else 0
                raw_score = max(0, min(raw_score, 2))
                signals.append({
                    "sid": expected_signal["sid"],
                    "signal_text": expected_signal["text"],
                    "weight": expected_signal["weight"],
                    "score_0to2_raw": raw_score,
                })
                has_positive = has_positive or raw_score > 0

            empty_zero_evidence = len(needed_section_ids) == 0 and not has_positive
            evidence_status = "ok"
            if len(needed_section_ids) == 0 and has_positive:
                evidence_status = "invalid_evidence"
            elif 0 < len(needed_section_ids) <= 1:
                evidence_status = "sparse_evidence"

            if evidence_status == "invalid_evidence" and has_positive:
                for signal in signals:
                    signal["score_0to2_raw"] = 0

            section_subs.append({
                "sub_id": expected_sub["sub_id"],
                "name": expected_sub["name"],
                "definition": expected_sub["definition"],
                "weight": expected_sub["weight"],
                "needed_section_ids": needed_section_ids,
                "evidence_count": len(needed_section_ids),
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


def _section_audit_payload(section: dict[str, Any], section_data: dict[str, Any]) -> dict[str, Any]:
    section_text_lookup = section_data["section_text_lookup"]
    referenced_ids: list[str] = []
    seen_ids: set[str] = set()
    for sub in section["sub_criteria"]:
        for section_id in sub["needed_section_ids"]:
            if section_id in seen_ids:
                continue
            seen_ids.add(section_id)
            referenced_ids.append(section_id)

    section_blocks: list[str] = []
    for section_id in referenced_ids:
        meta = section_text_lookup.get(section_id)
        if not meta:
            continue
        label = f"{section_id}: {meta['section']}"
        section_blocks.append(f"[{label}]\n{meta['text']}\n[{label}]")

    return {
        "evidence_context": "\n\n".join(section_blocks),
        "section_key": section["section_key"],
        "sub_criteria": [
            {
                "sub_id": sub["sub_id"],
                "sub_criterion": sub["name"],
                "evidence_status": sub["evidence_status"],
                "needed_section_ids": sub["needed_section_ids"],
                "signals": [
                    {
                        "sid": signal["sid"],
                        "signal_text": signal["signal_text"],
                        "stage1_score": signal["score_0to2_raw"],
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


def _aggregate_sub_criterion(sub: dict[str, Any], section_data: dict[str, Any]) -> dict[str, Any]:
    section_text_lookup = section_data["section_text_lookup"]
    total_weight = sum(signal["weight"] for signal in sub["signals"]) or 1.0
    weighted_sum = sum(signal["score_0to2_weighted"] * signal["weight"] for signal in sub["signals"])
    score_10 = round((weighted_sum / (2 * total_weight)) * 10, 2)
    avg_plausibility = round(
        sum(signal["plausibility_0to5"] for signal in sub["signals"]) / max(1, len(sub["signals"])),
        2,
    )
    weak_signal_count = sum(1 for signal in sub["signals"] if signal["plausibility_0to5"] <= 1)
    evidence = []
    for section_id in sub["needed_section_ids"]:
        meta = section_text_lookup.get(section_id)
        if not meta:
            continue
        evidence.append({
            "id": section_id,
            "section_id": section_id,
            "text": "",
            "section": meta["section"],
            "source_path": " | ".join(meta.get("source_paths", [])),
        })
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


def _aggregate_section(section: dict[str, Any], section_data: dict[str, Any]) -> dict[str, Any]:
    sub_criteria = [
        _aggregate_sub_criterion(sub, section_data) for sub in section["sub_criteria"]
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


def _write_stage1_artifacts(
    *,
    artifacts_dir: str | Path | None,
    doc_id: str,
    section_index: dict[str, str],
    raw_response: str,
    parsed: dict[str, Any],
    normalized_sections: list[dict[str, Any]],
) -> dict[str, str]:
    if artifacts_dir is None:
        return {}

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    raw_path = artifacts_path / f"{doc_id}_stage1_raw.txt"
    parsed_path = artifacts_path / f"{doc_id}_stage1_parsed.json"
    normalized_path = artifacts_path / f"{doc_id}_stage1_normalized.json"
    section_index_path = artifacts_path / f"{doc_id}_section_index.json"

    raw_path.write_text(raw_response, encoding="utf-8")
    parsed_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    normalized_path.write_text(
        json.dumps(normalized_sections, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    section_index_path.write_text(
        json.dumps(section_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "stage1_raw_response": str(raw_path),
        "stage1_parsed_response": str(parsed_path),
        "stage1_normalized_response": str(normalized_path),
        "section_index": str(section_index_path),
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
    section_data = build_section_index(application)

    messages = build_stage1_messages(
        application=application,
        rubric_sections=rubric_sections,
        section_index=section_data["section_index"],
    )
    schema = build_stage1_schema(rubric_sections)
    raw_response = stage1_client.generate_json(messages, schema=schema, max_tokens=8192)
    try:
        parsed = _safe_json_loads(raw_response)
    except Exception as exc:
        raise ValueError(
            "Stage 1 scorer did not return valid JSON. "
            f"Response preview: {_response_preview(raw_response)!r}"
        ) from exc

    if not isinstance(parsed, dict) or not _has_valid_stage1_shape(parsed, rubric_sections):
        raise ValueError(
            "Stage 1 scorer returned JSON without a valid section-object or `sub_criteria` structure. "
            f"Response preview: {_response_preview(raw_response)!r}"
        )

    sections = _normalize_stage1_output(parsed, rubric_sections, section_data)
    artifact_paths = _write_stage1_artifacts(
        artifacts_dir=artifacts_dir,
        doc_id=doc_id or "unknown",
        section_index=section_data["section_index"],
        raw_response=raw_response,
        parsed=parsed,
        normalized_sections=sections,
    )

    judge = judge or FallbackJudge()
    audit_payloads = {
        section["section_key"]: _section_audit_payload(section, section_data)
        for section in sections
    }
    audited = run_faithfulness_audit(audit_payloads, judge)
    for section in sections:
        _apply_plausibility(section, audited.get(section["section_key"], {}))

    features = {
        section["section_key"]: _aggregate_section(section, section_data)
        for section in sections
    }
    section_weights = {section["section_key"]: section["weight"] for section in sections}

    return {
        "doc_id": doc_id or "unknown",
        "run_info": {
            "ran_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
            "model": getattr(stage1_client, "model_name", "unknown"),
            "judge_model": getattr(judge, "model_name", getattr(judge, "__class__", type(judge)).__name__),
        },
        "pool_size": 0,
        "pool_lookup": {},
        "section_chunk_ids": {},
        "section_index": section_data["section_index"],
        "features": features,
        "overall": _aggregate_overall(features, section_weights),
        "debug": {
            "stage1_contract_version": "section_ids_v1",
            "artifacts": artifact_paths,
        },
    }
