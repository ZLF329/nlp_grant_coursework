"""
General (all-other) document parser: converts sections JSON to unified format.

Input:  sections JSON produced by document_parser (HybridDocumentParser etc.)
Output: unified JSON matching the fellowship parser's format keys.

Keys in output (only present if content found):
  - SUMMARY INFORMATION
  - LEAD APPLICANT & RESEARCH TEAM
  - APPLICATION DETAILS
  - SUMMARY BUDGET
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────────────────────── Target key constants ────────────────────────────
KEY_SUMMARY_INFO = "SUMMARY INFORMATION"
KEY_LEAD_TEAM    = "LEAD APPLICANT & RESEARCH TEAM"
KEY_APP_DETAILS  = "APPLICATION DETAILS"
KEY_BUDGET       = "SUMMARY BUDGET"

# APPLICATION DETAILS sub-keys (must match IC00458_after.json exactly)
KEY_ABSTRACT = "Scientific Abstract"
KEY_PES      = "Plain English Summary of Research"
KEY_CHANGES  = "Changes from Previous Stage"
KEY_WPCC     = "Patient & Public Involvement"
KEY_DETAILED_RESEARCH_PLAN = "Detailed Research Plan"
KEY_APPLICANT_CV = "Applicant CV"
KEY_APPLICANT_RESEARCH_BACKGROUND = "Applicant Research Background"

# ──────────────────────────── Keyword tables ──────────────────────────────────
_CHANGES_KW = [
    "changes from", "previous stage", "response to reviewer",
    "resubmission", "reviewer comments", "committee comments",
    "previous submission",
]
_WPCC_KW = [
    "working with people", "people and communit", "public involvement",
    "patient and public", "patient involvement", "lay involvement",
    "ppi summary", "community involvement",
]
_PES_KW = [
    "plain english", "plain language", "lay summary",
    "lay abstract", "lay description", "accessible summary",
]
_PLAN_KW = [
    "detailed research plan", "research plan", "project plan",
    "methodology", "methods", "work plan", "programme of work",
    # UKRI sections
    "approach", "reproducibility", "statistical design",
    "data management", "ethics and responsible",
]
_ABSTRACT_KW = [
    "scientific abstract", "abstract", "research summary",
    "project summary", "summary", "research area",
    "research question", "aims", "objectives", "background",
    "overview", "introduction", "project description",
    # UKRI sections
    "vision", "excellence", "originality", "innovation", "impact",
]
_APPLICANT_CV_KW = ["applicant cv"]
_APPLICANT_RESEARCH_BACKGROUND_KW = [
    "applicant research background",
    # UKRI sections
    "career development", "capability to deliver",
    "degree registration", "clinical activities",
]

_BUDGET_KW = [
    "budget", "cost", "financial", "finance",
    "funding breakdown", "expenditure", "resources requested",
    "direct costs", "indirect costs", "staff costs",
    # UKRI sections
    "resources and cost",
]

_LEAD_KW = [
    "lead applicant", "chief investigator",
    "principal investigator", "pi details", "lead researcher",
    # UKRI sections
    "core team",
]
_COAPPLICANT_KW = [
    "co-applicant", "co applicant", "coapplicant",
    "co-investigator", "team member",
]
_SUPERVISOR_KW = [
    "supervisor", "academic supervisor", "primary supervisor",
]
_MENTOR_KW = ["mentor"]
_HOST_KW   = ["host organisation", "host organization"]

_SUMMARY_INFO_KW = [
    "application summary", "summary information", "application information",
    "project information", "contracting", "grant reference",
    "project title", "application title", "application details overview",
    # UKRI "1. Details" section contains applicant/project metadata
    "details",
]

# ──────────────────────────── Block building ──────────────────────────────────

def _sections_to_blocks(sections: List[dict]) -> List[dict]:
    """
    Collapse alternating title/text/table sections into heading+content blocks.

    In the input JSON:
      type="title"  → content holds the heading string
      type="text"   → content holds body text
      type="table"  → content holds table data (string or list)
    """
    blocks: List[dict] = []
    current_heading: Optional[str] = None
    current_texts:   List[str]     = []

    def _flush_text():
        if current_heading is not None:
            blocks.append({
                "heading": current_heading,
                "content": "\n".join(t for t in current_texts if t),
                "type": "text",
            })

    for sec in sections:
        stype   = sec.get("type", "text")
        content = str(sec.get("content", "")).strip()

        if stype == "title":
            _flush_text()
            current_heading = content
            current_texts   = []
        elif stype == "text":
            current_texts.append(content)
        elif stype == "table":
            _flush_text()
            current_heading = None
            current_texts   = []
            # Tables are kept separately; they are not used in text extraction
            blocks.append({"heading": "", "content": content, "type": "table"})

    _flush_text()
    return blocks


# ──────────────────────────── Keyword helpers ─────────────────────────────────

def _matches(heading: str, keywords: List[str]) -> bool:
    h = heading.lower()
    return any(kw in h for kw in keywords)


def _app_details_subkey(heading: str) -> Optional[str]:
    """Return APPLICATION DETAILS sub-key for *heading*, or None.

    More-specific patterns are checked first to avoid false matches.
    """
    if _matches(heading, _CHANGES_KW):
        return KEY_CHANGES
    if _matches(heading, _WPCC_KW):
        return KEY_WPCC
    if _matches(heading, _APPLICANT_CV_KW):
        return KEY_APPLICANT_CV
    if _matches(heading, _APPLICANT_RESEARCH_BACKGROUND_KW):
        return KEY_APPLICANT_RESEARCH_BACKGROUND
    if _matches(heading, _PES_KW):
        return KEY_PES
    if _matches(heading, _PLAN_KW):
        return KEY_DETAILED_RESEARCH_PLAN
    if _matches(heading, _ABSTRACT_KW):
        return KEY_ABSTRACT
    return None


# ──────────────────────────── Section extractors ─────────────────────────────

def _extract_application_details(blocks: List[dict]) -> Optional[dict]:
    result: Dict[str, str] = {}

    for block in blocks:
        if block["type"] != "text" or not block["content"]:
            continue
        sub = _app_details_subkey(block["heading"])
        if sub is None:
            continue
        if sub in result:
            result[sub] = result[sub] + "\n" + block["content"]
        else:
            result[sub] = block["content"]

    return result if result else None


def _extract_budget(blocks: List[dict]) -> Optional[str]:
    parts = [
        block["content"]
        for block in blocks
        if block["type"] == "text"
        and _matches(block["heading"], _BUDGET_KW)
        and block["content"]
    ]
    return "\n\n".join(parts) if parts else None


def _extract_lead_team(blocks: List[dict]) -> Optional[dict]:
    result: Dict[str, Any] = {}

    for block in blocks:
        if block["type"] != "text" or not block["content"]:
            continue
        h = block["heading"]
        c = block["content"]

        if _matches(h, _LEAD_KW):
            result["Lead Applicant"] = c
        elif _matches(h, _COAPPLICANT_KW):
            result.setdefault("Co-Applicants", [])
            result["Co-Applicants"].append(c)
        elif _matches(h, _SUPERVISOR_KW):
            result.setdefault("Supervisors", [])
            result["Supervisors"].append(c)
        elif _matches(h, _MENTOR_KW):
            result["Mentor"] = c
        elif _matches(h, _HOST_KW):
            result["Host Organisation"] = c

    return result if result else None


def _is_classified(heading: str) -> bool:
    """Return True if *heading* is already handled by a known keyword extractor."""
    return (
        _app_details_subkey(heading) is not None
        or _matches(heading, _BUDGET_KW)
        or _matches(heading, _LEAD_KW)
        or _matches(heading, _COAPPLICANT_KW)
        or _matches(heading, _SUPERVISOR_KW)
        or _matches(heading, _MENTOR_KW)
        or _matches(heading, _HOST_KW)
        or _matches(heading, _SUMMARY_INFO_KW)
    )


def _extract_summary_info(blocks: List[dict]) -> Optional[dict]:
    """Best-effort: locate an explicit summary-info block and parse key:value pairs."""
    for block in blocks:
        if block["type"] != "text":
            continue
        if not _matches(block["heading"], _SUMMARY_INFO_KW):
            continue
        if not block["content"]:
            continue

        info: Dict[str, str] = {}
        for line in block["content"].splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                k = k.strip()
                v = v.strip()
                if k and v:
                    info[k] = v

        return info if info else {"details": block["content"]}

    return None


# ──────────────────────────── Public API ──────────────────────────────────────

def convert_to_unified_format(sections_json: dict) -> dict:
    """
    Convert a sections JSON (from HybridDocumentParser etc.) to the unified
    format used by the fellowship parser.

    Only keys for which content was found are included in the output.
    """
    sections = sections_json.get("sections", [])
    blocks   = _sections_to_blocks(sections)

    out: Dict[str, Any] = {}

    app_details = _extract_application_details(blocks)
    if app_details:
        out[KEY_APP_DETAILS] = app_details

    budget = _extract_budget(blocks)
    if budget:
        out[KEY_BUDGET] = budget

    lead_team = _extract_lead_team(blocks)
    if lead_team:
        out[KEY_LEAD_TEAM] = lead_team

    summary_info = _extract_summary_info(blocks)
    if summary_info:
        out[KEY_SUMMARY_INFO] = summary_info

    # ── Catch-all: rescue text from blocks not matched by any known extractor ──
    # Covers OCR output, unusual formats, or any section heading not in keyword
    # tables.  Content is stored under APPLICATION DETAILS["Raw Content"] so the
    # downstream NLP and LLM scorer still have something to work with.
    raw_parts = [
        block["content"]
        for block in blocks
        if block["type"] == "text"
        and block["content"]
        and not _is_classified(block["heading"])
    ]
    if raw_parts:
        details = out.setdefault(KEY_APP_DETAILS, {})
        if isinstance(details, dict):
            details.setdefault("Raw Content", "\n\n".join(raw_parts))

    return out


def convert_file(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Read a sections JSON file, convert to unified format, and write to disk.

    Returns the output file path.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_name(
        input_path.stem + "_unified.json"
    )

    with open(input_path, "r", encoding="utf-8") as f:
        sections_json = json.load(f)

    unified = convert_to_unified_format(sections_json)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, ensure_ascii=False, indent=2)

    return str(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python all_other_parser.py <input.json> [output.json]")
        sys.exit(1)

    out = convert_file(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"Written: {out}")
