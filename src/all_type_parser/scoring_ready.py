from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional


GUIDANCE_LINE_PATTERNS = [
    re.compile(r"^Reference:\s*", re.I),
    re.compile(r"^Page\s+\d+\s+of\s+\d+", re.I),
    re.compile(r"^\(?Word\s+Count", re.I),
    re.compile(r"^\(?Word\s+Limit", re.I),
    re.compile(r"^\[currently", re.I),
    re.compile(r"^Please\s+(describe|provide|outline|enter|use)\b", re.I),
    re.compile(r"^Use\s+this\s+section\b", re.I),
    re.compile(r"^Applicants?\s+should\b", re.I),
    re.compile(r"^The\s+following\s+must\s+be\b", re.I),
    re.compile(r"^The\s+individual\s+listed\b", re.I),
    re.compile(r"^This\s+section\s+must\s+be\b", re.I),
    re.compile(r"^The\s+applicant\s+is\s+required\b", re.I),
    re.compile(r"^Once\s+the\s+applicant\b", re.I),
    re.compile(r"^All\s+Participants?\b", re.I),
    re.compile(r"^Signatories?\s+must\b", re.I),
    re.compile(r"^Given\s+that\s+support\s+is\s+already\b", re.I),
    re.compile(r"^It\s+is\s+not\s+necessarily\s+expected\b", re.I),
    re.compile(r"^Research\s+support\s+is\s+referred\s+to\s+in\s+the\s+literature\b", re.I),
    re.compile(r"^This\s+section\s+includes\b", re.I),
    re.compile(r"^I\)\s+Salary\s+Costs\b", re.I),
    re.compile(r"^II\)\s+Travel,\s*Subsistence\b", re.I),
    re.compile(r"^III\)\s+Training\s+and\s+Development\b", re.I),
    re.compile(r"^IV\)\s+Other\s+Costs\b", re.I),
]

SUBSECTION_MAP = {
    "why is this research needed?": "background_and_rationale",
    "background": "background_and_rationale",
    "rationale": "background_and_rationale",
    "the area of research interest": "background_and_rationale",
    "outline of the proposed research": "methods",
    "aim": "aims",
    "aims": "aims",
    "objectives": "aims",
    "method": "methods",
    "methods": "methods",
    "what will happen?": "methods",
    "impact": "impact_and_dissemination",
    "impact/dissemination": "impact_and_dissemination",
    "sharing this research": "impact_and_dissemination",
    "dissemination": "impact_and_dissemination",
    "gap analysis": "gap_analysis",
    "training": "training_plan",
    "proposed training programme": "training_plan",
    "training and development plan": "training_plan",
    "research support": "research_support",
    "practice development": "practice_development",
    "professional practice development": "practice_development",
    "leadership": "leadership_development",
    "coaching": "coaching",
    "contextual factors": "contextual_factors",
    "patient and public involvement (ppi)": "prior_involvement",
    "please describe how patients/service users, carers and the public have been involved in developing this proposal": "prior_involvement",
    "please describe the ways in which patients/service users, carers and the public will be actively involved in the proposed research, including any training and support provided": "planned_involvement",
    "please describe the ways in which patients/service users and the public will be actively involved in the proposed research, including any training and support provided": "planned_involvement",
    "if it is considered not appropriate and meaningful to actively involve patients/service users, carers and the public in your proposed research, please justify why": "ppi_justification",
    "if it is considered not appropriate and meaningful to actively involve patients/services users, carers and the public in your proposed research, please justify why": "ppi_justification",
}

UNDERSERVED_GROUP_PATTERNS = [
    r"Black",
    r"African Caribbean",
    r"South Asian",
    r"Eastern European",
    r"underserved",
    r"seldom heard",
    r"socioeconomic deprivation",
    r"digital exclusion",
    r"disability",
    r"ethnic",
    r"equality",
    r"diversity",
    r"inclusion",
]


def _normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _clean_heading(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^\d+\.\s+", "", text)
    text = re.sub(r"\s*[-–]\s*\d+\s*word\s+limit\b.*$", "", text, flags=re.I)
    text = re.sub(r"\(\s*word\s+limit[^)]*\)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" :-").lower()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _detect_word_limit(text: str) -> Optional[int]:
    if not text:
        return None
    patterns = [
        re.compile(r"word\s+limit[:\s]+(\d+)", re.I),
        re.compile(r"(\d+)\s*word\s+limit", re.I),
        re.compile(r"\(word\s+limit[:\s]+(\d+)\)", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def _sentence_tokens(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    cleaned = []
    for sentence in sentences:
        s = re.sub(r"[^a-z0-9 ]+", "", sentence.lower()).strip()
        if s:
            cleaned.append(s)
    return cleaned


def _duplication_score(text: str) -> float:
    sentences = _sentence_tokens(text)
    if not sentences:
        return 0.0
    unique = len(set(sentences))
    return round(1 - (unique / max(len(sentences), 1)), 3)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    a = set(re.findall(r"\b[a-z]{3,}\b", (text_a or "").lower()))
    b = set(re.findall(r"\b[a-z]{3,}\b", (text_b or "").lower()))
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 3)


def _looks_like_answer_start(line: str) -> bool:
    if not line:
        return False
    if re.match(r"^(I|My|We|Our|This|These)\b", line):
        return True
    if re.match(r"^(Dr|Professor|Prof|Mr|Mrs|Ms)\b", line):
        return True
    if re.match(r"^(In\s+\d{4}|During|Following|To\s+|For\s+my)\b", line):
        return True
    if ":" in line and len(line.split()) <= 12:
        return True
    return False


def _clean_answer_text(text: str) -> str:
    raw_lines = [line.strip() for line in (text or "").splitlines()]
    lines: List[str] = []
    filtered_lines: List[str] = []
    started = False

    for line in raw_lines:
        if not line:
            continue
        if any(pattern.search(line) for pattern in GUIDANCE_LINE_PATTERNS):
            continue

        filtered_lines.append(line)

        lowered = line.lower()
        heading_key = SUBSECTION_MAP.get(_clean_heading(line))
        if not started:
            if _looks_like_answer_start(line):
                started = True
            elif heading_key:
                started = True
            elif len(line.split()) > 10 and not lowered.startswith((
                "the ",
                "this ",
                "applicants ",
                "how ",
                "why ",
                "an outline ",
                "use this section",
                "please ",
            )):
                started = True

        if started:
            lines.append(line)

    cleaned = _normalize_whitespace("\n".join(lines))
    if cleaned:
        return cleaned

    fallback = _normalize_whitespace("\n".join(filtered_lines))
    if any(_looks_like_answer_start(line) for line in filtered_lines):
        return fallback
    return ""


def _split_named_subsections(text: str) -> Dict[str, str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    result: Dict[str, List[str]] = {}
    current_key: Optional[str] = None

    for line in lines:
        heading_key = SUBSECTION_MAP.get(_clean_heading(line))
        if heading_key:
            current_key = heading_key
            result.setdefault(current_key, [])
            continue

        if current_key is None:
            continue
        result[current_key].append(line)

    return {
        key: _normalize_whitespace("\n".join(value))
        for key, value in result.items()
        if any(v.strip() for v in value)
    }


def _extract_meeting_frequency(text: str) -> List[str]:
    return re.findall(
        r"(every\s+\d+(?:-\d+)?\s+(?:weeks?|months?)|monthly|bi-weekly|quarterly|6-months?)",
        text or "",
        flags=re.I,
    )


def _extract_budget_line_items(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    current: Dict[str, str] = {}

    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("Description:"):
            if current:
                items.append(current)
            current = {"description": line.partition(":")[2].strip()}
        elif line.startswith("Item Description:"):
            current["item_description"] = line.partition(":")[2].strip()
        elif line.startswith("Type of cost:"):
            current["category"] = line.partition(":")[2].strip()
        elif line.startswith("Justification of Cost:"):
            current["justification"] = line.partition(":")[2].strip()
        elif re.search(r"£[\d,]+(?:\.\d+)?", line):
            current.setdefault("amount_hint", line)

    if current:
        items.append(current)

    if items:
        return items

    fallback_items: List[Dict[str, str]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if "£" in line or re.search(r"\bTotal Cost\b", line, re.I):
            fallback_items.append({"amount_hint": line})
    return fallback_items[:20]


def _extract_training_items(text: str) -> List[str]:
    items: List[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r"\b(course|module|training|conference|workshop|MSc|visit|coaching)\b", line, re.I):
            items.append(line)
    return items[:25]


def _extract_underserved_groups(*texts: str) -> List[str]:
    combined = "\n".join(text for text in texts if text)
    hits = []
    for pattern in UNDERSERVED_GROUP_PATTERNS:
        if re.search(rf"\b{pattern}\b", combined, re.I):
            hits.append(pattern)
    return hits


def _build_evidence(
    text: str,
    *,
    raw_text: Optional[str] = None,
    source_section: Optional[str] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    confidence: float = 0.75,
) -> Optional[Dict[str, Any]]:
    raw = _normalize_whitespace(raw_text if raw_text is not None else text)
    cleaned = _clean_answer_text(raw)
    if not cleaned:
        return None

    word_limit = _detect_word_limit(raw)
    payload: Dict[str, Any] = {
        "text": cleaned,
        "source_section": source_section,
        "page_start": page_start,
        "page_end": page_end,
        "confidence": round(confidence, 2),
        "word_count": _word_count(cleaned),
        "over_word_limit": bool(word_limit and _word_count(cleaned) > word_limit),
        "duplication_score": _duplication_score(cleaned),
        "likely_contains_guidance": cleaned == raw and any(
            pattern.search(raw) for pattern in GUIDANCE_LINE_PATTERNS
        ),
    }
    return payload


def _record_payload(record: Optional[Dict[str, Any]], fallback_text: str = "", confidence: float = 0.8) -> Optional[Dict[str, Any]]:
    if record:
        return _build_evidence(
            record.get("text", ""),
            raw_text=record.get("raw_text"),
            source_section=record.get("heading"),
            page_start=record.get("page_start"),
            page_end=record.get("page_end"),
            confidence=confidence,
        )
    if fallback_text:
        return _build_evidence(fallback_text, confidence=confidence - 0.1)
    return None


def _collect_fellowship_records(pdf_path: str) -> List[Dict[str, Any]]:
    from . import fellowships_parser as fp

    lines = fp.filter_fellowship_lines(fp.extract_lines_pdfplumber(pdf_path))
    section_titles = [
        fp.SECTION_APP_SUMMARY,
        fp.SECTION_LEAD_APPLICANT,
        fp.SECTION_SUPPORT_MENTORSHIP,
        fp.SECTION_PES,
        fp.SECTION_ABSTRACT,
        fp.SECTION_PPI,
        fp.SECTION_TRAINING,
        fp.SECTION_BUDGET_SUMMARY,
        fp.SECTION_BUDGET,
        fp.SECTION_PARTICIPANTS,
        fp.SECTION_APPLICANT_CV,
    ]
    records: List[Dict[str, Any]] = []

    for title in section_titles:
        section_lines = fp.slice_section(lines, title)
        if not section_lines and title != fp.SECTION_APP_SUMMARY:
            raw = fp._get_section_raw_text(pdf_path, title)
            if not raw:
                continue
            records.append({
                "heading": title,
                "clean_heading": _clean_heading(title),
                "text": _normalize_whitespace(raw),
                "raw_text": raw,
                "page_start": None,
                "page_end": None,
            })
            continue

        raw = fp._get_section_raw_text(pdf_path, title) if title == fp.SECTION_APP_SUMMARY else fp.parse_text_section(section_lines)
        page_start = min((line.page for line in section_lines), default=None)
        page_end = max((line.page for line in section_lines), default=None)
        records.append({
            "heading": title,
            "clean_heading": _clean_heading(title),
            "text": _normalize_whitespace(raw),
            "raw_text": raw,
            "page_start": page_start + 1 if page_start is not None else 1,
            "page_end": page_end + 1 if page_end is not None else 1,
        })

    return [record for record in records if record.get("text")]


def _collect_rfpb_records(pdf_path: str) -> List[Dict[str, Any]]:
    from . import RfPB_parser as rp

    lines = rp.filter_rfpb_lines(rp.extract_lines_pdfplumber(pdf_path))
    section_titles = [
        rp.SECTION_TEAM,
        rp.SECTION_ABSTRACT,
        rp.SECTION_PES,
        rp.SECTION_CHANGES,
        rp.SECTION_PLAN,
        rp.SECTION_PPI,
        rp.SECTION_BUDGET,
        rp.SECTION_CV_LEAD,
        rp.SECTION_CV_CO,
    ]
    records: List[Dict[str, Any]] = [{
        "heading": "Application Summary Information",
        "clean_heading": "application summary information",
        "text": _normalize_whitespace(rp._get_page1_raw_text(pdf_path)),
        "raw_text": rp._get_page1_raw_text(pdf_path),
        "page_start": 1,
        "page_end": 1,
    }]

    for title in section_titles:
        section_lines = rp.slice_section(lines, title)
        raw = rp._get_section_raw_text(pdf_path, title) if title in {rp.SECTION_TEAM, rp.SECTION_CV_LEAD, rp.SECTION_CV_CO} else rp.parse_text_section(section_lines)
        if not raw:
            continue
        page_start = min((line.page for line in section_lines), default=None)
        page_end = max((line.page for line in section_lines), default=None)
        records.append({
            "heading": title,
            "clean_heading": _clean_heading(title),
            "text": _normalize_whitespace(raw),
            "raw_text": raw,
            "page_start": page_start + 1 if page_start is not None else None,
            "page_end": page_end + 1 if page_end is not None else None,
        })

    return [record for record in records if record.get("text")]


def _collect_document_records(input_path: str) -> List[Dict[str, Any]]:
    from document_parser import HybridDocumentParser

    parser = HybridDocumentParser()
    parsed = parser.parse(input_path)
    records: List[Dict[str, Any]] = []
    current_heading: Optional[str] = None
    current_meta: Dict[str, Any] = {}

    for section in parsed.sections:
        if section.type == "title":
            current_heading = str(section.content).strip()
            current_meta = section.metadata or {}
        elif section.type == "text" and current_heading:
            page = current_meta.get("page")
            records.append({
                "heading": current_heading,
                "clean_heading": _clean_heading(current_heading),
                "text": _normalize_whitespace(str(section.content)),
                "raw_text": str(section.content),
                "page_start": page,
                "page_end": page,
                "metadata": section.metadata or {},
            })

    return records


def _is_rfpb_pdf(pdf_path: str) -> bool:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return False
            text = pdf.pages[0].extract_text() or ""
        return "RfPB" in "\n".join(text.splitlines()[:2])
    except Exception:
        return False


def _collect_records(input_path: str) -> tuple[str, List[Dict[str, Any]]]:
    ext = Path(input_path).suffix.lower()
    if ext == ".pdf":
        if _is_rfpb_pdf(input_path):
            return "rfpb_pdf", _collect_rfpb_records(input_path)
        records = _collect_fellowship_records(input_path)
        if records:
            return "fellowship_pdf", records
    return "document_fallback", _collect_document_records(input_path)


def _pick_record(records: List[Dict[str, Any]], *keywords: str) -> Optional[Dict[str, Any]]:
    for record in records:
        heading = record.get("clean_heading", "")
        if any(keyword in heading for keyword in keywords):
            return record
    return None


def _text_from_payload(payload: Optional[Dict[str, Any]]) -> str:
    return payload.get("text", "") if payload else ""


def _is_prompt_like_payload(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload:
        return False
    text = payload.get("text", "").strip().lower()
    if not text:
        return True
    if payload.get("word_count", 0) < 50 and re.match(r"^(please|use this section|how patient and public involvement)", text):
        return True
    if "{1100 words" in text or "{800 words" in text:
        return True
    return False


def _parse_award_type(input_path: str) -> str:
    """Extract award type from filename, e.g. 'IC00001_DF_Doctoral' -> 'DF_Doctoral'."""
    stem = Path(input_path).stem
    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 else stem


def _dedup_payload(
    payload: Optional[Dict[str, Any]],
    *references: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return None if payload text is identical to any reference payload's text."""
    if not payload:
        return None
    text = payload.get("text", "")
    for ref in references:
        if ref and ref.get("text", "") == text:
            return None
    return payload


def _strip_internal_fields(obj: Any) -> Any:
    """Recursively remove internal/debug fields before final output."""
    if isinstance(obj, dict):
        return {
            k: _strip_internal_fields(v)
            for k, v in obj.items()
            if k != "likely_contains_guidance"
        }
    if isinstance(obj, list):
        return [_strip_internal_fields(item) for item in obj]
    return obj


def _bundle_payload(
    records: List[Dict[str, Any]],
    *,
    source_label: str,
    confidence: float = 0.68,
) -> Optional[Dict[str, Any]]:
    bundle_text = _normalize_whitespace("\n".join(record.get("text", "") for record in records if record.get("text")))
    if not bundle_text:
        return None
    pages = [record.get("page_start") for record in records if record.get("page_start") is not None]
    page_end = [record.get("page_end") for record in records if record.get("page_end") is not None]
    return _build_evidence(
        bundle_text,
        source_section=source_label,
        page_start=min(pages) if pages else None,
        page_end=max(page_end) if page_end else None,
        confidence=confidence,
    )


def _form_quality_payload(evidence_map: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    over_limit = []
    word_counts: Dict[str, int] = {}
    texts: List[tuple[str, str]] = []

    for key, payload in evidence_map.items():
        if not payload:
            continue
        word_counts[key] = payload.get("word_count", 0)
        if payload.get("over_word_limit"):
            over_limit.append(key)
        texts.append((key, payload.get("text", "")))

    pairwise = []
    for idx, (key_a, text_a) in enumerate(texts):
        for key_b, text_b in texts[idx + 1:]:
            score = _jaccard_similarity(text_a, text_b)
            if score >= 0.2:
                pairwise.append({"section_a": key_a, "section_b": key_b, "similarity": score})

    return {
        "word_count_by_section": word_counts,
        "over_limit_sections": over_limit,
        "cross_section_similarity": pairwise[:15],
        "has_headings": True,
        "detected_heading_count": len(texts),
    }


def build_scoring_ready(input_path: str, unified_data: Dict[str, Any]) -> Dict[str, Any]:
    strategy, records = _collect_records(input_path)
    summary_info = unified_data.get("SUMMARY INFORMATION", {})
    team = unified_data.get("LEAD APPLICANT & RESEARCH TEAM", {})
    app_details = unified_data.get("APPLICATION DETAILS", {})
    budget_text = unified_data.get("SUMMARY BUDGET", "")
    support_mentorship_text = unified_data.get("SUPPORT AND MENTORSHIP", "")

    rec_summary = _pick_record(records, "application summary information", "summary")
    rec_pes = _pick_record(records, "plain english summary")
    rec_abstract = _pick_record(records, "scientific abstract")
    rec_plan = _pick_record(records, "detailed research plan", "research plan", "planned application")
    rec_training = _pick_record(records, "training development and research support", "training and development and research support", "training and development programme", "proposed training and development programme")
    rec_support = _pick_record(records, "support and mentorship", "research support", "primary academic supervisor", "supervisor")
    rec_host = _pick_record(records, "host organisations support statement", "host organisation")
    rec_ppi = _pick_record(records, "patient public involvement", "patient and public involvement", "working with people and communities")
    rec_budget = _pick_record(records, "budget summary", "detailed budget", "budget")
    rec_cv = _pick_record(records, "applicant cv", "lead applicant", "cv lead applicant")

    research_bundle = _bundle_payload(
        [
            record for record in records
            if any(keyword in record.get("clean_heading", "") for keyword in (
                "planned application",
                "research plan",
                "area of research interest",
                "outline of proposed research",
                "how patient and public involvement will be incorporated",
                "plain english summary",
                "scientific abstract",
            ))
        ],
        source_label="research_bundle",
        confidence=0.66,
    )
    training_bundle = _bundle_payload(
        [
            record for record in records
            if any(keyword in record.get("clean_heading", "") for keyword in (
                "training",
                "developing research literacy",
                "developing research collaborations",
                "communicating research",
                "research leadership",
                "professional practice development",
            ))
        ],
        source_label="training_bundle",
        confidence=0.66,
    )
    support_bundle = _bundle_payload(
        [
            record for record in records
            if any(keyword in record.get("clean_heading", "") for keyword in (
                "research support",
                "support and mentorship",
                "collaborations",
                "supervisor",
                "host organisation support statement",
            ))
        ],
        source_label="support_bundle",
        confidence=0.67,
    )

    pes_payload = _record_payload(rec_pes, app_details.get("Plain English Summary", ""))
    abstract_payload = _record_payload(rec_abstract, app_details.get("Scientific Abstract", ""))
    training_payload = _record_payload(rec_training, app_details.get("Training & Development and Research Support", ""))
    ppi_payload = _record_payload(rec_ppi, app_details.get("Working with People and Communities Summary", ""))
    budget_payload = _record_payload(rec_budget, budget_text)
    research_plan_payload = _record_payload(rec_plan) or research_bundle
    cv_payload = _record_payload(rec_cv)
    support_payload = _record_payload(rec_support) or support_bundle or _build_evidence(support_mentorship_text, source_section="SUPPORT AND MENTORSHIP", confidence=0.70)
    host_payload = _record_payload(rec_host, team.get("Host Organisation", "") if isinstance(team, dict) else "")

    if pes_payload and pes_payload.get("word_count", 0) < 40 and pes_payload.get("likely_contains_guidance"):
        pes_payload = None
    if _is_prompt_like_payload(abstract_payload):
        abstract_payload = None
    elif abstract_payload and abstract_payload.get("word_count", 0) < 40 and abstract_payload.get("likely_contains_guidance"):
        abstract_payload = None
    if training_payload and training_payload.get("word_count", 0) < 40 and training_payload.get("likely_contains_guidance"):
        training_payload = training_bundle

    research_splits: Dict[str, str] = {}
    for payload in (pes_payload, abstract_payload, research_plan_payload):
        if payload:
            research_splits.update(_split_named_subsections(payload["text"]))

    training_splits = _split_named_subsections(_text_from_payload(training_payload))
    ppi_splits = _split_named_subsections(_text_from_payload(ppi_payload))

    budget_line_items = _extract_budget_line_items(_text_from_payload(budget_payload))
    training_items = _extract_training_items(_text_from_payload(training_payload))
    underserved_groups = _extract_underserved_groups(
        _text_from_payload(ppi_payload),
        _text_from_payload(pes_payload),
        _text_from_payload(abstract_payload),
    )

    applicant_person = team.get("Lead Applicant", {}) if isinstance(team, dict) else {}
    co_applicants = team.get("Co-Applicants", []) if isinstance(team, dict) else []
    normalized_team: List[Dict[str, Any]] = []
    for person in co_applicants:
        if isinstance(person, dict):
            normalized_team.append(person)
        else:
            normalized_team.append({
                "Full Name": "",
                "Proposed Role": "Support",
                "Organisation": "",
                "Department": "",
                "ORCID": "",
                "Supporting Evidence": person,
            })

    support_text = "\n".join(
        person.get("Position", "") or person.get("Supporting Evidence", "")
        for person in normalized_team
    )
    support_text = _normalize_whitespace("\n".join(filter(None, [support_text, _text_from_payload(support_payload), _text_from_payload(host_payload)])))

    evidence_map = {
        "plain_english_summary": pes_payload,
        "scientific_abstract": abstract_payload,
        "research_plan": research_plan_payload,
        "training_plan": training_payload,
        "ppi": ppi_payload,
        "budget": budget_payload,
        "applicant_cv": cv_payload,
        "support_statement": host_payload,
    }

    scoring_ready: Dict[str, Any] = {
        "schema_version": "1.0",
        "document_path": str(Path(input_path).resolve()),
        "parser_strategy": strategy,
        "award_meta": {
            "application_title": summary_info.get("Application Title", ""),
            "award_type": _parse_award_type(input_path),
            "contracting_organisation": summary_info.get("Contracting Organisation", ""),
            "start_date": summary_info.get("Start Date", ""),
            "duration_months": summary_info.get("Duration (months)", ""),
            "source_section": rec_summary.get("heading") if rec_summary else "SUMMARY INFORMATION",
        },
        "applicant": {
            "lead_applicant": applicant_person if isinstance(applicant_person, dict) else {},
            "supporting_team": normalized_team,
            "evidence": cv_payload,
            "signals": {
                "publication_mentions": len(re.findall(r"\b(publication|peer-reviewed|journal)\b", _text_from_payload(cv_payload), re.I)),
                "grant_mentions": len(re.findall(r"\b(grant|funding|award)\b", _text_from_payload(cv_payload), re.I)),
                "leadership_mentions": len(re.findall(r"\b(lead|leadership|chair|director|mentor)\b", _text_from_payload(cv_payload), re.I)),
            },
        },
        "research": {
            "plain_english_summary": pes_payload,
            "scientific_abstract": abstract_payload,
            "background_and_rationale": _build_evidence(research_splits.get("background_and_rationale", "")) if research_splits.get("background_and_rationale") else None,
            "aims": _build_evidence(research_splits.get("aims", "")) if research_splits.get("aims") else None,
            "methods": _build_evidence(research_splits.get("methods", "")) if research_splits.get("methods") else research_plan_payload,
            "impact_and_dissemination": _build_evidence(research_splits.get("impact_and_dissemination", "")) if research_splits.get("impact_and_dissemination") else None,
            "inclusive_design": _build_evidence(
                "\n".join(
                    sentence for sentence in (
                        _text_from_payload(pes_payload) + "\n" + _text_from_payload(abstract_payload) + "\n" + _text_from_payload(ppi_payload)
                    ).splitlines()
                    if re.search(r"\b(inclusive|diverse|underserved|accessib|co-design|co design|equity|equality)\b", sentence, re.I)
                )
            ),
        },
        "training": {
            "overall_plan": training_payload or training_bundle,
            "gap_analysis": _build_evidence(training_splits.get("gap_analysis", "")) if training_splits.get("gap_analysis") else None,
            "training_plan": _dedup_payload(
                _build_evidence(training_splits.get("training_plan", "")) if training_splits.get("training_plan") else (training_payload or training_bundle),
                training_payload or training_bundle,
            ),
            "research_support": _dedup_payload(
                _build_evidence(training_splits.get("research_support", "")) if training_splits.get("research_support") else support_payload,
                training_payload or training_bundle,
            ),
            "practice_development": _build_evidence(training_splits.get("practice_development", "")) if training_splits.get("practice_development") else None,
            "leadership_development": _dedup_payload(
                _build_evidence(training_splits.get("leadership_development", "")) if training_splits.get("leadership_development") else None,
                training_payload or training_bundle,
            ),
            "training_items": training_items,
        },
        "support": {
            "supervision_and_mentorship": normalized_team,
            "support_text": _build_evidence(support_text, source_section="support_bundle", confidence=0.72) if support_text else None,
            "host_support_statement": host_payload,
            "meeting_frequency_mentions": _extract_meeting_frequency(support_text),
            "research_environment": _build_evidence(
                "\n".join(
                    line for line in _text_from_payload(host_payload).splitlines()
                    if re.search(r"\b(environment|infrastructure|facility|facilities|support|network|research culture|BRC|institute)\b", line, re.I)
                )
            ) if host_payload else None,
        },
        "ppi_and_inclusion": {
            "overall_ppi": ppi_payload,
            "prior_involvement": _dedup_payload(
                _build_evidence(ppi_splits.get("prior_involvement", "")) if ppi_splits.get("prior_involvement") else ppi_payload,
                ppi_payload,
            ),
            "planned_involvement": _build_evidence(ppi_splits.get("planned_involvement", "")) if ppi_splits.get("planned_involvement") else None,
            "underserved_groups_mentioned": underserved_groups,
            "costed_inclusion_present": bool(re.search(r"patient\s+and\s+public\s+involvement|ppi", _text_from_payload(budget_payload), re.I)),
        },
        "budget": {
            "summary": budget_payload,
            "line_items": budget_line_items,
            "justification": _build_evidence(
                "\n".join(
                    line for line in _text_from_payload(budget_payload).splitlines()
                    if re.search(r"\b(justification|because|required|covers|support|needed)\b", line, re.I)
                ),
                source_section=budget_payload.get("source_section") if budget_payload else "SUMMARY BUDGET",
                page_start=budget_payload.get("page_start") if budget_payload else None,
                page_end=budget_payload.get("page_end") if budget_payload else None,
                confidence=0.74,
            ) if budget_payload else None,
        },
    }

    scoring_ready["form_quality"] = _form_quality_payload(evidence_map)
    return _strip_internal_fields(scoring_ready)
