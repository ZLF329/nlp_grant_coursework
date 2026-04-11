"""
Fellowship PDF Parser (Blue-Box Format)

Parses NIHR Fellowship application PDFs (e.g., DF R11 Doctoral Fellowship).
Output JSON structure matches IC00458_after.json (same top-level keys; only
sections present in the document are included).

Section mapping
---------------
"1. Application Summary Information"    -> SUMMARY INFORMATION
"4. Plain English Summary of Research"  -> APPLICATION DETAILS["Plain English Summary"]
"5. Scientific Abstract"                -> APPLICATION DETAILS["Scientific Abstract"]
"2. Applicant CV"                       -> APPLICATION DETAILS["Applicant CV"]
"3. Applicant Research Background"      -> APPLICATION DETAILS["Applicant Research Background"]
"7. Patient & Public Involvement"       -> APPLICATION DETAILS["Working with People and Communities Summary"]
"9. Detailed Budget"                    -> SUMMARY BUDGET
"11. Participants and Signatories"      -> LEAD APPLICANT & RESEARCH TEAM (supervisors as Co-Applicants)

Lead Applicant name/title is extracted from the page-1 Summary overview.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import pdfplumber
import re
import os
import json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Blue-box filled rect band (same on every section-opening page)
BLUE_BOX_TOP_MIN: float = 69.0
BLUE_BOX_TOP_MAX: float = 86.0

# Page header ("DF R11") and footer thresholds
PAGE_HEADER_BOTTOM: float = 58.0   # discard lines whose bottom < this
PAGE_FOOTER_TOP:    float = 805.0  # discard lines whose top    > this

# Section headings for extraction into JSON output:
SECTION_APP_SUMMARY:  str = "Application Summary Information"
SECTION_PES:          str = "Plain English Summary of Research"
SECTION_ABSTRACT:     str = "Scientific Abstract"
SECTION_PLAN:         str = "Detailed Research Plan"
SECTION_PPI:          str = "Patient & Public Involvement"
SECTION_BUDGET:       str = "Detailed Budget"
SECTION_PARTICIPANTS: str = "Participants and Signatories"
SECTION_APPLICANT_CV: str = "Applicant CV"
SECTION_APPLICANT_RESEARCH_BACKGROUND: str = "Applicant Research Background"
SECTION_TRAINING:     str = "Training & Development and Research Support"

_STRIP_NUMBER_RE = re.compile(r'^\d+\.\s+')


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Line:
    text: str
    page: int
    x0: float
    top: float
    x1: float
    bottom: float
    page_height: float
    is_section_page: bool = False  # True iff this page carries a blue-box filled rect


# ---------------------------------------------------------------------------
# Core line extraction  (mirrors pdf_parser.py)
# ---------------------------------------------------------------------------

def normalize_heading(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s*:\s*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_lines_pdfplumber(
    pdf_path: str,
    y_tolerance: float = 3.0,
    x_tolerance: float = 1.0,
    keep_empty: bool = False,
) -> List[Line]:
    """
    Extract every word from the PDF and group nearby words into Line objects.
    Returns a list sorted by (page, top, x0).
    """
    lines_out: List[Line] = []

    with pdfplumber.open(pdf_path) as pdf:
        # Pre-scan: identify pages that genuinely carry a blue-box filled rect
        section_page_nums: set = {
            pno for pno, page in enumerate(pdf.pages)
            if any(
                r.get("fill") and BLUE_BOX_TOP_MIN <= r["top"] <= BLUE_BOX_TOP_MAX
                for r in page.rects
            )
        }

        for pno, page in enumerate(pdf.pages):
            words = page.extract_words(
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
                keep_blank_chars=False,
                use_text_flow=True,
            )
            if not words:
                continue

            buckets: Dict[int, List[dict]] = {}
            for w in words:
                key = int(round(w["top"] / y_tolerance))
                buckets.setdefault(key, []).append(w)

            for _, ws in sorted(buckets.items()):
                ws_sorted = sorted(ws, key=lambda w: w["x0"])
                text = " ".join(w["text"] for w in ws_sorted).strip()
                if not text and not keep_empty:
                    continue
                lines_out.append(Line(
                    text=text,
                    page=pno,
                    x0=min(w["x0"] for w in ws_sorted),
                    top=min(w["top"] for w in ws_sorted),
                    x1=max(w["x1"] for w in ws_sorted),
                    bottom=max(w["bottom"] for w in ws_sorted),
                    page_height=float(page.height),
                    is_section_page=(pno in section_page_nums),
                ))

    lines_out.sort(key=lambda ln: (ln.page, ln.top, ln.x0))
    return lines_out


def filter_fellowship_lines(lines: List[Line]) -> List[Line]:
    """Remove page headers (bottom < 58) and footers (top > 795)."""
    return [
        ln for ln in lines
        if ln.bottom >= PAGE_HEADER_BOTTOM and ln.top <= PAGE_FOOTER_TOP
    ]


# ---------------------------------------------------------------------------
# Section boundary detection  (mirrors pdf_parser.py's big-box detection)
# ---------------------------------------------------------------------------

def _strip_number(s: str) -> str:
    """Strip leading 'N. ' prefix from a heading string."""
    return _STRIP_NUMBER_RE.sub('', s.strip())


def is_fellowship_heading(line: Line) -> bool:
    """True when this line sits inside the blue-box band on a section-opening page.
    Purely visual — any filled-rect heading is recognised regardless of its text.
    """
    return (
        line.is_section_page
        and BLUE_BOX_TOP_MIN <= line.top <= BLUE_BOX_TOP_MAX
    )


def find_section_ranges(lines: List[Line]) -> List[int]:
    """Return indices of every blue-box section heading line."""
    return [i for i, ln in enumerate(lines) if is_fellowship_heading(ln)]


def list_section_titles(lines: List[Line]) -> List[str]:
    """Return all section heading texts in document order."""
    return [lines[i].text.strip() for i in find_section_ranges(lines)]


def slice_section(lines: List[Line], section_title: str) -> List[Line]:
    """
    Return lines belonging to `section_title` (the heading line itself is excluded).
    Slice ends at the next blue-box heading (or end of document).
    """
    section_idxs = find_section_ranges(lines)

    start_idx: Optional[int] = None
    for idx in section_idxs:
        if _strip_number(lines[idx].text.strip()) == section_title:
            start_idx = idx
            break

    if start_idx is None:
        return []

    pos = section_idxs.index(start_idx)
    content_start = start_idx + 1
    content_end = (
        section_idxs[pos + 1] if pos + 1 < len(section_idxs) else len(lines)
    )
    return lines[content_start:content_end]


# ---------------------------------------------------------------------------
# SUMMARY INFORMATION  (section 1 — two-column application form)
# ---------------------------------------------------------------------------

def _get_section_raw_text(pdf_path: str, target_heading: str) -> str:
    """
    Collect pdfplumber extract_text() output for all pages that belong to
    the section whose blue-box heading matches `target_heading`.
    Continuation pages (no blue-box rect) are included until the next section.
    """
    pages_text: List[str] = []
    in_section = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            rects = page.rects
            has_blue_box = any(
                r.get("fill") and BLUE_BOX_TOP_MIN <= r["top"] <= BLUE_BOX_TOP_MAX
                for r in rects
            )

            if has_blue_box:
                words = page.extract_words(x_tolerance=1, y_tolerance=3)
                heading_words = sorted(
                    [w for w in words
                     if BLUE_BOX_TOP_MIN <= w["top"] <= BLUE_BOX_TOP_MAX],
                    key=lambda w: w["x0"],
                )
                heading = _strip_number(" ".join(w["text"] for w in heading_words).strip())

                if heading == target_heading:
                    in_section = True
                    pages_text.append(page.extract_text() or "")
                elif in_section:
                    # Reached a new numbered section — stop
                    break
            elif in_section:
                # Continuation page (no blue box) — still part of this section
                pages_text.append(page.extract_text() or "")

    return "\n".join(pages_text)


def parse_summary_information(pdf_path: str) -> dict:
    """
    Parse 'Application Summary Information' into a SUMMARY INFORMATION dict.

    Extracted fields:
        Application Title          — from "Research Title" two-column row
        Contracting Organisation   — line after "Host organisation (...)"
        Start Date                 — date value after "Proposed start date"
        Duration (months)          — first integer after "Duration" block
    """
    raw = _get_section_raw_text(pdf_path, SECTION_APP_SUMMARY)
    out: dict = {}

    # Contracting Organisation: next non-empty line after "Host organisation ..." label
    m = re.search(
        r'Host organisation[^\n]*\n([^\n]+)',
        raw,
    )
    if m:
        candidate = m.group(1).strip()
        # Skip lines that are part of the label (e.g. "applicable)")
        if candidate and not re.match(r'^[Pp]artner|^applicable', candidate):
            out["Contracting Organisation"] = candidate

    # Application Title: appears split around the "Research Title" label.
    # extract_text order: value-part-1 \n Research Title \n value-part-2
    m = re.search(r'(.+)\nResearch Title\n(.+)', raw)
    if m:
        part1 = m.group(1).strip()
        part2 = m.group(2).strip()
        # part1 may also contain a trailing field-label line — keep only the last line
        part1_last_line = part1.split("\n")[-1].strip()
        out["Application Title"] = (part1_last_line + " " + part2).strip()

    # Duration (months): first stand-alone integer after the "Duration" block description
    m = re.search(r'\bDuration\b[\s\S]*?\n(\d+)\s*\n', raw)
    if m:
        out["Duration (months)"] = m.group(1)

    # Start Date: date (dd/mm/yyyy) that appears after "Proposed start date"
    m = re.search(
        r'Proposed start date[^\n]*\n([0-9]{2}/[0-9]{2}/[0-9]{4})',
        raw,
    )
    if m:
        out["Start Date"] = m.group(1)

    return out


# ---------------------------------------------------------------------------
# LEAD APPLICANT & RESEARCH TEAM
# ---------------------------------------------------------------------------

def _get_page1_lead_applicant(pdf_path: str) -> str:
    """
    Extract Lead Applicant name from the page-1 Summary section.
    On page 1, the two-column layout places h=12 labels and h=9 values
    on essentially the same y-coordinate, so extract_lines_pdfplumber
    merges them into a single line: "Lead Applicant Dr Weiyu Ye".
    """
    lines = extract_lines_pdfplumber(pdf_path)
    lines = filter_fellowship_lines(lines)

    # Page 1 only
    page1_lines = [ln for ln in lines if ln.page == 0]

    for ln in page1_lines:
        t = ln.text.strip()
        if t.startswith("Lead Applicant "):
            return t[len("Lead Applicant "):].strip()

    return ""


def _parse_participants_raw(pdf_path: str) -> List[dict]:
    """
    Parse section 11 'Participants and Signatories' from the raw extract_text output.

    The two-column form produces lines like:
        "Title Dr"
        "Forename(s) Satveer"
        "Surname Mahil"
        "Position Consultant Dermatologist and Adjunct Senior Lecturer"

    Role categories recognised as Co-Applicants:
        Doctoral Primary Supervisor, Supervisor
    """
    raw = _get_section_raw_text(pdf_path, SECTION_PARTICIPANTS)

    # Pattern: find each person block by Title → Forename(s) → Surname → Position
    person_pattern = re.compile(
        r'Title\s+(\S+)\s*\n'
        r'Forename\(s\)\s+(.+?)\s*\n'
        r'Surname\s+(.+?)\s*\n'
        r'Position\s+(.+?)(?=\nTitle|\nSupervisor|\nDoctoral|\nHost Organisation|\Z)',
        re.DOTALL,
    )

    # Role context: walk through raw text, track which role label precedes each person
    ROLE_HEADERS = [
        "Doctoral Primary Supervisor",
        "Supervisor",
    ]

    people = []

    # Split raw text into segments by role headers
    # Each segment: role_label + one or more person blocks
    segments = re.split(
        r'(?m)^(Doctoral Primary Supervisor|Supervisor)\s*\n',
        raw,
    )

    current_role = None
    for segment in segments:
        segment = segment.strip()
        if segment in ROLE_HEADERS:
            current_role = segment
            continue
        if current_role is None:
            continue
        # Find all persons in this segment
        for m in person_pattern.finditer(segment):
            title    = m.group(1).strip()
            forename = m.group(2).strip()
            surname  = m.group(3).strip()
            position = re.sub(r'\s+', ' ', m.group(4)).strip()
            full_name = f"{title} {forename} {surname}".strip()
            people.append({
                "Full Name":     full_name,
                "Proposed Role": current_role,
                "Organisation":  "",
                "Department":    "",
                "ORCID":         "",
                "Position":      position,
            })

    return people


def _extract_orcid_from_cv(pdf_path: str) -> str:
    """Extract the Lead Applicant's ORCID iD from the 'Applicant CV' section."""
    raw = _get_section_raw_text(pdf_path, SECTION_APPLICANT_CV)
    m = re.search(r'ORCID\s+iD\s+(\S+)', raw)
    return m.group(1) if m else ""


def parse_lead_applicant_research_team(pdf_path: str) -> dict:
    """
    Build LEAD APPLICANT & RESEARCH TEAM dict.

    Lead Applicant: name from page-1 summary; organisation from section 1.
    Co-Applicants:  supervisors from section 11.
    """
    lead_name = _get_page1_lead_applicant(pdf_path)

    # Get organisation from section 1 raw text
    raw_s1 = _get_section_raw_text(pdf_path, SECTION_APP_SUMMARY)
    organisation = ""
    m = re.search(r'Host organisation[^\n]*\n([^\n]+)', raw_s1)
    if m:
        candidate = m.group(1).strip()
        if candidate and not re.match(r'^[Pp]artner|^applicable', candidate):
            organisation = candidate

    orcid = _extract_orcid_from_cv(pdf_path)

    lead_applicant = {
        "Full Name":        lead_name,
        "Organisation":     organisation,
        "Department":       "",
        "Proposed Role":    "Lead Applicant",
        "ORCID":            orcid,
        "% FTE Commitment": "",
    } if lead_name else None

    co_applicants = _parse_participants_raw(pdf_path)

    return {
        "Lead Applicant":       lead_applicant,
        "Joint Lead Applicant": None,
        "Co-Applicants":        co_applicants,
    }


# ---------------------------------------------------------------------------
# APPLICATION DETAILS  (text sections)
# ---------------------------------------------------------------------------

def parse_text_section(lines: List[Line], join_with: str = "\n") -> str:
    """Concatenate all content lines in a section (PES, Abstract, WPCC, Budget)."""
    return join_with.join(ln.text for ln in lines if ln.text.strip())


def parse_application_details(lines: List[Line]) -> dict:
    """
    Build APPLICATION DETAILS dict from the relevant fellowship sections.

    Reads section 4 (PES), 5 (Abstract), 7 (PPI/WPCC) directly from
    the pre-filtered lines.
    """
    out: dict = {}

    cv_lines = slice_section(lines, SECTION_APPLICANT_CV)
    if cv_lines:
        out["Applicant CV"] = parse_text_section(cv_lines)

    research_background_lines = slice_section(lines, SECTION_APPLICANT_RESEARCH_BACKGROUND)
    if research_background_lines:
        out["Applicant Research Background"] = parse_text_section(research_background_lines)

    pes_lines = slice_section(lines, SECTION_PES)
    if pes_lines:
        out["Plain English Summary of Research"] = parse_text_section(pes_lines)

    abstract_lines = slice_section(lines, SECTION_ABSTRACT)
    if abstract_lines:
        out["Scientific Abstract"] = parse_text_section(abstract_lines)

    plan_lines = slice_section(lines, SECTION_PLAN)
    if plan_lines:
        out["Detailed Research Plan"] = parse_text_section(plan_lines)

    ppi_lines = slice_section(lines, SECTION_PPI)
    if ppi_lines:
        out["Patient & Public Involvement"] = parse_text_section(ppi_lines)

    training_lines = slice_section(lines, SECTION_TRAINING)
    if training_lines:
        out["Training & Development and Research Support"] = parse_text_section(training_lines)

    return out


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def extract_all_sections(pdf_path: str) -> dict:
    """
    Extract all relevant sections from a fellowship PDF and return a dict
    matching the IC00458_after.json structure.
    """
    # Step 1: extract and filter lines (used for text sections + section detection)
    lines = extract_lines_pdfplumber(pdf_path)
    lines = filter_fellowship_lines(lines)

    out: dict = {}

    # SUMMARY INFORMATION — regex-based on section 1 raw text
    summary_info = parse_summary_information(pdf_path)
    if summary_info:
        out["SUMMARY INFORMATION"] = summary_info

    # LEAD APPLICANT & RESEARCH TEAM
    team = parse_lead_applicant_research_team(pdf_path)
    if team.get("Lead Applicant") or team.get("Co-Applicants"):
        out["LEAD APPLICANT & RESEARCH TEAM"] = team

    # APPLICATION DETAILS — text sections from lines
    app_details = parse_application_details(lines)
    if app_details:
        out["APPLICATION DETAILS"] = app_details

    # SUMMARY BUDGET — join all lines from section 9
    budget_lines = slice_section(lines, SECTION_BUDGET)
    if budget_lines:
        out["SUMMARY BUDGET"] = parse_text_section(budget_lines)

    return out


# ---------------------------------------------------------------------------
# Save helpers  (mirrors pdf_parser.py)
# ---------------------------------------------------------------------------

def pdf_path_to_json_path(pdf_path: str, json_dir_name: str = "json_data") -> str:
    """
    Derive a sibling JSON path under a `json_data` subdirectory.

    Example:
        .../test_data/1.pdf  ->  .../test_data/json_data/1.json
    """
    pdf_dir  = os.path.dirname(pdf_path)
    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    json_dir = os.path.join(pdf_dir, json_dir_name)
    os.makedirs(json_dir, exist_ok=True)
    return os.path.join(json_dir, pdf_base + ".json")


def save_json(data: dict, json_path: str, indent: int = 2) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def extract_and_save(pdf_path: str) -> str:
    """
    Parse a fellowship PDF and write the result to a JSON file.
    Returns the path of the saved JSON.
    """
    sections = extract_all_sections(pdf_path)
    json_path = pdf_path_to_json_path(pdf_path)
    save_json(sections, json_path)
    return json_path
