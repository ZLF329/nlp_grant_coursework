"""
RfPB PDF Parser (Stage 2 Blue-Box Format)

Parses NIHR Research for Patient Benefit (RfPB) Stage 2 application PDFs.
Output JSON structure matches IC00458_after.json (same top-level keys; only
sections present in the document are included).

Section mapping
---------------
Page 1 summary table               -> SUMMARY INFORMATION
"1. The Research Team"             -> LEAD APPLICANT & RESEARCH TEAM
"3. Scientific abstract"           -> APPLICATION DETAILS["Scientific Abstract"]
"4. Plain English Summary"         -> APPLICATION DETAILS["Plain English Summary of Research"]
"5. Changes from first stage"      -> APPLICATION DETAILS["Changes from Previous Stage"]
"6. Detailed Research plan"        -> APPLICATION DETAILS["Detailed Research Plan"]
"7. Patient & Public Involvement"  -> APPLICATION DETAILS["Patient & Public Involvement"]
"8. Detailed Budget"               -> SUMMARY BUDGET

Key structural difference from fellowships_parser:
  Fellowship — section heading box is always at page TOP (y=69–86).
  RfPB       — section heading boxes can appear ANYWHERE on a page
               (multiple sections per page).  Detection is done per-line
               via in_section_box flag rather than a per-page flag.
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

# A filled rect must span at least this fraction of page width to count as a
# section-heading box.
SECTION_BOX_WIDTH_RATIO: float = 0.55

# Page header / footer thresholds (discard chrome outside this band)
PAGE_HEADER_BOTTOM: float = 58.0
PAGE_FOOTER_TOP:    float = 805.0

# Known section headings (without number prefix — matches any "N. Heading" variant)
SECTION_TEAM:     str = "The Research Team"
SECTION_HISTORY:  str = "History of application"
SECTION_ABSTRACT: str = "Scientific abstract"
SECTION_PES:      str = "Plain English Summary"
SECTION_CHANGES:  str = "Changes from first stage"
SECTION_PLAN:     str = "Detailed Research plan"
SECTION_PPI:      str = "Patient & Public Involvement"
SECTION_BUDGET:   str = "Detailed Budget"
SECTION_MGMT:     str = "Management & Governance"
SECTION_UPLOADS:  str = "Uploads"
SECTION_ADMIN:    str = "Administrative contact details"
SECTION_RD:       str = "Research & Development office contact"
SECTION_ACK:      str = "Acknowledgement review and submit"
SECTION_CV_LEAD:  str = "CV - Lead Applicant(s)"
SECTION_CV_CO:    str = "CV - Co-applicants"
KNOWN_SECTION_HEADINGS = {
    SECTION_TEAM,
    SECTION_HISTORY,
    SECTION_ABSTRACT,
    SECTION_PES,
    SECTION_CHANGES,
    SECTION_PLAN,
    SECTION_PPI,
    SECTION_BUDGET,
    SECTION_MGMT,
    SECTION_UPLOADS,
    SECTION_ADMIN,
    SECTION_RD,
    SECTION_ACK,
    SECTION_CV_LEAD,
    SECTION_CV_CO,
}


_STRIP_NUMBER_RE = re.compile(r'^\d+\.\s+')

def _strip_number(s: str) -> str:
    """Strip leading 'N. ' prefix from a heading string."""
    return _STRIP_NUMBER_RE.sub('', s.strip())


def _heading_key(s: str) -> str:
    """Normalize headings for robust RfPB section matching."""
    return re.sub(r'[^a-z0-9]+', '', _strip_number(s).lower())


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
    in_section_box: bool = False   # True iff this line overlaps a wide filled rect


# ---------------------------------------------------------------------------
# Section-box helpers
# ---------------------------------------------------------------------------

def _page_section_boxes(page) -> list:
    """Return all filled rects on this page that span ≥ SECTION_BOX_WIDTH_RATIO of the width."""
    threshold = SECTION_BOX_WIDTH_RATIO * float(page.width)
    return [
        r for r in page.rects
        if r.get("fill") and r.get("width", 0) >= threshold
    ]


def _overlaps_any(top: float, bottom: float, boxes: list) -> bool:
    """True if the vertical span [top, bottom] overlaps any box in `boxes`."""
    for b in boxes:
        if top <= b["bottom"] and bottom >= b["top"]:
            return True
    return False


# ---------------------------------------------------------------------------
# Core line extraction
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
    Each Line is annotated with in_section_box=True if it overlaps any wide
    filled rect on its page.
    Returns a list sorted by (page, top, x0).
    """
    lines_out: List[Line] = []

    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages):
            section_boxes = _page_section_boxes(page)

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
                top    = min(w["top"]    for w in ws_sorted)
                bottom = max(w["bottom"] for w in ws_sorted)
                lines_out.append(Line(
                    text=text,
                    page=pno,
                    x0=min(w["x0"] for w in ws_sorted),
                    top=top,
                    x1=max(w["x1"] for w in ws_sorted),
                    bottom=bottom,
                    page_height=float(page.height),
                    in_section_box=_overlaps_any(top, bottom, section_boxes),
                ))

    lines_out.sort(key=lambda ln: (ln.page, ln.top, ln.x0))
    return lines_out


def filter_rfpb_lines(lines: List[Line]) -> List[Line]:
    """Remove page headers (bottom < PAGE_HEADER_BOTTOM) and footers (top > PAGE_FOOTER_TOP)."""
    return [
        ln for ln in lines
        if ln.bottom >= PAGE_HEADER_BOTTOM and ln.top <= PAGE_FOOTER_TOP
    ]


# ---------------------------------------------------------------------------
# Section boundary detection
# ---------------------------------------------------------------------------

def is_rfpb_heading(line: Line) -> bool:
    """
    True when this line is a numbered section heading inside a wide blue box.
    Requires the line to overlap a wide filled rect AND the text to match a
    known RfPB section heading.
    """
    text = line.text.strip()
    if not line.in_section_box or not re.match(r'^\d+\.\s+', text):
        return False
    key = _heading_key(text)
    for known in KNOWN_SECTION_HEADINGS:
        known_key = _heading_key(known)
        if key == known_key or key.startswith(known_key):
            return True
    return False


def find_section_ranges(lines: List[Line]) -> List[int]:
    """Return indices of every section-heading line."""
    return [i for i, ln in enumerate(lines) if is_rfpb_heading(ln)]


def list_section_titles(lines: List[Line]) -> List[str]:
    """Return all section heading texts in document order."""
    return [lines[i].text.strip() for i in find_section_ranges(lines)]


def slice_section(lines: List[Line], section_title: str) -> List[Line]:
    """
    Return lines belonging to `section_title` (heading line excluded).
    Slice ends at the next section heading (or end of document).
    """
    section_idxs = find_section_ranges(lines)
    norm_target = _heading_key(section_title)

    start_idx: Optional[int] = None
    for idx in section_idxs:
        if _heading_key(lines[idx].text.strip()).startswith(norm_target):
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
# SUMMARY INFORMATION  (page-1 summary table)
# ---------------------------------------------------------------------------

def _get_page1_raw_text(pdf_path: str) -> str:
    """Return pdfplumber extract_text() output for page 1."""
    with pdfplumber.open(pdf_path) as pdf:
        if pdf.pages:
            return pdf.pages[0].extract_text() or ""
    return ""


def _get_section_raw_text(pdf_path: str, target_heading: str) -> str:
    """
    Collect pdfplumber extract_text() output for all pages belonging to the
    named section.  A page belongs to a section if it contains the target
    section-box heading; continuation pages are included until the next
    section-box heading is encountered.
    """
    norm_target = re.sub(r'\s+', ' ', target_heading.strip())
    pages_text: List[str] = []
    in_section = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            boxes = _page_section_boxes(page)

            if boxes:
                # Find all section headings on this page
                words = page.extract_words(x_tolerance=1, y_tolerance=3)
                headings_found: List[str] = []
                for box in boxes:
                    box_words = sorted(
                        [w for w in words
                         if w["top"] <= box["bottom"] and w["bottom"] >= box["top"]],
                        key=lambda w: w["x0"],
                    )
                    h = re.sub(r'\s+', ' ',
                               _strip_number(" ".join(w["text"] for w in box_words).strip()))
                    if h:
                        headings_found.append(h)

                if any(norm_target in h for h in headings_found):
                    in_section = True
                    pages_text.append(page.extract_text() or "")
                elif in_section:
                    # Any page that opens a new section box ends the current one
                    break
            elif in_section:
                pages_text.append(page.extract_text() or "")

    return "\n".join(pages_text)


def parse_summary_information(pdf_path: str) -> dict:
    """
    Extract SUMMARY INFORMATION fields from the page-1 summary table.

    Fields extracted:
        Application Title        — "Research Title" row
        Contracting Organisation — "Host organisation" row
        Start Date               — "Start Date" row
        End Date                 — "End Date" row
        Duration (months)        — "Grant Duration" row
        Total Cost to NIHR       — "Research Costs" row
    """
    raw = _get_page1_raw_text(pdf_path)
    out: dict = {}

    # Application Title
    m = re.search(r'Research Title[:\s]*\n?(.+)', raw)
    if m:
        out["Application Title"] = m.group(1).strip()

    # Contracting Organisation
    m = re.search(r'Host organisation[^\n]*\n([^\n]+)', raw)
    if m:
        candidate = m.group(1).strip()
        if candidate and not re.match(r'^[Pp]artner|^applicable', candidate):
            out["Contracting Organisation"] = candidate

    # Start Date
    m = re.search(r'Start Date[:\s]*\n?([0-9]{2}/[0-9]{2}/[0-9]{4})', raw)
    if m:
        out["Start Date"] = m.group(1)

    # End Date
    m = re.search(r'End Date[:\s]*\n?([0-9]{2}/[0-9]{2}/[0-9]{4})', raw)
    if m:
        out["End Date"] = m.group(1)

    # Duration (months) — from "Grant Duration" field
    m = re.search(r'(?:Grant\s+)?Duration[^\n]*\n?(\d+)\s*(?:months?)?', raw, re.IGNORECASE)
    if m:
        out["Duration (months)"] = m.group(1)

    # Total Cost to NIHR — from "Research Costs" field
    m = re.search(r'(?:Research\s+)?Costs?[^\n]*\n?([\£\$]?[\d,]+(?:\.\d{2})?)', raw, re.IGNORECASE)
    if m:
        out["Total Cost to NIHR"] = m.group(1).strip()

    return out


# ---------------------------------------------------------------------------
# LEAD APPLICANT & RESEARCH TEAM
# ---------------------------------------------------------------------------

def _get_page1_lead_applicant(pdf_path: str) -> str:
    """
    Extract Lead Applicant name from the page-1 summary table.
    Looks for a line of the form "Lead Applicant <Name>" or a two-column
    merge where the label and value appear on the same extracted line.
    """
    lines = extract_lines_pdfplumber(pdf_path)
    lines = filter_rfpb_lines(lines)
    page1_lines = [ln for ln in lines if ln.page == 0]

    for ln in page1_lines:
        t = ln.text.strip()
        if t.lower().startswith("lead applicant"):
            # Try to grab value from same line ("Lead Applicant Dr Jane Smith")
            rest = t[len("lead applicant"):].strip()
            if rest and not re.match(r'^[:\(]', rest):
                return rest
    return ""


def _parse_research_team_raw(pdf_path: str) -> List[dict]:
    """
    Parse section 1 'The Research Team' from raw extract_text output.

    Typical two-column layout produces blocks such as:
        Role  Lead Applicant
        Title  Dr
        Forename(s)  Jane
        Surname  Smith
        Position  Senior Lecturer
        Organisation  University of ...
    """
    raw = _get_section_raw_text(pdf_path, SECTION_TEAM)

    person_pattern = re.compile(
        r'Title\s+(\S+)\s*\n'
        r'Forename\(s\)\s+(.+?)\s*\n'
        r'Surname\s+(.+?)\s*\n'
        r'(?:Position\s+(.+?)\s*\n)?'
        r'(?:Organisation\s+(.+?)(?=\nTitle|\nRole|\nLead|\nCo-|\Z))?',
        re.DOTALL,
    )

    # Split by role labels to associate people with their roles
    role_split_re = re.compile(
        r'(?m)^(Lead Applicant|Co-[Aa]pplicant|Co [Aa]pplicant|'
        r'Collaborator|Statistician|Trial Manager|Research Nurse|Researcher)\s*\n'
    )

    segments = role_split_re.split(raw)
    people: List[dict] = []
    current_role = None

    for segment in segments:
        segment_stripped = segment.strip()
        # Check if this segment is a role header
        if re.match(r'^(Lead Applicant|Co-[Aa]pplicant|Co [Aa]pplicant|'
                    r'Collaborator|Statistician|Trial Manager|Research Nurse|Researcher)$',
                    segment_stripped):
            current_role = segment_stripped
            continue
        if current_role is None:
            continue
        for m in person_pattern.finditer(segment):
            title    = m.group(1).strip()
            forename = m.group(2).strip()
            surname  = m.group(3).strip()
            position = re.sub(r'\s+', ' ', (m.group(4) or "")).strip()
            org      = re.sub(r'\s+', ' ', (m.group(5) or "")).strip()
            full_name = f"{title} {forename} {surname}".strip()
            people.append({
                "Full Name":     full_name,
                "Proposed Role": current_role,
                "Organisation":  org,
                "Department":    "",
                "ORCID":         "",
                "Position":      position,
            })

    return people


def _extract_orcid_from_cv_lead(pdf_path: str) -> str:
    """Extract the Lead Applicant's ORCID iD from the 'CV - Lead Applicant(s)' section."""
    raw = _get_section_raw_text(pdf_path, SECTION_CV_LEAD)
    m = re.search(r'ORCID\s+iD\s+(\S+)', raw)
    return m.group(1) if m else ""


def parse_lead_applicant_research_team(pdf_path: str) -> dict:
    """
    Build LEAD APPLICANT & RESEARCH TEAM dict.

    Lead Applicant: name extracted from page-1; organisation from section 1.
    Co-Applicants:  all non-lead-applicant persons from section 1.
    """
    lead_name = _get_page1_lead_applicant(pdf_path)

    # Organisation from section 1 raw text
    raw_s1 = _get_section_raw_text(pdf_path, SECTION_TEAM)
    organisation = ""
    m = re.search(r'Organisation\s+([^\n]+)', raw_s1)
    if m:
        organisation = m.group(1).strip()

    orcid = _extract_orcid_from_cv_lead(pdf_path)

    lead_applicant = {
        "Full Name":        lead_name,
        "Organisation":     organisation,
        "Department":       "",
        "Proposed Role":    "Lead Applicant",
        "ORCID":            orcid,
        "% FTE Commitment": "",
    } if lead_name else None

    all_people = _parse_research_team_raw(pdf_path)
    co_applicants = [
        p for p in all_people
        if not p["Proposed Role"].lower().startswith("lead applicant")
    ]

    return {
        "Lead Applicant":       lead_applicant,
        "Joint Lead Applicant": None,
        "Co-Applicants":        co_applicants,
    }


# ---------------------------------------------------------------------------
# APPLICATION DETAILS  (text sections)
# ---------------------------------------------------------------------------

def parse_text_section(lines: List[Line], join_with: str = "\n") -> str:
    """Concatenate all non-empty content lines in a section."""
    return join_with.join(ln.text for ln in lines if ln.text.strip())


def parse_application_details(lines: List[Line]) -> dict:
    """
    Build APPLICATION DETAILS dict from the relevant RfPB sections.

    Section 3  -> Scientific Abstract
    Section 4  -> Plain English Summary of Research
    Section 5  -> Changes from Previous Stage
    Section 6  -> Detailed Research Plan
    Section 7  -> Patient & Public Involvement
    """
    out: dict = {}

    abstract_lines = slice_section(lines, SECTION_ABSTRACT)
    if abstract_lines:
        out["Scientific Abstract"] = parse_text_section(abstract_lines)

    pes_lines = slice_section(lines, SECTION_PES)
    if pes_lines:
        out["Plain English Summary of Research"] = parse_text_section(pes_lines)

    changes_lines = slice_section(lines, SECTION_CHANGES)
    if changes_lines:
        out["Changes from Previous Stage"] = parse_text_section(changes_lines)

    plan_lines = slice_section(lines, SECTION_PLAN)
    if plan_lines:
        out["Detailed Research Plan"] = parse_text_section(plan_lines)

    ppi_lines = slice_section(lines, SECTION_PPI)
    if ppi_lines:
        out["Patient & Public Involvement"] = parse_text_section(ppi_lines)

    return out


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def extract_all_sections(pdf_path: str) -> dict:
    """
    Extract all relevant sections from an RfPB Stage 2 PDF and return a dict
    matching the IC00458_after.json structure.

    Returns an empty dict if no RfPB section headings are detected (so the
    caller can fall through to the next parser in the pipeline).
    """
    # Extract and filter lines — used for section detection and text extraction
    lines = extract_lines_pdfplumber(pdf_path)
    lines = filter_rfpb_lines(lines)

    # Sanity check: if no known section headings found, this isn't an RfPB PDF
    if not find_section_ranges(lines):
        return {}

    out: dict = {}

    # SUMMARY INFORMATION — regex on page-1 raw text
    summary_info = parse_summary_information(pdf_path)
    if summary_info:
        out["SUMMARY INFORMATION"] = summary_info

    # LEAD APPLICANT & RESEARCH TEAM
    team = parse_lead_applicant_research_team(pdf_path)
    if team.get("Lead Applicant") or team.get("Co-Applicants"):
        out["LEAD APPLICANT & RESEARCH TEAM"] = team

    # APPLICATION DETAILS — sections 3, 4, 5, 6, 7
    app_details = parse_application_details(lines)
    if app_details:
        out["APPLICATION DETAILS"] = app_details

    # SUMMARY BUDGET — section 8
    budget_lines = slice_section(lines, SECTION_BUDGET)
    if budget_lines:
        out["SUMMARY BUDGET"] = parse_text_section(budget_lines)

    return out


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def pdf_path_to_json_path(pdf_path: str, json_dir_name: str = "json_data") -> str:
    """Derive a sibling JSON path under a `json_data` subdirectory."""
    pdf_dir  = os.path.dirname(pdf_path)
    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    json_dir = os.path.join(pdf_dir, json_dir_name)
    os.makedirs(json_dir, exist_ok=True)
    return os.path.join(json_dir, pdf_base + ".json")


def save_json(data: dict, json_path: str, indent: int = 2) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def extract_and_save(pdf_path: str) -> str:
    """Parse an RfPB Stage 2 PDF and write the result to a JSON file.
    Returns the path of the saved JSON.
    """
    sections = extract_all_sections(pdf_path)
    json_path = pdf_path_to_json_path(pdf_path)
    save_json(sections, json_path)
    return json_path
