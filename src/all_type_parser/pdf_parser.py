from dataclasses import dataclass
from typing import List, Optional, Dict, Set
import pdfplumber
import re
import os
import json


TARGET_HEADINGS = [
    "Scientific Abstract",
    "Plain English Summary",
    "Changes from Previous Stage",
    "Working with People and Communities Summary"
]


def normalize_heading(s: str) -> str:
    """
    Minimal normalization:
    - strip whitespace
    - remove trailing colon
    - collapse internal whitespace
    """
    if s is None:
        return ""

    s = s.strip()
    s = re.sub(r"\s*:\s*$", "", s)     # remove trailing colon
    s = re.sub(r"\s+", " ", s)         # collapse whitespace
    return s


@dataclass
class Line:
    text: str
    page: int                # 0-index page number
    x0: float
    top: float
    x1: float
    bottom: float
    page_height: float


def extract_lines_pdfplumber(
    pdf_path: str,
    y_tolerance: float = 3.0,
    x_tolerance: float = 1.0,
    keep_empty: bool = False,
) -> List[Line]:
    """
    Extract lines from a text-based PDF using pdfplumber.
    Returns a list of Line objects sorted by (page, top, x0).

    y_tolerance controls how strictly words are grouped into the same line.
    """
    lines_out: List[Line] = []

    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages):
            # extract_words gives word-level boxes;
            # use tolerances for grouping words
            words = page.extract_words(
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
                keep_blank_chars=False,
                use_text_flow=True,
            )

            if not words:
                continue

            # Group words into lines by proximity of "top"
            # We'll quantize top into buckets using y_tolerance
            buckets: Dict[int, List[dict]] = {}
            for w in words:
                key = int(round(w["top"] / y_tolerance))
                buckets.setdefault(key, []).append(w)

            for _, ws in sorted(buckets.items(), key=lambda kv: kv[0]):
                # Sort words left->right
                ws_sorted = sorted(ws, key=lambda w: w["x0"])
                text = " ".join(w["text"] for w in ws_sorted).strip()

                if not text and not keep_empty:
                    continue

                x0 = min(w["x0"] for w in ws_sorted)
                top = min(w["top"] for w in ws_sorted)
                x1 = max(w["x1"] for w in ws_sorted)
                bottom = max(w["bottom"] for w in ws_sorted)

                lines_out.append(Line(text=text, page=pno, x0=x0, top=top,
                                      x1=x1, bottom=bottom,
                                      page_height=float(page.height)))

    # Final stable sort
    lines_out.sort(key=lambda line: (line.page, line.top, line.x0))
    return lines_out


def filter_by_fixed_y(
    lines,
    header_y: float = 79.0,
    footer_y: float = 791.0,
):
    out = []
    for ln in lines:
        if ln.bottom < header_y:
            continue  # header
        if ln.top > footer_y:
            continue  # footer
        out.append(ln)
    return out


# def split_sections_by_headings(
#     lines: List[Line],
#     target_headings: List[str],
#     join_with: str = "\n",
#     keep_heading_line: bool = False,
# ) -> Dict[str, str]:
#     """
#     Split lines into sections using exact heading matches
#     (line.text == heading after minimal normalization).

#     Parameters
#     ----------
#     lines:
#         Output from extract_lines_pdfplumber(), sorted by (page, top, x0).
#     target_headings:
#         The small headings you want to extract, e.g. ["Scientific Abstract",
#           "Plain English Summary", ...]
#     join_with:
#         How to join multiple lines into a section string.
#     keep_heading_line:
#         If True, include the heading line itself as the first line of the
#         section content.

#     Returns
#     -------
#     Dict[str, str]
#         Mapping: heading -> extracted section text
#         ("" if heading present but no content).
#         Headings that never appear are omitted.
#     """
#     # Normalize target headings once and preserve order
#     normalized_targets = [normalize_heading(h) for h in target_headings]
#     heading_set: Set[str] = set(normalized_targets)

#     # Initialize output with all headings set to None (missing by default)
#     collected: Dict[str, Optional[List[str]]] = {
#         h: None for h in normalized_targets
#     }

#     current_heading: Optional[str] = None

#     for ln in lines:
#         text_norm = normalize_heading(ln.text)

#         # If this line is a heading we care about, start collecting for it
#         if text_norm in heading_set:
#             current_heading = text_norm
#             collected[current_heading] = []
#             if keep_heading_line:
#                 collected[current_heading].append(ln.text)
#             continue

#         # If currently inside a section, collect text
#         if current_heading is not None:
#             if ln.text.strip():
#                 collected[current_heading].append(ln.text)

#     # Finalize output:
#     # - None stays None (heading never appeared)
#     # - [] becomes "" (appeared but empty)
#     # - list[str] becomes joined string
#     out: Dict[str, Optional[str]] = {}
#     for h, parts in collected.items():
#         if parts is None:
#             out[h] = None
#         else:
#             out[h] = join_with.join(parts).strip()

#     return out


def is_big_box_heading(line: Line) -> bool:
    text = line.text.strip()

    # 1. Must be all uppercase (non-letter characters ignored implicitly)
    if not text or not text.upper() == text:
        return False

    # 2. Line height must match the grey box title height
    line_height = line.bottom - line.top
    if line_height < 12.3 or line_height > 12.5:
        # threshold can be tuned based on samples
        return False

    return True


def find_big_box_ranges(lines: list[Line]) -> list[int]:
    """
    Return the line indices (in order) corresponding to all grey-box headings.
    """
    idxs = []
    for i, ln in enumerate(lines):
        if is_big_box_heading(ln):
            idxs.append(i)
    return idxs


def slice_big_box(lines, box_title):
    """
    Slice lines between a grey-box heading (box_title) and the next grey-box heading.
    Reuses the same big-box detection as slice_application_details().
    """
    big_boxes = find_big_box_ranges(lines)

    start_idx = None
    for idx in big_boxes:
        if lines[idx].text.strip() == box_title:
            start_idx = idx
            break

    if start_idx is None:
        raise ValueError(f"Cannot find big box heading: {box_title}")

    pos = big_boxes.index(start_idx)
    if pos + 1 >= len(big_boxes):
        raise ValueError(f"{box_title} is the last big box; cannot find the next one")

    start = start_idx + 1
    end = big_boxes[pos + 1]
    return lines[start:end]


# def slice_application_details(lines: list[Line]) -> list[Line]:
#     big_boxes = find_big_box_ranges(lines)

#     # Find the index (in `lines`) corresponding
#     #  to the APPLICATION DETAILS grey box
#     app_idx_in_lines = None
#     for idx in big_boxes:
#         if lines[idx].text.strip() == "APPLICATION DETAILS":
#             app_idx_in_lines = idx
#             break

#     if app_idx_in_lines is None:
#         raise ValueError("Cannot find big box heading: APPLICATION DETAILS")

#     # Find its position in the big_boxes list and use the next
#     # grey box as the end boundary
#     try:
#         pos = big_boxes.index(app_idx_in_lines)
#     except ValueError:
#         # This should never happen (app_idx_in_lines comes from big_boxes)
#         raise ValueError("Internal error:" +
#                          " APPLICATION DETAILS index not in big_boxes")

#     if pos + 1 >= len(big_boxes):
#         raise ValueError("APPLICATION DETAILS is the last big box;" +
#                          " cannot find the next one")

#     start = app_idx_in_lines + 1
#     end = big_boxes[pos + 1]

#     return lines[start:end]


def parse_summary_information(lines):
    """
    Parse the 'SUMMARY INFORMATION' block into a structured dict.
    Handles cases where PDF merges multiple table cells into one text line.
    """

    def add_kv(out, k, v):
        if k in out:
            if isinstance(out[k], list):
                out[k].append(v)
            else:
                out[k] = [out[k], v]
        else:
            out[k] = v

    def is_noise_line(text):
        t = text.strip()
        return t in {"Costs"} or t.startswith("Partner Organisation")

    # Keys that should NOT accept continuation lines
    no_continuation_keys = {
        "Has this application been previously submitted to this or any other funding body?",
        "Duration (months)",
        "Start Date",
        "End Date",
    }

    # Regex to capture currency like "£335,383.93" or "£0.00"
    money_re = re.compile(r"(£\s?\d[\d,]*\.\d{2})")

    question_key = "Has this application been previously submitted to this or any other funding body?"
    question_anchor = "Has this application been"

    out = {}
    current_key = None

    for ln in lines:
        t = ln.text.strip()
        if not t or is_noise_line(t):
            continue

        # 1) Detect the yes/no question even if it appears mid-line (merged cells)
        if question_anchor in t:
            # Take the last token after the anchor as Yes/No if present
            # Example merged line: "... £0.00 Has this application been No previously ..."
            # We'll search for " Has this application been " then parse the next token as Yes/No.
            idx = t.find(question_anchor)
            tail = t[idx:].split()
            value = ""
            # tail looks like ["Has","this","application","been","No", ...]
            if len(tail) >= 5:
                cand = tail[4].strip()
                if cand in {"Yes", "No"}:
                    value = cand
                else:
                    # Fallback: last token on that line
                    last = tail[-1].strip()
                    if last in {"Yes", "No"}:
                        value = last

            if value:
                add_kv(out, question_key, value)
                current_key = question_key
            else:
                # If we can't confidently extract, still set key with empty value
                add_kv(out, question_key, "")
                current_key = question_key

            # Important: do NOT continue appending this line to some previous key
            # Also: we intentionally do not parse anything before the anchor here.
            continue

        # 2) Fix split label case: "Contracting Organisation Department"
        if t == "Contracting Organisation Department":
            current_key = "Department"
            if "Department" not in out:
                out["Department"] = ""
            continue

        # 3) Treatment costs keys: only keep the currency value (avoid merged junk)
        if t.startswith("NHS Excess Treatment"):
            m = money_re.search(t)
            value = m.group(1).replace(" ", "") if m else t[len("NHS Excess Treatment") :].strip()
            add_kv(out, "NHS Excess Treatment Costs", value)
            current_key = "NHS Excess Treatment Costs"
            continue

        if t.startswith("Non-NHS Excess Treatment"):
            m = money_re.search(t)
            value = m.group(1).replace(" ", "") if m else t[len("Non-NHS Excess Treatment") :].strip()
            add_kv(out, "Non-NHS Excess Treatment Costs", value)
            current_key = "Non-NHS Excess Treatment Costs"
            continue

        # 4) Other standard keys
        standard_keys = [
            "Application Title",
            "Contracting Organisation",
            "Department",
            "Start Date",
            "End Date",
            "Duration (months)",
            "Total Cost to NIHR",
            "NHS Support Costs",
        ]

        matched = False
        for k in standard_keys:
            if t == k or t.startswith(k + " "):
                value = t[len(k) :].strip()
                # Duration must be an integer; truncate any merged junk after it
                if k == "Duration (months)":
                    m = re.match(r"\d+", value)
                    value = m.group(0) if m else value
                # Contracting Organisation: stop at "Partner Organisation" if merged
                elif k == "Contracting Organisation":
                    partner_idx = value.find("Partner Organisation")
                    if partner_idx > 0:
                        value = value[:partner_idx].strip()
                add_kv(out, k, value)
                current_key = k
                matched = True
                break

        if matched:
            continue

        # 5) Continuation lines
        if current_key is not None and current_key not in no_continuation_keys:
            if isinstance(out.get(current_key), list):
                out[current_key][-1] = (out[current_key][-1] + " " + t).strip()
            else:
                out[current_key] = (out.get(current_key, "") + " " + t).strip()

    # 6) Post-fix: if Department is empty but Contracting Organisation is list of 2
    if (
        isinstance(out.get("Contracting Organisation"), list)
        and len(out["Contracting Organisation"]) == 2
        and (not out.get("Department"))
    ):
        out["Department"] = out["Contracting Organisation"][1]
        out["Contracting Organisation"] = out["Contracting Organisation"][0]

    return out


def parse_application_details(
    lines: List[Line],
    join_with: str = "\n",
    keep_heading_line: bool = False,
) -> Dict[str, str]:
    """
    Split lines into sections using exact heading matches
    (line.text == heading after minimal normalization).

    Parameters
    ----------
    lines:
        Output from extract_lines_pdfplumber(), sorted by (page, top, x0).
    target_headings:
        The small headings you want to extract, e.g. ["Scientific Abstract",
          "Plain English Summary", ...]
    join_with:
        How to join multiple lines into a section string.
    keep_heading_line:
        If True, include the heading line itself as the first line of the
        section content.

    Returns
    -------
    Dict[str, str]
        Mapping: heading -> extracted section text
        ("" if heading present but no content).
        Headings that never appear are omitted.
    """
    target_headings = [
    "Scientific Abstract",
    "Plain English Summary",
    "Changes from Previous Stage",
    "Working with People and Communities Summary"
    ]

    # Normalize target headings once and preserve order
    normalized_targets = [normalize_heading(h) for h in target_headings]
    heading_set: Set[str] = set(normalized_targets)

    # Initialize output with all headings set to None (missing by default)
    collected: Dict[str, Optional[List[str]]] = {
        h: None for h in normalized_targets
    }

    current_heading: Optional[str] = None

    for ln in lines:
        text_norm = normalize_heading(ln.text)

        # If this line is a heading we care about, start collecting for it
        if text_norm in heading_set:
            current_heading = text_norm
            collected[current_heading] = []
            if keep_heading_line:
                collected[current_heading].append(ln.text)
            continue

        # If currently inside a section, collect text
        if current_heading is not None:
            if ln.text.strip():
                collected[current_heading].append(ln.text)

    # Finalize output:
    # - None stays None (heading never appeared)
    # - [] becomes "" (appeared but empty)
    # - list[str] becomes joined string
    out: Dict[str, Optional[str]] = {}
    for h, parts in collected.items():
        if parts is None:
            out[h] = None
        else:
            out[h] = join_with.join(parts).strip()

    return out


def parse_lead_applicant_research_team(lines):
    """
    Parse the 'LEAD APPLICANT & RESEARCH TEAM' block into structured data.

    Expected structure:
      - Section labels: 'Lead Applicant', 'Joint Lead Applicant', 'Co-Applicants'
      - Repeated fields per person:
        Full Name, Organisation, Department, Proposed Role, ORCID, % FTE Commitment
    """
    section_labels = {"Lead Applicant", "Joint Lead Applicant", "Co-Applicants"}
    field_labels = [
        "Full Name",
        "Organisation",
        "Department",
        "Proposed Role",
        "ORCID",
        "% FTE Commitment",
    ]

    # Fields that must not accumulate continuation lines (single-line values only)
    no_continuation_fields = {"ORCID", "% FTE Commitment"}

    # Regex for a valid ORCID identifier (xxxx-xxxx-xxxx-xxxx, last char may be X)
    orcid_re = re.compile(r'\d{4}-\d{4}-\d{4}-\d{3}[\dX]')

    def starts_with_field(text):
        for f in field_labels:
            if text == f or text.startswith(f + " "):
                return f
        return None

    out = {
        "Lead Applicant": None,
        "Joint Lead Applicant": None,
        "Co-Applicants": [],
    }

    current_category = None
    current_person = None
    current_field = None

    def flush_person():
        nonlocal current_person, current_field
        if current_person is None:
            return

        for k, v in list(current_person.items()):
            if isinstance(v, str):
                current_person[k] = v.strip()

        if current_category in ("Lead Applicant", "Joint Lead Applicant"):
            out[current_category] = current_person
        elif current_category == "Co-Applicants":
            out["Co-Applicants"].append(current_person)

        current_person = None
        current_field = None

    for ln in lines:
        t = ln.text.strip()
        if not t:
            continue

        # New section label
        if t in section_labels:
            flush_person()
            current_category = t
            current_person = None
            current_field = None
            continue

        if current_category is None:
            continue

        f = starts_with_field(t)
        if f is not None:
            # In Co-Applicants, "Full Name" indicates a new person entry
            if (
                current_category == "Co-Applicants"
                and f == "Full Name"
                and current_person is not None
                and current_person.get("Full Name")
            ):
                flush_person()

            if current_person is None:
                current_person = {}

            value = t[len(f):].strip()
            # ORCID: extract only the standard xxxx-xxxx-xxxx-xxxx pattern
            if f == "ORCID":
                m = orcid_re.match(value)
                value = m.group(0) if m else value.split()[0] if value else ""
            # % FTE Commitment: keep only the first token (the percentage)
            elif f == "% FTE Commitment":
                value = value.split()[0] if value else ""
            current_person[f] = value
            current_field = f
            continue

        # Continuation line for the previous field (most commonly Proposed Role)
        if current_person is not None and current_field is not None \
                and current_field not in no_continuation_fields:
            # sep = "\n" if current_field == "Proposed Role" else " "
            sep = " "
            current_person[current_field] = (
                (current_person.get(current_field, "").rstrip() + sep + t).strip()
            )

    flush_person()
    return out


def parse_other_big_box(
    lines,
    join_with="\n",
    keep_heading_line=False,
):
    """
    Parse a generic big-box section by concatenating all text lines.

    Parameters
    ----------
    lines : List[Line]
        Lines belonging to a single big-box section.
    join_with : str
        String used to join lines (e.g. "\\n" or " ").
    keep_heading_line : bool
        Whether to keep the first line (often a sub-heading like
        'Justification of Costs') in the output.

    Returns
    -------
    str
        Concatenated text of the section.
    """
    texts = []

    for i, ln in enumerate(lines):
        t = ln.text.strip()
        if not t:
            continue

        # Optionally drop the first line (usually a local heading)
        if i == 0 and not keep_heading_line:
            continue

        texts.append(t)

    return join_with.join(texts)


def list_big_box_titles(lines):
    """
    Return big-box titles (grey box headings) in the order they appear.
    """
    idxs = find_big_box_ranges(lines)
    titles = []
    for i in idxs:
        t = lines[i].text.strip()
        if t:
            titles.append(t)
    return titles


def extract_all_big_box_sections(
    pdf_path,
    default_join_with=" ",
    default_keep_heading_line=False,
):
    """
    Extract ALL big-box sections from a NIHR PDF into a dict.

    Parsing strategy:
    - SUMMARY INFORMATION -> parse_summary_information
    - LEAD APPLICANT & RESEARCH TEAM -> parse_lead_applicant_research_team
    - APPLICATION DETAILS -> split by small headings (parse_application_details)
    - Everything else -> parse_other_big_box (join all text)
    """
    # Step 1: extract all lines
    lines = extract_lines_pdfplumber(pdf_path)

    # Step 2: remove header/footer
    lines = filter_by_fixed_y(lines)

    # Step 3: discover all big-box titles
    box_titles = list_big_box_titles(lines)

    out = {}

    for title in box_titles:
        # Slice lines for this big box
        try:
            block_lines = slice_big_box(lines, title)
        except Exception:
            # If slicing fails, skip (should be rare)
            continue

        # Dispatch to the right parser
        if title == "SUMMARY INFORMATION":
            out[title] = parse_summary_information(block_lines)

        elif title == "LEAD APPLICANT & RESEARCH TEAM":
            out[title] = parse_lead_applicant_research_team(block_lines)

        elif title == "APPLICATION DETAILS":
            # Use your existing splitter-based parser
            out[title] = parse_application_details(
                block_lines,
                join_with="\n",
                keep_heading_line=default_keep_heading_line,
            )

        else:
            # Fallback: just join all text
            out[title] = parse_other_big_box(
                block_lines,
                join_with=default_join_with,
                keep_heading_line=default_keep_heading_line,
            )

    return out


# def extract_application_detail_sections(
#     pdf_path: str,
#     target_headings: list[str] = TARGET_HEADINGS,
# ) -> dict[str, str]:
#     """
#     Extract specified sections from the
#     'APPLICATION DETAILS' block of a NIHR PDF.

#     The extraction pipeline is:
#     1. Extract all lines from the PDF.
#     2. Remove header and footer lines using fixed y-coordinate thresholds.
#     3. Slice the lines to keep only the content between the
#        'APPLICATION DETAILS' grey box and the next grey box.
#     4. Split the remaining lines by small section headings.

#     Parameters
#     ----------
#     pdf_path:
#         Path to the PDF file.
#     target_headings:
#         List of small section headings to extract
#         (e.g. 'Scientific Abstract', 'Plain English Summary').

#     Returns
#     -------
#     dict[str, str]
#         Mapping from section heading to extracted text.
#         Sections that do not appear in the document are omitted.
#         Sections that appear but have no content map to an empty string.
#     """
#     # Step 1: extract all lines from the PDF
#     lines = extract_lines_pdfplumber(pdf_path)

#     # Step 2: remove header and footer using fixed geometry
#     lines = filter_by_fixed_y(lines)

#     # Step 3: keep only the APPLICATION DETAILS block
#     app_lines = slice_application_details(lines)

#     # Step 4: split by small headings within APPLICATION DETAILS
#     sections = split_sections_by_headings(
#         app_lines,
#         target_headings=target_headings,
#     )

#     return sections


def pdf_path_to_json_path(pdf_path: str,
                          json_dir_name: str = "json_data") -> str:
    """
    Convert a PDF path to a JSON path under a `json_data` subdirectory.

    Example:
        D:\\msc_AI\\SWE_group_project\\data\\IC00494_after.pdf
        ->
        D:\\msc_AI\\SWE_group_project\\data\\json_data\\IC00494_after.json
    """
    pdf_dir = os.path.dirname(pdf_path)
    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]

    json_dir = os.path.join(pdf_dir, json_dir_name)
    os.makedirs(json_dir, exist_ok=True)

    return os.path.join(json_dir, pdf_base + ".json")


# def build_application_details_json(sections: dict[str, str]) -> dict:
#     return {
#         "APPLICATION DETAILS": sections
#     }


def save_json(data: dict, json_path: str, indent: int = 2) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def extract_and_save_all_big_boxes(
    pdf_path
):
    """
    Extract all big-box sections from a PDF and save to JSON.
    """
    # Step 1: extract all big-box sections
    sections = extract_all_big_box_sections(
        pdf_path=pdf_path
    )

    # Step 2: determine JSON output path
    json_path = pdf_path_to_json_path(pdf_path)

    # Step 3: save directly (sections is already a valid JSON object)
    save_json(sections, json_path)

    return json_path


# def extract_and_save_application_details(
#     pdf_path: str,
#     target_headings: list[str] = TARGET_HEADINGS,
# ) -> str:
#     """
#     Extract APPLICATION DETAILS sections from a PDF and
#     save them as a JSON file under a `json_data` subdirectory next to the PDF.
#     """
#     # Step 1: extract sections
#     sections = extract_application_detail_sections(
#         pdf_path=pdf_path,
#         target_headings=target_headings,
#     )

#     # Step 2: wrap JSON structure
#     json_data = build_application_details_json(sections)

#     # Step 3: determine JSON output path (json_data folder)
#     json_path = pdf_path_to_json_path(pdf_path)

#     # Step 4: save
#     save_json(json_data, json_path)

#     return json_path
