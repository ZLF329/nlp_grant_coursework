"""
All-Type Parser

Entry point for parsing any grant application file into the unified JSON format.

Pipeline for PDF files:
  1. fellowships_parser  — blue-box fellowship format (NIHR DF/AF)
     → if result is non-empty, done.
  2. pdf_parser          — generic big-box PDF format
     → if result is non-empty, done.
  3. document_parser + all_other_parser  — last-resort fallback (see below)

Pipeline for DOCX / other files:
  3. document_parser     — extracts a raw sections JSON
     all_other_parser    — maps it to the unified format

The output JSON always uses the same top-level keys as IC00458_after.json.
Only keys for which content was found are included.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# ── path setup so sub-parsers can be imported regardless of cwd ──────────────
_SRC = Path(__file__).resolve().parent.parent   # …/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ──────────────────────────── helpers ────────────────────────────────────────

def _is_empty(result: dict) -> bool:
    """Return True if the parser produced nothing useful."""
    if not result:
        return True
    # A non-empty dict must contain at least one non-empty value
    return all(
        (not v) or (isinstance(v, dict) and not v) or (isinstance(v, list) and not v)
        for v in result.values()
    )


def _json_output_path(input_path: str) -> str:
    """Derive the final JSON output path next to the input file."""
    p = Path(input_path)
    json_dir = p.parent / "json_data"
    json_dir.mkdir(exist_ok=True)
    return str(json_dir / (p.stem + ".json"))


def _save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ──────────────────────────── stage 1 — fellowships_parser ───────────────────

def _try_fellowships_parser(pdf_path: str) -> dict:
    try:
        from .fellowships_parser import extract_all_sections
        result = extract_all_sections(pdf_path)
        return result if result else {}
    except Exception as e:
        print(f"[all_type_parser] fellowships_parser failed: {e}")
        return {}


# ──────────────────────────── stage 2 — RfPB_parser ─────────────────────────

def _try_rfpb_parser(pdf_path: str) -> dict:
    try:
        from .RfPB_parser import extract_all_sections
        result = extract_all_sections(pdf_path)
        return result if result else {}
    except Exception as e:
        print(f"[all_type_parser] RfPB_parser failed: {e}")
        return {}


# ──────────────────────────── stage 3 — pdf_parser ───────────────────────────

def _try_pdf_parser(pdf_path: str) -> dict:
    try:
        from .pdf_parser import extract_all_big_box_sections
        result = extract_all_big_box_sections(pdf_path)
        return result if result else {}
    except Exception as e:
        print(f"[all_type_parser] pdf_parser failed: {e}")
        return {}


# ──────────────────────────── stage 3 — document_parser + all_other_parser ───

def _try_document_then_all_other(input_path: str) -> dict:
    """
    Parse with HybridDocumentParser → produce a temporary sections JSON →
    convert with all_other_parser → return unified dict.
    """
    try:
        from document_parser import HybridDocumentParser
        from .all_other_parser import convert_to_unified_format

        parser = HybridDocumentParser()
        parsed = parser.parse(input_path)

        # Serialise ParsedDocument to the sections-JSON dict expected by
        # convert_to_unified_format  (same schema as IC00009_after.json)
        sections_json = {
            "file_name": parsed.file_name,
            "file_type": parsed.file_type,
            "sections": [
                {
                    "title": sec.title,
                    "type":  sec.type,
                    "content": sec.content,
                }
                for sec in parsed.sections
            ],
        }

        # Write intermediate sections JSON, convert, then delete it
        p = Path(input_path)
        tmp_sections_path = p.parent / "json_data" / (p.stem + "_sections_tmp.json")
        tmp_sections_path.parent.mkdir(exist_ok=True)
        _save_json(sections_json, str(tmp_sections_path))

        result = convert_to_unified_format(sections_json)

        tmp_sections_path.unlink(missing_ok=True)

        return result if result else {}
    except Exception as e:
        print(f"[all_type_parser] document_parser/all_other_parser failed: {e}")
        return {}


# ──────────────────────────── RfPB pre-check ─────────────────────────────────

def _is_rfpb_pdf(pdf_path: str, n_lines: int = 2) -> bool:
    """
    Return True if the first `n_lines` text lines of the PDF contain 'RfPB'
    (case-sensitive).  Uses pdfplumber on page 1 only — fast, no full parse.
    """
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return False
            text = pdf.pages[0].extract_text() or ""
        first_lines = "\n".join(text.splitlines()[:n_lines])
        # print(first_lines)
        return "RfPB" in first_lines
    except Exception:
        return False


# ──────────────────────────── public API ─────────────────────────────────────

def parse(input_path: str) -> dict:
    """
    Parse any grant application file and return the unified JSON dict.

    For PDF files the three-stage pipeline is tried in order.
    For non-PDF files only stage 3 is used.
    """
    ext = Path(input_path).suffix.lower()

    if ext == ".pdf":
        # Fast pre-check: if page 1 mentions "RfPB", go straight to RfPB parser
        if _is_rfpb_pdf(input_path):
            print("[all_type_parser] detected RfPB PDF — using RfPB_parser directly")
            result = _try_rfpb_parser(input_path)
            if not _is_empty(result):
                print("[all_type_parser] ✓ RfPB_parser succeeded")
                return result
            print("[all_type_parser] RfPB_parser returned empty — falling back to document_parser")
        else:
            # Stage 1: fellowship blue-box parser
            result = _try_fellowships_parser(input_path)
            if not _is_empty(result):
                print("[all_type_parser] ✓ fellowships_parser succeeded")
                return result

            # Stage 2: generic big-box PDF parser
            result = _try_pdf_parser(input_path)
            if not _is_empty(result):
                print("[all_type_parser] ✓ pdf_parser succeeded")
                return result

            # Stage 3: RfPB Stage 2 parser (fallback for non-RfPB PDFs)
            result = _try_rfpb_parser(input_path)
            if not _is_empty(result):
                print("[all_type_parser] ✓ RfPB_parser succeeded")
                return result

            print("[all_type_parser] all PDF parsers returned empty — falling back to document_parser")

    # Stage 3: document_parser + all_other_parser (covers DOCX, other, and PDF fallback)
    result = _try_document_then_all_other(input_path)
    if not _is_empty(result):
        print("[all_type_parser] ✓ document_parser + all_other_parser succeeded")
    else:
        print("[all_type_parser] ✗ all parsers returned empty")
    return result


def parse_and_save(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Parse a file and write the unified JSON to disk.
    Returns the output JSON path.
    """
    result = parse(input_path)
    out = output_path or _json_output_path(input_path)
    _save_json(result, out)
    print(f"[all_type_parser] saved → {out}")
    return out


def parse_folder(folder_path: str, extensions: tuple = (".pdf", ".docx", ".doc")) -> list:
    """
    Parse all supported files in a folder and save each result to json_data/.
    Returns a list of output JSON paths.

    Parameters
    ----------
    folder_path : str
        Directory to scan (non-recursive).
    extensions : tuple
        File extensions to include. Defaults to pdf, docx, doc.
    """
    folder = Path(folder_path)
    files = [f for f in sorted(folder.iterdir()) if f.is_file() and f.suffix.lower() in extensions]

    if not files:
        print(f"[all_type_parser] no supported files found in {folder_path}")
        return []

    print(f"[all_type_parser] found {len(files)} file(s) in {folder_path}")
    saved: list = []
    for i, f in enumerate(files, 1):
        print(f"\n[all_type_parser] [{i}/{len(files)}] {f.name}")
        try:
            out = parse_and_save(str(f))
            saved.append(out)
        except Exception as e:
            print(f"[all_type_parser] ✗ failed: {e}")

    print(f"\n[all_type_parser] done — {len(saved)}/{len(files)} saved to json_data/")
    return saved


# ──────────────────────────── CLI ────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    # Run as a module so relative imports work:
    #   python -m all_type_parser.all_type_parser <file>
    if len(_sys.argv) < 2:
        print("Usage: python -m all_type_parser.all_type_parser <input_file> [output.json]")
        _sys.exit(1)

    out = parse_and_save(_sys.argv[1], _sys.argv[2] if len(_sys.argv) > 2 else None)
    print(f"Done: {out}")
