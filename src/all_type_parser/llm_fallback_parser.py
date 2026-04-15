"""
LLM Fallback Parser
===================
Last-resort parser for grant application PDFs that do not match any known
structural format (no blue-boxes, non-standard layout, scanned pages, etc.).

Strategy
--------
1. Try pdfplumber text extraction (fast, no GPU).
   - If text is rich enough (>= MIN_CHARS_PER_PAGE avg), send directly to LLM.
2. If text is sparse (scanned PDF), render pages to images via pdf2image and
   send them to glm-ocr (Ollama) for OCR, then concatenate the results.
3. Feed the extracted text to qwen3.5:27b (Ollama) and ask it to return a
   structured JSON matching the unified grant-application schema.

Environment variables (all optional — defaults match qwen3_ollama.py):
    OLLAMA_HOST          http://127.0.0.1:11434
    OLLAMA_MODEL         qwen3.5:27b      (structuring LLM)
    OLLAMA_OCR_MODEL     glm-ocr          (vision OCR model)
    OLLAMA_TIMEOUT       1200             (seconds)

Dependencies
------------
    pip install pdfplumber requests
    pip install pdf2image          # only needed for scanned PDFs
    # pdf2image also requires poppler on PATH:
    #   Ubuntu/Debian : apt install poppler-utils
    #   macOS         : brew install poppler
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Optional

import pdfplumber
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_HOST      = os.environ.get("OLLAMA_HOST",      "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL     = os.environ.get("OLLAMA_MODEL",     "qwen3.5:27b")
OLLAMA_OCR_MODEL = os.environ.get("OLLAMA_OCR_MODEL", "glm-ocr")
OLLAMA_TIMEOUT   = float(os.environ.get("OLLAMA_TIMEOUT", "1200"))

# Minimum average characters per page to skip OCR and use pdfplumber text directly
# Set high so we always use image OCR (glm-ocr) to capture table content
MIN_CHARS_PER_PAGE: int = 99999

# Page limits (token / cost guard)
MAX_PAGES_TEXT:  int = 60   # max pages whose text we send to the LLM
MAX_PAGES_IMAGE: int = 20   # max pages we OCR when in image mode

# DPI for pdf2image rendering
IMAGE_DPI: int = 150


# ---------------------------------------------------------------------------
# Stage 1 — text extraction via pdfplumber
# ---------------------------------------------------------------------------

def _extract_text_pdfplumber(pdf_path: str) -> tuple[str, float]:
    """
    Extract all text from a PDF using pdfplumber.

    Returns
    -------
    text : str
        Concatenated page texts (up to MAX_PAGES_TEXT pages).
    avg_chars_per_page : float
        Average character count per page — used to detect scanned PDFs.
    """
    pages_text: list[str] = []
    total_chars = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages[:MAX_PAGES_TEXT]:
            t = page.extract_text() or ""
            pages_text.append(t)
            total_chars += len(t)

    avg = total_chars / total_pages if total_pages else 0.0
    return "\n\n".join(pages_text), avg


def _extract_text_docx(docx_path: str) -> str:
    """
    Extract all text from a DOCX file using python-docx.

    Paragraphs and table cells are concatenated in document order.
    Returns a single string ready to be sent to the LLM.
    """
    from docx import Document

    doc = Document(docx_path)
    parts: list[str] = []

    # Paragraphs in document order
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)

    # Tables (append after paragraphs to preserve rough document order)
    for table in doc.tables:
        for row in table.rows:
            row_text = "\t".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                parts.append(row_text)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 2 — image OCR via glm-ocr (Ollama)
# ---------------------------------------------------------------------------

def _pdf_pages_to_base64(pdf_path: str, max_pages: int = MAX_PAGES_IMAGE) -> list[str]:
    """
    Render PDF pages to PNG images and return them as base64-encoded strings.
    Requires pdf2image and poppler on PATH.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise ImportError(
            "pdf2image is required for scanned PDF support. "
            "Install it with:  pip install pdf2image"
        ) from exc

    images = convert_from_path(pdf_path, dpi=IMAGE_DPI, last_page=max_pages)
    result: list[str] = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return result


def _ocr_image_ollama(b64_image: str, page_num: int) -> str:
    """
    Send a single base64-encoded image to glm-ocr via Ollama and return the
    raw extracted text.
    """
    payload = {
        "model": OLLAMA_OCR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Please extract all text from this image exactly as it appears, "
                    "preserving paragraph breaks and layout structure. "
                    "Output plain text only — no commentary."
                ),
                "images": [b64_image],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
        return (body.get("message") or {}).get("content") or ""
    except Exception as e:
        print(f"[llm_fallback_parser] OCR failed for page {page_num}: {e}")
        return ""


def _ocr_pdf(pdf_path: str) -> str:
    """
    Convert PDF pages to images, OCR each one with glm-ocr, and concatenate
    the results into a single text string.
    """
    print(
        f"[llm_fallback_parser] sparse text detected — "
        f"running image OCR (up to {MAX_PAGES_IMAGE} pages)"
    )
    b64_images = _pdf_pages_to_base64(pdf_path, max_pages=MAX_PAGES_IMAGE)
    page_texts: list[str] = []
    for i, b64 in enumerate(b64_images, 1):
        print(f"[llm_fallback_parser]   OCR page {i}/{len(b64_images)} …")
        text = _ocr_image_ollama(b64, page_num=i)
        if text.strip():
            page_texts.append(text)
    return "\n\n".join(page_texts)


# ---------------------------------------------------------------------------
# Stage 3 — structured extraction via qwen3.5:27b
# ---------------------------------------------------------------------------

# JSON schema for Ollama constrained generation
_UNIFIED_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string"},
        "SUMMARY INFORMATION": {
            "type": "object",
            "properties": {
                "Project Title":      {"type": "string"},
                "Chief Investigator": {"type": "string"},
                "Organisation":       {"type": "string"},
                "Start Date":         {"type": "string"},
                "Duration":           {"type": "string"},
                "Total Cost":         {"type": "string"},
            },
        },
        "LEAD APPLICANT & RESEARCH TEAM": {
            "type": "object",
            "properties": {
                "Lead Applicant": {
                    "type": "object",
                    "properties": {
                        "Full Name":        {"type": "string"},
                        "Organisation":     {"type": "string"},
                        "Department":       {"type": "string"},
                        "Proposed Role":    {"type": "string"},
                        "ORCID":            {"type": "string"},
                        "% FTE Commitment": {"type": "string"},
                    },
                },
                "Joint Lead Applicant": {"type": ["object", "null"]},
                "Co-Applicants": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Full Name":     {"type": "string"},
                            "Proposed Role": {"type": "string"},
                            "Organisation":  {"type": "string"},
                            "Department":    {"type": "string"},
                            "ORCID":         {"type": "string"},
                            "Position":      {"type": "string"},
                        },
                    },
                },
            },
        },
        "APPLICATION DETAILS": {
            "type": "object",
            "properties": {
                "Plain English Summary of Research":          {"type": "string"},
                "Scientific Abstract":                        {"type": "string"},
                "Applicant CV":                               {"type": "string"},
                "Applicant Research Background":              {"type": "string"},
                "Detailed Research Plan":                     {"type": "string"},
                "Patient & Public Involvement":               {"type": "string"},
                "Training & Development and Research Support": {"type": "string"},
            },
        },
        "SUMMARY BUDGET": {"type": "string"},
    },
}

_SYSTEM_PROMPT = """\
You are an expert at extracting structured information from grant application documents.
Given raw text from a grant application PDF, extract the information and return it as
valid JSON that matches the provided schema.

Rules:
- Only include fields for which you found actual content in the text.
- Do NOT invent or hallucinate any information.
- For string fields, copy or faithfully summarise the relevant passage.
- If a section is entirely absent from the text, omit that key entirely.
- Return ONLY the JSON object — no preamble, no markdown fences, no explanation.
"""

_USER_PROMPT_TEMPLATE = """\
Extract structured grant application information from the text below and return it as JSON.

--- BEGIN DOCUMENT ---
{text}
--- END DOCUMENT ---
"""


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL).strip()


def _extract_json_object(text: str) -> str:
    """Pull the outermost { … } JSON object from a string."""
    clean = (text or "").strip()
    first = clean.find("{")
    last  = clean.rfind("}")
    if first != -1 and last != -1 and last > first:
        return clean[first : last + 1]
    return clean


def _structure_with_llm(text: str) -> dict:
    """
    Send extracted text to qwen3.5:27b via Ollama and parse the structured
    JSON response into a Python dict.

    Uses Ollama's `format` parameter for constrained JSON generation,
    matching the pattern in qwen3_ollama.py.
    """
    # Guard against extremely long documents exceeding context window
    user_text = text[:120_000]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_PROMPT_TEMPLATE.format(text=user_text)},
        ],
        "stream": False,
        "format": _UNIFIED_SCHEMA,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 4096,
        },
        "think": False,
    }

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_HOST}. "
            "Make sure the server is running or set OLLAMA_HOST correctly."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Timed out waiting for Ollama at {OLLAMA_HOST}. "
            "Try increasing OLLAMA_TIMEOUT."
        ) from exc

    body = resp.json()
    raw = (body.get("message") or {}).get("content") or ""
    raw = _strip_think_tags(raw)
    raw = _extract_json_object(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[llm_fallback_parser] JSON parse error: {e}")
        print(f"[llm_fallback_parser] raw LLM output (first 500 chars): {raw[:500]}")
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_all_sections(input_path: str) -> dict:
    """
    Extract all grant-application sections from a PDF or DOCX file and return
    a unified JSON dict matching the IC00458_after.json schema.

    Pipeline for PDF
    ----------------
    1. pdfplumber text extraction
       - Rich text  (avg >= MIN_CHARS_PER_PAGE) → feed directly to LLM
       - Sparse text (scanned PDF)              → glm-ocr → feed to LLM
    2. qwen3.5:27b structured extraction → unified dict

    Pipeline for DOCX (and other non-PDF files)
    --------------------------------------------
    1. python-docx text extraction (paragraphs + tables)
    2. qwen3.5:27b structured extraction → unified dict
       (image OCR step is skipped — DOCX has selectable text by definition)

    Parameters
    ----------
    input_path : str
        Absolute or relative path to the input file (PDF or DOCX).

    Returns
    -------
    dict
        Unified JSON dict.  Empty dict if extraction failed completely.
    """
    ext = Path(input_path).suffix.lower()
    print(f"[llm_fallback_parser] processing: {input_path}")

    # ── DOCX branch: extract text with python-docx, skip image OCR ───────────
    if ext in (".docx", ".doc"):
        try:
            text = _extract_text_docx(input_path)
        except Exception as e:
            print(f"[llm_fallback_parser] python-docx extraction failed: {e}")
            return {}

        if not text.strip():
            print("[llm_fallback_parser] no text extracted from DOCX — giving up")
            return {}

        print(f"[llm_fallback_parser] DOCX: extracted {len(text):,} chars")

    # ── PDF branch: pdfplumber → (glm-ocr if sparse) ─────────────────────────
    else:
        # Stage 1: pdfplumber text extraction
        try:
            text, avg_chars = _extract_text_pdfplumber(input_path)
        except Exception as e:
            print(f"[llm_fallback_parser] pdfplumber failed: {e}")
            text, avg_chars = "", 0.0

        print(f"[llm_fallback_parser] avg chars/page: {avg_chars:.0f}")

        # Stage 2: image OCR if text is too sparse
        if avg_chars < MIN_CHARS_PER_PAGE:
            try:
                ocr_text = _ocr_pdf(input_path)
                if ocr_text.strip():
                    text = ocr_text
            except Exception as e:
                print(f"[llm_fallback_parser] image OCR failed: {e}")
                # Fall through — use whatever pdfplumber gave us (may be empty)

        if not text.strip():
            print("[llm_fallback_parser] no text available — giving up")
            return {}

    # ── Stage 3: structured extraction via LLM (both branches) ───────────────
    print(f"[llm_fallback_parser] sending {len(text):,} chars to {OLLAMA_MODEL} …")
    try:
        result = _structure_with_llm(text)
    except Exception as e:
        print(f"[llm_fallback_parser] LLM structuring failed: {e}")
        return {}

    if result:
        result.setdefault("doc_type", "llm_fallback")
        print("[llm_fallback_parser] ✓ extraction complete")
    else:
        print("[llm_fallback_parser] ✗ LLM returned empty result")

    return result
