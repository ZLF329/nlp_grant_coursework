"""
Re-generate SCORING_READY for all existing JSON files without re-parsing PDFs.

Usage:
    python rebuild_scoring_ready.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make src importable
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from all_type_parser.scoring_ready import build_scoring_ready

JSON_DIR = _ROOT / "Data" / "json_data"
DATA_DIR = _ROOT / "Data"

UNIFIED_KEYS = {
    "SUMMARY INFORMATION",
    "LEAD APPLICANT & RESEARCH TEAM",
    "APPLICATION DETAILS",
    "SUMMARY BUDGET",
}


def rebuild(json_path: Path) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Extract unified_data (everything except SCORING_READY)
    unified_data = {k: v for k, v in data.items() if k != "SCORING_READY"}

    # Find the original source file (PDF or DOCX)
    stem = json_path.stem
    source_path = None
    for ext in (".pdf", ".docx", ".doc"):
        candidate = DATA_DIR / (stem + ext)
        if candidate.exists():
            source_path = candidate
            break

    if source_path is None:
        print(f"  [SKIP] no source file found for {stem}")
        return

    print(f"  rebuilding {stem} (source: {source_path.name}) ...", end=" ", flush=True)
    try:
        new_sr = build_scoring_ready(str(source_path), unified_data)
        data["SCORING_READY"] = new_sr
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        sr_kb = len(json.dumps(new_sr)) // 1000
        print(f"done  (SCORING_READY: {sr_kb}KB)")
    except Exception as e:
        print(f"FAILED: {e}")


def main() -> None:
    json_files = sorted(JSON_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON file(s) in {JSON_DIR}\n")
    for jf in json_files:
        rebuild(jf)
    print("\nAll done.")


if __name__ == "__main__":
    main()
