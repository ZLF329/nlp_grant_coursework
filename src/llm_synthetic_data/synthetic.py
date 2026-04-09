"""
Synthetic data generator for NIHR grant feature JSONs.

Uses a local Ollama LLM (via the feature_extractions package) to produce
structurally identical synthetic feature files conditioned on funding outcome.

Usage
-----
    python -m llm_synthetic_data.synthetic \
        --input  data/features/           \
        --output data/synthetic/          \
        --host   http://localhost:11434   \
        --model  qwen2.5:14b             \
        --num    5
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from feature_extractions.clients.ollama_client import (
    OllamaClient,
    OllamaConfig,
    extract_json_object,
)
from feature_extractions.utils.io_utils import iter_json_files, load_json, write_json

log = logging.getLogger(__name__)

# Feature sections that carry criteria + overall
FEATURE_SECTIONS = [
    "wpcc",
    "proposed_research",
    "application_form",
    "general",
    "training_development",
    "sites_support",
]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _funding_guidance(funding_result: int) -> str:
    """Return prompt text describing the expected quality profile."""
    if funding_result == 1:
        return (
            "This is a SUCCESSFUL grant application (funding_result=1). "
            "Generate criteria that mostly show exists=\"positive\" and quality=\"good\", "
            "with strong, detailed evidence quotes. Most criteria should have evidence. "
            "The overall scores should reflect a strong application."
        )
    return (
        "This is a FAILED grant application (funding_result=0). "
        "Generate criteria where several show exists=\"negative\" and quality=\"unknown\" or \"bad\". "
        "Evidence should be weaker, sparser, or missing for some criteria. "
        "The overall scores should reflect a weaker application with fewer positive/good items."
    )


def _build_section_prompt(
    section_name: str,
    example_section: Dict[str, Any],
    funding_result: int,
) -> str:
    """Build a prompt for synthesising one feature section."""
    guidance = _funding_guidance(funding_result)
    example_json = json.dumps(example_section, indent=2, ensure_ascii=False)

    return (
        "Return ONLY valid JSON. No markdown fences, no explanation.\n\n"
        "You are generating synthetic research grant application feature data.\n"
        f"{guidance}\n\n"
        f'Below is a real example of the "{section_name}" feature section:\n\n'
        f"{example_json}\n\n"
        f'Generate a NEW synthetic "{section_name}" section that:\n'
        "1. Has the EXACT same JSON structure (keys: \"criteria\" list and \"overall\" dict).\n"
        "2. Uses DIFFERENT criterion names, evidence IDs, and quotes — do NOT copy the example.\n"
        "3. Evidence quotes should sound like realistic NIHR grant application text.\n"
        "4. Evidence IDs follow the pattern used (e.g. PES_001, SEC_001, WPCC_001, BUD_001).\n"
        "5. The \"overall\" dict must be numerically consistent with the criteria list:\n"
        "   - total_items = number of criteria\n"
        "   - positive_items = count where exists=\"positive\"\n"
        "   - good_items = count where exists=\"positive\" AND quality=\"good\"\n"
        "   - exists_points = positive_items\n"
        "   - quality_points = good_items\n"
        "6. Keep a similar number of criteria items as the example (±1).\n"
        "7. If the example section has empty criteria, you may return empty criteria too.\n\n"
        "Return ONLY the JSON object."
    )


def _build_orcid_prompt(
    example_orcid: Dict[str, Any],
    funding_result: int,
) -> str:
    """Build a prompt for synthesising the orcid section."""
    guidance = _funding_guidance(funding_result)
    example_json = json.dumps(example_orcid, indent=2, ensure_ascii=False)

    return (
        "Return ONLY valid JSON. No markdown fences, no explanation.\n\n"
        "You are generating synthetic research team (ORCID) data for a grant application.\n"
        f"{guidance}\n\n"
        'Below is a real example of the "orcid" section:\n\n'
        f"{example_json}\n\n"
        'Generate a NEW synthetic "orcid" section that:\n'
        "1. Has the EXACT same JSON structure.\n"
        "2. Uses DIFFERENT researcher names, ORCID IDs, and roles.\n"
        "3. Names should be realistic but fictional.\n"
        "4. ORCID IDs should follow the format XXXX-XXXX-XXXX-XXXX with plausible digits.\n"
        "5. team_metrics values should be numerically plausible:\n"
        "   - For successful grants: higher avg_citations, h_index, works_total\n"
        "   - For failed grants: lower metrics overall\n"
        "6. Keep a similar team_size as the example (±1).\n\n"
        "Return ONLY the JSON object."
    )


# ---------------------------------------------------------------------------
# Section-level synthesis
# ---------------------------------------------------------------------------

def synthesise_section(
    client: OllamaClient,
    section_name: str,
    example_section: Dict[str, Any],
    funding_result: int,
    num_predict: int = 2000,
) -> Dict[str, Any]:
    """Generate one synthetic feature section via LLM."""
    prompt = _build_section_prompt(section_name, example_section, funding_result)
    raw = client.generate(prompt, num_predict=num_predict)
    try:
        return json.loads(extract_json_object(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Failed to parse %s section, using fallback copy: %s", section_name, exc)
        return copy.deepcopy(example_section)


def synthesise_orcid(
    client: OllamaClient,
    example_orcid: Dict[str, Any],
    funding_result: int,
    num_predict: int = 1500,
) -> Dict[str, Any]:
    """Generate synthetic ORCID / team data via LLM."""
    prompt = _build_orcid_prompt(example_orcid, funding_result)
    raw = client.generate(prompt, num_predict=num_predict)
    try:
        return json.loads(extract_json_object(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Failed to parse orcid section, using fallback copy: %s", exc)
        return copy.deepcopy(example_orcid)


# ---------------------------------------------------------------------------
# Top-level overall aggregation
# ---------------------------------------------------------------------------

def _compute_overall(features: Dict[str, Any]) -> Dict[str, int]:
    """Sum per-section overalls into the top-level overall dict."""
    totals: Dict[str, int] = {
        "exists_points": 0,
        "quality_points": 0,
        "total_items": 0,
        "positive_items": 0,
        "good_items": 0,
    }
    for sec_name in FEATURE_SECTIONS:
        sec_overall = features.get(sec_name, {}).get("overall", {})
        for key in totals:
            totals[key] += int(sec_overall.get(key, 0))
    return totals


# ---------------------------------------------------------------------------
# Full-document synthesis
# ---------------------------------------------------------------------------

def synthesise_one(
    client: OllamaClient,
    source_doc: Dict[str, Any],
    synth_id: str,
) -> Dict[str, Any]:
    """Generate a complete synthetic feature JSON from one source document."""
    funding_result = source_doc.get("funding_result", 1)
    source_features = source_doc.get("features", {})

    # --- synthesise each feature section ---
    synth_features: Dict[str, Any] = {}
    for sec_name in FEATURE_SECTIONS:
        example = source_features.get(sec_name, {"criteria": [], "overall": {}})
        log.info("    [%s] generating ...", sec_name)
        synth_features[sec_name] = synthesise_section(
            client, sec_name, example, funding_result,
        )

    # --- synthesise orcid if present ---
    if "orcid" in source_features:
        log.info("    [orcid] generating ...")
        synth_features["orcid"] = synthesise_orcid(
            client, source_features["orcid"], funding_result,
        )

    return {
        "doc_id": synth_id,
        "source": {"synthetic_from": source_doc.get("doc_id", "unknown")},
        "run_info": {
            "ran_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": client.cfg.host,
            "model": client.cfg.model,
            "type": "synthetic",
        },
        "funding_result": funding_result,
        "overall": _compute_overall(synth_features),
        "features": synth_features,
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    input_path: str,
    output_dir: str,
    *,
    host: str = "http://localhost:11434",
    model: str = "qwen2.5:14b",
    num_per_source: int = 1,
    temperature: float = 0.7,
    cache_dir: Optional[str] = None,
) -> List[Path]:
    """
    Read source feature JSONs and generate synthetic copies.

    Parameters
    ----------
    input_path : str
        Path to a single .json file or a directory of .json files.
    output_dir : str
        Directory to write synthetic output files.
    host : str
        Ollama server address.
    model : str
        Model name available on the Ollama server.
    num_per_source : int
        Number of synthetic files to generate per source file.
    temperature : float
        Sampling temperature (higher = more diverse outputs).
    cache_dir : str | None
        Optional cache directory for LLM responses.

    Returns
    -------
    list[Path]
        Paths of all generated synthetic JSON files.
    """
    cfg = OllamaConfig(
        host=host,
        model=model,
        temperature=temperature,
        num_predict=2000,
        timeout_sec=300,
    )
    client = OllamaClient(cfg, cache_dir=cache_dir)

    source_files = iter_json_files(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for src_path in source_files:
        log.info("Source: %s", src_path.name)
        doc = load_json(src_path)

        for i in range(num_per_source):
            synth_id = f"SYNTH_{src_path.stem}_{i:03d}"
            log.info("  Generating %s ...", synth_id)

            result = synthesise_one(client, doc, synth_id)
            out_path = out / f"{synth_id}.features.json"
            write_json(out_path, result)
            written.append(out_path)
            log.info("  Wrote %s", out_path)

    return written


# ---------------------------------------------------------------------------
# Batch directory helpers
# ---------------------------------------------------------------------------

def _make_client(
    host: str = "http://localhost:11434",
    model: str = "qwen2.5:14b",
    temperature: float = 0.7,
    cache_dir: Optional[str] = None,
) -> OllamaClient:
    """Create an OllamaClient with sensible defaults for synthesis."""
    cfg = OllamaConfig(
        host=host,
        model=model,
        temperature=temperature,
        num_predict=2000,
        timeout_sec=300,
    )
    return OllamaClient(cfg, cache_dir=cache_dir)


def process_directory(
    input_dir: str,
    output_dir: str,
    *,
    host: str = "http://localhost:11434",
    model: str = "qwen2.5:14b",
    num_per_source: int = 1,
    temperature: float = 0.7,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process every .json in a directory and return a summary report.

    Returns
    -------
    dict with keys:
        total_source   – number of source files found
        total_generated – number of synthetic files written
        successful     – list of source stems that succeeded
        failed         – list of (source stem, error message) that failed
        output_files   – list of Path objects written
    """
    client = _make_client(host, model, temperature, cache_dir)

    src_dir = Path(input_dir).resolve()
    source_files = sorted(src_dir.glob("*.json"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "total_source": len(source_files),
        "total_generated": 0,
        "successful": [],
        "failed": [],
        "output_files": [],
    }

    for idx, src_path in enumerate(source_files, 1):
        log.info("[%d/%d] %s", idx, len(source_files), src_path.name)
        try:
            doc = load_json(src_path)
            for i in range(num_per_source):
                synth_id = f"SYNTH_{src_path.stem}_{i:03d}"
                log.info("  -> %s", synth_id)
                result = synthesise_one(client, doc, synth_id)
                out_path = out / f"{synth_id}.features.json"
                write_json(out_path, result)
                summary["output_files"].append(out_path)
                summary["total_generated"] += 1
            summary["successful"].append(src_path.stem)
        except Exception as exc:
            log.error("  FAILED: %s", exc)
            summary["failed"].append((src_path.stem, str(exc)))

    # write summary report alongside outputs
    report_path = out / "_synthesis_report.json"
    write_json(report_path, {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "model": model,
        "temperature": temperature,
        "num_per_source": num_per_source,
        "total_source": summary["total_source"],
        "total_generated": summary["total_generated"],
        "successful": summary["successful"],
        "failed": summary["failed"],
    })
    log.info(
        "Summary: %d source -> %d generated, %d failed.  Report: %s",
        summary["total_source"],
        summary["total_generated"],
        len(summary["failed"]),
        report_path,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic NIHR grant feature data using a local LLM.",
    )
    p.add_argument("--input", required=True,
                    help="Source feature .json file or directory.")
    p.add_argument("--output", required=True,
                    help="Output directory for synthetic files.")
    p.add_argument("--host", default="http://localhost:11434",
                    help="Ollama server address (default: localhost:11434).")
    p.add_argument("--model", default="qwen2.5:14b",
                    help="Ollama model name (default: qwen2.5:14b).")
    p.add_argument("--num", type=int, default=1,
                    help="Synthetic copies per source file (default: 1).")
    p.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature (default: 0.7).")
    p.add_argument("--cache-dir", default=None,
                    help="Cache directory for LLM responses.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    paths = generate_synthetic_dataset(
        input_path=args.input,
        output_dir=args.output,
        host=args.host,
        model=args.model,
        num_per_source=args.num,
        temperature=args.temperature,
        cache_dir=args.cache_dir,
    )
    log.info("Done — generated %d synthetic file(s).", len(paths))


if __name__ == "__main__":
    main()
