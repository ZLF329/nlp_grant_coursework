"""
Grant AI web pipeline server.

End-to-end flow when the user uploads a PDF:
  1. Save the upload to <project>/data/uploads/.
  2. all_type_parser.parse_and_save → data/uploads/json_data/<stem>.json
  3. nlp_feature.extract_nlp_features on the parsed JSON
  4. ORCID enrichment for team members with valid ORCID IDs
  5. qwen3_ollama.score_application on the parsed JSON (section evidence belief + final signal scoring)
  6. Combine into a single result JSON and expose via /result/<job_id>.

The static HTML in web/public is served as-is and drives the flow via:
    POST /upload          → {job_id}
    GET  /progress/<id>   → progress + step list
    GET  /result/<id>     → {features_json: <result>, nlp_features: ...}
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import traceback
import uuid
from pathlib import Path

import requests as _requests

from flask import Flask, jsonify, request, send_from_directory

# ── path wiring ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PARSED_DIR = UPLOAD_DIR / "json_data"        # all_type_parser writes here
RESULT_DIR = DATA_DIR / "results"
for d in (UPLOAD_DIR, PARSED_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

PUBLIC_DIR = Path(__file__).resolve().parent / "public"
CRITERIA_PATH = PROJECT_ROOT / "criteria_points.json"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

# ── pipeline imports (lazy where heavy) ───────────────────────────────────────
from src.all_type_parser.all_type_parser import parse_and_save           # noqa: E402
from src.feature_eng.nlp_feature import extract_nlp_features              # noqa: E402
from ORCID.orcid_features import compute_features as compute_orcid_features  # noqa: E402
from ORCID.orcid_features import fetch_orcid_profile                       # noqa: E402
from ORCID.orcid_features import openalex_cited_by_for_dois                # noqa: E402

# Ollama scorer is imported lazily — keeping a single shared instance avoids
# reloading the model on every request.
_scorer_lock = threading.Lock()
_scorer = None
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")


def _get_scorer():
    global _scorer
    with _scorer_lock:
        if _scorer is None:
            from qwen3_ollama import _Scorer  # noqa: WPS433
            _scorer = _Scorer()
        return _scorer


# ── job state ─────────────────────────────────────────────────────────────────
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

STEP_TEMPLATE = [
    ("stage0", "Stage 0 · Parse + Base Features"),
    ("stage1", "Stage 1 · ORCID Enrichment"),
    ("stage2", "Stage 2 · Drawback-Aware Final Scoring"),
]


def _extract_team_members(parsed: dict) -> list[dict]:
    team_section = parsed.get("LEAD APPLICANT & RESEARCH TEAM", {}) or {}
    members: list[dict] = []

    def add_member(role_group: str, member: dict | None):
        if not isinstance(member, dict) or not member:
            return
        raw_orcid = str(member.get("ORCID") or "").strip()
        match = ORCID_RE.search(raw_orcid)
        members.append({
            "role_group": role_group,
            "name": str(member.get("Full Name") or "").strip() or role_group,
            "organisation": str(member.get("Organisation") or "").strip(),
            "proposed_role": str(member.get("Proposed Role") or "").strip(),
            "orcid": match.group(0) if match else "",
        })

    add_member("Lead Applicant", team_section.get("Lead Applicant"))
    add_member("Joint Lead Applicant", team_section.get("Joint Lead Applicant"))
    for co_applicant in team_section.get("Co-Applicants", []) or []:
        add_member("Co-Applicant", co_applicant)
    return members


def _run_orcid_enrichment(parsed: dict) -> dict:
    members = _extract_team_members(parsed)
    enriched_members: list[dict] = []
    works_recent_5y_values: list[float] = []
    h_index_values: list[float] = []
    citation_values: list[float] = []
    success_count = 0

    for member in members:
        entry = dict(member)
        # Only enrich Lead Applicant via ORCID
        if entry.get("role_group") not in ("Lead Applicant", "Joint Lead Applicant"):
            entry["status"] = "skipped"
            enriched_members.append(entry)
            continue
        orcid = entry.get("orcid") or ""
        if not orcid:
            entry["status"] = "missing_orcid"
            enriched_members.append(entry)
            continue

        try:
            profile = fetch_orcid_profile(orcid, max_works=100)
            # Enrich with citation counts from OpenAlex (parallel, capped at 40 DOIs)
            doi2citedby: dict = {}
            try:
                dois = [w["doi"] for w in profile.get("works", []) if w.get("doi")][:40]
                if dois:
                    doi2citedby = openalex_cited_by_for_dois(
                        dois, max_workers=6, timeout_sec=12, max_retries=1
                    )
            except Exception as _oa_exc:
                print(f"[WARN] OpenAlex fetch skipped for {orcid}: {_oa_exc}")
            features = compute_orcid_features(profile, doi2citedby=doi2citedby if dois else None)
            entry["status"] = "ok"
            entry["summary"] = {
                "works_total": features.get("outputs", {}).get("works_total", 0),
                "works_recent_5y": features.get("outputs", {}).get("works_recent_5y", 0),
                "funding_total": features.get("recognition", {}).get("funding_total", 0),
                "h_index": features.get("impact", {}).get("h_index"),
                "citations_total": features.get("impact", {}).get("citations_total"),
            }
            entry["profile"] = profile.get("person", {})
            success_count += 1

            works_recent_5y = entry["summary"]["works_recent_5y"]
            works_recent_5y_values.append(float(works_recent_5y or 0))

            h_index = entry["summary"]["h_index"]
            if h_index is not None:
                h_index_values.append(float(h_index))

            citations_total = entry["summary"]["citations_total"]
            if citations_total is not None:
                citation_values.append(float(citations_total))
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)

        enriched_members.append(entry)

    total_members = len(members)
    orcid_members = sum(1 for member in members if member.get("orcid"))
    return {
        "team_metrics": {
            "team_size": total_members,
            "orcid_count": orcid_members,
            "orcid_coverage_ratio": round(orcid_members / total_members, 2) if total_members else 0.0,
            "resolved_profiles": success_count,
            "avg_h_index": round(sum(h_index_values) / len(h_index_values), 2) if h_index_values else 0.0,
            "avg_citations": round(sum(citation_values) / len(citation_values), 2) if citation_values else 0.0,
            "avg_works_recent_5y": round(sum(works_recent_5y_values) / len(works_recent_5y_values), 2) if works_recent_5y_values else 0.0,
        },
        "members": enriched_members,
    }


def _new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "progress": 0,
            "phase": "stage0",
            "detail": "Stage 0: Receiving file…",
            "steps": [
                {"key": k, "title": t, "status": "pending", "progress": 0}
                for k, t in STEP_TEMPLATE
            ],
            "result": None,
            "error": None,
        }
    return job_id


def _update(job_id: str, *, step_key: str | None = None, step_status: str | None = None,
            progress: int | None = None, detail: str | None = None,
            status: str | None = None, error: str | None = None,
            result: dict | None = None):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        if step_key:
            job["phase"] = step_key
            for s in job["steps"]:
                if s["key"] == step_key and step_status:
                    s["status"] = step_status
                    s["progress"] = 100 if step_status == "done" else (50 if step_status == "running" else s["progress"])
        if progress is not None:
            job["progress"] = progress
        if detail is not None:
            job["detail"] = detail
        if status is not None:
            job["status"] = status
        if error is not None:
            job["error"] = error
        if result is not None:
            job["result"] = result


# ── pipeline ──────────────────────────────────────────────────────────────────

def _run_pipeline(job_id: str, upload_path: Path):
    try:
        _update(job_id, step_key="stage0", step_status="running",
                progress=10, detail="Stage 0: Parsing PDF…")
        parsed_path = parse_and_save(str(upload_path))
        parsed = json.loads(Path(parsed_path).read_text(encoding="utf-8"))

        _update(job_id, step_key="stage0", step_status="running",
                progress=28, detail="Stage 0: Extracting NLP features…")
        try:
            nlp_features = extract_nlp_features(parsed)
        except Exception as e:
            nlp_features = {"error": str(e)}
        _update(job_id, step_key="stage0", step_status="done",
                progress=40, detail="Stage 0 complete")

        _update(job_id, step_key="stage1", step_status="running",
                progress=48, detail="Stage 1: Enriching team ORCID data…")
        orcid_features = _run_orcid_enrichment(parsed)
        _update(job_id, step_key="stage1", step_status="done",
                progress=62,
                detail=f"Stage 1 complete · {orcid_features['team_metrics']['resolved_profiles']} ORCID profile(s) resolved")

        _update(job_id, step_key="stage2", step_status="running",
                progress=70, detail="Stage 2: Drawback-aware final signal scoring…")
        from qwen3_ollama import score_application
        scorer = _get_scorer()
        scored = score_application(
            parsed,
            CRITERIA_PATH,
            doc_id=upload_path.stem,
            scorer=scorer,
            artifacts_dir=RESULT_DIR,
        )
        _update(job_id, step_key="stage2", step_status="done",
                progress=92, detail="Stage 2 complete · assembling result…")

        result = dict(scored)
        result["nlp_features"] = nlp_features
        result.setdefault("features", {})
        if orcid_features["team_metrics"]["resolved_profiles"] > 0:
            result["features"]["orcid"] = orcid_features

        # Persist for later inspection
        out_path = RESULT_DIR / f"{job_id}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")

        _update(job_id, progress=100,
                detail="Done", status="done",
                result={"features_json": result, "nlp_features": nlp_features})
    except Exception as e:
        traceback.print_exc()
        _update(job_id, status="error", error=str(e), detail=f"Error: {e}")


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path="")


@app.get("/")
def root():
    return send_from_directory(str(PUBLIC_DIR), "index.html")


@app.get("/result")
def result_page():
    return send_from_directory(str(PUBLIC_DIR), "result.html")


@app.post("/upload")
def upload():
    f = request.files.get("pdf")
    if not f:
        return jsonify({"error": "no file"}), 400

    job_id = _new_job()
    safe_name = Path(f.filename or "upload.pdf").name
    target = UPLOAD_DIR / f"{job_id}_{safe_name}"
    f.save(str(target))

    threading.Thread(target=_run_pipeline, args=(job_id, target), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.get("/progress/<job_id>")
def progress(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "unknown job"}), 404
        return jsonify({k: v for k, v in job.items() if k != "result"})


@app.get("/result/<job_id>")
def result(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "unknown job"}), 404
        if job["status"] != "done" or not job["result"]:
            return jsonify({"error": "not ready"}), 409
        return jsonify(job["result"])


@app.get("/history")
def history():
    entries = []
    for p in sorted(RESULT_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            entries.append({
                "job_id":   p.stem,
                "doc_id":   data.get("doc_id", p.stem),
                "ran_at":   (data.get("run_info") or {}).get("ran_at_utc"),
                "score":    (data.get("overall") or {}).get("final_score_0to100"),
                "model":    (data.get("run_info") or {}).get("scorer_model"),
            })
        except Exception:
            pass
    return jsonify(entries)


@app.get("/history/<job_id>")
def history_item(job_id: str):
    p = RESULT_DIR / f"{job_id}.json"
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(json.loads(p.read_text(encoding="utf-8")))


@app.get("/status")
def status():
    with JOBS_LOCK:
        all_jobs = list(JOBS.values())
    running = sum(1 for j in all_jobs if j["status"] == "running")
    queued  = sum(1 for j in all_jobs if j["status"] == "pending")

    # Ollama GPU info via /api/ps
    gpu: dict = {}
    try:
        r = _requests.get(f"{OLLAMA_HOST}/api/ps", timeout=3)
        if r.status_code == 200:
            models = r.json().get("models") or []
            size_vram = sum(m.get("size_vram", 0) for m in models)
            gpu = {
                "models_loaded": [m.get("name") for m in models],
                "vram_used_gb": round(size_vram / 1024**3, 2),
                "model_active": len(models) > 0,
            }
        else:
            gpu = {"error": f"ollama /api/ps returned {r.status_code}"}
    except Exception as e:
        gpu = {"error": str(e)}

    return jsonify({
        "active_jobs": running,
        "queued_jobs": queued,
        "gpu": gpu,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"[server] Grant AI pipeline listening on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
