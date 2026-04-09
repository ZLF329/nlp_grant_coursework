"""
Grant AI web pipeline server.

End-to-end flow when the user uploads a PDF:
  1. Save the upload to <project>/data/uploads/.
  2. all_type_parser.parse_and_save → data/uploads/json_data/<stem>.json
  3. nlp_feature.extract_nlp_features on the parsed JSON
  4. qwen3_vllm.score_application on the parsed JSON (vLLM)
  5. Combine into a single result JSON and expose via /result/<job_id>.

The static HTML in web/public is served as-is and drives the flow via:
    POST /upload          → {job_id}
    GET  /progress/<id>   → progress + step list
    GET  /result/<id>     → {features_json: <result>, nlp_features: ...}
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
import uuid
from pathlib import Path

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

# ── pipeline imports (lazy where heavy) ───────────────────────────────────────
from src.all_type_parser.all_type_parser import parse_and_save           # noqa: E402
from src.feature_eng.nlp_feature import extract_nlp_features              # noqa: E402

# vLLM scorer is imported lazily — keeping a single shared instance avoids
# reloading the 30B model on every request.
_scorer_lock = threading.Lock()
_scorer = None


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
    ("upload",          "Upload file"),
    ("parse_pdf",       "Parse PDF"),
    ("run_nlp",         "Run NLP feature extraction"),
    ("prepare_features", "Prepare LLM scoring"),
    ("run_orcid",       "Run ORCID extraction"),
    ("run_non_orcid",   "Run Qwen3 vLLM scoring"),
    ("load_features",   "Load feature output"),
    ("build_checklist", "Build checklist"),
    ("finish",          "Complete"),
]


def _new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "progress": 0,
            "phase": "upload",
            "detail": "Receiving file…",
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
        _update(job_id, step_key="upload", step_status="done",
                progress=10, detail="File uploaded")

        # 1. Parse PDF → data/uploads/json_data/<stem>.json
        _update(job_id, step_key="parse_pdf", step_status="running",
                progress=15, detail="Parsing PDF…")
        parsed_path = parse_and_save(str(upload_path))
        parsed = json.loads(Path(parsed_path).read_text(encoding="utf-8"))
        _update(job_id, step_key="parse_pdf", step_status="done", progress=30)

        # 2. NLP features
        _update(job_id, step_key="run_nlp", step_status="running",
                progress=35, detail="Extracting NLP features…")
        try:
            nlp_features = extract_nlp_features(parsed)
        except Exception as e:
            nlp_features = {"error": str(e)}
        _update(job_id, step_key="run_nlp", step_status="done", progress=50)

        # 3. Qwen3 vLLM scoring
        _update(job_id, step_key="prepare_features", step_status="done",
                progress=55, detail="Preparing LLM prompts…")
        _update(job_id, step_key="run_orcid", step_status="skipped",
                progress=58, detail="ORCID step skipped (not configured)")

        _update(job_id, step_key="run_non_orcid", step_status="running",
                progress=60, detail="Scoring with Qwen3 (Ollama)…")
        from qwen3_ollama import score_application
        scorer = _get_scorer()
        scored = score_application(parsed, CRITERIA_PATH,
                                   doc_id=upload_path.stem, scorer=scorer)
        _update(job_id, step_key="run_non_orcid", step_status="done", progress=85)

        # 4. Combine
        _update(job_id, step_key="load_features", step_status="done", progress=90)
        _update(job_id, step_key="build_checklist", step_status="done",
                progress=95, detail="Assembling result…")

        result = dict(scored)
        result["nlp_features"] = nlp_features

        # Persist for later inspection
        out_path = RESULT_DIR / f"{job_id}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")

        _update(job_id, step_key="finish", step_status="done", progress=100,
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"[server] Grant AI pipeline listening on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
