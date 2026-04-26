# NIHR Grant Application Automated Scoring System

This project implements a structured scoring pipeline for NIHR grant applications using a local large language model backend through Ollama. The input is a grant application PDF. The output is a structured JSON file containing rubric-level scores, evidence sources, strengths, limitations, and an overall score.

---

## Core Idea

### Why not send the full application directly to the model?

NIHR applications can contain tens of thousands of words, often exceeding or crowding the model context window. Even when the full text fits, the model may struggle to check each scoring signal against the right evidence because attention is spread across the entire document.

This system addresses that problem by first splitting the application into a traceable chunk pool, then using a two-stage pipeline for evidence discovery and final scoring. The model only sees the context most relevant to the current scoring dimension.

---

## Pipeline Architecture

```text
PDF
 |
 v
[1] Parser
 |   Extract text by application section and generate structured JSON.
 |
 v
[2] Chunk Pool Construction
 |   Split each section into text chunks (default: <= 1200 characters).
 |   Assign each chunk a unique chunk_id and preserve its source section.
 |
 v
[3] Stage 1 - Belief Accumulation
 |   Scan the application section by section.
 |   Ask the model to identify evidence chunks relevant to each rubric sub-criterion.
 |   Output good/bad evidence chunk IDs and short implications.
 |   Merge results into a global belief_state.
 |
 v
[4] Dynamic Context Selection
 |   For each scoring dimension, combine rule-based section priors with
 |   evidence-derived sections from the belief_state.
 |   Deduplicate chunks and preserve original document order.
 |
 v
[5] Stage 2 - Final Scoring
 |   Build a scoped version of the application for each rubric section.
 |   Ask the model to score each signal from 0 to 5.
 |   Return pros, drawbacks, and grounded evidence IDs.
 |
 v
[6] Aggregated Output
     signal -> sub_criterion -> section -> overall
     Compute weighted averages and apply doc_type-specific exclusions.
```

---

## Key Modules

### Parser Layer

Located in `src/all_type_parser/`. The parser automatically detects the PDF type and routes the file to the appropriate parser.

| File | Target format | doc_type |
|---|---|---|
| `fellowships_parser.py` | NIHR Fellowship applications (doctoral/postdoctoral) | `fellowship` |
| `RfPB_parser.py` | Research for Patient Benefit Stage 2 applications | `rfpb` |
| `all_other_parser.py` | Fallback parser for other formats | `unknown` |

The parser output is a structured JSON object. The top level includes a `doc_type` field, which is later used for scoring adaptation.

Key layout differences between RfPB and Fellowship applications:

- Fellowship applications usually have fixed blue boxes near the top of each page, making section boundaries relatively clear.
- RfPB Stage 2 applications may place blue boxes at variable positions, and a single page can contain multiple sections, so line-level section detection is required.

---

### Chunk Pool (`src/pool/build_pool.py`)

The chunk pool converts the parsed JSON into fixed-size evidence units. By default, each text chunk is limited to 1200 characters. Each chunk stores:

- `chunk_id`: unique ID, such as `secdrp__001_a`
- `parser_section`: source section, such as `Detailed Research Plan`
- `source_path`: original JSON path, such as `APPLICATION DETAILS > Detailed Research Plan`

The pool also adds derived chunks:

- **Application Context**: application-level metadata such as title, applicant, organisation, and other contextual fields.
- **Plain English NLP Analysis**: readability and plain-English indicators, including sentence length, Flesch-Kincaid estimates, jargon proxy density, content coverage, and lexical overlap with the detailed research plan.
- **Application Form Analysis**: structural indicators derived from parser output, including word counts, list markers, table-like budget lines, repeated text, transition phrases, and cross-section lexical overlap. This chunk supports the application form quality dimension.

---

### Stage 1 - Belief Accumulation

**Motivation:** The same passage can be relevant to multiple scoring dimensions. For example, a CV section can support both research experience and leadership trajectory. Stage 1 scans the document first and builds a mapping from rubric sub-criteria to supporting or negative evidence chunks.

**Execution:**

- Iterate through parsed application sections, excluding the derived Application Form scoring chunk.
- For each section, send the current section chunks, the current global belief state, and the rubric to the model.
- The model returns findings in this shape:

  ```json
  {
    "sub_id": "g.4",
    "evidence": {
      "good_evidence_ids": ["secac__001_a"],
      "bad_evidence_ids": []
    },
    "implication": "The CV lists publications and prior funded projects, supporting research output quality."
  }
  ```

- Findings are merged by `sub_id` into `belief_state.subcriteria_beliefs`.
- Evidence accumulates across sections and is reused during final scoring.

**Flattened design:** Each finding maps directly to a parent `sub_id` rather than nesting at the signal level. This reduces token cost and allows Stage 1 to cover more rubric items within the context budget.

---

### Dynamic Context Selection

Final scoring does not reuse the same full-document prompt for every rubric section. Instead, the system builds a scoped context dynamically for each scoring dimension.

The selection process combines two sources:

1. **Rule-based section priors** from `SECTION_TO_PARSER_SECTIONS`, which define where evidence is expected to appear for each rubric dimension.
2. **Evidence-driven expansion** from the Stage 1 `belief_state`, which adds parser sections that actually contain evidence for the target rubric sub-criteria.

For example:

- `proposed_research` retrieves the application context, plain English summary, scientific abstract, detailed research plan, previous-stage changes, PPI/WPCC sections, and budget information.
- `training_development` retrieves training, development, support, and mentorship sections.
- `application_form` retrieves the derived Application Form Analysis chunk.
- `general` uses the full application because applicant quality evidence can appear across many sections.

After selection, duplicate chunks are removed and the remaining chunks are ordered according to their original document position. If no scoped context can be constructed, the system falls back to the full application text.

---

### Stage 2 - Final Scoring

**Execution:**

- For each major rubric section, identify relevant parser sections from the belief state and rule-based mappings.
- Build a scoped version of the application text from the chunk pool.
- Send the scoped application text, the target rubric section, and the final belief state to the model.
- The model scores every signal from 0 to 5 and returns:
  - `used_chunk_ids`
  - `pros`
  - `drawbacks`
  - signal-level scores

The model output is constrained by JSON schemas so that all required signals are scored and all score values are valid integers from 0 to 5.

---

### Scoring Rubric (`criteria_points.json`)

The rubric contains six major scoring sections. Each section contains sub-criteria, and each sub-criterion contains signal-level scoring items.

| Section | key | Description |
|---|---|---|
| General | `general` | Applicant experience, outputs, leadership potential, and career trajectory |
| Proposed Research | `proposed_research` | Research question, design, methodological rigour, feasibility, impact, and resources |
| Training and Development | `training_development` | Training plan, mentorship, career development, and skill acquisition |
| Sites and Support | `sites_support` | Institutional capability, supervision, infrastructure, and research culture |
| Working with People and Communities | `wpcc` | PPI/WPCC design, representativeness, depth of involvement, and feedback mechanisms |
| Application Form | `application_form` | Structural completeness, logical flow, formatting, repetition, and coherence |

Score aggregation path:

```text
signal score (0-5)
 -> weighted sub_criterion average
 -> sub_criterion score (0-10)
 -> weighted section average
 -> overall score
```

---

### `doc_type` Adaptation

Different NIHR application types do not share exactly the same scoring expectations. The pipeline uses `doc_type` to exclude criteria that are not applicable to a given application format.

#### RfPB (Research for Patient Benefit)

RfPB is a project grant rather than an individual career-development fellowship. Compared with Fellowship applications:

- It does not require a personal career-development narrative.
- It does not contain a training and development section in the Fellowship sense.
- Training-related Fellowship criteria should not reduce the score of an RfPB application.

For `doc_type=rfpb`, the pipeline applies these exclusions:

| Excluded item | Exclusion scope | Reason |
|---|---|---|
| `g.1` Common Characteristics of Good Applications | Excluded from the General section average | Contains Fellowship-oriented signals such as training plan quality |
| `g.2` Tell Us Why You Need This Award | Excluded from the General section average | Asks about personal career-development need, which is not applicable to a project grant |
| `training_development` | Excluded from the overall score | RfPB applications do not have a matching Fellowship-style training section |

Excluded sub-criteria can still appear in the output with `excluded_reason: "not_applicable_for_doc_type"`. They are visible for review but do not affect the averaged scores.

---

## Directory Structure

```text
.
|-- criteria_points.json          # Rubric definition
|-- qwen3_ollama.py               # Main Ollama scoring entry point
|-- score_experiments.ipynb       # Score stability experiments
|-- src/
|   |-- all_type_parser/
|   |   |-- all_type_parser.py    # Parser router
|   |   |-- fellowships_parser.py # Fellowship PDF parser
|   |   |-- RfPB_parser.py        # RfPB Stage 2 PDF parser
|   |   |-- pdf_parser.py         # Generic PDF parser
|   |   |-- pdf_utils.py          # PDF utility functions
|   |-- pool/
|   |   |-- build_pool.py         # Chunk pool construction
|   |-- scoring/
|       |-- pipeline.py           # Two-stage scoring pipeline
|-- data/
|   |-- successful/               # Example successful application PDFs
|   |-- unsuccessful/             # Example unsuccessful application PDFs
|   |-- experiments/              # Experiment datasets
```

---

## Usage

The recommended way to run the full pipeline is the **end-to-end web server on a Linux/GPU host** (RunPod, EC2, university cluster, or any Ubuntu/Debian machine). The steps below mirror `Runpod_Instruction.ipynb` exactly and take you from a fresh pod to a browser-accessible scoring service.

### A. Run the web pipeline on a fresh Linux/RunPod host

Run each block in a shell on the host (RunPod web terminal, SSH, or local Linux). Pick a pod template that exposes HTTP port `8000`.

**Step 1 — Install OS packages**

```bash
sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources
apt-get update && apt-get install -y git python3 python3-venv python3-pip curl
```

(The `sed` line swaps to a faster apt mirror; remove it if you are outside mainland China or on a non-Ubuntu base image.)

**Step 2 — Install Ollama and pull the model**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 3
ollama pull qwen3.5:27b
```

`ollama serve &` starts the model server in the background. The pull is ~17 GB and takes a few minutes on a fast network. Switch the model tag if you want a smaller footprint (e.g. `qwen3:4b`).

**Step 3 — Clone the project**

```bash
git clone https://github.com/ZLF329/nlp_grant_coursework.git
cd nlp_grant_coursework
```

**Step 4 — Launch the web server**

```bash
chmod +x start.sh
PORT=8000 ./start.sh
```

`start.sh` creates a virtual environment, installs Python dependencies, downloads the spaCy `en_core_web_sm` model and NLTK `punkt` data, then starts the Flask server on port 8000.

**Step 5 — Open the UI**

- Locally: navigate to `http://localhost:8000`.
- On RunPod: in the pod's **Connect** panel, pick the HTTP service exposed on port 8000 to get a public proxy URL such as `https://<pod-id>-8000.proxy.runpod.net`.

Upload a PDF, watch the progress bar through the three pipeline stages, and view the scored result.

**`start.sh` configurable environment variables**

| Variable      | Default                       | Purpose                                       |
|---------------|-------------------------------|-----------------------------------------------|
| `PORT`        | `8000`                        | HTTP port for the Flask server                |
| `PYTHON`      | `python3`                     | Python interpreter used to create the venv    |
| `VENV_DIR`    | `.venv`                       | Path of the virtual environment               |
| `OLLAMA_HOST` | `http://127.0.0.1:11434`      | URL of the Ollama service (read by the app)   |
| `OLLAMA_MODEL`| `qwen3.5:27b`                 | Model name passed to Ollama                   |

The script is idempotent: subsequent runs reuse the venv and skip dependency installation (a `.deps_installed` marker is written inside the venv).

---

### B. Optional — direct CLI usage of the parser and scorer

If you want to call the underlying components without the web server (e.g. for batch processing or experiments), they are exposed as plain Python modules.

**Parse a PDF**

```bash
python -m src.all_type_parser.all_type_parser path/to/application.pdf
```

The parser auto-detects the document type where possible and writes a structured JSON file under a `json_data/` directory next to the input PDF.

**Score a parsed application**

Requires a running Ollama server with the selected model available locally.

```bash
OLLAMA_MODEL=qwen3.5:27b python qwen3_ollama.py \
  data/successful/json_data/IC00029_RfPB.json \
  --criteria criteria_points.json \
  --out data/successful/json_data/IC00029_RfPB_scored.json
```

**Run experiments**

Use `score_experiments.ipynb` for scoring experiments, including:

- repeated scoring of the same PDF to inspect variance
- A/B group comparisons between application sets
- score distribution and hypothesis-test exploration

---

## macOS — Native Ollama (recommended on Mac, much faster than Docker)

> **Why not Docker on Mac?** Docker Desktop on macOS runs containers inside a Linux VM, which **cannot access the Mac's Metal GPU**. LLM inference is forced onto CPU and ends up 5–10× slower than the native Ollama app. On Apple Silicon a single PDF can take 30–60 minutes through Docker; through native Ollama it typically finishes in 2–10 minutes (depending on model size).

Use this path when running on a MacBook (Intel or Apple Silicon). The web server still runs locally as a Python process — only Ollama is moved out of the container.

### Step 1 — Install native Ollama

Either route works:

```bash
# Homebrew
brew install ollama

# or download the .dmg from https://ollama.com/download and drag to Applications
```

### Step 2 — Start Ollama and pull the model

```bash
ollama serve &
ollama pull qwen3.5:27b      # paper configuration, ~17 GB
# or, for faster local testing:
ollama pull qwen3:4b         # ~2.6 GB
```

You can verify the model is loaded with:

```bash
ollama list
curl http://localhost:11434/api/tags
```

### Step 3 — Install OS-level dependencies for the parser

The parser uses `pdf2image` (poppler) and `pytesseract` (tesseract):

```bash
brew install poppler tesseract
```

### Step 4 — Launch the web server

```bash
cd /path/to/nlp_grant_coursework
chmod +x start.sh
./start.sh                                  # uses qwen3.5:27b by default
# or pick a different model:
OLLAMA_MODEL=qwen3:4b ./start.sh
```

`start.sh` creates a virtualenv on first run, installs dependencies, then serves at `http://localhost:8000`.

### Step 5 — Use it

Open `http://localhost:8000` in a browser, upload a PDF, and watch the three pipeline stages complete. Inference runs on Metal GPU automatically — no flags needed.

### Stopping

- Web server: `Ctrl+C` in its terminal
- Ollama background process: `pkill ollama` (or close the menu-bar Ollama app if installed via .dmg)

---

## Docker Deployment (All-in-One)

> **Note for macOS users:** prefer the **macOS — Native Ollama** section above. Docker on Mac forces CPU-only inference, which is several times slower than native Ollama on the same hardware. Use Docker on Mac only to verify the image builds correctly before delivering it to a Linux/GPU host.

The image bundles the Flask web server, all Python dependencies, the Ollama runtime, and the LLM weights, so a single `docker run` is enough to start the full pipeline.

### Files

- `Dockerfile` — image definition
- `docker-entrypoint.sh` — starts Ollama in the background, waits for it, pulls the model on first start if needed, then launches the web server
- `.dockerignore` — keeps notebooks, datasets, and dev artefacts out of the image

### Build arguments

| Build arg     | Default        | Purpose                                                      |
|---------------|----------------|--------------------------------------------------------------|
| `OLLAMA_MODEL`| `qwen3.5:27b`  | Ollama model tag baked into the image and used at runtime    |
| `BAKE_MODEL`  | `1`            | If `1`, pre-download the model during build (offline-ready). If `0`, pull on first container start instead. |

---

### Path A — Production image with `qwen3.5:27b` (paper configuration)

This is the configuration used in the paper and the recommended build to deliver to the marker. The image is fully self-contained (no network needed at runtime) but is large.

**Requirements**
- Disk: ~25 GB during build, final image ~20 GB
- Network: must reach `ollama.com` (model download) and Docker Hub during build
- For inference: a GPU host is strongly recommended; CPU-only is far too slow for `27b`

**Build**

```bash
docker build -t grant-ai .
```

(Default args, equivalent to `--build-arg OLLAMA_MODEL=qwen3.5:27b --build-arg BAKE_MODEL=1`. Expect 30–60 min depending on network.)

**Run (CPU)**

```bash
docker run --rm -p 8000:8000 grant-ai
```

**Run (NVIDIA GPU)**

```bash
docker run --rm --gpus all -p 8000:8000 grant-ai
```

Requires the NVIDIA Container Toolkit on the host. For maximum throughput, rebuild from a CUDA base image (e.g. `nvidia/cuda:12.4.0-runtime-ubuntu22.04`) instead of `python:3.11-slim`.

**Persist uploads and results across restarts**

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" grant-ai
```

**Distribute the image as a single offline file**

```bash
# on the build host
docker save grant-ai | gzip > grant-ai.tar.gz
# on the target machine
gunzip -c grant-ai.tar.gz | docker load
docker run --rm -p 8000:8000 grant-ai
```

Open `http://localhost:8000` after the container reports `Grant AI pipeline listening on http://0.0.0.0:8000`.

---

### Path B — Lightweight Mac local test with `qwen3:4b`

Use this to verify the Docker pipeline end-to-end on a laptop (especially Apple Silicon, where the `27b` model is too slow for interactive testing).

> Note: there is no `qwen3.5:4b` in the Ollama registry. The 4-billion-parameter option in the Qwen3 family is `qwen3:4b` (~2.6 GB).

**Requirements**
- Disk: ~10 GB during build, final image ~8 GB
- Docker Desktop running
- Network access during build

**Build**

```bash
docker build -t grant-ai-test --build-arg OLLAMA_MODEL=qwen3:4b .
```

(Takes roughly 5–15 min; most of the time is `pip install` and the 2.6 GB model pull. The pip layer is cached on rebuilds.)

**Run**

```bash
docker run --rm -p 8000:8000 grant-ai-test
```

Then open `http://localhost:8000`.

**Override the model at runtime (optional)**

```bash
docker run --rm -p 8000:8000 -e OLLAMA_MODEL=qwen3:8b grant-ai-test
```

If the requested model is not already inside the image, the entrypoint pulls it on first start (requires network).

---

### Common operations

**Stop a running container**

```bash
docker stop grant-ai            # by name (only if you used --name)
# or find it:
docker ps                       # list running containers
docker stop <container_id>
```

If you started the container with `--rm` and **without** `-d`, it runs in the foreground — just press `Ctrl+C` to stop it.

**Remove the image**

```bash
docker rmi grant-ai-test
```

**Inspect the container interactively (for debugging)**

```bash
docker run --rm -it grant-ai-test bash
# inside:
ollama list      # confirm the baked model is present
ls /app          # confirm project files
```

---

## Output Structure

The top-level scored JSON has the following shape:

```json
{
  "doc_id": "IC00029_RfPB",
  "run_info": {
    "scorer_model": "qwen3.5:35b",
    "ran_at_utc": "..."
  },
  "pool_lookup": {
    "chunk_id": {
      "text": "...",
      "parser_section": "..."
    }
  },
  "belief_state": {
    "subcriteria_beliefs": {
      "pr.1": {
        "good_evidence_ids": ["..."],
        "bad_evidence_ids": []
      }
    }
  },
  "features": {
    "general": {
      "score_10": 7.33,
      "sub_criteria": [
        {
          "sub_id": "g.3",
          "score_10": 8.0,
          "counts_toward_section_average": true,
          "signals": [
            {
              "sid": "g.3.a",
              "score": 4
            }
          ],
          "pros": "...",
          "drawbacks": "...",
          "evidence": [
            {
              "id": "secac__001_a",
              "text": "..."
            }
          ]
        }
      ]
    }
  },
  "overall": {
    "score_10": 8.44,
    "final_score_0to100": 84.4
  },
  "debug": {
    "doc_type": "rfpb",
    "excluded_sections": ["training_development"],
    "excluded_sub_ids": ["g.1", "g.2"]
  }
}
```

The output is designed to be auditable: each score can be inspected together with the evidence chunks and model rationale used to produce it.
