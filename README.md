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

### 1. Parse a PDF

```bash
python -m src.all_type_parser.all_type_parser path/to/application.pdf
```

The parser automatically detects the document type where possible and writes a structured JSON file under a `json_data/` directory next to the input PDF.

### 2. Score an application

The scoring step requires a running Ollama server with the selected model available locally.

```bash
OLLAMA_MODEL=qwen3.5:35b python qwen3_ollama.py \
  data/successful/json_data/IC00029_RfPB.json \
  --criteria criteria_points.json \
  --out data/successful/json_data/IC00029_RfPB_scored.json
```

### 3. Run experiments

Use `score_experiments.ipynb` for scoring experiments, including:

- repeated scoring of the same PDF to inspect variance
- A/B group comparisons between application sets
- score distribution and hypothesis-test exploration

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
