"""
Two-Pass Adaptive RAG retriever for grant scoring.

Pipeline:
  1. chunk_application      — slice APPLICATION DETAILS (and any later sections)
                              into fixed-size chunks; pre-APPLICATION DETAILS
                              content is returned as "always_text" (not chunked,
                              always fed to the LLM in full).
  2. build_index            — embed chunks and build a cosine HNSW index.
  3. pass1_recall           — for each rubric criterion, run ANN top-K.
  4. rerank                 — cross-encoder rerank the top-K candidates (CPU).
  5. allocate_budget        — distribute the global char budget across criteria
                              using the entropy of each criterion's rerank score
                              distribution (high entropy = evidence spread out =
                              gets a bigger share).
  6. pack_by_budget         — greedy fill of each criterion's budget with the
                              highest-ranked chunks.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── tunables ──────────────────────────────────────────────────────────────────
CHUNK_SIZE     = int(os.environ.get("RAG_CHUNK_SIZE", "400"))
CHUNK_OVERLAP  = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))
TOP_K          = int(os.environ.get("RAG_TOP_K", "20"))
TEMPERATURE    = float(os.environ.get("RAG_TEMPERATURE", "1.0"))
EMBED_MODEL    = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL   = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Sections that are always passed to the LLM in full (not chunked / retrieved).
# Anything else under the parsed JSON's top-level keys gets chunked.
ALWAYS_INCLUDE_KEYS = ("SUMMARY INFORMATION", "LEAD APPLICANT & RESEARCH TEAM")

# Lazy-loaded singletons (avoid reloading models per request).
_embed_model = None
_reranker = None


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: int
    text: str
    section: str          # parser section name (e.g. "Detailed Research Plan")
    section_order: int    # order of the section in the parsed doc
    char_offset: int      # start offset within the section


# ── helpers ───────────────────────────────────────────────────────────────────

def _stringify(value: Any) -> str:
    """Turn any parser value into a single string blob."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _slice_chunks(text: str, section: str, section_order: int,
                  start_id: int) -> list[Chunk]:
    """Fixed-size sliding-window chunker."""
    out: list[Chunk] = []
    if not text:
        return out
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    cid = start_id
    for offset in range(0, len(text), step):
        piece = text[offset:offset + CHUNK_SIZE]
        if not piece.strip():
            continue
        out.append(Chunk(
            chunk_id=cid,
            text=piece,
            section=section,
            section_order=section_order,
            char_offset=offset,
        ))
        cid += 1
        if offset + CHUNK_SIZE >= len(text):
            break
    return out


# ── 1. chunking ───────────────────────────────────────────────────────────────

def chunk_application(application: dict) -> tuple[str, list[Chunk]]:
    """
    Returns (always_text, chunks).
      always_text — pre-APPLICATION DETAILS content concatenated, always sent
                    to the LLM in full.
      chunks      — fixed-size chunks from APPLICATION DETAILS and any sections
                    that come after it.
    """
    always_parts: list[str] = []
    chunks: list[Chunk] = []
    section_order = 0
    next_id = 0

    for top_key, top_value in application.items():
        if top_key in ALWAYS_INCLUDE_KEYS:
            blob = _stringify(top_value)
            if blob.strip():
                always_parts.append(f"=== {top_key} ===\n{blob}")
            continue

        if top_key == "APPLICATION DETAILS" and isinstance(top_value, dict):
            for sub_name, sub_val in top_value.items():
                text = _stringify(sub_val)
                new = _slice_chunks(text, sub_name, section_order, next_id)
                chunks.extend(new)
                next_id += len(new)
                section_order += 1
            continue

        # Anything else (SUMMARY BUDGET, etc.) — chunk under its top-level name
        text = _stringify(top_value)
        new = _slice_chunks(text, top_key, section_order, next_id)
        chunks.extend(new)
        next_id += len(new)
        section_order += 1

    always_text = "\n\n".join(always_parts)
    return always_text, chunks


# ── 2. index ──────────────────────────────────────────────────────────────────

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[rag] loading embed model {EMBED_MODEL}", flush=True)
        _embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _embed_model


def build_index(chunks: list[Chunk]):
    """
    Returns (hnsw_index, embeddings_np). For empty input returns (None, None).
    """
    if not chunks:
        return None, None
    import hnswlib
    model = _get_embed_model()
    texts = [c.text for c in chunks]
    embs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    dim = embs.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(chunks), ef_construction=100, M=16)
    index.add_items(embs, ids=np.arange(len(chunks)))
    index.set_ef(64)
    return index, embs


# ── 3. recall ─────────────────────────────────────────────────────────────────

def _criterion_query(crit: dict) -> str:
    parts = [crit["name"]]
    for it in crit.get("sub_items", []):
        parts.append(it.get("name", ""))
        parts.append(it.get("definition", ""))
        sigs = it.get("signals") or []
        if sigs:
            parts.append(" ".join(sigs))
    return "\n".join(p for p in parts if p)


def pass1_recall(index, chunks: list[Chunk], criteria: list[dict],
                 top_k: int = TOP_K) -> dict[str, list[tuple[Chunk, float]]]:
    if index is None or not chunks:
        return {c["key"]: [] for c in criteria}
    model = _get_embed_model()
    queries = [_criterion_query(c) for c in criteria]
    q_embs = model.encode(queries, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True)
    k = min(top_k, len(chunks))
    labels, dists = index.knn_query(q_embs, k=k)
    out: dict[str, list[tuple[Chunk, float]]] = {}
    for crit, lab_row, dist_row in zip(criteria, labels, dists):
        # cosine distance → similarity
        out[crit["key"]] = [
            (chunks[int(idx)], float(1.0 - d))
            for idx, d in zip(lab_row, dist_row)
        ]
    return out


# ── 4. rerank ─────────────────────────────────────────────────────────────────

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"[rag] loading reranker {RERANK_MODEL} (cpu)", flush=True)
        _reranker = CrossEncoder(RERANK_MODEL, device="cpu")
    return _reranker


def rerank(criteria: list[dict],
           recall: dict[str, list[tuple[Chunk, float]]]
           ) -> dict[str, list[tuple[Chunk, float]]]:
    reranker = _get_reranker()
    out: dict[str, list[tuple[Chunk, float]]] = {}
    for crit in criteria:
        cands = recall.get(crit["key"], [])
        if not cands:
            out[crit["key"]] = []
            continue
        query = _criterion_query(crit)
        pairs = [(query, ch.text) for (ch, _) in cands]
        scores = reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip([c for (c, _) in cands], scores),
                        key=lambda x: x[1], reverse=True)
        out[crit["key"]] = [(c, float(s)) for (c, s) in ranked]
    return out


# ── 5. budget allocation ──────────────────────────────────────────────────────

def _entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr - arr.max()              # numerical stability
    p = np.exp(arr)
    p = p / p.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def allocate_budget(reranked: dict[str, list[tuple[Chunk, float]]],
                    total_budget: int,
                    temperature: float = TEMPERATURE) -> dict[str, int]:
    keys = list(reranked.keys())
    entropies = {k: _entropy([s for (_, s) in reranked[k]]) for k in keys}

    # softmax over entropies
    ent_vals = np.array([entropies[k] for k in keys], dtype=np.float64)
    if ent_vals.size == 0 or ent_vals.sum() == 0:
        weights = np.ones_like(ent_vals) / max(1, len(keys))
    else:
        scaled = ent_vals / max(1e-9, temperature)
        scaled = scaled - scaled.max()
        w = np.exp(scaled)
        weights = w / w.sum()

    budgets = {k: int(round(total_budget * float(w))) for k, w in zip(keys, weights)}

    # Zero-out criteria with no candidates so we don't waste budget on them.
    for k in keys:
        if not reranked[k]:
            budgets[k] = 0

    print(f"[rag] entropy/budget allocation (temperature={temperature}, "
          f"total={total_budget}):", flush=True)
    for k in keys:
        print(f"[rag]   {k:<22} entropy={entropies[k]:.3f}  "
              f"budget={budgets[k]} chars  (cands={len(reranked[k])})",
              flush=True)
    return budgets


# ── 6. packing ────────────────────────────────────────────────────────────────

def pack_by_budget(reranked: list[tuple[Chunk, float]],
                   budget: int) -> list[Chunk]:
    selected: list[Chunk] = []
    used = 0
    for ch, _ in reranked:
        size = len(ch.text)
        if used + size > budget:
            continue
        selected.append(ch)
        used += size
        if used >= budget:
            break
    return selected


# ── 7. orchestration ──────────────────────────────────────────────────────────

def retrieve_for_application(application: dict,
                             criteria: list[dict],
                             total_budget: int
                             ) -> tuple[str, dict[str, list[Chunk]]]:
    """
    criteria: list of {"key": str, "name": str, "sub_items": list[dict]}.
    Returns (always_text, {criterion_key: [Chunk, ...]}).
    """
    always_text, chunks = chunk_application(application)
    print(f"[rag] chunked application → {len(chunks)} chunks "
          f"(always_text={len(always_text)} chars)", flush=True)

    index, _ = build_index(chunks)
    recall = pass1_recall(index, chunks, criteria, top_k=TOP_K)
    reranked = rerank(criteria, recall)
    budgets = allocate_budget(reranked, total_budget, temperature=TEMPERATURE)

    selected: dict[str, list[Chunk]] = {}
    for crit in criteria:
        k = crit["key"]
        chosen = pack_by_budget(reranked[k], budgets[k])
        selected[k] = chosen
        print(f"[rag]   {k:<22} packed {len(chosen)} chunks "
              f"({sum(len(c.text) for c in chosen)}/{budgets[k]} chars)",
              flush=True)

    return always_text, selected
