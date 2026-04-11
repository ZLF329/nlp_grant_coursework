from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_CHARS = 1200
APPLICATION_DETAILS_KEY = "APPLICATION DETAILS"
SUMMARY_BUDGET_KEY = "SUMMARY BUDGET"
APPLICATION_CONTEXT_SECTION = "Application Context"


@dataclass(frozen=True)
class PoolChunk:
    chunk_id: str
    text: str
    parser_section: str
    source_path: str


def _slug_initials(name: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", name.lower())
    initials = "".join(part[0] for part in parts if part)
    return f"sec{initials or 'x'}"


def _stringify_leaf(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, indent=2).strip()


def _split_long_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", clean) if part.strip()]
    if len(paragraphs) <= 1:
        return [clean[i:i + max_chars].strip() for i in range(0, len(clean), max_chars)]

    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        for idx in range(0, len(paragraph), max_chars):
            piece = paragraph[idx:idx + max_chars].strip()
            if piece:
                chunks.append(piece)
    return chunks


def _child_path(path: list[str], key: Any) -> list[str]:
    if isinstance(key, int):
        return [*path, f"[{key}]"]
    return [*path, str(key)]


def _iter_leaves(value: Any, path: list[str], parser_section: str) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for key, child in value.items():
            out.extend(_iter_leaves(child, _child_path(path, key), parser_section))
        return out
    if isinstance(value, list):
        out = []
        for idx, child in enumerate(value):
            out.extend(_iter_leaves(child, _child_path(path, idx), parser_section))
        return out

    text = _stringify_leaf(value)
    return [(text, " > ".join(path))] if text else []


def _format_combined_context(entries: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"{source_path}:\n{text}"
        for text, source_path in entries
        if text.strip()
    )


def build_chunk_pool(application: dict[str, Any], max_chars: int = MAX_CHARS) -> dict[str, Any]:
    section_slug_map: dict[str, str] = {}
    used_slugs: dict[str, str] = {}
    section_counters: dict[str, int] = {}
    pool_lookup: dict[str, dict[str, str]] = {}
    section_chunk_ids: dict[str, list[str]] = {}

    def get_slug(section_name: str) -> str:
        base_slug = _slug_initials(section_name)
        existing_owner = used_slugs.get(base_slug)
        if existing_owner is None or existing_owner == section_name:
            used_slugs[base_slug] = section_name
            section_slug_map.setdefault(section_name, base_slug)
            return section_slug_map[section_name]

        suffix = 2
        while True:
            candidate = f"{base_slug}{suffix}"
            existing = used_slugs.get(candidate)
            if existing is None or existing == section_name:
                used_slugs[candidate] = section_name
                section_slug_map.setdefault(section_name, candidate)
                return section_slug_map[section_name]
            suffix += 1

    def add_leaf(parser_section: str, source_path: str, text: str, *, split: bool = True) -> None:
        slug = get_slug(parser_section)
        section_counters[slug] = section_counters.get(slug, 0) + 1
        base_id = f"{slug}__{section_counters[slug]:03d}"
        pieces = _split_long_text(text, max_chars=max_chars) if split else [text.strip()]
        ids: list[str] = []
        if len(pieces) == 1:
            ids = [base_id]
        else:
            ids = [f"{base_id}_{chr(97 + idx)}" for idx in range(len(pieces))]
        for chunk_id, piece in zip(ids, pieces):
            pool_lookup[chunk_id] = {
                "text": piece,
                "parser_section": parser_section,
                "source_path": source_path,
            }
            section_chunk_ids.setdefault(parser_section, []).append(chunk_id)

    combined_context_entries: list[tuple[str, str]] = []
    for root_key, root_value in application.items():
        root_name = str(root_key)
        if root_name == APPLICATION_DETAILS_KEY and isinstance(root_value, dict):
            for child_key, child_value in root_value.items():
                parser_section = str(child_key)
                for leaf_text, source_path in _iter_leaves(
                    child_value,
                    [root_name, str(child_key)],
                    parser_section,
                ):
                    add_leaf(parser_section, source_path, leaf_text)
        elif root_name == SUMMARY_BUDGET_KEY:
            for leaf_text, source_path in _iter_leaves(root_value, [root_name], root_name):
                add_leaf(root_name, source_path, leaf_text)
        elif isinstance(root_value, dict):
            for child_key, child_value in root_value.items():
                child_name = str(child_key)
                if child_name == SUMMARY_BUDGET_KEY:
                    for leaf_text, source_path in _iter_leaves(
                        child_value,
                        [root_name, child_name],
                        child_name,
                    ):
                        add_leaf(child_name, source_path, leaf_text)
                    continue
                combined_context_entries.extend(
                    _iter_leaves(child_value, [root_name, child_name], APPLICATION_CONTEXT_SECTION)
                )
        else:
            combined_context_entries.extend(
                _iter_leaves(root_value, [root_name], APPLICATION_CONTEXT_SECTION)
            )

    combined_context = _format_combined_context(combined_context_entries)
    if combined_context:
        add_leaf(
            APPLICATION_CONTEXT_SECTION,
            APPLICATION_CONTEXT_SECTION,
            combined_context,
            split=False,
        )

    pool_index_lines = [
        f'{chunk_id}: {json.dumps(meta["text"], ensure_ascii=False)}'
        for chunk_id, meta in pool_lookup.items()
    ]

    return {
        "pool_lookup": pool_lookup,
        "pool_index_text": "\n".join(pool_index_lines),
        "section_chunk_ids": section_chunk_ids,
        "id_to_text": {chunk_id: meta["text"] for chunk_id, meta in pool_lookup.items()},
        "id_to_parser_section": {
            chunk_id: meta["parser_section"] for chunk_id, meta in pool_lookup.items()
        },
    }


def write_pool_artifacts(
    *,
    pool_lookup: dict[str, dict[str, str]],
    pool_index_text: str,
    artifacts_dir: str | Path,
    doc_id: str,
) -> dict[str, str]:
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    pool_json_path = artifacts_path / f"{doc_id}_pool.json"
    pool_index_path = artifacts_path / f"{doc_id}_pool_index.txt"
    pool_json_path.write_text(
        json.dumps(pool_lookup, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pool_index_path.write_text(pool_index_text, encoding="utf-8")
    return {
        "pool_json": str(pool_json_path),
        "pool_index": str(pool_index_path),
    }
