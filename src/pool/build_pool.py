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
APPLICATION_FORM_ANALYSIS_SECTION = "Application Form Analysis"
PLAIN_ENGLISH_ANALYSIS_SECTION = "Plain English NLP Analysis"


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


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _normalized_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in (text or "").splitlines():
        clean = re.sub(r"\s+", " ", line).strip().lower()
        if len(clean.split()) >= 4:
            lines.append(clean)
    return lines


def _sentence_tokens(text: str) -> list[str]:
    sentences: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        clean = re.sub(r"[^a-z0-9 ]+", "", sentence.lower()).strip()
        if len(clean.split()) >= 5:
            sentences.append(clean)
    return sentences


def _duplication_rate(items: list[str]) -> float:
    if not items:
        return 0.0
    return round(1 - (len(set(items)) / len(items)), 3)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    words_a = set(re.findall(r"\b[a-z]{3,}\b", (text_a or "").lower()))
    words_b = set(re.findall(r"\b[a-z]{3,}\b", (text_b or "").lower()))
    if not words_a or not words_b:
        return 0.0
    return round(len(words_a & words_b) / len(words_a | words_b), 3)


def _sentences_for_readability(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or "")
        if sentence.strip()
    ]


def _words_for_readability(text: str) -> list[str]:
    return re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text or "")


def _syllable_count(word: str) -> int:
    clean = re.sub(r"[^a-z]", "", (word or "").lower())
    if not clean:
        return 0
    clean = re.sub(r"e$", "", clean)
    groups = re.findall(r"[aeiouy]+", clean)
    return max(1, len(groups))


def _flesch_kincaid_grade(text: str) -> float:
    sentences = _sentences_for_readability(text)
    words = _words_for_readability(text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_syllable_count(word) for word in words)
    return round(
        0.39 * (len(words) / len(sentences))
        + 11.8 * (syllables / len(words))
        - 15.59,
        2,
    )


def _flesch_reading_ease(text: str) -> float:
    sentences = _sentences_for_readability(text)
    words = _words_for_readability(text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_syllable_count(word) for word in words)
    return round(
        206.835
        - 1.015 * (len(words) / len(sentences))
        - 84.6 * (syllables / len(words)),
        2,
    )


def _technical_terms(text: str) -> list[str]:
    stopwords = {
        "because", "between", "different", "important", "research", "summary",
        "treatment", "patients", "people", "project", "condition", "currently",
    }
    terms: list[str] = []
    for word in _words_for_readability(text):
        clean = word.lower().strip("'")
        if len(clean) < 11 or clean in stopwords:
            continue
        if clean not in terms:
            terms.append(clean)
    return terms


def _format_plain_english_analysis(
    section_chunk_ids: dict[str, list[str]],
    pool_lookup: dict[str, dict[str, str]],
) -> str:
    summary_section = None
    for candidate in ("Plain English Summary of Research", "Plain English Summary"):
        if candidate in section_chunk_ids:
            summary_section = candidate
            break
    if not summary_section:
        return ""

    summary_text = "\n\n".join(
        pool_lookup[chunk_id]["text"]
        for chunk_id in section_chunk_ids.get(summary_section, [])
    )
    if not summary_text.strip():
        return ""

    detailed_text = "\n\n".join(
        pool_lookup[chunk_id]["text"]
        for chunk_id in section_chunk_ids.get("Detailed Research Plan", [])
    )
    words = _words_for_readability(summary_text)
    sentences = _sentences_for_readability(summary_text)
    sentence_lengths = [
        len(_words_for_readability(sentence))
        for sentence in sentences
    ]
    avg_sentence_length = round(sum(sentence_lengths) / len(sentence_lengths), 2) if sentence_lengths else 0.0
    long_sentence_ratio = round(
        sum(1 for length in sentence_lengths if length >= 30) / len(sentence_lengths),
        3,
    ) if sentence_lengths else 0.0
    terms = _technical_terms(summary_text)
    jargon_density = round((len(terms) / len(words)) * 100, 2) if words else 0.0
    alignment = _jaccard_similarity(summary_text, detailed_text) if detailed_text else 0.0
    coverage_terms = {
        "problem": r"\b(problem|condition|burden|currently|uncertainty|need)\b",
        "objectives": r"\b(aim|objective|will|project|develop|evaluate|identify)\b",
        "methods": r"\b(method|data|dataset|model|interview|review|analysis|study)\b",
        "beneficiaries": r"\b(patient|people|clinician|public|service|nhs)\b",
        "impact": r"\b(benefit|improve|impact|personalised|save|reduce|support)\b",
    }
    coverage_hits = [
        label for label, pattern in coverage_terms.items()
        if re.search(pattern, summary_text, flags=re.IGNORECASE)
    ]

    return "\n".join([
        "Plain English Summary NLP analysis derived from parser output.",
        "Use these metrics as supporting evidence for pr.1, but also read the raw Plain English Summary and "
        "Detailed Research Plan in application_text. Do not claim readability, jargon, alignment, or sentence "
        "coherence evidence is missing solely because Stage 1 did not provide NLP/coherence findings.",
        "",
        "Readability and sentence structure metrics:",
        f"- source_section={summary_section}",
        f"- word_count={len(words)}",
        f"- sentence_count={len(sentences)}",
        f"- avg_sentence_length_words={avg_sentence_length}",
        f"- long_sentence_ratio_30_words={long_sentence_ratio}",
        f"- flesch_kincaid_grade_estimate={_flesch_kincaid_grade(summary_text)}",
        f"- flesch_reading_ease_estimate={_flesch_reading_ease(summary_text)}",
        "",
        "Jargon proxy metrics:",
        f"- unexplained_jargon_proxy_density_pct={jargon_density}",
        f"- technical_terms_sample={terms[:12]}",
        "- Jargon proxy is based on long/difficult-looking terms; final scoring must still judge whether terms "
        "are explained clearly in the summary text.",
        "",
        "Alignment and content coverage metrics:",
        f"- lexical_overlap_with_detailed_research_plan={alignment}",
        f"- lay_summary_coverage_hits={coverage_hits}",
        "- Alignment metric is lexical only; final scoring must compare the actual plain-English claims with "
        "the detailed proposal content.",
    ])


def _format_application_form_analysis(
    section_chunk_ids: dict[str, list[str]],
    pool_lookup: dict[str, dict[str, str]],
) -> str:
    section_rows: list[tuple[str, list[str], str]] = []
    for section_name, chunk_ids in section_chunk_ids.items():
        if section_name == APPLICATION_FORM_ANALYSIS_SECTION:
            continue
        text = "\n\n".join(pool_lookup[chunk_id]["text"] for chunk_id in chunk_ids)
        if text.strip():
            section_rows.append((section_name, chunk_ids, text))

    if not section_rows:
        return ""

    all_text = "\n\n".join(text for _, _, text in section_rows)
    non_budget_text = "\n\n".join(
        text
        for section_name, _, text in section_rows
        if "budget" not in section_name.lower()
    ) or all_text
    all_lines = _normalized_lines(all_text)
    non_budget_sentences = _sentence_tokens(non_budget_text)
    bullet_marker_count = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", all_text))
    numbered_heading_count = len(re.findall(r"(?m)^\s*\d+(?:\.\d+)*[.)]?\s+[A-Z][^\n]{3,120}$", all_text))
    table_like_line_count = len(re.findall(r"(?im)\b(year\s+1|year\s+2|year\s+3|total cost|total \(|£)\b", all_text))
    emphasis_marker_count = len(re.findall(r"(\*\*|__|<b>|</b>|\b[A-Z][A-Z /&-]{8,}\b)", all_text))
    transition_count = len(re.findall(
        r"\b(however|therefore|furthermore|moreover|in addition|to do this|for example|"
        r"as a result|this will|this project|aligns? with|building on|in phase|phase \d)\b",
        all_text,
        flags=re.IGNORECASE,
    ))
    objective_method_link_count = len(re.findall(
        r"\b(aims?|objectives?|research questions?|methods?|workstreams?|work packages?|"
        r"phase \d|project plan|data analysis|impact|dissemination|budget|justification)\b",
        all_text,
        flags=re.IGNORECASE,
    ))

    section_summary_lines = [
        f"- {section_name}: words={_word_count(text)}"
        for section_name, chunk_ids, text in section_rows
    ]

    overlap_rows: list[tuple[str, str, float]] = []
    for idx, (section_a, _, text_a) in enumerate(section_rows):
        for section_b, _, text_b in section_rows[idx + 1:]:
            score = _jaccard_similarity(text_a, text_b)
            if score >= 0.18:
                overlap_rows.append((section_a, section_b, score))
    overlap_rows = sorted(overlap_rows, key=lambda row: row[2], reverse=True)[:8]
    overlap_lines = [
        f"- {section_a} <-> {section_b}: lexical_overlap={score}"
        for section_a, section_b, score in overlap_rows
    ] or ["- No high cross-section lexical overlap detected at threshold 0.18."]

    return "\n".join([
        "Application form structural analysis derived from parser output.",
        "Use this single derived chunk as evidence for Application Form criteria af.*.",
        "",
        "Section coverage and hierarchy:",
        *section_summary_lines,
        "",
        "Formatting and structure indicators:",
        f"- parser_sections_detected={len(section_rows)}",
        f"- bullet_or_numbered_list_markers={bullet_marker_count}",
        f"- numbered_heading_like_lines={numbered_heading_count}",
        f"- table_like_budget_lines={table_like_line_count}",
        f"- extracted_emphasis_markers={emphasis_marker_count}",
        "- Parser limitation: bold/emphasis may be lost during text extraction; use extracted headings, "
        "section labels, list markers, and table structure as the available evidence.",
        "",
        "Duplication and repetition indicators:",
        f"- duplicate_sentence_rate_excluding_budget={_duplication_rate(non_budget_sentences)}",
        f"- repeated_line_rate={_duplication_rate(all_lines)}",
        *overlap_lines,
        "",
        "Logical flow and coherence indicators:",
        f"- transition_phrase_count={transition_count}",
        f"- objective_method_budget_link_terms={objective_method_link_count}",
        "- Section order moves from applicant/context, plain summary and abstract, research plan, PPI, "
        "training/support, and budget where those sections are present.",
    ])


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

    plain_english_analysis = _format_plain_english_analysis(section_chunk_ids, pool_lookup)
    if plain_english_analysis:
        add_leaf(
            PLAIN_ENGLISH_ANALYSIS_SECTION,
            PLAIN_ENGLISH_ANALYSIS_SECTION,
            plain_english_analysis,
            split=False,
        )

    application_form_analysis = _format_application_form_analysis(section_chunk_ids, pool_lookup)
    if application_form_analysis:
        add_leaf(
            APPLICATION_FORM_ANALYSIS_SECTION,
            APPLICATION_FORM_ANALYSIS_SECTION,
            application_form_analysis,
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
