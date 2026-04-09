"""
Stitch retrieved chunks back into a structured evidence string preserving
the original section order and intra-section offsets.
"""
from __future__ import annotations

from .retriever import Chunk, CHUNK_SIZE


def stitch_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(no evidence retrieved)"

    # Sort by (section_order, char_offset) so the LLM sees evidence in
    # document order, grouped by section.
    ordered = sorted(chunks, key=lambda c: (c.section_order, c.char_offset))

    blocks: list[str] = []
    by_section: dict[tuple[int, str], list[Chunk]] = {}
    for ch in ordered:
        by_section.setdefault((ch.section_order, ch.section), []).append(ch)

    for (_, section_name), section_chunks in by_section.items():
        header = f"=== {section_name} ({len(section_chunks)} excerpt"
        header += "s" if len(section_chunks) != 1 else ""
        header += ") ==="
        lines = [header]

        prev_end = -1
        for ch in section_chunks:
            # If this chunk is contiguous with the previous one, merge silently.
            if prev_end >= 0 and ch.char_offset <= prev_end:
                lines.append(ch.text)
            else:
                lines.append(f"[§@{ch.char_offset}] {ch.text}")
            prev_end = ch.char_offset + len(ch.text)
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
