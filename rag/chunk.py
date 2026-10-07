"""Heading-aware Markdown chunker.

Most RAG failures I've debugged start here, not in the vector index. Two rules:
1. Never split across a heading. A chunk that mixes two sections retrieves badly for both.
2. Keep the heading path with the chunk ("Backups > Restore"), so the embedding and the
   keyword index both know what section the text came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_no: int
    heading: str
    content: str


def _approx_tokens(text: str) -> int:
    # ~0.75 words per token for English prose. Close enough for sizing chunks.
    return int(len(text.split()) / 0.75)


def _split_long(paragraphs: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Pack paragraphs into windows of ~max_tokens, carrying ~overlap_tokens of tail words."""
    out: list[str] = []
    cur: list[str] = []
    for p in paragraphs:
        candidate = "\n\n".join(cur + [p])
        if cur and _approx_tokens(candidate) > max_tokens:
            out.append("\n\n".join(cur))
            tail_words = " ".join(cur).split()[-int(overlap_tokens * 0.75) :] if overlap_tokens else []
            cur = ([" ".join(tail_words)] if tail_words else []) + [p]
        else:
            cur.append(p)
    if cur:
        out.append("\n\n".join(cur))
    # A single giant paragraph still needs splitting by words.
    final: list[str] = []
    max_words = int(max_tokens * 0.75)
    step = max(1, max_words - int(overlap_tokens * 0.75))
    for piece in out:
        words = piece.split()
        if len(words) <= max_words:
            final.append(piece)
        else:
            for i in range(0, len(words), step):
                final.append(" ".join(words[i : i + max_words]))
                if i + max_words >= len(words):
                    break
    return final


def chunk_markdown(text: str, max_tokens: int = 350, overlap_tokens: int = 50) -> list[Chunk]:
    sections: list[tuple[str, list[str]]] = []
    path: list[str] = []
    buf: list[str] = []
    current_heading = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            sections.append((current_heading, paras))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            path[:] = path[: level - 1] + [title]
            current_heading = " > ".join(path)
        else:
            buf.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading, paras in sections:
        for piece in _split_long(paras, max_tokens, overlap_tokens):
            chunks.append(Chunk(chunk_no=len(chunks), heading=heading, content=piece))
    return chunks
