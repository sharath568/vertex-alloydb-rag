"""Idempotent ingestion: re-running it only re-embeds documents whose content changed.

Usage:
    python -m rag.ingest data/docs            # tenant taken from front matter, default "default"
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

from .chunk import chunk_markdown
from .config import Settings, get_settings
from .db import apply_schema, build_scann_index, pool
from .embed import Embedder, get_embedder

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[m.end() :]


def ingest_text(
    *,
    doc_id: str,
    text: str,
    source_uri: str,
    settings: Settings,
    embedder: Embedder,
) -> str:
    """Returns 'skipped' | 'indexed'."""
    meta, body = parse_front_matter(text)
    tenant = meta.get("tenant", "default")
    title = meta.get("title") or Path(doc_id).stem.replace("-", " ").title()
    content_hash = hashlib.sha256(text.encode()).hexdigest()

    with pool(settings).connection() as conn:
        row = conn.execute(
            "SELECT content_hash FROM documents WHERE doc_id = %s", (doc_id,)
        ).fetchone()
        if row and row[0] == content_hash:
            return "skipped"

        chunks = chunk_markdown(body, settings.chunk_tokens, settings.chunk_overlap)
        # Embed the heading together with the text: it is cheap context that helps retrieval a lot.
        vectors = embedder.embed_documents(
            [f"{title}\n{c.heading}\n{c.content}" for c in chunks]
        )

        with conn.transaction():
            conn.execute(
                """
                INSERT INTO documents (doc_id, tenant_id, title, source_uri, content_hash)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE
                  SET tenant_id = EXCLUDED.tenant_id, title = EXCLUDED.title,
                      source_uri = EXCLUDED.source_uri, content_hash = EXCLUDED.content_hash,
                      updated_at = now()
                """,
                (doc_id, tenant, title, source_uri, content_hash),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (doc_id, tenant_id, chunk_no, heading, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (doc_id, tenant, c.chunk_no, c.heading, c.content, np.asarray(v, dtype=np.float32))
                        for c, v in zip(chunks, vectors)
                    ],
                )
    return "indexed"


def ingest_dir(
    path: str | Path, settings: Settings | None = None, embedder: Embedder | None = None
) -> dict:
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    apply_schema(settings)
    stats = {"indexed": 0, "skipped": 0}
    root = Path(path)
    for f in sorted(root.rglob("*.md")):
        doc_id = str(f.relative_to(root))
        result = ingest_text(
            doc_id=doc_id,
            text=f.read_text(),
            source_uri=f"docs/{doc_id}",
            settings=settings,
            embedder=embedder,
        )
        stats[result] += 1
    if settings.alloydb:
        stats["vector_index"] = build_scann_index(settings)
    return stats


if __name__ == "__main__":
    print(ingest_dir(sys.argv[1] if len(sys.argv) > 1 else "data/docs"))
