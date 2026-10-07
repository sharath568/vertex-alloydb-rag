"""Hybrid retrieval in one SQL statement: vector + full-text, fused with Reciprocal Rank Fusion.

Why hybrid: embeddings are great at "how do I restore last night's backup?" and bad at exact
tokens like an error code (ERR_4021), a flag name, or a ticket id. Full-text search is the
opposite. RRF merges the two rankings without having to calibrate their scores against each other.

Why the tenant filter lives in both CTEs: it is a security boundary, not a ranking hint.
Every candidate set is filtered before ranking, so another tenant's text can never be retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings
from .db import pool
from .embed import Embedder

HYBRID_SQL = """
-- The nearest-neighbour subquery orders by the raw distance operator (embedding <=> param),
-- ascending. That's the only shape the HNSW / ScaNN index can serve. Two common variants
-- silently fall back to a full scan: ORDER BY 1 - (embedding <=> q) DESC ("similarity"), and
-- passing the vector through a MATERIALIZED CTE. tests/test_retrieval.py checks the plan.
WITH semantic AS (
    SELECT chunk_id,
           row_number() OVER (ORDER BY embedding <=> %(qvec)s::vector) AS rnk
    FROM (
        SELECT chunk_id, embedding
        FROM chunks
        WHERE tenant_id = %(tenant)s
        ORDER BY embedding <=> %(qvec)s::vector
        LIMIT %(candidates)s
    ) nn
),
keyword AS (
    SELECT chunk_id,
           row_number() OVER (ORDER BY score DESC) AS rnk
    FROM (
        SELECT chunk_id, ts_rank_cd(tsv, websearch_to_tsquery('english', %(qtext)s)) AS score
        FROM chunks
        WHERE tenant_id = %(tenant)s
          AND tsv @@ websearch_to_tsquery('english', %(qtext)s)
        ORDER BY score DESC
        LIMIT %(candidates)s
    ) kw
),
fused AS (
    SELECT chunk_id,
           coalesce(1.0 / (%(rrf_k)s + s.rnk), 0) + coalesce(1.0 / (%(rrf_k)s + k.rnk), 0) AS rrf,
           s.rnk AS semantic_rank,
           k.rnk AS keyword_rank
    FROM semantic s FULL OUTER JOIN keyword k USING (chunk_id)
)
SELECT c.chunk_id, c.doc_id, d.title, d.source_uri, c.heading, c.content,
       f.rrf, f.semantic_rank, f.keyword_rank,
       1 - (c.embedding <=> %(qvec)s::vector) AS similarity
FROM fused f
JOIN chunks c USING (chunk_id)
JOIN documents d ON d.doc_id = c.doc_id
ORDER BY f.rrf DESC, similarity DESC
LIMIT %(top_k)s
"""


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    title: str
    source_uri: str
    heading: str
    content: str
    rrf: float
    semantic_rank: int | None
    keyword_rank: int | None
    similarity: float


def retrieve(
    question: str,
    *,
    tenant: str,
    settings: Settings,
    embedder: Embedder,
    top_k: int | None = None,
) -> list[Hit]:
    qvec = np.asarray(embedder.embed_query(question), dtype=np.float32)
    params = {
        "qvec": qvec,
        "qtext": question,
        "tenant": tenant,
        "candidates": settings.candidates,
        "rrf_k": settings.rrf_k,
        "top_k": top_k or settings.top_k,
    }
    with pool(settings).connection() as conn:
        rows = conn.execute(HYBRID_SQL, params).fetchall()
    return [
        Hit(
            chunk_id=r[0], doc_id=r[1], title=r[2], source_uri=r[3], heading=r[4],
            content=r[5], rrf=float(r[6]), semantic_rank=r[7], keyword_rank=r[8],
            similarity=float(r[9]),
        )
        for r in rows
    ]
