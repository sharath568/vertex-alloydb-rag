def test_ingest_is_idempotent(settings, embedder, indexed):
    from rag.ingest import ingest_dir

    assert indexed["indexed"] == 6
    again = ingest_dir("data/docs", settings, embedder)
    assert again == {"indexed": 0, "skipped": 6}


def test_keyword_side_finds_exact_error_codes(settings, embedder, indexed):
    from rag.retrieve import retrieve

    hits = retrieve("ERR_4021", tenant="larkspur", settings=settings, embedder=embedder)
    assert hits[0].doc_id == "larkspur/incidents.md"
    assert hits[0].keyword_rank == 1


def test_tenant_isolation(settings, embedder, indexed):
    from rag.retrieve import retrieve

    # Both tenants have a "Backups" section. Harbor must only ever see Harbor's.
    hits = retrieve("how long are backups kept", tenant="harbor", settings=settings, embedder=embedder)
    assert hits and all(h.doc_id.startswith("harbor/") for h in hits)


def test_eval_meets_baseline(indexed):
    from eval.run_eval import run

    report = run(k=5)
    assert report["tenant_leaks"] == 0
    # Fake lexical embedder baseline. With gemini-embedding-001 you should see ~1.0.
    assert report["hit_rate@5"] >= 0.8


def test_vector_index_is_used_and_antipattern_is_not(settings, indexed):
    """Guard against query shapes that silently skip the vector index.

    Uses a throwaway tenant with 5,000 random vectors so the planner has a reason to use HNSW.
    """
    import numpy as np
    import psycopg
    from pgvector.psycopg import register_vector
    from rag.retrieve import HYBRID_SQL

    rng = np.random.default_rng(0)

    def unit():
        v = rng.standard_normal(768)
        return (v / np.linalg.norm(v)).astype(np.float32)

    params = {"qvec": unit(), "qtext": "backups", "tenant": "bulk",
              "candidates": 40, "rrf_k": 60, "top_k": 5}
    antipattern = """SELECT chunk_id FROM chunks WHERE tenant_id = %(tenant)s
                     ORDER BY 1 - (embedding <=> %(qvec)s::vector) DESC LIMIT %(candidates)s"""

    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        conn.execute("INSERT INTO documents VALUES ('bulk/x.md', 'bulk', 'Bulk', 'x', 'h')")
        # Bulk-load without the index, then build it once: far faster than 5,000 incremental inserts.
        conn.execute("DROP INDEX chunks_embedding_hnsw")
        with conn.cursor() as cur, cur.copy(
            "COPY chunks (doc_id, tenant_id, chunk_no, content, embedding) FROM STDIN"
        ) as copy:
            for i in range(5000):
                vec = "[" + ",".join(f"{x:.5f}" for x in unit()) + "]"
                copy.write_row(("bulk/x.md", "bulk", i, "filler text", vec))
        conn.execute("SET maintenance_work_mem = '256MB'")
        conn.execute("CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
        conn.execute("ANALYZE chunks")
        explain = lambda sql: "\n".join(r[0] for r in conn.execute("EXPLAIN " + sql, params))
        good, bad = explain(HYBRID_SQL), explain(antipattern)
        conn.rollback()  # leave the shared test data untouched

    assert "chunks_embedding_hnsw" in good, good
    assert "chunks_embedding_hnsw" not in bad, bad
