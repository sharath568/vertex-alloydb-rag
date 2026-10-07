-- Works on plain Postgres 16 + pgvector (local) and on AlloyDB.
-- The vector size must match EMBEDDING_DIM (768 by default).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id       text PRIMARY KEY,               -- stable id, e.g. the file path
    tenant_id    text NOT NULL,
    title        text NOT NULL,
    source_uri   text NOT NULL,
    content_hash text NOT NULL,                  -- skip re-embedding unchanged docs
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_id      text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    tenant_id   text NOT NULL,                   -- denormalized so filters never need a join
    chunk_no    int  NOT NULL,
    heading     text NOT NULL DEFAULT '',
    content     text NOT NULL,
    embedding   vector(768) NOT NULL,
    -- Keyword side of hybrid search. Generated, so it can never drift from content.
    tsv         tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', coalesce(heading, '')), 'A') ||
                    setweight(to_tsvector('english', content), 'B')
                ) STORED,
    UNIQUE (doc_id, chunk_no)
);

CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx    ON chunks USING gin (tsv);

-- Local / plain Postgres: HNSW from pgvector. On AlloyDB, run 002_alloydb_scann.sql instead.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
