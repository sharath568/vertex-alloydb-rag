-- AlloyDB only. Run AFTER the initial bulk load (rag.ingest does this for you when ALLOYDB=true).
-- ScaNN learns its partitioning from the rows already in the table, so build it on real data.
-- Docs: https://docs.cloud.google.com/alloydb/docs/ai/create-scann-index

CREATE EXTENSION IF NOT EXISTS alloydb_scann CASCADE;

-- Auto-tuned ScaNN index with cosine distance.
CREATE INDEX IF NOT EXISTS chunks_embedding_scann
    ON chunks USING scann (embedding cosine);

-- Only drop HNSW once ScaNN exists, so there is never a window with no vector index.
DROP INDEX IF EXISTS chunks_embedding_hnsw;

-- Optional: let AlloyDB compute embeddings in SQL (google_ml_integration).
-- Useful for backfills and for apps that only send text. Check the output size of the
-- model you call: it must match the vector(768) column, or store it in its own column.
--
-- CREATE EXTENSION IF NOT EXISTS google_ml_integration;
-- SELECT google_ml.embedding(model_id => 'gemini-embedding-001',
--                            content  => 'how do I rotate the db password?')::vector;
