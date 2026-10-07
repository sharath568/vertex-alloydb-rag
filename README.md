# vertex-alloydb-rag

Production-minded RAG on Google Cloud: **Gemini embeddings on Vertex AI**, **hybrid search in AlloyDB** (vector + full-text in one SQL query), **grounded answers with checked citations**, and an **eval you can run on every change**. Deploys to Cloud Run.

Companion code for the article *"From demo to production: RAG on Vertex AI and AlloyDB AI"*.

```
question ──► gemini-embedding-001 (RETRIEVAL_QUERY, 768d, normalized)
                 │
                 ▼
        ┌──────────────── AlloyDB (one SQL statement) ────────────────┐
        │  semantic top-40  (ScaNN / HNSW, cosine, WHERE tenant_id=…) │
        │  keyword  top-40  (tsvector + websearch_to_tsquery)          │
        │  fuse with Reciprocal Rank Fusion  →  top-5                  │
        └──────────────────────────────────────────────────────────────┘
                 │
     best similarity < MIN_SIMILARITY ? ──yes──► refuse (no LLM call)
                 │ no
                 ▼
        Gemini (JSON schema: answer, citations[], grounded)
                 │
     drop citations that weren't in the context; no valid citation ► refuse
                 ▼
          answer + sources
```

## What makes it "production"

| Problem you hit after the demo | What this repo does |
|---|---|
| Exact tokens (error codes, flags, ticket ids) don't retrieve | Hybrid search: pgvector/ScaNN + Postgres full-text, fused with RRF in one query |
| One customer's docs show up in another's answers | `tenant_id` filter inside both candidate sets, before ranking. Tested |
| Confident answers when nothing relevant was found | Similarity gate refuses before calling the model |
| Citations that point at nothing | Structured output + citation validation |
| Re-embedding everything on every run | Content-hash idempotent ingestion |
| A query change quietly stops using the vector index | Test that checks the query plan |
| "Is it better or worse?" after a tweak | `eval/run_eval.py`: hit rate@k, MRR, tenant leaks, latency |

## Quickstart (local, 5 minutes, no cloud needed)

```bash
docker compose up -d --wait                       # Postgres 16 + pgvector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
docker compose exec -T db psql -U rag -c "CREATE DATABASE rag_test"
pytest -q                                         # offline: fake embedder + scripted model

EMBEDDER=fake python -m rag.ingest data/docs
EMBEDDER=fake python -m eval.run_eval
```

## With Gemini on Vertex AI

```bash
gcloud auth application-default login
cp .env.example .env       # set GOOGLE_CLOUD_PROJECT, keep EMBEDDER=gemini
set -a; source .env; set +a
python -m rag.models       # confirm the model IDs your project can see
python -m rag.ingest data/docs
python -m eval.run_eval    # expect hit_rate@5 close to 1.0 on the sample docs
uvicorn rag.api:app --port 8080

curl -s localhost:8080/ask -H 'Content-Type: application/json' -H 'X-Tenant-Id: larkspur' \
  -d '{"question":"What does ERR_4021 mean?"}' | jq
```

## Deploy to AlloyDB + Cloud Run

```bash
PROJECT_ID=my-project DB_PASSWORD='choose-a-strong-one' ./deploy/deploy.sh
```

The script enables APIs, sets up private services access, creates an AlloyDB cluster and primary, a least-privilege runtime service account, stores the DSN in Secret Manager, runs ingestion as a Cloud Run job (which builds the ScaNN index after loading), and deploys the API as a private Cloud Run service with Direct VPC egress. By default it creates an [AlloyDB free trial cluster](https://cloud.google.com/alloydb/docs/free-trial-cluster) (30 days, no instance charge, one per project); set `ALLOYDB_TRIAL=false` for a paid cluster. Run `./deploy/teardown.sh` when you're done so nothing keeps billing.

## Layout

```
rag/
  config.py     env-driven settings
  embed.py      Gemini embedder (task types + normalization) and an offline fake
  chunk.py      heading-aware Markdown chunker
  ingest.py     idempotent ingestion; builds ScaNN on AlloyDB after load
  retrieve.py   the hybrid RRF query
  answer.py     gate → Gemini JSON → citation validation
  api.py        FastAPI for Cloud Run
sql/            schema (pgvector HNSW) and AlloyDB ScaNN
eval/           golden questions + retrieval metrics
tests/          offline tests, incl. tenant isolation and query-plan checks
deploy/         AlloyDB + Cloud Run deploy and teardown scripts
data/docs/      sample docs for two fictional tenants
```

## Tuning notes

- `EMBEDDING_DIM` must match `vector(768)` in `sql/001_schema.sql`. Change both together.
- `MIN_SIMILARITY` depends on the embedding model and your data. Run the eval, look at the similarity of correct vs. unanswerable questions, and set it between them.
- On pgvector 0.8+, the pool sets `hnsw.iterative_scan = relaxed_order` so filtered vector searches don't come back short.

## License

Apache-2.0
