.PHONY: db test ingest eval serve
db:      ; docker compose up -d --wait && docker compose exec -T db psql -U rag -c "CREATE DATABASE rag_test" || true
test:    ; pytest -q
ingest:  ; python -m rag.ingest data/docs
eval:    ; python -m eval.run_eval
serve:   ; uvicorn rag.api:app --reload --port 8080
