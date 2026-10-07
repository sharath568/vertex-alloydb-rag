#!/usr/bin/env bash
# Delete everything deploy.sh created that can cost money. Safe to run more than once.
# Keeps: the service account and the private services access range (both free).
set -uo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
CLUSTER="${CLUSTER:-rag-cluster}"
SERVICE="${SERVICE:-rag-api}"

gcloud config set project "$PROJECT_ID" >/dev/null

echo "== Cloud Run service and job"
gcloud run services delete "$SERVICE" --region="$REGION" --quiet 2>/dev/null || true
gcloud run jobs delete rag-ingest --region="$REGION" --quiet 2>/dev/null || true
gcloud run jobs delete rag-eval --region="$REGION" --quiet 2>/dev/null || true

echo "== AlloyDB cluster (and its instances)"
gcloud alloydb clusters delete "$CLUSTER" --region="$REGION" --force --quiet 2>/dev/null || true

echo "== Secret"
gcloud secrets delete rag-database-url --quiet 2>/dev/null || true

echo "== Container images left by 'gcloud run deploy --source'"
gcloud artifacts repositories delete cloud-run-source-deploy --location="$REGION" --quiet 2>/dev/null || true

echo
echo "Left running (should be empty):"
gcloud alloydb clusters list --format='value(name)' 2>/dev/null
gcloud run services list --region="$REGION" --format='value(metadata.name)' 2>/dev/null
