#!/usr/bin/env bash
# End-to-end deploy: AlloyDB + Cloud Run service + Cloud Run ingest job.
# Run step by step the first time (copy/paste each block); every step is idempotent-ish but slow ones
# (AlloyDB cluster/instance creation) take 10-20 minutes.
#
# Cost: by default this creates an AlloyDB *free trial* cluster (8 vCPU, 30 days, no instance charge,
# one per project). Set ALLOYDB_TRIAL=false for a paid 2 vCPU cluster, which bills by the hour
# (roughly $0.35+/hour) until you delete it. Cloud Run scales to zero; Gemini calls bill per token.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
CLUSTER="${CLUSTER:-rag-cluster}"
INSTANCE="${INSTANCE:-rag-primary}"
NETWORK="${NETWORK:-default}"
SERVICE="${SERVICE:-rag-api}"
SA_NAME="${SA_NAME:-rag-runtime}"
ALLOYDB_TRIAL="${ALLOYDB_TRIAL:-true}"
LABELS="${LABELS:-app=vertex-alloydb-rag,purpose=gde}"
DB_PASSWORD="${DB_PASSWORD:?set DB_PASSWORD (postgres user password for the new cluster)}"
SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"

echo "== 1. APIs"
gcloud services enable aiplatform.googleapis.com alloydb.googleapis.com run.googleapis.com \
  servicenetworking.googleapis.com secretmanager.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com compute.googleapis.com

echo "== 2. Private services access (AlloyDB has private IPs only by default)"
gcloud compute addresses describe alloydb-psa --global >/dev/null 2>&1 || \
  gcloud compute addresses create alloydb-psa --global --purpose=VPC_PEERING \
    --prefix-length=16 --network="$NETWORK"
gcloud services vpc-peerings connect --service=servicenetworking.googleapis.com \
  --ranges=alloydb-psa --network="$NETWORK" || true

echo "== 3. AlloyDB cluster + primary instance (slow)"
if [[ "$ALLOYDB_TRIAL" == "true" ]]; then
  SUBSCRIPTION=TRIAL;    CPU=8      # trial clusters are fixed at 8 vCPU, free
else
  SUBSCRIPTION=STANDARD; CPU=2        # smallest paid shape; billed hourly
fi
gcloud alloydb clusters describe "$CLUSTER" --region="$REGION" >/dev/null 2>&1 || \
  gcloud alloydb clusters create "$CLUSTER" --region="$REGION" --network="$NETWORK" \
    --password="$DB_PASSWORD" --subscription-type="$SUBSCRIPTION"
gcloud alloydb instances describe "$INSTANCE" --cluster="$CLUSTER" --region="$REGION" >/dev/null 2>&1 || \
  gcloud alloydb instances create "$INSTANCE" --cluster="$CLUSTER" --region="$REGION" \
    --instance-type=PRIMARY --cpu-count="$CPU"
DB_IP=$(gcloud alloydb instances describe "$INSTANCE" --cluster="$CLUSTER" --region="$REGION" \
  --format='value(ipAddress)')
echo "AlloyDB private IP: $DB_IP"

echo "== 4. Runtime service account (least privilege)"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name="RAG runtime"
for role in roles/aiplatform.user roles/secretmanager.secretAccessor roles/alloydb.client; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$SA" \
    --role="$role" --condition=None >/dev/null
done

echo "== 5. Connection string in Secret Manager (never in env vars or the image)"
DSN="postgresql://postgres:${DB_PASSWORD}@${DB_IP}:5432/postgres"
if gcloud secrets describe rag-database-url >/dev/null 2>&1; then
  printf '%s' "$DSN" | gcloud secrets versions add rag-database-url --data-file=-
else
  printf '%s' "$DSN" | gcloud secrets create rag-database-url --data-file=-
fi

COMMON_FLAGS=(
  --region="$REGION" --service-account="$SA"
  --network="$NETWORK" --subnet=default --vpc-egress=private-ranges-only
  --set-secrets=DATABASE_URL=rag-database-url:latest
  --set-env-vars=GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT="$PROJECT_ID",GOOGLE_CLOUD_LOCATION="$REGION",ALLOYDB=true
)

echo "== 6. Ingest job (creates schema + ScaNN index, embeds data/docs)"
gcloud run jobs deploy rag-ingest --source . "${COMMON_FLAGS[@]}" \
  --command=python --args=-m,rag.ingest,data/docs --max-retries=0 --labels="$LABELS"
gcloud run jobs execute rag-ingest --region="$REGION" --wait

echo "== 6b. Eval job on AlloyDB (prints hit rate, MRR and p95 to the job logs)"
gcloud run jobs deploy rag-eval --source . "${COMMON_FLAGS[@]}" \
  --command=python --args=-m,eval.run_eval --max-retries=0 --labels="$LABELS"
gcloud run jobs execute rag-eval --region="$REGION" --wait
echo "Eval results:"
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=rag-eval" \
  --limit=50 --freshness=15m --order=asc --format='value(textPayload)'

echo "== 7. API service (private: callers need roles/run.invoker)"
gcloud run deploy "$SERVICE" --source . "${COMMON_FLAGS[@]}" --no-allow-unauthenticated \
  --cpu=1 --memory=1Gi --concurrency=40 --min-instances=0 --max-instances=2 \
  --labels="$LABELS"

URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')
echo
echo "Try it:"
echo "curl -s -X POST $URL/ask -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" \\"
echo "  -H 'Content-Type: application/json' -H 'X-Tenant-Id: larkspur' \\"
echo "  -d '{\"question\": \"How do I roll back a Cloud Run deploy?\"}' | jq"
echo
echo "When you're done: ./deploy/teardown.sh  (deletes the cluster, service, job and images)"
