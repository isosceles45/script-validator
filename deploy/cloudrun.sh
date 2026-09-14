#!/usr/bin/env bash
#
# Deploy the TFS Creative Script Validator to Cloud Run.
#
# Builds remotely via Cloud Build, so local Docker is not required. One image
# serves the API and the frontend; the manual corpus ships inside it as an
# embedded vector store, so there is no database to provision for retrieval.
#
#   ./deploy/cloudrun.sh
#
# Override any of the settings below via environment variables.

set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${GCP_REGION:-asia-south1}"
SERVICE="${SERVICE_NAME:-tfs-script-validator}"
BUCKET="${RUNS_BUCKET:-${PROJECT}-script-validator-runs}"
COLLECTION="${FIRESTORE_COLLECTION:-script_validator_runs}"
PROVIDER="${PROVIDER:-openai}"
SECRET="${SECRET_NAME:-script-validator-llm-key}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight --

command -v gcloud >/dev/null || die "gcloud not found. Install the Google Cloud CLI first."
[[ -n "$PROJECT" ]] || die "No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"

# The corpus is baked into the image at build time. Without it the service
# deploys perfectly and marks every single claim unverifiable -- the worst
# possible failure, because it looks like it is working.
[[ -s data/store.sqlite3 ]] || die "data/store.sqlite3 is missing or empty. Run 'make ingest' first."
[[ -s data/eval/golden.yaml ]] || die "data/eval/golden.yaml is missing -- retrieval eval would report no_golden_set on every run."

# Existing on disk is not enough: `gcloud run deploy --source .` uploads what
# .gcloudignore allows, and with no .gcloudignore it derives exclusions from
# .gitignore -- which excludes the 12MB store. The build then fails deep in
# Cloud Build with "file not found in build context", which reads like a
# Dockerfile bug. Check the actual upload set instead.
UPLOAD="$(gcloud meta list-files-for-upload . 2>/dev/null || true)"
if [[ -n "$UPLOAD" ]]; then
  grep -qx "data/store.sqlite3" <<<"$UPLOAD" \
    || die "data/store.sqlite3 exists but is excluded from the source upload. Check .gcloudignore -- without that file gcloud falls back to .gitignore, which excludes it."
  grep -qx "data/eval/golden.yaml" <<<"$UPLOAD" \
    || die "data/eval/golden.yaml is excluded from the source upload. Check .gcloudignore."
fi

case "$PROVIDER" in
  openai) KEY="${OPENAI_API_KEY:-}" ; KEY_ENV="OPENAI_API_KEY" ;;
  gemini) KEY="${GOOGLE_API_KEY:-}" ; KEY_ENV="GOOGLE_API_KEY" ;;
  *) die "PROVIDER must be 'openai' or 'gemini', got '$PROVIDER'" ;;
esac
# Read from .env if not already exported, so the script works straight after
# a local run without re-exporting anything.
if [[ -z "$KEY" && -f .env ]]; then
  KEY="$(grep -E "^${KEY_ENV}=" .env | head -1 | cut -d= -f2- || true)"
fi
[[ -n "$KEY" ]] || die "$KEY_ENV is not set (checked the environment and .env)."

say "Deploying '$SERVICE' to $REGION in project '$PROJECT' (provider: $PROVIDER)"

# ------------------------------------------------------------------- setup --

say "Enabling required APIs"
gcloud services enable \
  run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  storage.googleapis.com firestore.googleapis.com secretmanager.googleapis.com \
  --project "$PROJECT" --quiet

say "Ensuring runs bucket gs://$BUCKET"
if ! gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" \
    --project "$PROJECT" --location "$REGION" --uniform-bucket-level-access
else
  echo "    already exists"
fi

say "Ensuring Firestore database (native mode)"
if ! gcloud firestore databases describe --database='(default)' --project "$PROJECT" >/dev/null 2>&1; then
  gcloud firestore databases create --location "$REGION" --project "$PROJECT" --quiet
else
  echo "    already exists"
fi

say "Storing the $PROVIDER API key in Secret Manager"
# The key is passed to Cloud Run as a mounted secret, never as a plain env var:
# env vars are visible to anyone with run.services.get on the project.
if gcloud secrets describe "$SECRET" --project "$PROJECT" >/dev/null 2>&1; then
  printf '%s' "$KEY" | gcloud secrets versions add "$SECRET" --data-file=- --project "$PROJECT" --quiet
else
  printf '%s' "$KEY" | gcloud secrets create "$SECRET" --data-file=- --project "$PROJECT" --quiet
fi

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

say "Granting the runtime service account access"
gcloud secrets add-iam-policy-binding "$SECRET" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/secretmanager.secretAccessor" \
  --project "$PROJECT" --quiet >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/storage.objectAdmin" --quiet >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/datastore.user" --quiet >/dev/null

# ------------------------------------------------------------------ deploy --

say "Building and deploying (Cloud Build -- no local Docker needed)"
# --memory 1Gi: the whole embedded corpus is loaded into a numpy matrix on first
# search and cached for the instance's lifetime.
# --concurrency 8: each request runs several concurrent LLM calls; packing more
#   than this onto one instance makes latency spiky for no throughput gain.
# --timeout 300: a long script with many claims can take ~60s; 300 leaves room
#   without letting a hung provider call pin an instance indefinitely.
gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 8 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars "PROVIDER=${PROVIDER},RUNS_BACKEND=cloud,GCP_PROJECT=${PROJECT},RUNS_BUCKET=${BUCKET},FIRESTORE_COLLECTION=${COLLECTION},DATA_DIR=/app/data,DB_PATH=/app/data/store.sqlite3,LOG_LEVEL=INFO" \
  --set-secrets "${KEY_ENV}=${SECRET}:latest" \
  --quiet

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)')"

# ------------------------------------------------------------- smoke check --

say "Verifying the deployment"
HEALTH="$(curl -fsS --max-time 60 "$URL/health" || true)"
if [[ -z "$HEALTH" ]]; then
  die "Deployed, but $URL/health did not respond. Check: gcloud run services logs read $SERVICE --region $REGION"
fi
echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"

# A degraded service usually means the corpus did not make it into the image,
# which is silent at request time -- every claim just comes back unverifiable.
if ! grep -q '"status": *"ok"' <<<"$HEALTH"; then
  die "Service is up but reports degraded. Most likely the embedded corpus is empty or the API key secret is not readable."
fi

say "Deployed successfully"
cat <<EOF

  Frontend + API : $URL
  API docs       : $URL/docs
  Corpus         : $URL/corpus
  Run history    : $URL/runs
  Metrics        : $URL/metrics

  Run artifacts  : gs://$BUCKET/runs/
  Run index      : Firestore collection '$COLLECTION'
  Logs           : gcloud run services logs read $SERVICE --region $REGION

  Re-deploy after adding manuals:
      make ingest && ./deploy/cloudrun.sh

EOF
