#!/usr/bin/env bash
# Put the app on Cloud Run, with the data bucket mounted.
#
# Run it from the project root. It needs the gcloud CLI logged in -- which is a
# separate thing from the application-default credentials the Python client
# uses, so a working `sync_data.py` does not mean this will work:
#
#   gcloud auth login
#   gcloud config set project <your project>
#
# What this sets up and why:
#
#   --source .            builds with Cloud Build, so no local Docker is needed
#   --add-volume/--mount  the data bucket appears at /gcs, which is what makes
#                         the corpus readable and the odds record durable --
#                         Cloud Run's own filesystem does not survive the
#                         request that wrote to it
#   --memory 4Gi          measured, not guessed. The build process itself peaks
#                         near 440MB, but files read through the GCS mount are
#                         cached by the kernel, and that cache counts against the
#                         container's limit -- so memory climbs for the whole run
#                         rather than settling. At 2Gi it crossed the limit about
#                         fifteen minutes in and the container was killed, twice.
#                         That reaches the phone only as "lost connection": the
#                         kill takes down the event stream that would have
#                         carried the reason.
#   --timeout 900         a cold report build takes a few minutes
#   --min-instances 0     scale to nothing when unused; a cold start costs a
#                         slower first request and no money in between
#   --no-cpu-throttling   the one that is not optional. Cloud Run allocates
#                         CPU only while a request is in flight, and a report
#                         build runs after the response that started it --
#                         throttled, it logged one line and then sat for ten
#                         minutes. It bills CPU for an instance's whole life
#                         rather than per request, which scaling to zero is
#                         what bounds.
#
# ACCESS_TOKEN is required on every request once set. It is not authentication
# and does not pretend to be: it stops a URL that leaks from becoming a
# five-hundred-credit-a-month odds quota that anybody can drain.

set -euo pipefail

SERVICE="${SERVICE:-guards-report}"
REGION="${REGION:-us-central1}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
BUCKET="${BUCKET:-$(grep -E '^GCS_BUCKET=' .env | cut -d= -f2- | tr -d '\r')}"
ODDS_KEY="${ODDS_KEY:-$(grep -E '^ODDS_API_KEY=' .env | cut -d= -f2- | tr -d '\r')}"
# Read from .env before generating one. A fresh token on every deploy is a
# redeploy that silently invalidates the bookmark on your phone, which reads
# as the service having broken rather than as the token having changed.
TOKEN="${ACCESS_TOKEN:-$(grep -E '^ACCESS_TOKEN=' .env 2>/dev/null | cut -d= -f2- | tr -d '\r')}"

if [[ -z "${PROJECT}" || "${PROJECT}" == "(unset)" ]]; then
  echo "No project set. Run: gcloud config set project <id>" >&2
  exit 1
fi
if [[ -z "${BUCKET}" ]]; then
  echo "No GCS_BUCKET found in .env" >&2
  exit 1
fi
if [[ -z "${TOKEN}" ]]; then
  # Generated rather than defaulted: a deployment with a guessable token is a
  # deployment with none, and a blank one leaves the service wide open.
  TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
  echo "Generated an access token. Save it -- it is the only thing between the"
  echo "URL and your odds quota:"
  echo
  echo "    ${TOKEN}"
  echo
fi

echo "Deploying ${SERVICE} to ${REGION} in ${PROJECT}, bucket ${BUCKET} ..."

gcloud run deploy "${SERVICE}" \
  --source . \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --no-cpu-throttling \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --min-instances 0 \
  --max-instances 2 \
  --concurrency 4 \
  --add-volume "name=data,type=cloud-storage,bucket=${BUCKET}" \
  --add-volume-mount "volume=data,mount-path=/gcs" \
  --set-env-vars "GCP_PROJECT=${PROJECT},GCS_BUCKET=${BUCKET},DATA_DIR=/gcs/data,RAW_ARCHIVE_DIR=/gcs/data/raw,OUTPUT_DIR=/gcs/out,ODDS_API_KEY=${ODDS_KEY},ACCESS_TOKEN=${TOKEN}"

URL="$(gcloud run services describe "${SERVICE}" \
  --project "${PROJECT}" --region "${REGION}" --format='value(status.url)')"

echo
echo "Deployed. Bookmark this on your phone -- the token is in the link, and"
echo "the app sets a cookie from it so the rest of the site works without it:"
echo
echo "    ${URL}/?k=${TOKEN}"
echo
