# Container for Cloud Run.
#
# Code only. The corpus and the fitted models are ~90MB for a current-season
# build and change every day, so baking them in would mean rebuilding the image
# to refresh data. The bucket is already the backup for all of it, so it is
# mounted at /gcs instead and the app reads its data straight out of there.
#
# That also solves the write problem. Cloud Run's filesystem does not survive
# the request that wrote to it, and this app writes: every odds capture and
# every flagged play. Through the mount those land in the bucket, which is the
# only reason the closing-line record can span more than one request.

FROM python:3.13-slim

# Faster, quieter, and no .pyc litter in a layer that is thrown away anyway.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so a code change does not re-resolve the whole tree.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY scripts/ ./scripts/

# Where the mounted bucket appears, and where the app looks for its data. Both
# are overridable at deploy time; these are the values the deploy script sets.
ENV DATA_DIR=/gcs/data \
    RAW_ARCHIVE_DIR=/gcs/data/raw \
    OUTPUT_DIR=/tmp/out

# Cloud Run supplies PORT and expects every interface, which `serve` reads.
# Nothing is exposed by declaring it, but it documents the contract.
EXPOSE 8080

CMD ["python", "-m", "guards_report.cli", "serve"]
