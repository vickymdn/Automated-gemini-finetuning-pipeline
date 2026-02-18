# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Cloud Run Job container for the fine-tuning pipeline.
#
# Build:  docker build -t fine-tuning-pipeline .
# Run:    docker run --env-file .env fine-tuning-pipeline
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Keeps Python from generating .pyc files and ensures stdout/stderr are
# flushed immediately (important for Cloud Run log streaming).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# The pipeline entrypoint is pipeline/main.py
CMD ["python", "pipeline/main.py"]
