"""
main.py — Pipeline orchestrator.

This is the container's entrypoint. It runs three gates before starting
any training to ensure we only fire Vertex AI jobs when it's worthwhile:

  Gate 1 — Distributed Lock   : Is a training job already running?
  Gate 2 — Sample Validation  : Do we have enough matched data pairs?
  Gate 3 — Delta Check        : Have enough NEW samples arrived since
                                the last training run?

If all gates pass, it runs:
  Step 1 → data_converter.run_jsonl_generation()
  Step 2 → tuning.run_fine_tuning()
  Step 3 → Saves metadata to GCS for the next delta check.

Usage:
  python pipeline/main.py          # Manual / local execution
  # or via Docker CMD in Cloud Run Job
"""

import os
import json
import logging
from datetime import datetime

from google.cloud import storage

from data_converter import run_jsonl_generation
from tuning import run_fine_tuning
from config import (
    GCS_BUCKET_NAME,
    GCS_PROJECT_ID,
    GCS_RAW_DATA_FOLDER,
    GCS_LABEL_FOLDER,
    MODEL_META_FOLDER,
    MIN_SAMPLES,
    SUPPORTED_RAW_EXTENSIONS,
    SUPPORTED_LABEL_EXTENSIONS,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Runtime configuration (overridable via Cloud Run env vars / .env.yaml)
# ──────────────────────────────────────────────────────────────────────────────
BUCKET_NAME       = os.getenv("BUCKET_NAME",        GCS_BUCKET_NAME)
RAW_DATA_FOLDER   = os.getenv("RAW_DATA_FOLDER",    GCS_RAW_DATA_FOLDER)
LABEL_FOLDER      = os.getenv("LABEL_FOLDER",       GCS_LABEL_FOLDER)
META_FOLDER       = os.getenv("MODEL_META_FOLDER",  MODEL_META_FOLDER)
MIN_DELTA         = int(os.getenv("MIN_SAMPLES",    str(MIN_SAMPLES)))

LOCK_FILE         = f"{META_FOLDER}/training.lock.json"


# ──────────────────────────────────────────────────────────────────────────────
# GCS helpers
# ──────────────────────────────────────────────────────────────────────────────

def _list_files(bucket, prefix: str, extensions: list = None) -> list:
    """List blobs under a GCS prefix, optionally filtered by extension."""
    files = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        if extensions and not any(blob.name.lower().endswith(e) for e in extensions):
            continue
        files.append(blob.name)
    return files


# ──────────────────────────────────────────────────────────────────────────────
# Gate 1 — Distributed lock
# ──────────────────────────────────────────────────────────────────────────────

def is_locked(bucket) -> bool:
    """Return True if a training.lock.json file exists in GCS."""
    locked = bucket.blob(LOCK_FILE).exists()
    if locked:
        try:
            data = json.loads(bucket.blob(LOCK_FILE).download_as_text())
            logger.warning(f"Lock active since: {data.get('locked_at')} "
                           f"by: {data.get('locked_by')}")
        except Exception:
            logger.warning("Lock file exists but could not be read.")
    return locked


def create_lock(bucket) -> None:
    """Write a lock file to GCS to signal training is in progress."""
    payload = {
        "locked_at": datetime.utcnow().isoformat(),
        "locked_by": "training_pipeline",
    }
    bucket.blob(LOCK_FILE).upload_from_string(
        json.dumps(payload, indent=2),
        content_type="application/json",
    )
    logger.info(f"Lock created: gs://{BUCKET_NAME}/{LOCK_FILE}")


def remove_lock(bucket) -> None:
    """Delete the GCS lock file."""
    try:
        bucket.blob(LOCK_FILE).delete()
        logger.info("Lock released.")
    except Exception as e:
        logger.error(f"Failed to release lock: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Gate 2 — Sample validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_samples(bucket) -> tuple:
    """
    Count matched (raw file, label file) pairs in GCS.

    A match is defined as a raw file and a label file sharing the same
    stem (filename without extension) in their respective GCS folders.

    Returns:
        (int, list): count of matched pairs, sorted list of matched stems.
    """
    raw_files   = _list_files(bucket, RAW_DATA_FOLDER, SUPPORTED_RAW_EXTENSIONS)
    label_files = _list_files(bucket, LABEL_FOLDER,    SUPPORTED_LABEL_EXTENSIONS)

    raw_stems   = {os.path.splitext(os.path.basename(f))[0] for f in raw_files}
    label_stems = {os.path.splitext(os.path.basename(f))[0] for f in label_files}

    matched = sorted(raw_stems & label_stems)
    logger.info(f"Validation — raw: {len(raw_stems)}, labels: {len(label_stems)}, "
                f"matched: {len(matched)}")
    return len(matched), matched


# ──────────────────────────────────────────────────────────────────────────────
# Gate 3 — Incremental delta check
# ──────────────────────────────────────────────────────────────────────────────

def get_previous_sample_count(bucket) -> int:
    """
    Read the sample_size from the most recent successful training metadata file.
    Returns 0 if no previous training metadata exists.
    """
    try:
        blobs = list(bucket.list_blobs(prefix=f"{META_FOLDER}/model_metadata_"))
        if not blobs:
            logger.info("No previous training metadata found — this is the first run.")
            return 0

        # Metadata filenames embed a UTC timestamp → lexicographic sort finds latest
        blobs.sort(key=lambda b: b.name, reverse=True)
        metadata = json.loads(blobs[0].download_as_text())

        prev_count = metadata.get("sample_size", 0)
        logger.info(f"Previous training: {prev_count} samples "
                    f"(from {blobs[0].name})")
        return prev_count

    except Exception as e:
        logger.warning(f"Could not read previous metadata ({e}). Treating as first run.")
        return 0


def should_start_training(current: int, previous: int) -> tuple:
    """
    Decide whether to fire a new training job.

    First run  : train if current >= MIN_DELTA.
    Subsequent : train if (current - previous) >= MIN_DELTA.

    Returns:
        (bool, str): (go_ahead, human-readable reason)
    """
    delta = current - previous

    logger.info(f"Training gate — current: {current}, previous: {previous}, "
                f"delta: {delta}, required: {MIN_DELTA}")

    if previous == 0:
        if current >= MIN_DELTA:
            return True,  f"First training — {current} samples (≥ {MIN_DELTA})"
        return False, f"First training — waiting for {MIN_DELTA} samples, have {current}"

    if delta >= MIN_DELTA:
        return True,  f"{delta} new samples accumulated (≥ {MIN_DELTA})"
    return False, f"Only {delta}/{MIN_DELTA} new samples — waiting for more data"


# ──────────────────────────────────────────────────────────────────────────────
# Metadata persistence
# ──────────────────────────────────────────────────────────────────────────────

def upload_model_metadata(bucket, data: dict) -> str:
    """
    Write a training run metadata JSON to GCS.
    Filename embeds a UTC timestamp so files sort chronologically.
    Returns the GCS blob name.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    blob_name = f"{META_FOLDER}/model_metadata_{timestamp}.json"
    bucket.blob(blob_name).upload_from_string(
        json.dumps(data, indent=2, default=str),
        content_type="application/json",
    )
    logger.info(f"Metadata saved: gs://{BUCKET_NAME}/{blob_name}")
    return blob_name


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(event_data: dict = None) -> dict:
    """
    Full pipeline entry point.

    Args:
        event_data: Optional dict from the triggering event (GCS object
                    metadata). Used for logging/metadata only.

    Returns:
        dict with status ("success" | "waiting" | "skipped") and details.

    Raises:
        Exception: Re-raised after logging and saving failure metadata.
    """
    logger.info("=" * 60)
    logger.info("Training Pipeline — Start")
    logger.info("=" * 60)

    event_data   = event_data or {}
    event_type   = event_data.get("eventType", "manual_trigger")
    logger.info(f"Trigger type: {event_type}")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # ── Gate 1: Lock ──────────────────────────────────────────────────────────
    if is_locked(bucket):
        msg = "Training already in progress — skipping this trigger."
        logger.warning(msg)
        return {"status": "skipped", "reason": "training_in_progress", "message": msg}

    # ── Gate 2: Sample validation ─────────────────────────────────────────────
    matched_count, matched_files = validate_samples(bucket)

    # ── Gate 3: Delta check ───────────────────────────────────────────────────
    previous_count          = get_previous_sample_count(bucket)
    go_ahead, reason        = should_start_training(matched_count, previous_count)

    if not go_ahead:
        logger.info(f"Waiting: {reason}")
        return {
            "status":           "waiting",
            "current_samples":  matched_count,
            "previous_samples": previous_count,
            "delta":            matched_count - previous_count,
            "required_delta":   MIN_DELTA,
            "message":          reason,
        }

    # ── All gates passed — start training ────────────────────────────────────
    logger.info(f"All gates passed. Reason: {reason}")
    create_lock(bucket)
    training_start = datetime.utcnow()

    try:
        # Step 1: Convert data to JSONL and upload to GCS
        logger.info("Step 1/2 — JSONL conversion")
        run_jsonl_generation()

        # Step 2: Fine-tune Gemini on Vertex AI
        logger.info("Step 2/2 — Vertex AI SFT")
        model_info = run_fine_tuning()

        training_end = datetime.utcnow()
        duration_s   = (training_end - training_start).total_seconds()

        # Step 3: Persist metadata for next delta check
        metadata = {
            "status":                    "success",
            "sample_size":               matched_count,       # ← used by next delta check
            "previous_sample_size":      previous_count,
            "sample_delta":              matched_count - previous_count,
            "training_start_time":       training_start.isoformat(),
            "training_end_time":         training_end.isoformat(),
            "training_duration_seconds": duration_s,
            "trigger_event_type":        event_type,
            "trigger_event_data": {
                "bucket":    event_data.get("bucket"),
                "object":    event_data.get("name"),
                "timestamp": event_data.get("timeCreated"),
            },
            "model_details":    model_info,
            "sample_files":     matched_files[:10],
            "config": {
                "bucket_name":          BUCKET_NAME,
                "raw_data_folder":      RAW_DATA_FOLDER,
                "label_folder":         LABEL_FOLDER,
                "min_samples_delta":    MIN_DELTA,
                "project_id":           GCS_PROJECT_ID,
            },
        }
        meta_file = upload_model_metadata(bucket, metadata)

        logger.info(f"Pipeline complete in {duration_s:.0f}s. Metadata: {meta_file}")
        logger.info("=" * 60)

        return {
            "status":               "success",
            "sample_count":         matched_count,
            "previous_count":       previous_count,
            "sample_delta":         matched_count - previous_count,
            "duration_seconds":     duration_s,
            "metadata_file":        meta_file,
            "model_info":           model_info,
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

        # Save failure metadata so the next run can still read a valid previous count
        try:
            upload_model_metadata(bucket, {
                "status":               "failed",
                "sample_size":          previous_count,   # keep previous count on failure
                "previous_sample_size": previous_count,
                "training_start_time":  training_start.isoformat(),
                "failure_time":         datetime.utcnow().isoformat(),
                "error":                str(e),
                "error_type":           type(e).__name__,
                "trigger_event_type":   event_type,
            })
        except Exception as meta_err:
            logger.error(f"Also failed to save failure metadata: {meta_err}")

        raise

    finally:
        # The lock MUST always be released, even on failure
        remove_lock(bucket)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Manual / local execution entry point."""
    print("=" * 60)
    print("Gemini Fine-Tuning Pipeline — Manual Run")
    print("=" * 60)
    print(f"  Bucket         : {BUCKET_NAME}")
    print(f"  Raw data folder: {RAW_DATA_FOLDER}")
    print(f"  Label folder   : {LABEL_FOLDER}")
    print(f"  Min delta      : {MIN_DELTA}")
    print("=" * 60)

    try:
        result = run_training_pipeline()
        print("\nResult:")
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"\nPipeline error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
