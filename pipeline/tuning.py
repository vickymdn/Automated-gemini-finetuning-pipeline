"""
tuning.py — Vertex AI Supervised Fine-Tuning (SFT) wrapper.

Provides a clean, parameterised function to launch a Gemini SFT job,
poll it to completion, and return a structured result dict.

Called by main.py via run_fine_tuning(), but also usable standalone
from notebooks or scripts.
"""

import time
import json
import logging

import vertexai
from google.cloud import aiplatform
from vertexai.tuning import sft

from config import (
    GCS_BUCKET_NAME,
    GCS_PROJECT_ID,
    GCS_JSONL_FOLDER,
    DEFAULT_TRAIN_FILE,
    DEFAULT_VAL_FILE,
    GCP_LOCATION,
    SOURCE_MODEL,
)

logger = logging.getLogger(__name__)


def fine_tune_gemini_model(
    project_id: str = None,
    location: str = None,
    bucket_name: str = None,
    jsonl_folder: str = None,
    train_file: str = None,
    val_file: str = None,
    source_model: str = None,
    poll_interval_seconds: int = 60,
    verbose: bool = True,
) -> dict:
    """
    Launch a Vertex AI Gemini SFT job and wait for it to complete.

    All parameters fall back to values in config.py if not provided,
    which in turn fall back to environment variables.

    Args:
        project_id            : GCP project ID.
        location              : GCP region (e.g., "us-central1").
        bucket_name           : GCS bucket holding the JSONL datasets.
        jsonl_folder          : GCS folder path within the bucket for JSONL files.
        train_file            : Filename of the training JSONL.
        val_file              : Filename of the validation JSONL.
        source_model          : Base Gemini model to fine-tune.
        poll_interval_seconds : How often (seconds) to poll the job status.
        verbose               : Print progress to stdout.

    Returns:
        dict with keys:
            tuned_model_name  — Name of the registered tuned model.
            endpoint_name     — Deployed endpoint name.
            experiment        — Experiment name/ID.
            state             — Final job state string.
            resource_name     — Full resource path of the tuning job.
            job_info          — Lightweight job metadata dict.

    Raises:
        Exception: Propagates any Vertex AI error after logging details.
    """
    # ── Resolve configuration ─────────────────────────────────────────────────
    PROJECT_ID   = project_id   or GCS_PROJECT_ID
    LOCATION     = location     or GCP_LOCATION
    BUCKET_NAME  = bucket_name  or GCS_BUCKET_NAME
    JSONL_FOLDER = jsonl_folder or GCS_JSONL_FOLDER
    TRAIN_FILE   = train_file   or DEFAULT_TRAIN_FILE
    VAL_FILE     = val_file     or DEFAULT_VAL_FILE
    MODEL_NAME   = source_model or SOURCE_MODEL

    train_dataset = f"gs://{BUCKET_NAME}/{JSONL_FOLDER}/{TRAIN_FILE}"
    val_dataset   = f"gs://{BUCKET_NAME}/{JSONL_FOLDER}/{VAL_FILE}"

    # ── Initialise Vertex AI ──────────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("Initialising Vertex AI")
        print("=" * 60)
        print(f"  Project  : {PROJECT_ID}")
        print(f"  Location : {LOCATION}")
        print(f"  Model    : {MODEL_NAME}")
        print(f"  Train    : {train_dataset}")
        print(f"  Val      : {val_dataset}")
        print("=" * 60)

    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # ── Launch SFT job ────────────────────────────────────────────────────────
    if verbose:
        print("\nLaunching supervised fine-tuning job...")

    try:
        sft_job = sft.train(
            source_model=MODEL_NAME,
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
        )

        if verbose:
            print(f"  Job created  : {sft_job.resource_name}")
            print(f"  Initial state: {sft_job.state}")

        # ── Poll until complete ───────────────────────────────────────────────
        if verbose:
            print("\nMonitoring progress (this typically takes 30–90 minutes)...")
            print("-" * 60)

        elapsed = 0
        while not sft_job.has_ended:
            time.sleep(poll_interval_seconds)
            sft_job.refresh()
            elapsed += poll_interval_seconds
            if verbose:
                mins = elapsed // 60
                print(f"  [{mins:3d} min] State: {sft_job.state}")

        # ── Job finished ──────────────────────────────────────────────────────
        if verbose:
            print("\n" + "=" * 60)
            print("Training complete!")
            print("=" * 60)
            print(f"  Tuned model : {sft_job.tuned_model_name}")
            print(f"  Endpoint    : {sft_job.tuned_model_endpoint_name}")
            print(f"  Experiment  : {sft_job.experiment}")
            print(f"  Final state : {sft_job.state}")
            print("=" * 60)

        return {
            "tuned_model_name": sft_job.tuned_model_name,
            "endpoint_name":    sft_job.tuned_model_endpoint_name,
            "experiment":       str(sft_job.experiment),
            "state":            str(sft_job.state),
            "resource_name":    sft_job.resource_name,
            "job_info": {
                "name":      sft_job.resource_name,
                "state":     str(sft_job.state),
                "has_ended": sft_job.has_ended,
            },
        }

    except Exception as e:
        logger.error(f"Fine-tuning job failed: {e}", exc_info=True)
        if verbose:
            print(f"\n✗ Error: {e}")
            print("\nTroubleshooting checklist:")
            print(f"  1. Verify JSONL files exist:")
            print(f"       {train_dataset}")
            print(f"       {val_dataset}")
            print("  2. Confirm files are valid JSONL (one JSON object per line).")
            print("  3. Check IAM: service account needs roles/aiplatform.user.")
            print("  4. Verify the source model name is correct.")
        raise


def run_fine_tuning() -> dict:
    """
    Entry point called by the orchestrator (main.py).
    Runs fine-tuning and writes the result to model_output.json.
    """
    result = fine_tune_gemini_model()
    with open("model_output.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Model output saved to model_output.json")
    return result


if __name__ == "__main__":
    # Allow direct execution for testing:  python tuning.py
    logging.basicConfig(level=logging.INFO)
    run_fine_tuning()
