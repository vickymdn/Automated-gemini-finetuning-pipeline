"""
cloud_function/main.py — GCS upload event trigger.

Deployed as a Google Cloud Function (gen2) with an Eventarc trigger
on your GCS bucket. When a new file is finalised (uploaded), this
function fires the Cloud Run Job that runs the full training pipeline.

The function is deliberately thin — all intelligence (locking, delta
checks, validation) lives inside the Cloud Run Job. This function's
only responsibility is to trigger the job.

Deployment command:
  gcloud functions deploy trigger-fine-tuning \\
    --gen2 \\
    --runtime=python311 \\
    --region=us-central1 \\
    --source=./cloud_function \\
    --entry-point=trigger_job_on_upload \\
    --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \\
    --trigger-event-filters="bucket=YOUR_BUCKET_NAME" \\
    --set-env-vars GCP_PROJECT=YOUR_PROJECT_ID,REGION=us-central1,JOB_NAME=YOUR_JOB_NAME \\
    --service-account=YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com
"""

import os
import json
import logging

import functions_framework
from google.cloud import run_v2

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("GCP_PROJECT", "your-gcp-project-id")
REGION     = os.environ.get("REGION",      "us-central1")
JOB_NAME   = os.environ.get("JOB_NAME",   "your-cr-fine-tuning-job")


@functions_framework.http
def trigger_job_on_upload(request):
    """
    HTTP handler invoked by Eventarc when a new object is finalised in GCS.

    The Eventarc trigger converts the GCS event into an HTTP POST to this
    function. We parse the event payload for logging purposes and then
    trigger the Cloud Run Job.

    Args:
        request: Flask Request object containing the CloudEvent payload.

    Returns:
        (str, int): HTTP response body and status code.
    """
    # ── Parse event payload (for logging only) ────────────────────────────────
    try:
        payload      = request.get_json(silent=True) or {}
        bucket_name  = payload.get("bucket",  "unknown")
        object_name  = payload.get("name",    "unknown")
        event_time   = payload.get("timeCreated", "unknown")

        logger.info(f"GCS event received:")
        logger.info(f"  Bucket : {bucket_name}")
        logger.info(f"  Object : {object_name}")
        logger.info(f"  Time   : {event_time}")
    except Exception as e:
        logger.warning(f"Could not parse event payload: {e}")

    # ── Fire the Cloud Run Job ────────────────────────────────────────────────
    try:
        client   = run_v2.JobsClient()
        job_path = f"projects/{PROJECT_ID}/locations/{REGION}/jobs/{JOB_NAME}"

        logger.info(f"Triggering Cloud Run Job: {job_path}")
        operation = client.run_job(
            request=run_v2.RunJobRequest(name=job_path)
        )
        logger.info(f"Job execution started. Operation: {operation.operation.name}")

    except Exception as e:
        logger.error(f"Failed to trigger Cloud Run Job: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)}), 500

    return json.dumps({
        "status":  "triggered",
        "job":     JOB_NAME,
        "project": PROJECT_ID,
        "region":  REGION,
    }), 200
