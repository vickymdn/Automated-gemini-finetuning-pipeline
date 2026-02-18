# Automated Gemini Fine-Tuning Pipeline on Vertex AI

An event-driven, production-ready pipeline that automatically fine-tunes a Gemini model on Google Cloud whenever new training data is uploaded to GCS.

## How It Works

```
New file uploaded to GCS
        │
        ▼
  Cloud Function              ← Eventarc trigger fires on GCS object.finalized
        │
        ▼
  Cloud Run Job               ← Containerised pipeline (this repo)
        │
        ├─ Gate 1: Lock       ← Is a training job already running?
        ├─ Gate 2: Validate   ← Enough matched data pairs?
        ├─ Gate 3: Delta      ← Enough NEW samples since last run?
        │
        ├─ data_converter.py  ← Your raw data → Vertex AI JSONL
        ├─ tuning.py          ← Vertex AI SFT job (Gemini)
        └─ Metadata → GCS     ← Saved for next delta comparison
```

---

## Project Structure

```
gemini-finetuning-pipeline/
├── pipeline/
│   ├── config.py           ← All configuration (env-var backed)
│   ├── data_converter.py   ← ✏️  CUSTOMISE THIS for your data format
│   ├── tuning.py           ← Vertex AI SFT wrapper
│   └── main.py             ← Orchestrator with all three gates
├── cloud_function/
│   ├── main.py             ← GCS event → Cloud Run Job trigger
│   └── requirements.txt
├── Dockerfile
├── requirements.txt
├── .env.yaml               ← Cloud Run Job environment variables
├── .gitlab-ci.yml          ← CI/CD: build + deploy on push
└── .gitignore
```

---

## Quickstart

### 1. Prerequisites

- A GCP project with billing enabled
- APIs enabled: Cloud Run, Vertex AI, Cloud Functions, Eventarc, Cloud Storage
- A GCS bucket for training data and model metadata
- A service account with the following roles:
  - `roles/storage.objectAdmin`
  - `roles/aiplatform.user`
  - `roles/run.invoker`
  - `roles/run.developer` (for CI/CD deployment)

### 2. Configure

Edit `.env.yaml` with your values:

```yaml
PROJECT_ID:       "your-gcp-project-id"
BUCKET_NAME:      "your-ml-bucket"
RAW_DATA_FOLDER:  "raw_training_data"
LABEL_FOLDER:     "raw_training_data"
JSONL_FOLDER:     "training_datasets"
MODEL_META_FOLDER: "trained_model"
MIN_SAMPLES:      "10"
SOURCE_MODEL:     "gemini-2.0-flash-001"
```

### 3. Implement Your Data Converter

Open `pipeline/data_converter.py` and implement two functions:

**`build_prompt()`** — Return the instruction/system prompt for your task:
```python
def build_prompt():
    return "Extract all named entities as JSON."
```

**`sample_to_jsonl_entry()`** — Convert one raw file + its label into a Vertex AI JSONL entry:
```python
def sample_to_jsonl_entry(raw_file_path, label_data, gcs_uri=None):
    return {
        "contents": [
            {"role": "user",  "parts": [{"text": build_prompt()}]},
            {"role": "model", "parts": [{"text": json.dumps(label_data)}]}
        ]
    }
```

For multimodal tasks (images, PDFs, audio), uncomment the `file_data` block in the function and set `upload_raw_files=True` when calling `generate_dataset()`.

### 4. Build and Deploy the Cloud Run Job

```bash
# Authenticate
gcloud auth configure-docker us-docker.pkg.dev

# Build and push
export IMAGE="us-docker.pkg.dev/YOUR_PROJECT/YOUR_REPO/fine-tuning-pipeline:latest"
docker build -t $IMAGE .
docker push $IMAGE

# Deploy Cloud Run Job
gcloud run jobs deploy your-cr-fine-tuning-job \
  --region=us-central1 \
  --image=$IMAGE \
  --memory=2Gi \
  --cpu=1 \
  --task-timeout=120m \
  --service-account=YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com \
  --env-vars-file=.env.yaml
```

### 5. Deploy the Cloud Function

```bash
gcloud functions deploy trigger-fine-tuning \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=./cloud_function \
  --entry-point=trigger_job_on_upload \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=YOUR_BUCKET_NAME" \
  --set-env-vars="GCP_PROJECT=YOUR_PROJECT_ID,REGION=us-central1,JOB_NAME=your-cr-fine-tuning-job" \
  --service-account=YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com
```

### 6. Test End-to-End

```bash
# Upload a sample file to trigger the pipeline
gsutil cp your-sample-file.json gs://your-ml-bucket/raw_training_data/

# Or trigger the Cloud Run Job manually
gcloud run jobs execute your-cr-fine-tuning-job --region=us-central1

# Monitor logs
gcloud run jobs executions list --job=your-cr-fine-tuning-job --region=us-central1
```

---

## Data Format

Your raw data and label files must share the same **filename stem**. For example:
```
raw_training_data/
  sample_001.png     ← raw file
  sample_001.json    ← label file (must have the same stem)
  sample_002.txt
  sample_002.json
```

The pipeline counts these matched pairs and uses the count for the delta gate.

### Vertex AI JSONL Schema

**Text-only:**
```json
{
  "contents": [
    {"role": "user",  "parts": [{"text": "Your prompt"}]},
    {"role": "model", "parts": [{"text": "Expected output"}]}
  ]
}
```

**Multimodal (image + text):**
```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {"file_data": {"mime_type": "image/png", "file_uri": "gs://bucket/image.png"}},
        {"text": "Your prompt"}
      ]
    },
    {"role": "model", "parts": [{"text": "Expected output"}]}
  ]
}
```

---

## The Three Training Gates

| Gate | What it checks | If it fails |
|---|---|---|
| **Lock** | Is `trained_model/training.lock.json` present in GCS? | Skip — a job is already running |
| **Sample validation** | Are there matched (raw + label) pairs in GCS? | Wait — log the count |
| **Delta check** | Has the number of matched pairs grown by `MIN_SAMPLES` since the last successful training? | Wait — log the gap |

The lock file is **always** deleted after a run (success or failure) via a `finally` block, so the system recovers automatically.

---

## Metadata

After every training run (success or failure), a JSON file is saved to GCS:

```
gs://your-ml-bucket/trained_model/model_metadata_YYYYMMDD_HHMMSS.json
```

Example contents:
```json
{
  "status": "success",
  "sample_size": 120,
  "previous_sample_size": 100,
  "sample_delta": 20,
  "training_start_time": "2025-03-01T10:00:00",
  "training_end_time":   "2025-03-01T11:23:45",
  "training_duration_seconds": 5025,
  "model_details": {
    "tuned_model_name": "projects/.../models/...",
    "endpoint_name": "projects/.../endpoints/...",
    "state": "JobState.JOB_STATE_SUCCEEDED"
  }
}
```

The `sample_size` field from the latest metadata file is what the next run uses for the delta check.

---

## CI/CD

Push to the `dev` branch → GitLab CI automatically builds and deploys the Cloud Run Job. Edit `.gitlab-ci.yml` to add a production stage.

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set GCP credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Run the pipeline manually
python pipeline/main.py

# Test just the data converter
python -c "from pipeline.data_converter import run_jsonl_generation; run_jsonl_generation()"

# Test just the tuning step
python -c "from pipeline.tuning import run_fine_tuning; run_fine_tuning()"
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `PROJECT_ID` | — | GCP project ID |
| `BUCKET_NAME` | — | GCS bucket for all data and metadata |
| `RAW_DATA_FOLDER` | `raw_training_data` | GCS prefix for raw input files |
| `LABEL_FOLDER` | `raw_training_data` | GCS prefix for label/annotation files |
| `JSONL_FOLDER` | `training_datasets` | GCS prefix for generated JSONL datasets |
| `MODEL_META_FOLDER` | `trained_model` | GCS prefix for metadata + lock file |
| `MIN_SAMPLES` | `10` | Minimum new samples (delta) before training fires |
| `SPLIT_RATIO` | `0.9` | Train/val split ratio |
| `SOURCE_MODEL` | `gemini-2.0-flash-001` | Base Gemini model for fine-tuning |
| `TRAIN_FILE` | `sft_train.jsonl` | Training JSONL filename |
| `VAL_FILE` | `sft_val.jsonl` | Validation JSONL filename |
| `LOCATION` | `us-central1` | GCP region for Vertex AI |
