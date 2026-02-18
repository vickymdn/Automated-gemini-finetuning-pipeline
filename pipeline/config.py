"""
config.py — Central configuration for the fine-tuning pipeline.

All values here serve as defaults and can be overridden via environment variables
in Cloud Run Job's .env.yaml or at the command line.
"""

import os

# ==========================================
# GCS Configuration
# ==========================================
GCS_BUCKET_NAME   = os.getenv("BUCKET_NAME", "your-ml-bucket")
GCS_PROJECT_ID    = os.getenv("PROJECT_ID",  "your-gcp-project-id")

# Input: where your raw training data lives (e.g., images, text files, CSVs)
GCS_RAW_DATA_FOLDER = os.getenv("RAW_DATA_FOLDER", "raw_training_data")

# Input: where label/annotation files live (can be same as RAW_DATA_FOLDER)
GCS_LABEL_FOLDER  = os.getenv("LABEL_FOLDER", "raw_training_data")

# Output: where the generated JSONL datasets are uploaded
GCS_JSONL_FOLDER  = os.getenv("JSONL_FOLDER", "training_datasets")

# Output: where training metadata and the lock file are stored
MODEL_META_FOLDER = os.getenv("MODEL_META_FOLDER", "trained_model")

# ==========================================
# Vertex AI Configuration
# ==========================================
GCP_LOCATION  = os.getenv("LOCATION",    "us-central1")
SOURCE_MODEL  = os.getenv("SOURCE_MODEL", "gemini-2.0-flash-001")

# ==========================================
# Dataset Configuration
# ==========================================
DEFAULT_TRAIN_FILE  = os.getenv("TRAIN_FILE",    "sft_train.jsonl")
DEFAULT_VAL_FILE    = os.getenv("VAL_FILE",      "sft_val.jsonl")
DEFAULT_SPLIT_RATIO = float(os.getenv("SPLIT_RATIO", "0.9"))

# ==========================================
# Training Gate
# ==========================================
# Minimum number of NEW samples since the last training run
# required before a new fine-tuning job is triggered.
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "10"))

# ==========================================
# File Extensions
# ==========================================
# Adjust these to match your raw data format
SUPPORTED_RAW_EXTENSIONS   = [".png", ".jpg", ".jpeg", ".txt", ".pdf"]
SUPPORTED_LABEL_EXTENSIONS = [".json"]
