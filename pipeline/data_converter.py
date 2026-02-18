"""
data_converter.py — Use-case specific data converter.

This is the ONE file you need to customise for your project.

It reads your raw data from GCS, transforms each sample into the
Vertex AI Supervised Fine-Tuning (SFT) JSONL format, splits the
dataset into train/val sets, and uploads the JSONL files back to GCS.

──────────────────────────────────────────────────
VERTEX AI JSONL FORMAT (text-only example)
──────────────────────────────────────────────────
{
  "contents": [
    {
      "role": "user",
      "parts": [{ "text": "Your prompt here" }]
    },
    {
      "role": "model",
      "parts": [{ "text": "Your expected model output here" }]
    }
  ]
}

──────────────────────────────────────────────────
MULTIMODAL EXAMPLE (image + text)
──────────────────────────────────────────────────
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "file_data": {
            "mime_type": "image/png",
            "file_uri": "gs://your-bucket/images/sample.png"
          }
        },
        { "text": "Describe this image." }
      ]
    },
    {
      "role": "model",
      "parts": [{ "text": "Expected model response here." }]
    }
  ]
}

──────────────────────────────────────────────────
OPTIONAL: systemInstruction
──────────────────────────────────────────────────
You can add a top-level "systemInstruction" key if needed:
{
  "systemInstruction": {
    "parts": [{ "text": "You are a helpful assistant." }]
  },
  "contents": [ ... ]
}
"""

import json
import random
import tempfile
import os
from pathlib import Path

from google.cloud import storage

from config import (
    GCS_BUCKET_NAME,
    GCS_PROJECT_ID,
    GCS_RAW_DATA_FOLDER,
    GCS_JSONL_FOLDER,
    DEFAULT_TRAIN_FILE,
    DEFAULT_VAL_FILE,
    DEFAULT_SPLIT_RATIO,
    SUPPORTED_RAW_EXTENSIONS,
)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 ── CUSTOMISE THIS FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt():
    """
    Return the system/user prompt string that will be sent to the model.

    Customise this for your task. Some examples:
      • "Extract all named entities from the following text as JSON."
      • "Classify the sentiment of this review as positive, negative, or neutral."
      • "Answer the question based only on the provided context."
    """
    return "Your task prompt here."


def sample_to_jsonl_entry(raw_file_path: Path, label_data: dict, gcs_uri: str = None) -> dict:
    """
    Convert one raw sample + its label into a Vertex AI JSONL entry.

    Args:
        raw_file_path : Path to the local raw file (image, text, etc.)
        label_data    : Parsed label/annotation dict loaded from the label file
        gcs_uri       : GCS URI of the raw file (for multimodal tasks)

    Returns:
        dict: A single JSONL-compatible dict in Vertex AI SFT format.

    ── HOW TO CUSTOMISE ──────────────────────────────────────────────────────
    TEXT-ONLY task:
        Replace the user parts list with just [{"text": your_prompt}]
        and remove the file_data block.

    MULTIMODAL task (image/PDF/audio):
        Keep the file_data block and set gcs_uri to the uploaded file's URI.
        Vertex AI will load the file directly from GCS during training.

    CHAT / MULTI-TURN task:
        Extend the contents list with alternating user/model turns.
    ──────────────────────────────────────────────────────────────────────────
    """
    prompt = build_prompt()

    # Build the target output string from the label data.
    # Adjust this to match whatever format your model should produce.
    target_output = json.dumps(label_data, indent=2)

    # ── Text-only entry ───────────────────────────────────────────────────────
    entry = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            },
            {
                "role": "model",
                "parts": [{"text": target_output}]
            }
        ]
    }

    # ── Uncomment the block below for a MULTIMODAL entry ─────────────────────
    # Determine MIME type from file extension
    # mime_map = {
    #     ".png":  "image/png",
    #     ".jpg":  "image/jpeg",
    #     ".jpeg": "image/jpeg",
    #     ".pdf":  "application/pdf",
    #     ".mp3":  "audio/mpeg",
    #     ".mp4":  "video/mp4",
    # }
    # mime_type = mime_map.get(raw_file_path.suffix.lower(), "application/octet-stream")
    #
    # entry = {
    #     "contents": [
    #         {
    #             "role": "user",
    #             "parts": [
    #                 {
    #                     "file_data": {
    #                         "mime_type": mime_type,
    #                         "file_uri":  gcs_uri          # must be a gs:// URI
    #                     }
    #                 },
    #                 {"text": prompt}
    #             ]
    #         },
    #         {
    #             "role": "model",
    #             "parts": [{"text": target_output}]
    #         }
    #     ]
    # }
    # ─────────────────────────────────────────────────────────────────────────

    return entry


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 ── CUSTOMISE label loading IF your format differs from JSON
# ──────────────────────────────────────────────────────────────────────────────

def load_label(label_path: Path) -> dict:
    """
    Load and parse a label/annotation file.

    Default: reads a JSON file. Override this if your labels are in a
    different format (CSV row, plain text, XML, etc.).
    """
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# GCS HELPERS  (no need to modify these)
# ──────────────────────────────────────────────────────────────────────────────

def _gcs_client(project_id: str):
    return storage.Client(project=project_id)


def download_folder_from_gcs(bucket_name: str, gcs_folder: str, project_id: str) -> str:
    """
    Download all raw + label files from a GCS folder to a temp directory.
    Returns the local directory path.
    """
    local_dir = Path(tempfile.mkdtemp())
    client = _gcs_client(project_id)
    bucket = client.bucket(bucket_name)

    print(f"  Downloading gs://{bucket_name}/{gcs_folder}/ → {local_dir}")
    count = 0
    for blob in bucket.list_blobs(prefix=gcs_folder):
        if blob.name.endswith("/"):
            continue
        filename = blob.name.split("/")[-1]
        local_file = local_dir / filename
        blob.download_to_filename(str(local_file))
        count += 1

    print(f"  Downloaded {count} files.")
    return str(local_dir)


def upload_raw_file_to_gcs(local_path: Path, bucket_name: str,
                             gcs_folder: str, project_id: str) -> str:
    """Upload a single raw file to GCS. Returns its gs:// URI."""
    client = _gcs_client(project_id)
    bucket = client.bucket(bucket_name)
    blob_name = f"{gcs_folder}/{local_path.name}"
    blob = bucket.blob(blob_name)

    ext = local_path.suffix.lower()
    content_type_map = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".pdf":  "application/pdf",
        ".txt":  "text/plain",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")
    blob.upload_from_filename(str(local_path), content_type=content_type)
    return f"gs://{bucket_name}/{blob_name}"


def upload_jsonl_to_gcs(local_path: str, bucket_name: str,
                         gcs_folder: str, project_id: str) -> str:
    """Upload a JSONL file to GCS. Returns its gs:// URI."""
    client = _gcs_client(project_id)
    bucket = client.bucket(bucket_name)
    filename = Path(local_path).name
    blob_name = f"{gcs_folder}/{filename}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path, content_type="application/jsonl")
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"  Uploaded → {gcs_uri}")
    return gcs_uri


# ──────────────────────────────────────────────────────────────────────────────
# CORE DATASET BUILDER  (orchestrates download → convert → split → upload)
# ──────────────────────────────────────────────────────────────────────────────

def create_vertex_jsonl(
    input_folder: str,
    output_train: str,
    output_val: str,
    gcs_bucket: str,
    gcs_raw_folder: str,
    gcs_jsonl_folder: str,
    project_id: str,
    split_ratio: float = 0.9,
    upload_to_gcs: bool = True,
    download_from_gcs_first: bool = True,
    upload_raw_files: bool = False,   # Set True for multimodal tasks
):
    """
    Full dataset creation pipeline:
      1. (Optional) Download raw files from GCS to a local temp dir.
      2. Pair each raw file with its label file.
      3. (Optional) Upload raw files to GCS and get their gs:// URIs.
      4. Convert each pair to a Vertex AI JSONL entry.
      5. Shuffle and split into train/val.
      6. Write JSONL files locally.
      7. (Optional) Upload JSONL files to GCS.
    """
    if download_from_gcs_first:
        print("Downloading raw data from GCS...")
        input_folder = download_folder_from_gcs(gcs_bucket, gcs_raw_folder, project_id)

    folder_path = Path(input_folder)
    dataset = []

    # Find all label files (JSON by default)
    label_files = list(folder_path.glob("*.json"))
    print(f"\nFound {len(label_files)} label files. Processing...")

    for idx, label_file in enumerate(label_files, 1):
        print(f"\n[{idx}/{len(label_files)}] {label_file.name}")

        # Find the matching raw file
        raw_file = None
        for ext in SUPPORTED_RAW_EXTENSIONS:
            candidate = label_file.with_suffix(ext)
            if candidate.exists():
                raw_file = candidate
                break

        if raw_file is None:
            print(f"  ⚠ Skipping: no matching raw file found for {label_file.stem}")
            continue

        # Load label
        try:
            label_data = load_label(label_file)
        except Exception as e:
            print(f"  ✗ Failed to load label: {e}")
            continue

        # Upload raw file to GCS (required for multimodal; skip for text-only)
        gcs_uri = None
        if upload_raw_files:
            try:
                gcs_uri = upload_raw_file_to_gcs(raw_file, gcs_bucket, gcs_raw_folder, project_id)
                print(f"  ✓ Uploaded raw file → {gcs_uri}")
            except Exception as e:
                print(f"  ✗ Upload failed: {e}")
                continue

        # Convert to JSONL entry
        try:
            entry = sample_to_jsonl_entry(raw_file, label_data, gcs_uri)
            dataset.append(entry)
        except Exception as e:
            print(f"  ✗ Conversion failed: {e}")
            continue

    if not dataset:
        raise RuntimeError("No valid samples were converted. Check your data and label files.")

    # Shuffle and split
    random.shuffle(dataset)
    split_idx = max(1, int(len(dataset) * split_ratio))
    if split_idx >= len(dataset):
        split_idx = len(dataset) - 1  # Ensure at least 1 val sample

    train_data = dataset[:split_idx]
    val_data   = dataset[split_idx:]

    # Ensure at least 1 validation sample
    if not val_data:
        val_data = [train_data[-1]]

    print(f"\n  Dataset summary:")
    print(f"    Total:      {len(dataset)}")
    print(f"    Train:      {len(train_data)}")
    print(f"    Validation: {len(val_data)}")

    # Write JSONL locally
    for path, data, label in [(output_train, train_data, "train"),
                               (output_val, val_data, "val")]:
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"  Wrote {label} → {path}")

    # Upload JSONL to GCS
    if upload_to_gcs:
        print("\nUploading JSONL files to GCS...")
        train_uri = upload_jsonl_to_gcs(output_train, gcs_bucket, gcs_jsonl_folder, project_id)
        val_uri   = upload_jsonl_to_gcs(output_val,   gcs_bucket, gcs_jsonl_folder, project_id)
        print(f"\n  Train URI : {train_uri}")
        print(f"  Val URI   : {val_uri}")
        return train_uri, val_uri

    return output_train, output_val


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE  (called by main.py)
# ──────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    input_folder=None,
    output_train=None,
    output_val=None,
    gcs_bucket=None,
    gcs_raw_folder=None,
    gcs_jsonl_folder=None,
    project_id=None,
    split_ratio=None,
    upload_to_gcs=True,
    download_from_gcs_first=True,
    upload_raw_files=False,
):
    """Programmatic interface. Falls back to config.py for all unset params."""
    input_folder     = input_folder     or "local_data"
    output_train     = output_train     or DEFAULT_TRAIN_FILE
    output_val       = output_val       or DEFAULT_VAL_FILE
    gcs_bucket       = gcs_bucket       or GCS_BUCKET_NAME
    gcs_raw_folder   = gcs_raw_folder   or GCS_RAW_DATA_FOLDER
    gcs_jsonl_folder = gcs_jsonl_folder or GCS_JSONL_FOLDER
    project_id       = project_id       or GCS_PROJECT_ID
    split_ratio      = split_ratio      or DEFAULT_SPLIT_RATIO

    return create_vertex_jsonl(
        input_folder=input_folder,
        output_train=output_train,
        output_val=output_val,
        gcs_bucket=gcs_bucket,
        gcs_raw_folder=gcs_raw_folder,
        gcs_jsonl_folder=gcs_jsonl_folder,
        project_id=project_id,
        split_ratio=split_ratio,
        upload_to_gcs=upload_to_gcs,
        download_from_gcs_first=download_from_gcs_first,
        upload_raw_files=upload_raw_files,
    )


def run_jsonl_generation():
    """Entry point called by the orchestrator (main.py)."""
    print("=" * 60)
    print("JSONL Dataset Generation")
    print("=" * 60)
    generate_dataset()
    print("\nJSONL generation complete.")
