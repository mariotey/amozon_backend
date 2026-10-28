"""Manual dataset preparation and ML experiment settings; no initialization I/O.

Shared resource names come from api_config without importing FastAPI or creating
clients. REPO_ROOT is available here for notebook and local workflow paths.
"""
from pathlib import Path
from .api_config import (
    REPO_ROOT, MODEL_ARTEFACT_DIR, MODEL_ARTEFACT_BUCKET, MODELREGISTRY_TABLE_NAME,
    USER_TABLE_NAME, ITEM_TABLE_NAME, REVIEW_TABLE_NAME, TOOLS,
)

DATA_DIR: Path = REPO_ROOT / "data"
DATA_OUTPUT_DIR: Path = DATA_DIR / "output"

# Prefix extraction is intentional; this is not a random sample.
DATASET_NAME = 'McAuley-Lab/Amazon-Reviews-2023'
DATASET_CATEGORY = 'Software'
DATASET_SPLIT_PERCENTAGE = 5
TRAIN_FILENAME = 'train.parquet'
VALIDATION_FILENAME = 'validation.parquet'
TEST_FILENAME = 'test.parquet'
