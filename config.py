"""
Configuration Constants for the Recommendation System
"""
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────────────────────

# Repository root directory
REPO_ROOT: Path = Path(__file__).resolve().parent

# Local data directories.
DATA_DIR: Path = REPO_ROOT / "data"
DATA_INPUT_DIR: Path = DATA_DIR / "input"
DATA_OUTPUT_DIR: Path = DATA_DIR / "output"

# Local model artefact directory.
MODEL_ARTEFACT_DIR: Path = REPO_ROOT / "models"

# ── Supabase Configuration ─────────────────────────────────────────────────────────────────────

# Storage bucket used for model artefacts.
MODEL_ARTEFACT_BUCKET: str = "ModelArtefacts"

# Database Table Names
USER_TABLE_NAME = "User"
REVIEW_TABLE_NAME = "Review"
ITEM_TABLE_NAME = "Item"
MODELREGISTRY_TABLE_NAME = "ModelRegistry"

# ── Amazon Reviews Dataset ─────────────────────────────────────────────────────────────────────

# Retrieved from: https://amazon-reviews-2023.github.io/
# NOTE: Use 2023 dataset as it is larger, more descriptive, more granular and cleaner compared to
#       previous datasets.
# NOTE: This dataset (McAuley-Lab/Amazon-Reviews-2023, configuration: raw_review_Software) was
#       extracted using this notebook on 11 Februrary 2026, based on the latest locally cached
#       version available at that time. As the dataset may be updated, restructured, or modified
#       by the authors in the future, results generated from this notebook may differ depending
#       on the dataset version used.
DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"

# NOTE: There are 33 categories + Unknown. Software is a subset.
DATASET_CATEGORY = "Software"

# Percentage of the dataset split used during data extraction.
DATASET_SPLIT_PERCENTAGE = 5

# ── Local Data Filenames ───────────────────────────────────────────────────────────────────────

USER_REVIEW_FILENAME = "review_data.parquet"
ITEM_METADATA_FILENAME = "meta_data.parquet"

USER_FILENAME = "user.parquet"
ITEM_FILENAME = "item.parquet"
USER_ITEM_INTERACT_FILENAME = "user-item-interaction.parquet"

# ── Recommendation Tools ───────────────────────────────────────────────────────────────────────

TOOLS = [
    "collaborative_filtering",
    "content_based_filtering"
]