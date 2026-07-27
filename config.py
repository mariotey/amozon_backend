"""
Configuration Constants for the Recommendation System
"""
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = REPO_ROOT / "data"
DATA_INPUT_DIR: Path = DATA_DIR / "input"
DATA_OUTPUT_DIR: Path = DATA_DIR / "output"

MODEL_ARTEFACT_DIR: Path = REPO_ROOT / "models"
MODEL_ARTEFACT_BUCKET: str = "ModelArtefacts"

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

# Supabase Table Name
USER_TABLE_NAME = "User"
REVIEW_TABLE_NAME = "Review"
ITEM_TABLE_NAME = "Item"
MODELREGISTRY_TABLE_NAME = "ModelRegistry"

# Data Filename
USER_REVIEW_FILENAME = "review_data.parquet"
ITEM_METADATA_FILENAME = "meta_data.parquet"

USER_FILENAME = "user.parquet"
ITEM_FILENAME = "item.parquet"
USER_ITEM_INTERACT_FILENAME = "user-item-interaction.parquet"

TOOLS = [
    "collaborative_filtering",
    "content_based_filtering"
]