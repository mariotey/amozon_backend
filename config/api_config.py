"""Application serving settings and shared resource names; no initialization I/O."""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
MODEL_ARTEFACT_DIR: Path = REPO_ROOT / "models"

USER_TABLE_NAME = "User"
REVIEW_TABLE_NAME = "Review"
ITEM_TABLE_NAME = "Item"
MODELREGISTRY_TABLE_NAME = "ModelRegistry"
MODEL_ARTEFACT_BUCKET = "ModelArtefacts"

# Explicit published versions; local bundles still take precedence.
MODEL_IDS = {
    "collaborative_filtering": "45b867d2-094a-4603-9566-0ef25278dbd0",
    "content_based_filtering": "1fa76797-5438-4271-8aa0-0b74d64df9d6",
}
TOOLS = list(MODEL_IDS)
