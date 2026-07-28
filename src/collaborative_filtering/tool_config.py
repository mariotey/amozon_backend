"""
Configuration constants for the collaborative filtering module.
"""
from pathlib import Path

MODEL_NAME: str = Path(__file__).parent.name

# ── Artefact Filenames ─────────────────────────────────────────────────────────────────────────
MODEL_ID: str = "xxxxx"

MODEL_ARTEFACTS = {
    "als_model_filename": "als_model.joblib",
    "user_item_matrix_filename": "cf_user_item_matrix.npz",
    "uid_mapping_filename": "idx_to_userid_mapping.json",
    "itempasin_mapping_filename": "idx_to_itempasin_mapping.json"
}

# ── ALS Model hyperparameters ──────────────────────────────────────────────────────────────────
ALS_FACTORS: int = 50
ALS_REG: float = 0.01
ALS_ITERA: int = 20

# ── Recommendation hyperparameters ─────────────────────────────────────────────────────────────
DEFAULT_TOP_N: int = 10
