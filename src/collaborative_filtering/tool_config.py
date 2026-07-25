"""
Configuration constants for the collaborative filtering module.
"""

# ── ALS Model hyperparameters ──────────────────────────────────────────────────────────────────
ALS_FACTORS: int = 50
ALS_REG: float = 0.01
ALS_ITERA: int = 20

# ── Recommendation hyperparameters ─────────────────────────────────────────────────────────────
DEFAULT_TOP_N: int = 10

# ── Artefact Filenames ─────────────────────────────────────────────────────────────────────────
ALS_MODEL_FILENAME: str = "als_model.joblib"
USER_ITEM_MATRIX_FILENAME: str = "cf_user_item_matrix.npz"
USERID_MAPPING_FILENAME: str = "idx_to_userid_mapping.json"
ITEMPASIN_MAPPING_FILENAME: str = "idx_to_itempasin_mapping.json"