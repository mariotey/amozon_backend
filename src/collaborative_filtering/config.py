"""Configuration constants for the collaborative filtering module."""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR: Path = REPO_ROOT / "etl" / "data" / "output"
USER_PARQUET: Path = DATA_DIR / "user.parquet"
ITEM_PARQUET: Path = DATA_DIR / "item.parquet"
USER_ITEM_PARQUET: Path = DATA_DIR / "user-item-interaction.parquet"

# ── Artifact paths ────────────────────────────────────────────────────────────
MODELS_DIR: Path = REPO_ROOT / "backend"/ "models"
ALS_MODEL_PATH: Path = MODELS_DIR / "als_model.joblib"
USER_ITEM_MATRIX_PATH: Path = MODELS_DIR / "cf_user_item_matrix.npz"
USERID_MAPPING_PATH: Path = MODELS_DIR / "idx_to_userid_mapping.json"
ITEMPASIN_MAPPING_PATH: Path = MODELS_DIR / "idx_to_itempasin_mapping.json"

# ── ALS Model hyperparameters ────────────────────────────────────────────────────
ALS_FACTORS: int = 50
ALS_REG: float = 0.01
ALS_ITERA: int = 20

# ── Recommendation hyperparameters ────────────────────────────────────────────
DEFAULT_TOP_N: int = 10