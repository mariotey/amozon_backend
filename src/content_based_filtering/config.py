"""Configuration constants for the content-based filtering module."""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# ── Artifact paths ────────────────────────────────────────────────────────────
MODELS_DIR: Path = REPO_ROOT / "backend" / "models"
TFIDF_PATH: Path = MODELS_DIR / "cb_tfidf.joblib"
ITEM_MATRIX_PATH: Path = MODELS_DIR / "cb_item_matrix.npz"
META_PATH: Path = MODELS_DIR / "cb_meta.parquet"

# ── TF-IDF hyperparameters ────────────────────────────────────────────────────
TFIDF_MAX_FEATURES: int = 10_000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
TFIDF_SUBLINEAR_TF: bool = True

# ── Recommendation hyperparameters ────────────────────────────────────────────
DEFAULT_TOP_N: int = 10
CANDIDATE_POOL_MULTIPLIER: int = 5
MIN_RATING_THRESHOLD: int = 3
CATEGORY_BOOST: float = 0.1
FREE_PREFERENCE_THRESHOLD: float = 0.5
