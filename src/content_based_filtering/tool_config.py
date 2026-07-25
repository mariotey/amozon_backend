"""
Configuration constants for the content-based filtering module.
"""

# ── TF-IDF hyperparameters ─────────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES: int = 10_000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
TFIDF_SUBLINEAR_TF: bool = True

# ── Recommendation hyperparameters ─────────────────────────────────────────────────────────────
DEFAULT_TOP_N: int = 10
CANDIDATE_POOL_MULTIPLIER: int = 5
MIN_RATING_THRESHOLD: int = 3
CATEGORY_BOOST: float = 0.1
FREE_PREFERENCE_THRESHOLD: float = 0.5

# ── Artefact Filenames ─────────────────────────────────────────────────────────────────────────
TFIDF_FILENAME: str = "cb_tfidf.joblib"
ITEM_MATRIX_FILENAME: str = "cb_item_matrix.npz"
META_FILENAME: str = "cb_meta.parquet"
