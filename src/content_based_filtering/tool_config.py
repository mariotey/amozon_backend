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
MODEL_NAME: str = "content_based_filtering"
MODEL_ID: str = "xxxxx"

MODEL_ARTEFACTS = {
    "tfidf_filename": "cb_tfidf.joblib",
    "item_matrix_filename": "cb_item_matrix.npz",
    "meta_filename": "cb_meta.parquet"
}
