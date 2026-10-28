"""Default settings and the single serving-bundle filename."""
from pathlib import Path

MODEL_NAME: str = Path(__file__).parent.name

# ── Artefact Filenames ─────────────────────────────────────────────────────────────────────────

MODEL_FILENAME = "model.joblib"

# ── TF-IDF hyperparameters ─────────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES: int = 10_000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
TFIDF_SUBLINEAR_TF: bool = True

# ── Recommendation hyperparameters ─────────────────────────────────────────────────────────────
DEFAULT_TOP_N: int = 10
CANDIDATE_POOL_MULTIPLIER: int = 5
MIN_RATING_THRESHOLD: int = 4
CATEGORY_BOOST: float = 0.1
FREE_PREFERENCE_THRESHOLD: float = 0.5
