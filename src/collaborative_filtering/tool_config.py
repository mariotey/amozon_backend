"""Default settings and the single serving-bundle filename."""

from pathlib import Path

MODEL_NAME: str = Path(__file__).parent.name

# ── Artefact Filenames ─────────────────────────────────────────────────────────────────────────

MODEL_FILENAME = "model.joblib"

# ── ALS Model hyperparameters ──────────────────────────────────────────────────────────────────
ALS_FACTORS: int = 50
ALS_REG: float = 0.01
ALS_ITERA: int = 20
ALS_RANDOM_STATE: int = 42
ALS_NUM_THREADS: int = 1

# Initial baseline: explicit likes (4/5), dislikes (1/2), and neutral (3).
# The confidence scale is a documented starting value, not a tuned optimum.
ALS_POSITIVE_RATING_THRESHOLD: float = 4.0
ALS_NEGATIVE_RATING_THRESHOLD: float = 2.0
ALS_CONFIDENCE_SCALE: float = 20.0

# ── Recommendation hyperparameters ─────────────────────────────────────────────────────────────
DEFAULT_TOP_N: int = 10
