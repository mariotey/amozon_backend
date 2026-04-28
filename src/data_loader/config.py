"""Configuration constants for the content-based filtering module."""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR: Path = REPO_ROOT / "etl" / "data" / "output"
META_PARQUET: Path = DATA_DIR / "meta_data.parquet"
USER_PARQUET: Path = DATA_DIR / "user.parquet"
ITEM_PARQUET: Path = DATA_DIR / "item.parquet"
USER_ITEM_PARQUET: Path = DATA_DIR / "user-item-interaction.parquet"
