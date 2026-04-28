"""Data loading utilities for the collaborative filtering module."""

import logging
from typing import Optional
import pandas as pd
from collaborative_filtering.config import ITEM_PARQUET, USER_PARQUET, USER_ITEM_PARQUET

logger = logging.getLogger(__name__)


def load_parquet_data(
    parquet_path: str,
    compulsory_cols: Optional[list[str]] = []
) -> pd.DataFrame:
    """Loads a parquet data based on specified path.

    Args:
    - parquet_path(str): The pathway of the targeted parquet file.

    Returns:
    - pd.DataFrame: A DataFrame of the targeted parquet file

    Raises:
    - FileNotFoundError: If the interaction parquet file is missing.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Required data file not found: {parquet_path}"
        )

    logger.info(f"Loading data from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    missing = set(compulsory_cols) - set(df.columns)

    if missing:
        raise ValueError(
            f"DataFrame loaded from {USER_ITEM_PARQUET} is missing columns: {missing}"
        )

    logger.info(f"Loaded {parquet_path} ({len(df)})")
    return df

def load_user() -> pd.DataFrame:
    """Loads user data.

    Returns:
    - pd.DataFrame: A DataFrame containing information of items.
    """
    return load_parquet_data(
        USER_PARQUET
    )

def load_item() -> pd.DataFrame:
    """Loads items data.

    Returns:
    - pd.DataFrame: A DataFrame containing information of items.
    """
    return load_parquet_data(
        ITEM_PARQUET,
        compulsory_cols = ["parent_asin", "is_free"]
    )


def load_user_item() -> pd.DataFrame:
    """Loads user-item interaction data.

    Returns:
    - pd.DataFrame: A DataFrame containing information of user-item interaction.
    """
    return load_parquet_data(
        USER_ITEM_PARQUET,
        compulsory_cols = ["user_id", "parent_asin", "review_rating", "recency_weight"]
    )