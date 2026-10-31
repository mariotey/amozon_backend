"""Prepare ratings and fit ALS in memory. Saving is an explicit notebook step."""

import implicit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from utils.review_history import latest_reviews
from .tool_config import (
    ALS_CONFIDENCE_SCALE,
    ALS_FACTORS,
    ALS_ITERA,
    ALS_NEGATIVE_RATING_THRESHOLD,
    ALS_NUM_THREADS,
    ALS_POSITIVE_RATING_THRESHOLD,
    ALS_RANDOM_STATE,
    ALS_REG,
)


def prepare_training_data(items: pd.DataFrame, history: pd.DataFrame) -> tuple:
    """
    Build aligned IDs and signed confidences; neutral entries remain seen.

    Args:
    - items(pd.DataFrame): Product metadata for the supplied workflow.
    - history(pd.DataFrame): Available reviews, including ratings and seen products.

    Returns:
    - tuple: CSR interaction matrix, ordered user IDs, and ordered product IDs.
    """
    if "parent_asin" not in items or items.empty or history.empty:
        raise ValueError("ALS requires nonempty item metadata and history")
    if (
        not items.parent_asin.is_unique
        or not items.parent_asin.map(
            lambda value: isinstance(value, str) and bool(value.strip()) and value == value.strip()
        ).all()
    ):
        raise ValueError("Item IDs must be unique, nonempty, trimmed strings")
    if "review_date" not in history:
        raise ValueError("ALS history requires review_date")
    past = latest_reviews(history)
    past = past.loc[past.parent_asin.isin(items.parent_asin)]
    if past.empty:
        raise ValueError("History has no products in the catalogue")
    users, products = sorted(past.user_id.unique()), sorted(past.parent_asin.unique())
    user_index = {value: i for i, value in enumerate(users)}
    item_index = {value: i for i, value in enumerate(products)}
    ratings = past.review_rating.to_numpy(dtype=float)
    confidence = 1 + ALS_CONFIDENCE_SCALE * np.abs(ratings - 3) / 2
    weights = np.where(
        ratings >= ALS_POSITIVE_RATING_THRESHOLD,
        confidence,
        np.where(ratings <= ALS_NEGATIVE_RATING_THRESHOLD, -confidence, 0),
    )
    matrix = csr_matrix(
        (
            weights.astype(np.float32),
            (
                past.user_id.map(user_index).to_numpy(),
                past.parent_asin.map(item_index).to_numpy(),
            ),
        ),
        shape=(len(users), len(products)),
    )
    matrix.sort_indices()
    # Keep explicit zeros: their column indices exclude neutral reviewed products.
    return matrix, users, products


def fit(
    items: pd.DataFrame,
    history: pd.DataFrame,
    *,
    factors: int = ALS_FACTORS,
    regularization: float = ALS_REG,
    iterations: int = ALS_ITERA,
    random_state: int = ALS_RANDOM_STATE,
) -> dict:
    """
    Fit only the supplied history; return a complete named serving bundle.

    Args:
    - items(pd.DataFrame): Product metadata for the supplied workflow.
    - history(pd.DataFrame): Available reviews, including ratings and seen products.
    - factors(int): Number of latent ALS factors.
    - regularization(float): ALS regularization strength.
    - iterations(int): Number of ALS fitting iterations.
    - random_state(int): Random seed for reproducible ALS fitting.

    Returns:
    - dict: ALS model, interaction matrix, ordered user IDs, and ordered product IDs.
    """
    matrix, users, products = prepare_training_data(items, history)
    if not np.any(matrix.data > 0):
        raise ValueError("ALS requires at least one positive historical preference")
    with threadpool_limits(limits=1, user_api="blas"):
        model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=1.0,
            random_state=random_state,
            use_gpu=False,
            num_threads=ALS_NUM_THREADS,
        )
        model.fit(matrix, show_progress=False)
    return {"model": model, "interactions": matrix, "user_ids": users, "item_ids": products}
