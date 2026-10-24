"""
Core recommendation logic for collaborative filtering.

Artifact order: ALS model, CSR user-item matrix, index-to-user map, and
index-to-product map. Matrix rows represent users and columns products.
Filtering removes seen items and exhausted-result padding without reordering
the surviving model-ranked candidates.
"""
from typing import Any
import logging
import numpy as np
import pandas as pd
import scipy.sparse
from .tool_config import (
    DEFAULT_TOP_N
)

logger = logging.getLogger(__name__)

def recommend_for_user(
    user_id: str,
    artefacts: tuple[Any, ...],
    n: int = DEFAULT_TOP_N,
    *,
    user_index: dict[str, int] | None = None,
) -> list[str]:
    """
    Generate ranked ALS collaborative recommendations for a known user.

    Excludes already-seen items and invalid padding returned when candidates
    are exhausted. Returns fewer than n items if fewer candidates remain.

    Args:
    - user_id (str): The user identifier
    - artefacts (tuple): Loaded artefacts of the tool
    - n (int): Number of recommendations to return

    Returns:
    - list: A list containing the parent_asin of the recommended products for the given user id

    Raises:
    - ValueError: If n is below 1 or the user is absent from the model mapping.
    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    model, user_item_matrix, idx_to_userid, idx_to_itempasin = artefacts

    # The API reuses its model-owned inverse map; standalone callers keep the fallback.
    userid_to_idx = user_index if user_index is not None else {v: k for k, v in idx_to_userid.items()}

    # Check if user_id is valid or not
    user_idx = userid_to_idx.get(user_id)

    if user_idx is None:
        raise ValueError(f"Unknown user_id: {user_id}")

    # Reuse the same sparse row for inference and seen-item filtering.
    user_items = user_item_matrix[np.int32(user_idx)]
    recomm_idx, recomm_scores = model.recommend(
        userid=np.int32(user_idx),
        user_items=user_items,
        N=n
    )

    # Returns the parent_asin of the recommended items
    # implicit pads exhausted results with minimum-score entries. These can
    # include already-seen products, so only return finite, unseen candidates.
    seen_idx = set(user_items.indices)
    return [
        idx_to_itempasin[i]
        for i, score in zip(recomm_idx, recomm_scores)
        if i in idx_to_itempasin and i not in seen_idx and np.isfinite(score)
        and score > -np.finfo(np.float32).max
    ]
