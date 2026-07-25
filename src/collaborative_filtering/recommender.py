"""Core recommendation logic for collaborative filtering."""

import logging
import numpy as np
import pandas as pd
import scipy.sparse
from .model import load_artifacts
from .tool_config import (
    DEFAULT_TOP_N
)

logger = logging.getLogger(__name__)

def recommend_for_user(
    user_id: str,
    artifacts,
    n: int = DEFAULT_TOP_N,
) -> list[str]:
    """Generate top-N content-based recommendations for a user.

    Excludes already-seen items, boosts the user's preferred category,
    and optionally filters out paid items for users who prefer free software.

    Args:
    - user_id: The user identifier.
    - n: Number of recommendations to return.

    Returns:
    - list: A list containing the parent_asin of the recommended products for the given user id.

    Raises:
    - ValueError: If n is not a positive integer.
    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    model, user_item_matrix, idx_to_userid, idx_to_itempasin = artifacts

    # Create a mapping of user_id to idx
    userid_to_idx = {v: k for k, v in idx_to_userid.items()}

    # Check if user_id is valid or not
    user_idx = userid_to_idx.get(user_id)

    if user_idx is None:
        raise ValueError(f"Unknown user_id: {user_id}")

    # Recommend using the idx of the targeted user
    recomm_idx, recomm_scores = model.recommend(
        userid=np.int32(user_idx),
        user_items=user_item_matrix[np.int32(user_idx)],
        N=n
    )

    # Returns the parent_asin of the recommended items
    return [idx_to_itempasin[i] for i in recomm_idx]