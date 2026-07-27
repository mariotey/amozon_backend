"""Core recommendation logic for content-based filtering."""
from typing import Any
import logging
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from .tool_config import (
    DEFAULT_TOP_N,
    CANDIDATE_POOL_MULTIPLIER,
    MIN_RATING_THRESHOLD,
    CATEGORY_BOOST,
    FREE_PREFERENCE_THRESHOLD
)

logger = logging.getLogger(__name__)

def build_user_profile(
    user_id: str,
    user_item_df: pd.DataFrame,
    item_matrix: scipy.sparse.csr_matrix,
    item_to_idx: dict[str, int],
) -> np.ndarray | None:
    """Build a weighted TF-IDF profile vector for a user from their review history.

    Args:
    - user_id (str): The user identifier
    - user_item_df (pd.DataFrame): DataFrame of user-item interactions
    - item_matrix (scipy.sparse.csr_matrix): Sparse TF-IDF item matrix (n_items × n_features)
    - item_to_idx (dict[str, int]): Mapping from parent_asin to row index in item_matrix

    Returns:
    - np.ndarray: A (1 × n_features) numpy array representing the user profile, or None if the
                  user has no qualifying history
    """
    hist = user_item_df[user_item_df["user_id"] == user_id].copy()
    hist = hist[
        hist["parent_asin"].isin(item_to_idx)
        & (hist["review_rating"] >= MIN_RATING_THRESHOLD)
    ]

    if hist.empty:
        logger.warning("No positive history found for user: %s", user_id)
        return None

    weights = (hist["review_rating"] / 5).values * hist["recency_weight"].values
    indices = hist["parent_asin"].map(item_to_idx).values
    profile = (weights @ item_matrix[indices]) / (weights.sum() + 1e-9)
    return np.asarray(profile).reshape(1, -1)

def build_item_index_maps(
    meta_df: pd.DataFrame
) -> tuple[dict[str, int], dict[int, str]]:
    """Build mappings between item IDs and their corresponding matrix indices.

    Args:
    - meta_df (pd.DataFrame): DataFrame containing a "parent_asin" column

    Returns:
    - tuple:
        - item_to_idx: Maps each parent ASIN to its integer index
        - idx_to_item: Maps each integer index back to its parent ASIN
    """
    item_to_idx = {
        asin: idx
        for idx, asin in enumerate(meta_df["parent_asin"])
    }
    idx_to_item = {
        idx: asin
        for asin, idx in item_to_idx.items()
    }

    return item_to_idx, idx_to_item

def recommend_for_user(
    user_id: str,
    user_item_df: pd.DataFrame,
    artefacts: tuple[Any, ...],
    n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Generate top-N content-based recommendations for a user.

    Excludes already-seen items, boosts the user's preferred category, and optionally filters out
    paid items for users who prefer free software.

    Args:
    - user_id (str): The user identifier
    - user_item_df (pd.DataFrame): DataFrame of user-item interactions
    - artefacts (tuple): Loaded artefacts of the tool
    - n (int): Number of recommendations to return

    Returns:
    - pd.DataFrame: A DataFrame with columns ["parent_asin", "score", "item_title",
                    "main_category", "is_free"]. Empty if the user has no history

    Raises:
    - ValueError: If n is not a positive integer.
    """
    _, item_matrix, meta_df = artefacts
    item_to_idx, idx_to_item = build_item_index_maps(meta_df)

    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    profile = build_user_profile(user_id, user_item_df, item_matrix, item_to_idx)
    if profile is None:
        return pd.DataFrame()

    scores: np.ndarray = cosine_similarity(profile, item_matrix).flatten()

    seen_asins = set(
        user_item_df.loc[user_item_df["user_id"] == user_id, "parent_asin"]
    )
    seen_idx = [item_to_idx[a] for a in seen_asins if a in item_to_idx]
    scores[seen_idx] = -1

    # Derive user preferences from positive history
    pos_hist = user_item_df[
        (user_item_df["user_id"] == user_id)
        & (user_item_df["review_rating"] >= MIN_RATING_THRESHOLD)
        & (user_item_df["parent_asin"].isin(item_to_idx))
    ]
    hist_meta = pos_hist.merge(
        meta_df[["parent_asin", "main_category", "is_free"]],
        on="parent_asin",
        how="left",
    )
    top_category: str | None = (
        hist_meta["main_category"].value_counts().index[0]
        if not hist_meta.empty
        else None
    )
    prefers_free: bool = (
        hist_meta["is_free"].mean() > FREE_PREFERENCE_THRESHOLD
        if not hist_meta.empty
        else False
    )

    pool_size = n * CANDIDATE_POOL_MULTIPLIER
    pool_idx: np.ndarray = np.argsort(scores)[::-1][:pool_size]
    pool = pd.DataFrame(
        {
            "parent_asin": [idx_to_item[i] for i in pool_idx],
            "score": scores[pool_idx],
        }
    ).merge(
        meta_df[["parent_asin", "item_title", "main_category", "is_free"]],
        on="parent_asin",
    )

    if top_category:
        pool["score"] += CATEGORY_BOOST * (
            pool["main_category"] == top_category
        ).astype(float)
        logger.debug("Applied category boost for: %s", top_category)

    if prefers_free:
        pool = pool[pool["is_free"]]
        logger.debug("Applied free-only filter for user: %s", user_id)

    recommendations = pool.sort_values("score", ascending=False).head(n).reset_index(drop=True)

    logger.info("Returning %d recommendations for user: %s", len(recommendations), user_id)

    return recommendations

def similar_items(
    parent_asin: str,
    artefacts: tuple[Any, ...],
    n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Find the top-N items most similar to a given item using cosine similarity.

    Args:
    - parent_asin (str): The ASIN of the seed item
    - artefacts (tuple): Loaded artefacts of the tool
    - n (int): Number of similar items to return

    Returns:
    - pd.DataFrame: A DataFrame with columns ["parent_asin", "score", "item_title",
                    "main_category"]. Empty if the ASIN is not in the index.

    Raises:
    - ValueError: If n is not a positive integer
    """
    _, item_matrix, meta_df = artefacts
    item_to_idx, idx_to_item = build_item_index_maps(meta_df)

    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    if parent_asin not in item_to_idx:
        logger.warning("Unknown item ASIN: %s", parent_asin)
        return pd.DataFrame()

    idx = item_to_idx[parent_asin]
    scores: np.ndarray = cosine_similarity(
        item_matrix[idx], item_matrix
    ).flatten()
    scores[idx] = -1

    top_idx: np.ndarray = np.argsort(scores)[::-1][:n]
    result = pd.DataFrame(
        {
            "parent_asin": [idx_to_item[i] for i in top_idx],
            "score": scores[top_idx],
        }
    ).merge(meta_df[["parent_asin", "item_title", "main_category"]], on="parent_asin")

    logger.info("Returning %d similar items for ASIN: %s", len(result), parent_asin)

    return result
