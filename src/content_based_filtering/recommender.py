"""
Core recommendation logic for content-based filtering.

Matrix rows and metadata rows must stay aligned. Positive reviews create a
rating/recency-weighted user profile. Cosine similarity selects a bounded pool;
seen items are excluded, category preference boosts scores, and a free-only
preference can reduce results. Fewer than n results is valid.
"""
import logging
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot
from utils.review_history import latest_reviews
from tqdm import tqdm
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
    """
    Build a weighted TF-IDF profile vector for a user from their review history.

    Args:
    - user_id (str): The user identifier
    - user_item_df (pd.DataFrame): DataFrame of user-item interactions
    - item_matrix (scipy.sparse.csr_matrix): Sparse TF-IDF item matrix (n_items × n_features)
    - item_to_idx (dict[str, int]): Mapping from parent_asin to row index in item_matrix

    Returns:
    - np.ndarray: A (1 × n_features) numpy array representing the user profile, or None if the
                  user has no qualifying history
    """
    hist = latest_reviews(user_item_df.loc[user_item_df["user_id"] == user_id])
    # Mapping a callable avoids constructing a Series from the entire item map.
    indices = hist["parent_asin"].map(item_to_idx.get)
    positive = indices.notna() & hist["review_rating"].ge(MIN_RATING_THRESHOLD)

    if not positive.any():
        logger.warning("No positive history found for user: %s", user_id)
        return None

    return _weighted_profile(hist.loc[positive], item_matrix, indices.loc[positive].to_numpy(dtype=np.intp))


def _weighted_profile(hist, item_matrix, indices) -> np.ndarray:
    """Average known, positive item vectors using the original rating/recency weights."""
    weights = (hist["review_rating"] / 5).values * hist["recency_weight"].values
    profile = (weights @ item_matrix[indices]) / (weights.sum() + 1e-9)
    return np.asarray(profile).reshape(1, -1)


def _normalize_items(item_matrix, profile_dtype):
    """Match cosine_similarity's common dtype before normalizing catalogue rows."""
    dtype = np.float32 if profile_dtype == item_matrix.dtype == np.float32 else np.float64
    return normalize(item_matrix.astype(dtype, copy=False), copy=True)

def build_item_index_maps(
    meta_df: pd.DataFrame
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build mappings between item IDs and their corresponding matrix indices.

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
    artefacts: dict,
    n: int = DEFAULT_TOP_N,
    *,
    index_maps: tuple[dict[str, int], dict[int, str]] | None = None,
) -> pd.DataFrame:
    """
    Generate top-N content-based recommendations for a user.

    Excludes already-seen items, boosts the user's preferred category, and optionally filters out
    paid items for users who prefer free software.

    Args:
    - user_id (str): The user identifier
    - user_item_df (pd.DataFrame): DataFrame of user-item interactions
    - artefacts (dict): Item matrix and row-aligned metadata
    - n (int): Number of recommendations to return

    Returns:
    - pd.DataFrame: A DataFrame with columns ["parent_asin", "score", "item_title",
                    "main_category", "is_free"]. Empty if the user has no history

    Raises:
    - ValueError: If n is not a positive integer.
    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    # A newer dislike/neutral rating replaces an older like of the same product.
    # Keep every distinct reviewed product available for seen-item exclusion.
    history = latest_reviews(user_item_df.loc[user_item_df["user_id"] == user_id])
    item_to_idx, _ = index_maps if index_maps is not None else build_item_index_maps(artefacts["metadata"])
    return _recommend_from_history(user_id, history, artefacts, n, item_to_idx)


def _recommend_from_history(user_id, history, artefacts, n, item_to_idx, normalized_items=None):
    """Shared ranking for one resolved history; evaluation reuses normalized items."""
    item_matrix, meta_df = artefacts["item_matrix"], artefacts["metadata"]
    indices = history["parent_asin"].map(item_to_idx.get)
    positive = indices.notna() & history["review_rating"].ge(MIN_RATING_THRESHOLD)
    if not positive.any():
        logger.warning("No positive history found for user: %s", user_id)
        return pd.DataFrame()

    positive_idx = indices.loc[positive].to_numpy(dtype=np.intp)
    profile = _weighted_profile(history.loc[positive], item_matrix, positive_idx)
    if normalized_items is None:
        normalized_items = _normalize_items(item_matrix, profile.dtype)
    # Exactly the cosine calculation, without copying/normalizing the catalogue per user.
    scores = safe_sparse_dot(normalize(profile), normalized_items.T, dense_output=True).ravel()
    seen_idx = indices.dropna().to_numpy(dtype=np.intp)

    # Metadata rows already correspond to matrix indices; no catalogue-wide joins.
    hist_meta = meta_df.iloc[positive_idx]
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
    # Exclude seen products before selecting the pool; a low score alone still
    # returns them when the requested count exceeds the unseen candidates.
    eligible = np.ones(len(scores), dtype=bool)
    eligible[seen_idx] = False
    candidate_idx = np.flatnonzero(eligible & np.isfinite(scores))
    pool_idx: np.ndarray = candidate_idx[np.argsort(scores[candidate_idx])[::-1]][:pool_size]
    pool = meta_df.iloc[pool_idx][["parent_asin", "item_title", "main_category", "is_free"]].copy()
    pool.insert(1, "score", scores[pool_idx])

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
    artefacts: dict,
    n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """
    Find the top-N items most similar to a given item using cosine similarity.

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
    item_matrix, meta_df = artefacts["item_matrix"], artefacts["metadata"]
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

    eligible = np.flatnonzero(np.isfinite(scores) & (np.arange(len(scores)) != idx))
    top_idx: np.ndarray = eligible[np.argsort(scores[eligible])[::-1]][:n]
    result = pd.DataFrame(
        {
            "parent_asin": [idx_to_item[i] for i in top_idx],
            "score": scores[top_idx],
        }
    ).merge(meta_df[["parent_asin", "item_title", "main_category"]], on="parent_asin")

    logger.info("Returning %d similar items for ASIN: %s", len(result), parent_asin)

    return result


def evaluate(artefacts: dict, data, k: int = DEFAULT_TOP_N) -> dict:
    """Prepare shared inputs once, then evaluate the same ranking used for serving."""
    from src.metrics import evaluate_predictions

    users = data.targets.user_id.unique()
    item_to_idx, _ = build_item_index_maps(artefacts["metadata"])
    predictions = {}
    if len(users):
        # Resolve only evaluated users' histories once, including rating updates.
        history = latest_reviews(data.history.loc[data.history.user_id.isin(users)])
        histories = {user: frame for user, frame in history.groupby("user_id", sort=False)}
        weights = (history["review_rating"] / 5).values * history["recency_weight"].values
        profile_dtype = np.result_type(weights.dtype, artefacts["item_matrix"].dtype)
        normalized_items = _normalize_items(artefacts["item_matrix"], profile_dtype)
        empty_history = history.iloc[:0]
        for user in tqdm(users):
            rows = _recommend_from_history(
                user, histories.get(user, empty_history), artefacts, k, item_to_idx, normalized_items,
            )
            predictions[user] = rows.parent_asin.tolist() if not rows.empty else []

    result = evaluate_predictions(predictions, data.targets, k)
    result["summary"].update(excluded_targets=data.excluded_targets, excluded_users=data.excluded_users)

    return result
