"""
Core recommendation logic for collaborative filtering.

Named bundle: ALS model, CSR interaction matrix, user IDs, and product IDs. Matrix rows represent users and columns products.
Filtering removes seen items and exhausted-result padding without reordering
the surviving model-ranked candidates.
"""
import numpy as np
from .tool_config import (
    DEFAULT_TOP_N
)


def recommend_for_user(
    user_id: str,
    artefacts: dict,
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
    - artefacts (dict): Named model bundle
    - n (int): Number of recommendations to return

    Returns:
    - list: A list containing the parent_asin of the recommended products for the given user id

    Raises:
    - ValueError: If n is below 1 or the user is absent from the model mapping.
    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    model = artefacts["model"]
    user_item_matrix = artefacts["interactions"]
    user_ids, product_ids = artefacts["user_ids"], artefacts["item_ids"]

    # The API reuses its model-owned inverse map; standalone callers keep the fallback.
    userid_to_idx = user_index if user_index is not None else {value: i for i, value in enumerate(user_ids)}

    # Check if user_id is valid or not
    user_idx = userid_to_idx.get(user_id)

    if user_idx is None:
        raise ValueError(f"Unknown user_id: {user_id}")

    # Reuse the same sparse row for inference and seen-item filtering.
    user_items = user_item_matrix[np.int32(user_idx)]
    # An all-neutral/disliked history has no positive preference to personalize
    # from. Avoid presenting arbitrary tied scores as recommendations.
    if not np.any(user_items.data > 0):
        return []
    available = len(product_ids) - len(user_items.indices)
    if available <= 0:
        return []
    recomm_idx, recomm_scores = model.recommend(
        userid=np.int32(user_idx),
        user_items=user_items,
        N=min(n, available)
    )

    # Returns the parent_asin of the recommended items
    # implicit pads exhausted results with minimum-score entries. These can
    # include already-seen products, so only return finite, unseen candidates.
    seen_idx = set(user_items.indices)
    return [
        product_ids[i]
        for i, score in zip(recomm_idx, recomm_scores)
        if 0 <= i < len(product_ids) and i not in seen_idx and np.isfinite(score)
        and score > -np.finfo(np.float32).max
    ]


def evaluate(artefacts: dict, data, k: int = DEFAULT_TOP_N) -> dict:
    """Evaluate CF alone on its prepared supported population; never fit or save."""
    from src.metrics import evaluate_predictions

    index = {value: i for i, value in enumerate(artefacts["user_ids"])}
    predictions = {
        user: recommend_for_user(user, artefacts, n=k, user_index=index) if user in index else []
        for user in data.targets.user_id.unique()
    }
    result = evaluate_predictions(predictions, data.targets, k)
    result["summary"].update(excluded_targets=data.excluded_targets, excluded_users=data.excluded_users)
    return result
