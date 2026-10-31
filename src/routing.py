"""Application-only CF, content, then popular fallback with explicit resources."""

from contextlib import nullcontext

from src.collaborative_filtering import recommender as collaborative
from src.content_based_filtering import recommender as content
from src.content_based_filtering.tool_config import MIN_RATING_THRESHOLD
from utils.review_history import latest_reviews


def get_user_recommendations(
    user_id: str,
    n: int = 10,
    *,
    model_artefacts: dict,
    history_loader,
    item_loader,
    popular_loader,
    inference_lock=None,
    user_index: dict | None = None,
    content_indexes: tuple | None = None,
) -> dict:
    """
    Empty recommendations permit fallback; dependency/inference failures propagate.

    Args:
    - user_id(str): Shopper identifier used to select history or recommendations.
    - n(int): Maximum number of results to return.
    - model_artefacts(dict): Collaborative and content bundles keyed by algorithm name.
    - history_loader(Callable): Fetch one shopper's full inference history as a DataFrame.
    - item_loader(Callable): Fetch card records for ranked product IDs.
    - popular_loader(Callable): Fetch popular records while excluding seen products.
    - inference_lock(Lock | None): Optional lock serializing numeric inference.
    - user_index(dict | None): Optional cached user-to-ALS-row map.
    - content_indexes(tuple | None): Optional cached content product index maps.

    Returns:
    - dict: User ID, engine name, fallback reason, and ranked unseen records.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a nonempty string")
    if not 1 <= n <= 100:
        raise ValueError("n must be between 1 and 100")
    user_id = user_id.strip()
    cf = model_artefacts["collaborative_filtering"]
    cb = model_artefacts["content_based_filtering"]
    known_users = user_index if user_index is not None else cf["user_ids"]
    reasons = []
    if user_id in known_users:
        with inference_lock if inference_lock is not None else nullcontext():
            ids = collaborative.recommend_for_user(user_id, cf, n=n, user_index=user_index)
        # Fetch card metadata outside the numeric inference lock; preserve rank.
        catalogue = {row["parent_asin"]: row for row in item_loader(ids)} if ids else {}
        items = [catalogue[product] for product in ids if product in catalogue]
        if items:
            return {
                "user_id": user_id,
                "engine_used": "collaborative",
                "reason": "User is present in the trained ALS model",
                "items": items,
            }
        reasons.append("Collaborative filtering returned no items")
    else:
        reasons.append("User is absent from the trained ALS model")
    history = latest_reviews(history_loader(user_id))
    qualifying = history.loc[
        history.review_rating.ge(MIN_RATING_THRESHOLD)
        & history.parent_asin.isin(cb["metadata"].parent_asin)
    ]
    if not qualifying.empty:
        with inference_lock if inference_lock is not None else nullcontext():
            items = content.recommend_for_user(
                user_id,
                history,
                cb,
                n=n,
                index_maps=content_indexes,
            ).to_dict(orient="records")
        if items:
            return {
                "user_id": user_id,
                "engine_used": "content_based",
                "items": items,
                "reason": "; ".join(reasons + ["User has qualifying positive history"]),
            }
        reasons.append("Content-based filtering returned no items")
    else:
        reasons.append("User has no qualifying positive history in the content model")
    items = popular_loader(n, exclude_ids=set(history.parent_asin))
    reasons.append(
        "Showing unseen popular products because personalised recommendations are unavailable"
    )
    if not items:
        reasons.append("No unseen popular products remain")
    return {
        "user_id": user_id,
        "engine_used": "popular",
        "reason": "; ".join(reasons),
        "items": items,
    }
