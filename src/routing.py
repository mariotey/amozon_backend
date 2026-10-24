"""
Deterministic CF-first routing for user recommendations.

The router calls the existing Python handlers directly. Users in the trained ALS
mapping are offered collaborative recommendations first. Content-based filtering
is tried next when positive reviews overlap the TF-IDF index. Popular unseen items
provide the final fallback. Empty results permit fallback; model/data errors propagate.
"""
import logging
from typing import Any
import pandas as pd
from src.collaborative_filtering import main as collaborative
from src.collaborative_filtering import tool_config as cf_config
from src.content_based_filtering import main as content
from src.content_based_filtering.tool_config import MIN_RATING_THRESHOLD

logger = logging.getLogger(__name__)

_ITEM_COLUMNS = ["parent_asin", "item_title", "main_category", "is_free"]
_USER_MAPPING_POSITION = list(cf_config.MODEL_ARTEFACTS).index("uid_mapping_filename")


def _popular_items(
    item_df: pd.DataFrame,
    history: pd.DataFrame,
    n: int,
) -> list[dict[str, Any]]:
    """
    Rank unseen catalogue items by quality, review count, and a stable ID tie-breaker.

    Args:
    - item_df (pd.DataFrame): Catalogue including quality_score and num_reviews.
    - history (pd.DataFrame): All reviews belonging to the requested user.
    - n (int): Maximum number of items to return.

    Returns:
    - list[dict[str, Any]]: Ranked item metadata; empty if no unseen items remain.
    """
    unseen = item_df[~item_df["parent_asin"].isin(history["parent_asin"])]
    ranked = unseen.sort_values(
        ["quality_score", "num_reviews", "parent_asin"],
        ascending=[False, False, True],
        na_position="last",
    )
    return ranked.head(n)[_ITEM_COLUMNS].to_dict(orient="records")


def get_user_recommendations(
    user_id: str,
    n: int = 10,
) -> dict[str, Any]:
    """
    Select the first eligible engine that produces recommendations for a user.

    Eligibility and fallback order:
    1. Collaborative: user exists in the trained ALS mapping.
    2. Content-based: at least one review meets MIN_RATING_THRESHOLD and its item
       exists in the TF-IDF index.
    3. Popular: rank unseen items without requiring a personalised model profile.

    Uses the shared data/model caches initialized by the API. A user missing from
    both models is valid and receives the popularity fallback. No minimum review
    count beyond the content engine's actual prerequisites is imposed.

    Args:
    - user_id (str): Non-empty user identifier; surrounding whitespace is stripped.
    - n (int): Maximum number of results, from 1 to 100.

    Returns:
    - dict[str, Any]: user_id, engine_used, reason, and items. Items may be empty
                      when no unseen catalogue candidates remain.

    Raises:
    - ValueError: If the identifier is blank or n is outside the supported range.
    - Exception: Unexpected data/model failures propagate to the caller rather
                 than being treated as missing user history.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if not 1 <= n <= 100:
        raise ValueError("n must be between 1 and 100")
    user_id = user_id.strip()

    data = collaborative.get_data_loader()
    models = collaborative.get_models_loader().model_artefacts
    cf_artefacts = models[cf_config.MODEL_NAME]
    known_users = cf_artefacts[_USER_MAPPING_POSITION].values()
    reasons: list[str] = []

    if user_id in known_users:
        items = collaborative.get_user_recommendations(user_id, n=n)
        if items:
            logger.info("Selected collaborative filtering for user %s", user_id)
            return {
                "user_id": user_id,
                "engine_used": "collaborative",
                "reason": "User is present in the trained ALS model",
                "items": items,
            }
        reasons.append("Collaborative filtering returned no items")
    else:
        reasons.append("User is absent from the trained ALS model")

    history = data.user_item_df[data.user_item_df["user_id"] == user_id]
    content_meta = models[content.MODEL_NAME][2]
    qualifying = history[
        (history["review_rating"] >= MIN_RATING_THRESHOLD)
        & history["parent_asin"].isin(content_meta["parent_asin"])
    ]

    if not qualifying.empty:
        items = content.get_user_recommendations(user_id, n=n)
        if items:
            logger.info("Selected content-based filtering for user %s", user_id)
            return {
                "user_id": user_id,
                "engine_used": "content_based",
                "reason": "; ".join(reasons + ["User has qualifying positive history"]),
                "items": items,
            }
        reasons.append("Content-based filtering returned no items")
    else:
        reasons.append("User has no qualifying positive history in the content model")

    logger.info("Selected popularity fallback for user %s", user_id)
    return {
        "user_id": user_id,
        "engine_used": "popular",
        "reason": "; ".join(reasons),
        "items": _popular_items(data.item_df, history, n),
    }
