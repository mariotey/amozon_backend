"""
Deterministic CF-first routing for user recommendations.

The router calls the existing Python handlers directly. Users in the trained ALS
mapping are offered collaborative recommendations first. Content-based filtering
is tried next when positive reviews overlap the TF-IDF index. If neither engine
produces recommendations, items and engine_used are None; model/data errors propagate.

The API supplies shared model artifacts, a filtered history loader, and a bounded
metadata loader. Legacy standalone callers retain cached full-data behavior.
Empty engine results permit fallback; dependency/inference failures propagate.
Unknown users remain valid. Popular browsing is separate.
"""
import logging
from contextlib import nullcontext
from typing import Any
from src.collaborative_filtering import main as collaborative
from src.collaborative_filtering import tool_config as cf_config
from src.content_based_filtering import main as content
from src.content_based_filtering.tool_config import MIN_RATING_THRESHOLD

logger = logging.getLogger(__name__)

_USER_MAPPING_POSITION = list(cf_config.MODEL_ARTEFACTS).index("uid_mapping_filename")


def get_user_recommendations(
    user_id: str,
    n: int = 10,
    *,
    model_artefacts: dict | None = None,
    history_loader=None,
    item_loader=None,
    inference_lock=None,
    user_index: dict[str, int] | None = None,
    content_indexes: tuple[dict[str, int], dict[int, str]] | None = None,
) -> dict[str, Any]:
    """
    Select the first eligible engine that produces recommendations for a user.

    Eligibility and fallback order:
    1. Collaborative: user exists in the trained ALS mapping.
    2. Content-based: at least one review meets MIN_RATING_THRESHOLD and its item
       exists in the TF-IDF index.
    3. Return None for items and engine_used if neither engine produces items.

    Serving calls provide explicit model_artefacts/history_loader/item_loader and
    never initialize a bulk DataLoader. History is fetched only after CF cannot
    produce results and remains local to this call. No minimum review count beyond
    the content engine's actual prerequisites is imposed.

    Args:
    - user_id (str): Non-empty user identifier; surrounding whitespace is stripped.
    - n (int): Maximum number of results, from 1 to 100.

    Returns:
    - dict[str, Any]: user_id, engine_used, reason, and items. Items and engine_used are None
                      when neither personalised engine produces recommendations.

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

    serving = model_artefacts is not None
    if serving and (history_loader is None or item_loader is None):
        raise ValueError("Serving inference requires explicit history and item loaders")
    models = model_artefacts if serving else collaborative.get_models_loader().model_artefacts
    cf_artefacts = models[cf_config.MODEL_NAME]
    known_users = user_index if user_index is not None else cf_artefacts[_USER_MAPPING_POSITION].values()
    reasons: list[str] = []

    if user_id in known_users:
        if serving:
            items = collaborative.get_user_recommendations(
                user_id, n=n, artefacts=cf_artefacts, item_loader=item_loader,
                user_index=user_index, inference_lock=inference_lock,
            )
        else:
            with inference_lock if inference_lock is not None else nullcontext():
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

    if serving:
        history = history_loader(user_id)
    else:
        data = collaborative.get_data_loader()
        history = data.user_item_df[data.user_item_df["user_id"] == user_id]
    content_meta = models[content.MODEL_NAME][2]
    qualifying = history[
        (history["review_rating"] >= MIN_RATING_THRESHOLD)
        & history["parent_asin"].isin(content_meta["parent_asin"])
    ]

    if not qualifying.empty:
        with inference_lock if inference_lock is not None else nullcontext():
            items = content.get_user_recommendations(user_id, n=n, **(
                {"artefacts": models[content.MODEL_NAME], "history": history,
                 "index_maps": content_indexes} if serving else {}
            ))
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

    logger.info("No personalised recommendations for user %s", user_id)
    return {
        "user_id": user_id,
        "engine_used": None,
        "reason": "; ".join(reasons),
        "items": None,
    }
