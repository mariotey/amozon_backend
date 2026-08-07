"""
User analysis backing the `analyze_user` tool.

Pure pandas over the interaction and item tables — no LLM involved, so this is directly
testable offline. The output is a verdict (`available_engines`), not raw statistics: the rating
threshold and free-preference cut-off are applied here rather than being described to the model
in a prompt, where they could be ignored under paraphrase.

Thresholds are imported from the content-based tool's config so that a change there cannot
silently desynchronise routing from the engine's actual behaviour.
"""
import logging
import pandas as pd
from src.content_based_filtering.tool_config import (
    MIN_RATING_THRESHOLD,
    FREE_PREFERENCE_THRESHOLD,
)
from .models import UserAnalysis
from .orchestration_config import (
    COLLABORATIVE_TOOL,
    CONTENT_BASED_TOOL,
    POPULAR_TOOL,
)
from .resources import get_item_df, get_known_cf_user_ids, get_user_item_df

logger = logging.getLogger(__name__)

def _positive_history_with_metadata(
    history: pd.DataFrame,
    item_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter a user's review history to positively-rated rows and attach item metadata.

    Args:
    - history (pd.DataFrame): All interaction rows for a single user
    - item_df (pd.DataFrame): Item metadata table

    Returns:
    - pd.DataFrame: Positively-rated rows joined to "main_category" and "is_free". Empty when the
                    user has no qualifying history.
    """
    positive = history[history["review_rating"] >= MIN_RATING_THRESHOLD]

    if positive.empty:
        return positive

    return positive.merge(
        item_df[["parent_asin", "main_category", "is_free"]],
        on="parent_asin",
        how="left",
    )

def analyze_user_profile(
    user_id: str
) -> UserAnalysis:
    """
    Determine which recommendation engines can serve a user, and summarise their taste profile.

    Args:
    - user_id (str): The user identifier

    Returns:
    - UserAnalysis: Engine availability plus the profile fields used to phrase the final answer.

    Raises:
    - ValueError: If user_id is empty
    - FileNotFoundError: If the collaborative-filtering artefacts have not been built

    Notes:
    - `content_based_recommendations` is offered whenever the user has at least one positively-
      rated review. That is a necessary but not strictly sufficient condition: the engine also
      requires the rated item to be present in the TF-IDF index. Confirming that would mean
      loading the full content-based artefacts here, so the residual case is instead absorbed by
      the tool wrapper, which reports an empty result as `status="unavailable"`.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")

    user_id = user_id.strip()

    user_item_df = get_user_item_df()
    item_df = get_item_df()

    history = user_item_df[user_item_df["user_id"] == user_id]
    positive = _positive_history_with_metadata(history, item_df)

    # Taste-profile fields. These do not affect routing; they let the agent explain its answer.
    top_category: str | None = None
    prefers_free: bool = False

    if not positive.empty:
        categories = positive["main_category"].dropna()

        if not categories.empty:
            top_category = str(categories.value_counts().index[0])

        prefers_free = bool(positive["is_free"].mean() > FREE_PREFERENCE_THRESHOLD)

    # Engine availability. `popular_items` is always offered, so the list is never empty and the
    # agent always has a path to a non-empty answer.
    available_engines: list[str] = []
    unavailable: dict[str, str] = {}

    if user_id in get_known_cf_user_ids():
        available_engines.append(COLLABORATIVE_TOOL)
    else:
        unavailable[COLLABORATIVE_TOOL] = "user is not present in the trained ALS matrix"

    if len(positive) >= 1:
        available_engines.append(CONTENT_BASED_TOOL)
    else:
        unavailable[CONTENT_BASED_TOOL] = (
            f"user has no reviews rated {MIN_RATING_THRESHOLD} or above, "
            "so no taste profile can be built"
        )

    available_engines.append(POPULAR_TOOL)

    logger.info(
        "Analysed user %s: %d reviews, %d positive, engines=%s",
        user_id, len(history), len(positive), available_engines,
    )

    return UserAnalysis(
        user_id=user_id,
        num_reviews=int(len(history)),
        num_positive_reviews=int(len(positive)),
        top_category=top_category,
        prefers_free=prefers_free,
        available_engines=available_engines,
        unavailable=unavailable,
    )
