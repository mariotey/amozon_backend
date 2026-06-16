"""
LangChain tool definitions for the recommendation agent.

Each tool wraps a pipeline's get_user_recommendations function and exposes
it to the LangGraph ReAct agent via the @tool decorator.  The docstring is
the agent's primary signal for when to call each tool — keep it accurate.
"""

import logging
from typing import Any

from langchain_core.tools import tool

from collaborative_filtering.main import get_user_recommendations as _cf_recommend
from content_based_filtering.main import get_user_recommendations as _cb_recommend

logger = logging.getLogger(__name__)


@tool
def collaborative_filtering_recommend(user_id: str, n: int = 10) -> list[dict[str, Any]]:
    """Recommend items using Collaborative Filtering (ALS).

    Best for users with an established rating or purchase history.
    Returns a list of dicts with keys: parent_asin, item_title,
    main_category, is_free.

    Args:
        user_id: Identifier of the target user.
        n: Number of recommendations to return (default: 10).

    Raises:
        ValueError: If the user is not present in the CF training set.
            The agent should fall back to content_based_recommend in this case.
        FileNotFoundError: If CF model artifacts have not been built.
    """
    logger.info("CF tool called for user: %s (n=%d)", user_id, n)
    return _cf_recommend(user_id, n)


@tool
def content_based_recommend(user_id: str, n: int = 10) -> list[dict[str, Any]]:
    """Recommend items using Content-Based Filtering (TF-IDF).

    Works for new, cold-start, or unknown users.  Returns a list of dicts
    with keys: parent_asin, item_title, main_category, is_free, score.
    Returns an empty list if the user has no qualifying interaction history
    (ratings >= 3).

    Args:
        user_id: Identifier of the target user.
        n: Number of recommendations to return (default: 10).

    Raises:
        FileNotFoundError: If CB model artifacts have not been built.
    """
    logger.info("CB tool called for user: %s (n=%d)", user_id, n)
    return _cb_recommend(user_id, n)


TOOLS = [collaborative_filtering_recommend, content_based_recommend]
