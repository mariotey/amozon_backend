"""
Agent-facing tools.

Thin wrappers around the recommender handlers that already exist in
`src/collaborative_filtering/main.py` and `src/content_based_filtering/main.py`. No recommendation
logic lives here; the wrappers exist to do three things the handlers deliberately do not:

1. Convert failures into data. An uncaught exception inside a tool aborts the agent run, so a
   user the engine cannot serve is reported as `status="unavailable"` with a reason, leaving the
   agent free to fall back to another engine.
2. Normalise the output schema. Collaborative filtering returns no relevance score and
   content-based filtering does; both are emitted through one envelope so the agent phrases
   answers consistently regardless of which engine fired.
3. Bound `n`, so a malformed query cannot request an unbounded result set.

Each function's signature and docstring become the tool's JSON schema and description, so both are
written for the model to read.

NOTE: The docstrings below use standard Google-style indented sections rather than the repo's
      usual `- name (type): ...` bullets. Pydantic AI parses these with griffe, which requires a
      section body to be indented under its header; an unindented body yields an empty block and
      raises IndexError while the agent is being constructed.
"""
import logging
from pydantic_ai import ModelRetry
from src.collaborative_filtering.main import (
    get_user_recommendations as collaborative_handler,
)
from src.content_based_filtering.main import (
    get_user_recommendations as content_based_handler,
)
from .analysis import analyze_user_profile
from .models import RecommendationResult, UserAnalysis, to_recommended_items
from .orchestration_config import (
    COLLABORATIVE_TOOL,
    CONTENT_BASED_TOOL,
    DEFAULT_TOP_N,
    MAX_TOP_N,
    POPULAR_TOOL,
)
from .popular import get_popular_items

logger = logging.getLogger(__name__)

def _clamp(
    n: int
) -> int:
    """
    Constrain a requested result count to the supported range.

    Args:
    - n (int): Requested number of results

    Returns:
    - int: `n` clamped to [1, MAX_TOP_N].
    """
    return max(1, min(int(n), MAX_TOP_N))

# ── Tools ──────────────────────────────────────────────────────────────────────────────────────

def analyze_user(
    user_id: str
) -> UserAnalysis:
    """
    Inspect a user's review history and report which recommendation engines can serve them.

    Call this first, before any recommendation tool. The returned `available_engines` list is
    authoritative: calling an engine absent from it will not produce recommendations.

    Args:
        user_id (str): The user identifier taken from the query

    Returns:
        UserAnalysis: Engine availability, plus the user's review counts, preferred category
            and whether they favour free products.
    """
    try:
        return analyze_user_profile(user_id)

    except ValueError as exc:
        # A malformed identifier is recoverable: ask the model to supply a corrected one.
        raise ModelRetry(f"Invalid user_id: {exc}") from exc

def collaborative_recommendations(
    user_id: str,
    n: int = DEFAULT_TOP_N,
) -> RecommendationResult:
    """
    Recommend products using collaborative filtering, based on what similar users liked.

    Only usable when `analyze_user` lists "collaborative_recommendations" in `available_engines`;
    the underlying model can only score users it was trained on. Returns no relevance scores.

    Args:
        user_id (str): The user identifier
        n (int): Number of recommendations to return (default: 10, maximum: 50)

    Returns:
        RecommendationResult: Recommended products, or status "unavailable" with the reason
            this engine could not serve the user.
    """
    n = _clamp(n)

    try:
        records = collaborative_handler(user_id=user_id, n=n)

    except ValueError as exc:
        # Raised for an unknown user_id, and for an invalid user_id or n.
        logger.info("Collaborative filtering unavailable for %s: %s", user_id, exc)

        return RecommendationResult(
            status="unavailable",
            engine=COLLABORATIVE_TOOL,
            reason=f"collaborative filtering cannot serve this user ({exc})",
        )

    except (FileNotFoundError, OSError) as exc:
        logger.error("Collaborative filtering artefacts unavailable: %s", exc)

        return RecommendationResult(
            status="unavailable",
            engine=COLLABORATIVE_TOOL,
            reason="the collaborative filtering model artefacts are not available",
        )

    if not records:
        return RecommendationResult(
            status="unavailable",
            engine=COLLABORATIVE_TOOL,
            reason="collaborative filtering produced no recommendations for this user",
        )

    return RecommendationResult(
        status="ok",
        engine=COLLABORATIVE_TOOL,
        items=to_recommended_items(records),
    )

def content_based_recommendations(
    user_id: str,
    n: int = DEFAULT_TOP_N,
) -> RecommendationResult:
    """
    Recommend products using content-based filtering, based on the text of items the user rated
    highly.

    Only usable when `analyze_user` lists "content_based_recommendations" in `available_engines`;
    it needs at least one positively-rated review to build a taste profile from. Returns a
    relevance score per item.

    Args:
        user_id (str): The user identifier
        n (int): Number of recommendations to return (default: 10, maximum: 50)

    Returns:
        RecommendationResult: Recommended products, or status "unavailable" with the reason
            this engine could not serve the user.
    """
    n = _clamp(n)

    try:
        records = content_based_handler(user_id=user_id, n=n)

    except ValueError as exc:
        logger.info("Content-based filtering rejected input for %s: %s", user_id, exc)

        return RecommendationResult(
            status="unavailable",
            engine=CONTENT_BASED_TOOL,
            reason=f"content-based filtering cannot serve this user ({exc})",
        )

    except (FileNotFoundError, OSError) as exc:
        logger.error("Content-based filtering artefacts unavailable: %s", exc)

        return RecommendationResult(
            status="unavailable",
            engine=CONTENT_BASED_TOOL,
            reason="the content-based filtering model artefacts are not available",
        )

    # The handler signals "no qualifying history" by returning an empty list rather than raising.
    if not records:
        return RecommendationResult(
            status="unavailable",
            engine=CONTENT_BASED_TOOL,
            reason="the user has no positively-rated history to build a taste profile from",
        )

    return RecommendationResult(
        status="ok",
        engine=CONTENT_BASED_TOOL,
        items=to_recommended_items(records),
    )

def popular_items(
    n: int = DEFAULT_TOP_N
) -> RecommendationResult:
    """
    Recommend the highest-quality products overall, ignoring any user history.

    Always available. Use it when no personalised engine can serve the user, and say plainly in
    the answer that the results are general recommendations rather than personalised ones.

    Args:
        n (int): Number of recommendations to return (default: 10, maximum: 50)

    Returns:
        RecommendationResult: The most highly-rated products, without relevance scores.
    """
    n = _clamp(n)

    try:
        items = get_popular_items(n=n)

    except (FileNotFoundError, OSError) as exc:
        logger.error("Item metadata unavailable: %s", exc)

        return RecommendationResult(
            status="unavailable",
            engine=POPULAR_TOOL,
            reason="the item catalogue is not available",
        )

    return RecommendationResult(
        status="ok",
        engine=POPULAR_TOOL,
        items=items,
    )
