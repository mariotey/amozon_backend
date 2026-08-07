"""
Popularity fallback backing the `popular_items` tool.

This is the only recommendation path that requires no user history at all, and therefore the only
one that can serve a brand-new user for whom both trained engines are unavailable.

It needs no model artefacts: `item.parquet` already carries `quality_score` (a Wilson lower bound
on the positive-review ratio) and `popularity_segment` from the feature-engineering step, so this
is a filter-sort-slice over item metadata.
"""
import logging
from .models import RecommendedItem, to_recommended_items
from .orchestration_config import (
    DEFAULT_TOP_N,
    POPULAR_EXCLUDED_SEGMENTS,
    POPULAR_RANK_COLUMNS,
)
from .resources import get_item_df

logger = logging.getLogger(__name__)

_OUTPUT_COLUMNS = ["parent_asin", "item_title", "main_category", "is_free"]

def get_popular_items(
    n: int = DEFAULT_TOP_N
) -> list[RecommendedItem]:
    """
    Return the top-N highest-quality items overall, ignoring any user history.

    Items in the excluded popularity segments are dropped first, because `quality_score` is not
    meaningful for products with too few reviews. If that filter leaves fewer than N items, the
    unfiltered ranking is used instead so the caller always receives a full result set.

    Args:
    - n (int): Number of items to return (default: DEFAULT_TOP_N)

    Returns:
    - list[RecommendedItem]: Recommended products, best first. `score` is None — this engine
                             ranks by absolute quality, not by relevance to a user.

    Raises:
    - ValueError: If n is not a positive integer
    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    item_df = get_item_df()

    ranked = item_df.sort_values(
        list(POPULAR_RANK_COLUMNS), ascending=False
    )

    eligible = ranked[~ranked["popularity_segment"].isin(POPULAR_EXCLUDED_SEGMENTS)]

    if len(eligible) < n:
        logger.warning(
            "Only %d items outside segments %s; falling back to the unfiltered ranking",
            len(eligible), POPULAR_EXCLUDED_SEGMENTS,
        )
        eligible = ranked

    recommendations = to_recommended_items(
        eligible.head(n)[_OUTPUT_COLUMNS].to_dict(orient="records")
    )

    logger.info("Returning %d popular items", len(recommendations))

    return recommendations
