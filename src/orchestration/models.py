"""
Pydantic schemas for the orchestration layer.

These models define the JSON contract the LLM sees. Tool return types are serialised into the
model context, so field names and docstrings here are effectively prompt text: keep them
self-explanatory.
"""
from typing import Any, Literal
import pandas as pd
from pydantic import BaseModel, Field

class UserAnalysis(BaseModel):
    """
    Verdict on which recommendation engines can serve a given user.

    Returned by the `analyze_user` tool. `available_engines` is the authoritative routing signal —
    the threshold logic behind it lives in Python, so the model never has to reason about rating
    cut-offs or matrix membership.
    """
    user_id: str = Field(description="The user identifier that was analysed.")
    num_reviews: int = Field(description="Total reviews this user has written.")
    num_positive_reviews: int = Field(
        description="Reviews rated at or above the positive-rating threshold."
    )
    top_category: str | None = Field(
        default=None,
        description="Most frequent product category across the user's positively-rated history.",
    )
    prefers_free: bool = Field(
        default=False,
        description="True when the majority of the user's positively-rated items are free.",
    )
    available_engines: list[str] = Field(
        description="Names of the recommendation tools that can serve this user. Never empty.",
    )
    unavailable: dict[str, str] = Field(
        default_factory=dict,
        description="Tool name to the reason it cannot serve this user.",
    )

class RecommendedItem(BaseModel):
    """A single recommended product."""
    parent_asin: str = Field(description="Product identifier.")
    item_title: str | None = Field(default=None, description="Product title.")
    main_category: str | None = Field(default=None, description="Product category.")
    is_free: bool | None = Field(default=None, description="Whether the product is free.")
    score: float | None = Field(
        default=None,
        description=(
            "Relevance score, where the engine produces one. Collaborative filtering and the "
            "popularity fallback do not, so this is null for those engines."
        ),
    )

def to_recommended_items(
    records: list[dict[str, Any]]
) -> list[RecommendedItem]:
    """
    Convert raw handler output into the shared recommendation schema.

    The existing handlers return `DataFrame.to_dict(orient="records")`, which yields NumPy scalars
    and pandas missing values. Neither validates cleanly against the optional fields, so both are
    normalised here: missing values become None, and NumPy scalars are unboxed to their Python
    equivalents.

    Unrecognised keys are dropped, which is what lets a single schema absorb the differing column
    sets of the two engines (content-based emits "score", collaborative does not).

    Args:
    - records (list[dict[str, Any]]): Raw per-item dictionaries from a recommender handler

    Returns:
    - list[RecommendedItem]: Validated items, in the order supplied.
    """
    items: list[RecommendedItem] = []

    for record in records:
        cleaned: dict[str, Any] = {}

        for key, value in record.items():
            if key not in RecommendedItem.model_fields:
                continue

            if pd.isna(value):
                cleaned[key] = None
                continue

            # NumPy scalars (np.bool_, np.float64, ...) unbox to native Python types.
            cleaned[key] = value.item() if hasattr(value, "item") else value

        items.append(RecommendedItem.model_validate(cleaned))

    return items

class RecommendationResult(BaseModel):
    """
    Unified envelope returned by every recommendation tool.

    Recommendation tools never raise: a user the engine cannot serve comes back as
    `status="unavailable"` with a `reason`, so the agent can recover by calling another tool.
    """
    status: Literal["ok", "unavailable"] = Field(
        description="'ok' when items were produced, 'unavailable' when this engine cannot serve.",
    )
    engine: str = Field(description="Name of the tool that produced this result.")
    items: list[RecommendedItem] = Field(
        default_factory=list,
        description="Recommended products, best first. Empty when status is 'unavailable'.",
    )
    reason: str | None = Field(
        default=None,
        description="Why the engine could not serve this user. Null when status is 'ok'.",
    )

class AgentResponse(BaseModel):
    """Final structured response returned to the caller of the orchestration agent."""
    answer: str = Field(
        description="One or two sentences explaining the recommendations in plain language.",
    )
    user_id: str | None = Field(
        default=None,
        description="The user the recommendations are for, or null if none was supplied.",
    )
    engine_used: str | None = Field(
        default=None,
        description="Name of the recommendation tool whose results are reported below.",
    )
    items: list[RecommendedItem] = Field(
        default_factory=list,
        description="The recommended products, copied verbatim from the tool result.",
    )
