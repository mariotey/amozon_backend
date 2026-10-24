"""
FastAPI entry points for the Amozon recommendation tools.

The default user endpoint delegates deterministic CF-first selection to src.routing.
The API exposes recommendations, catalogue browsing, shoppers, reviews, and a
deployment health check. Inference is serialized to limit temporary memory usage.

Run locally:
    uv run uvicorn app:app --reload

Developer flow: lifespan loads one shared DataLoader and ModelsLoader before
requests are accepted. Both engine modules receive these same cached objects;
shutdown clears their references. The recommendation lock serializes inference
to reduce temporary memory pressure. Catalogue routes read the startup snapshot,
while user profiles and review pages query Supabase per request. HTTP handlers
validate inputs and serialize pandas values; src.routing owns engine selection.
"""
from contextlib import asynccontextmanager
import json
from threading import Lock
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from config import REVIEW_TABLE_NAME, USER_TABLE_NAME
from utils.supabase_utils import SUPABASE_CLIENT
from data_loader import DataLoader
from models_loader import ModelsLoader
from src import routing
from src.collaborative_filtering import main as collaborative
from src.content_based_filtering import main as content

_inference_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load shared inference resources before the server accepts requests.

    Both tools receive the same datasets and model loader. Resources remain fixed
    for the process lifetime; restarting reloads them using the existing cache policy.
    Startup fails if required Supabase resources or local artefacts cannot be loaded.
    """
    data = DataLoader(inference_only=True)
    models = ModelsLoader()
    for tool in (collaborative, content):
        tool._data_loader_obj = data
        tool._models_loader_obj = models
    try:
        yield
    finally:
        for tool in (collaborative, content):
            tool._data_loader_obj = None
            tool._models_loader_obj = None


app = FastAPI(title="Amozon Recommendation API", lifespan=lifespan)


def _validate_identifier(identifier: str) -> str:
    """Strip surrounding whitespace or raise HTTP 422 for a blank identifier."""
    if not identifier.strip():
        raise HTTPException(status_code=422, detail="ID must not be blank")
    return identifier.strip()


def _serialize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert pandas/NumPy scalars and missing metadata into JSON-compatible values."""
    return json.loads(pd.DataFrame(items).to_json(orient="records"))


CARD_FIELDS = [
    "parent_asin", "item_title", "main_category", "price", "is_free",
    "avg_rating", "num_reviews",
]
DETAIL_FIELDS = CARD_FIELDS + [
    "store", "popularity_segment", "quality_score", "description",
]


def _product(parent_asin: str) -> pd.Series:
    """Find cached product metadata or raise HTTP 404 for an unknown ID."""
    catalogue = collaborative.get_data_loader().item_df
    matches = catalogue[catalogue["parent_asin"] == parent_asin]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    return matches.iloc[0]


def _card_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich ranked results with card metadata while preserving engine fields."""
    catalogue = collaborative.get_data_loader().item_df[CARD_FIELDS].set_index("parent_asin")
    enriched = []
    for item in items:
        metadata = (
            catalogue.loc[item["parent_asin"]].to_dict()
            if item["parent_asin"] in catalogue.index else {}
        )
        merged = {**item, **metadata}
        # Preserve engine-specific fields such as recommendation scores.
        enriched.append({**merged, **{key: merged.get(key) for key in CARD_FIELDS}})
    return _serialize_items(enriched)


@app.get("/health")
def health() -> dict[str, str]:
    """Report process health after startup resource loading succeeds."""
    return {"status": "ok"}


@app.get("/api/v1/users/{user_id}/recommendations")
def user_recommendations(
    user_id: str,
    n: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    """
    Try collaborative then content-based recommendations; return null items if unavailable.

    Args:
    - user_id (str): User identifier, including users unknown to the trained models.
    - n (int): Maximum number of recommendations, from 1 to 100 (default 10).

    Returns:
    - dict[str, Any]: user_id, engine_used, reason, and items. Unavailable recommendations
                      have null items and engine_used, with HTTP 200. Unexpected inference errors remain
                      server errors and do not silently trigger fallback.
    """
    user_id = _validate_identifier(user_id)
    with _inference_lock:
        result = routing.get_user_recommendations(user_id, n=n)
        if result["items"] is not None:
            result["items"] = _card_items(result["items"])
        return result


@app.get("/api/v1/items/{parent_asin}")
def item_details(parent_asin: str) -> dict[str, Any]:
    """Return product card and detail-page metadata, or 404 for an unknown ID."""
    product = _product(_validate_identifier(parent_asin))
    return _serialize_items([{key: product.get(key) for key in DETAIL_FIELDS}])[0]


@app.get("/api/v1/items/{parent_asin}/reviews")
def item_reviews(
    parent_asin: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    """Fetch one review page from Supabase, newest first with stable review-ID ties."""
    parent_asin = _validate_identifier(parent_asin)
    _product(parent_asin)
    start = (page - 1) * page_size
    response = (
        SUPABASE_CLIENT.table(REVIEW_TABLE_NAME)
        .select(
            "review_id,user_id,parent_asin,review_title,review_text,review_rating,"
            "review_date,verified_purchase,helpful_vote", count="exact",
        )
        .eq("parent_asin", parent_asin)
        .order("review_date", desc=True, nullsfirst=False)
        .order("review_id")
        .range(start, start + page_size - 1)
        .execute()
    )
    total = response.count or 0
    return {
        "items": _serialize_items(response.data),
        "page": page, "page_size": page_size,
        "total_count": total, "page_count": (total + page_size - 1) // page_size,
    }


@app.get("/api/v1/users/{user_id}")
def user_details(user_id: str) -> dict[str, Any]:
    """Fetch a shopper's prepared profile from Supabase without loading all users."""
    user_id = _validate_identifier(user_id)
    response = (
        SUPABASE_CLIENT.table(USER_TABLE_NAME)
        .select(
            "user_id,num_reviews,avg_rating_given,min_rating_given,max_rating_given,"
            "first_review_date,last_review_date,days_active,user_segment,"
            "num_verified_purchases,verified_purchase_ratio,avg_price_purchased,"
            "free_app_ratio,total_helpful_votes_received"
        )
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="User not found")
    return _serialize_items(response.data)[0]

#################################################################################################
# Helper Endpoints for Frontend
#################################################################################################

@app.get("/api/v1/randomusers")
def random_users(n: int = Query(5, ge=1, le=100)) -> list[dict[str, Any]]:
    """Sample distinct shoppers with review history, up to the available count."""
    users = collaborative.get_data_loader().user_item_df["user_id"].dropna().drop_duplicates()
    return [{"user_id": str(user)} for user in users.sample(n=min(n, len(users)))]


@app.get("/api/v1/popular_items")
def popular_items(limit: int = Query(5, ge=1, le=100)) -> list[dict[str, Any]]:
    """Rank by review count, then average rating, with stable product-ID ties."""
    catalogue = collaborative.get_data_loader().item_df
    ranked = catalogue.sort_values(
        ["num_reviews", "avg_rating", "parent_asin"],
        ascending=[False, False, True], na_position="last",
    )
    return _serialize_items(ranked.head(limit)[CARD_FIELDS].to_dict(orient="records"))
