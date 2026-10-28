"""Pure in-memory feature preparation for the Supabase publication notebook.

Import and transformation perform no local reads/writes or remote operations.
"""
import ast
import json
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd



RECENCY_HALF_LIFE_DAYS = 365.25


def _missing(value: Any) -> bool:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return False
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def safe_len(value: Any) -> int:
    """Count distinct nonmissing entries in list-like values."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return len({safe_json_numpy(entry) for entry in value if not _missing(entry)})
    return 0


def safe_join(value: Any) -> str:
    """Flatten text collections, accepting prepared strings and NumPy values."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return ",".join(part for entry in value if (part := safe_join(entry)))
    if isinstance(value, dict):
        return safe_json_numpy(value)
    return "" if _missing(value) else str(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_value(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(entry) for entry in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    return None if _missing(value) else value


def safe_json_numpy(value: Any) -> str:
    """Serialize nested collection metadata; preserve already serialized text."""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(_json_value(value), sort_keys=True)
    return "" if _missing(value) else str(value)


def count_review_images(value: Any) -> int:
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return 0
        return len(parsed) if isinstance(parsed, (list, tuple)) else 0
    return 0


def wilson_lower_bound(pos: int | float, n: int, confidence: float = .95) -> float:
    """Conservative positive-review proportion; zero when no reviews exist."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n == 0:
        return 0.
    if not 0 <= pos <= n:
        raise ValueError("positive count must be between zero and n")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    proportion = pos / n
    return float((proportion + z*z/(2*n) - z * np.sqrt(
        (proportion * (1-proportion) + z*z/(4*n)) / n
    )) / (1+z*z/n))


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _utc_dates(values) -> pd.Series:
    dates = pd.to_datetime(values, utc=True, format="mixed", errors="coerce")
    if dates.isna().any():
        raise ValueError("review_date contains missing or invalid timestamps")
    return dates


def _reference_time(dates: pd.Series, as_of) -> pd.Timestamp:
    reference = dates.max() if as_of is None else pd.to_datetime(as_of, utc=True)
    if pd.isna(reference):
        raise ValueError("as_of must be a valid timestamp")
    if (dates > reference).any():
        raise ValueError("Reviews after as_of must be excluded before preparation")
    return reference


def _count_media(value: Any, field: str) -> int:
    if not isinstance(value, dict):
        return 0
    entries = value.get(field, [])
    return sum(bool(entry) for entry in entries) if isinstance(entries, (list, tuple, np.ndarray)) else 0


def transform_data(
    review_df: pd.DataFrame, meta_df: pd.DataFrame, *, as_of=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return review, user, item tables without I/O or input mutation.

    The default reference time is the latest supplied review, never wall-clock
    time. Callers evaluating an earlier cutoff must supply only its past history.
    Repeated user/product rows are retained for serving; model training applies
    its explicit interaction policy separately.
    """
    _require_columns(review_df, {
        "user_id", "parent_asin", "review_date", "review_rating", "review_text",
        "review_title", "review_images", "helpful_vote", "verified_purchase",
    }, "reviews")
    _require_columns(meta_df, {
        "parent_asin", "item_title", "main_category", "price", "categories",
        "description", "features", "details", "item_images", "item_videos",
    }, "metadata")
    if review_df.empty:
        raise ValueError("reviews must not be empty")
    reviews = review_df.copy(deep=True)
    metadata = meta_df.copy(deep=True)
    for frame, fields in ((reviews, ("user_id", "parent_asin")), (metadata, ("parent_asin",))):
        for field in fields:
            if not frame[field].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f"{field} must contain nonempty string IDs")
    if metadata.parent_asin.duplicated().any():
        raise ValueError("metadata parent_asin must be unique")
    ratings = pd.to_numeric(reviews.review_rating, errors="coerce")
    if not (np.isfinite(ratings).all() and ratings.between(1, 5).all()):
        raise ValueError("review_rating must be finite and between 1 and 5")
    reviews["review_rating"] = ratings
    if "review_id" not in reviews:
        reviews = reviews.reset_index(names="review_id")
    ratings = reviews["review_rating"]
    reviews["review_id"] = reviews["review_id"].astype(str)
    if reviews.review_id.duplicated().any():
        raise ValueError("review_id must be unique")
    reviews["review_date"] = _utc_dates(reviews.review_date)
    reference = _reference_time(reviews.review_date, as_of)
    for field in ("review_text", "review_title"):
        reviews[field] = reviews[field].astype("string").fillna("").str.replace("\x00", "", regex=False)
    reviews["review_text_length"] = reviews.review_text.str.len()
    reviews["review_title_length"] = reviews.review_title.str.len()
    reviews["review_word_count"] = reviews.review_text.str.split().str.len()
    reviews["is_extreme_rating"] = ratings.isin([1., 5.])
    reviews["is_positive"] = ratings >= 4.
    reviews["is_negative"] = ratings <= 2.
    reviews["num_review_img"] = reviews.review_images.map(count_review_images)
    reviews["review_images"] = reviews.review_images.map(safe_json_numpy)
    age_days = (reference - reviews.review_date).dt.total_seconds() / 86400
    reviews["days_since_review"] = np.floor(age_days).astype("int64")
    reviews["review_year"] = reviews.review_date.dt.year
    reviews["review_month"] = reviews.review_date.dt.month
    # Exactly half the weight after one year, retaining fractional-day precision.
    reviews["recency_weight"] = np.exp(-np.log(2) * age_days / RECENCY_HALF_LIFE_DAYS)

    prices = pd.to_numeric(metadata.price, errors="coerce")
    metadata["price"] = prices.where(np.isfinite(prices) & (prices >= 0))
    users = (
        reviews.merge(metadata[["parent_asin", "price"]], on="parent_asin", how="left", validate="many_to_one")
        .assign(is_free=lambda f: f.price.eq(0))
        .groupby("user_id", sort=True)
        .agg({
            "review_id": "count", "review_rating": ["mean", "std", "min", "max"],
            "review_date": ["min", "max"], "recency_weight": "sum",
            "helpful_vote": ["sum", "mean"], "verified_purchase": "sum",
            "review_text_length": "mean", "review_word_count": "mean", "num_review_img": "sum",
            "is_extreme_rating": "mean", "is_positive": "mean", "is_negative": "mean",
            "price": "mean", "is_free": "mean",
        }).reset_index()
    )
    users.columns = ["_".join(column).strip("_") for column in users.columns]
    users = users.rename(columns={
        "review_id_count": "num_reviews", "review_rating_mean": "avg_rating_given",
        "review_rating_std": "rating_std", "review_rating_min": "min_rating_given",
        "review_rating_max": "max_rating_given", "review_date_min": "first_review_date",
        "review_date_max": "last_review_date", "recency_weight_sum": "total_recency_weight",
        "helpful_vote_sum": "total_helpful_votes_received", "helpful_vote_mean": "avg_helpful_votes_per_review",
        "verified_purchase_sum": "num_verified_purchases", "review_text_length_mean": "avg_review_length",
        "review_word_count_mean": "avg_review_words", "num_review_img_sum": "total_review_images",
        "is_extreme_rating_mean": "extreme_rating_ratio", "is_positive_mean": "positive_rating_ratio",
        "is_negative_mean": "negative_rating_ratio", "price_mean": "avg_price_purchased",
        "is_free_mean": "free_app_ratio",
    })
    users["days_active"] = (users.last_review_date - users.first_review_date).dt.days + 1
    users["reviews_per_day"] = users.num_reviews / users.days_active
    users["verified_purchase_ratio"] = users.num_verified_purchases / users.num_reviews
    users["user_segment"] = pd.cut(users.num_reviews, [0, 1, 10, np.inf], labels=["one_time", "occasional", "power_user"])
    users["rating_std"] = users.rating_std.fillna(0)
    users["is_discriminating"] = users.rating_std > 1.

    items = metadata[metadata.parent_asin.isin(reviews.parent_asin)].copy()
    if items.empty:
        raise ValueError("No reviewed products have matching metadata")
    items["num_item_img"] = items.item_images.map(lambda value: sum(_count_media(value, field) for field in ("hi_res", "large", "thumb")))
    items["num_item_videos"] = items.item_videos.map(lambda value: _count_media(value, "url"))
    items["has_price_info"] = items.price.notna()
    items["is_free"] = items.price.eq(0)
    items["price_bucket"] = pd.cut(
        items.price, [0, 1, 10, 25, 50, 100, np.inf], labels=["0-1", "1-10", "10-25", "25-50", "50-100", "100+"],
    ).astype("string").fillna("unknown")
    items.loc[items.is_free, "price_bucket"] = "free"
    items["price"] = items.price.fillna(-1)
    items["num_categories"] = items.categories.map(safe_len)
    for field in ("categories", "description", "features"):
        items[field] = items[field].map(safe_join)
    for field in ("details", "item_images", "item_videos"):
        items[field] = items[field].map(safe_json_numpy)
    popularity = reviews.groupby("parent_asin", sort=True).agg({
        "user_id": "count", "review_rating": ["mean", "std", "min", "max"],
        "review_date": ["min", "max"], "recency_weight": "sum",
        "helpful_vote": ["sum", "mean"], "verified_purchase": "sum",
        "is_positive": "mean", "is_negative": "mean",
    }).reset_index()
    popularity.columns = ["_".join(column).strip("_") for column in popularity.columns]
    popularity = popularity.rename(columns={
        "user_id_count": "num_reviews", "review_rating_mean": "avg_rating", "review_rating_std": "rating_std",
        "review_rating_min": "min_rating", "review_rating_max": "max_rating", "review_date_min": "first_review_date",
        "review_date_max": "last_review_date", "recency_weight_sum": "total_recency_weight",
        "helpful_vote_sum": "total_helpful_votes", "helpful_vote_mean": "avg_helpful_votes",
        "verified_purchase_sum": "num_verified_reviews", "is_positive_mean": "positive_review_ratio",
        "is_negative_mean": "negative_review_ratio",
    })
    items = items.merge(popularity, on="parent_asin", how="left", validate="one_to_one")
    items["days_on_platform"] = (items.last_review_date - items.first_review_date).dt.days + 1
    items["reviews_per_day"] = items.num_reviews / items.days_on_platform
    items["verified_review_ratio"] = items.num_verified_reviews / items.num_reviews
    items["popularity_segment"] = pd.cut(items.num_reviews, [0, 1, 10, 100, np.inf], labels=["cold_start", "low_coverage", "medium", "popular"])
    items["quality_score"] = items.apply(lambda row: wilson_lower_bound(row.positive_review_ratio * row.num_reviews, row.num_reviews), axis=1)
    for frame in (users, items):
        for field in ("first_review_date", "last_review_date"):
            frame[field] = frame[field].dt.strftime("%Y-%m-%d")
    reviews["review_date"] = reviews.review_date.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    reviews = reviews.sort_values(["review_date", "review_id"], kind="stable").reset_index(drop=True)
    items = items.sort_values("parent_asin", kind="stable").reset_index(drop=True)
    reviews.attrs["as_of"] = reference.isoformat()
    return reviews, users, items
