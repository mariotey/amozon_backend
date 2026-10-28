"""Pure, shared review validation and deterministic user/product resolution."""
import numpy as np
import pandas as pd


DUPLICATE_POLICY = (
    "Latest review_date per user/product; lexical greatest review_id when present; "
    "lowest rating breaks any remaining timestamp/ID tie"
)


def latest_reviews(history: pd.DataFrame) -> pd.DataFrame:
    """Return one validated review per user/product without changing the input.

    Duplicate pairs require dates; unique histories may omit the date column for
    standalone compatibility. Provided dates must be valid and retain sub-day UTC
    precision. IDs are arbitrary strings, so review_id only breaks timestamp ties
    deterministically and is never treated as a chronological sequence.
    """
    required = {"user_id", "parent_asin", "review_rating"}
    if missing := required - set(history.columns):
        raise ValueError(f"Review history is missing required columns: {sorted(missing)}")
    result = history.copy().reset_index(drop=True)
    for name in ("user_id", "parent_asin"):
        valid = result[name].map(
            lambda value: isinstance(value, str) and bool(value.strip())
            and value == value.strip()
        )
        if not valid.all():
            raise ValueError(f"Review {name} must contain non-empty, whitespace-trimmed string IDs")
    if result["review_rating"].map(
        lambda value: isinstance(value, (bool, np.bool_, complex, np.complexfloating))
    ).any():
        raise ValueError("Review ratings must be real numbers, not booleans or complex values")
    try:
        result["review_rating"] = pd.to_numeric(result["review_rating"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Review ratings must be numeric and between 1 and 5") from exc
    ratings = result["review_rating"].to_numpy(dtype=float)
    if not np.isfinite(ratings).all() or not ((ratings >= 1) & (ratings <= 5)).all():
        raise ValueError("Review ratings must be finite and between 1 and 5")

    keys = ["user_id", "parent_asin"]
    if "review_date" not in result.columns:
        if result.duplicated(keys).any():
            raise ValueError("Duplicate user/product reviews require review_date to select the latest review")
        return result.sort_values(keys, kind="stable").reset_index(drop=True)
    if result["review_date"].map(
        lambda value: isinstance(value, (int, float, np.number)) and not pd.isna(value)
    ).any():
        raise ValueError("Review dates must be timestamps, not numbers with an unspecified unit")
    try:
        result["review_date"] = pd.to_datetime(
            result["review_date"], format="mixed", errors="raise", utc=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Review dates must be valid timestamps") from exc
    if result["review_date"].isna().any():
        raise ValueError("Review dates must not be missing")

    result["_latest_review_id_tie"] = (
        result["review_id"].fillna("").astype(str) if "review_id" in result.columns else ""
    )
    result["_latest_rating_tie"] = -result["review_rating"]
    return (
        result.sort_values(
            keys + ["review_date", "_latest_review_id_tie", "_latest_rating_tie"], kind="stable",
        )
        .drop_duplicates(keys, keep="last")
        .drop(columns=["_latest_review_id_tie", "_latest_rating_tie"])
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
