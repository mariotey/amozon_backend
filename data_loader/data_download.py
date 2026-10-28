"""Explicitly download Amazon review and matching metadata."""
import ast
import json
from typing import Any

from datasets import load_dataset
import numpy as np
import pandas as pd

from config.ml_config import DATASET_NAME, DATASET_CATEGORY, DATASET_SPLIT_PERCENTAGE


CATEGORY_REPLACEMENTS = {
    "accounting": "accounting & finance",
    "antivirus": "antivirus & security",
    "education": "education & reference",
    "free one-day shipping i software": "free one-day shipping for software",
    "free one-day shipping on select software with your citi card": "free one-day shipping for software",
    "medicine": "medicine & health sciences",
    "photography": "photography & graphic design",
    "programming": "programming & web development",
    "spreadsheet": "spreadsheet & database",
    "training": "training & tutorials",
}


def parse_categories(value: Any) -> list[str]:
    """Normalize collection-valued and legacy string-valued categories."""
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = [value]
    if not isinstance(value, (list, tuple, np.ndarray)):
        return []
    categories = []
    for entry in value:
        if isinstance(entry, str) and entry.strip():
            normalized = entry.strip().lower().replace("'s", "")
            categories.append(CATEGORY_REPLACEMENTS.get(normalized, normalized))
    return sorted(set(categories))


def parse_videos(value: Any) -> dict:
    """Normalize empty video fields without mutating the source value."""
    if not isinstance(value, dict):
        return {}
    return {
        key: [] if isinstance(entries, (list, np.ndarray))
        and len(entries) == 1 and entries[0] == "" else entries
        for key, entries in value.items()
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_value(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(entry) for entry in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    return None if value is None or pd.isna(value) else value


def _normalize_reviews(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    for field in ("title", "text", "asin", "parent_asin", "user_id"):
        frame[field] = frame[field].astype("string").str.strip().str.lower()
    frame["images"] = frame["images"].map(lambda value: json.dumps(_json_value(value)))
    # Keep sub-day information for deterministic ordering and later time splits.
    frame["review_date"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame[[
        "asin", "parent_asin", "user_id", "review_date", "title", "text",
        "images", "verified_purchase", "helpful_vote", "rating",
    ]]
    return frame.rename(columns={
        "title": "review_title", "text": "review_text", "images": "review_images",
        "rating": "review_rating",
    })


def _normalize_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for field in ("parent_asin", "title", "main_category", "store"):
        frame[field] = frame[field].astype("string").str.strip().str.lower()
    frame["categories"] = frame["categories"].map(parse_categories)
    frame["videos"] = frame["videos"].map(parse_videos)
    frame["rating_number"] = pd.to_numeric(frame["rating_number"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame[[
        "parent_asin", "title", "main_category", "categories", "description", "features",
        "details", "images", "videos", "rating_number", "store", "price",
    ]]
    return frame.rename(columns={
        "title": "item_title", "images": "item_images", "videos": "item_videos",
        "rating_number": "item_rating",
    })


def download_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normalized review/metadata DataFrames without exporting local files.

    The datasets library may maintain its own download/Arrow cache. No project
    raw or transformed Parquet snapshots are written by this function.
    """
    if not 0 < DATASET_SPLIT_PERCENTAGE <= 100:
        raise ValueError(
            "DATASET_SPLIT_PERCENTAGE must be between 0 and 100"
        )

    review_split = f"full[:{DATASET_SPLIT_PERCENTAGE}%]"

    reviews = load_dataset(
        DATASET_NAME,
        f"raw_review_{DATASET_CATEGORY}",
        split=review_split,
    )

    metadata = load_dataset(
        DATASET_NAME,
        f"raw_meta_{DATASET_CATEGORY}",
        split="full",
    )

    return _normalize_reviews(reviews.to_pandas()), _normalize_metadata(metadata.to_pandas())
