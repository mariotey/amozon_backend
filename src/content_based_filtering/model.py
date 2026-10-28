"""Build reproducible TF-IDF artifacts from raw or prepared product text."""
from typing import Any
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .tool_config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_SUBLINEAR_TF,
)

logger = logging.getLogger(__name__)

_TEXT_FIELDS = ("item_title", "description", "features")
_REQUIRED_COLUMNS = {"parent_asin", "main_category", "is_free", *_TEXT_FIELDS}


def _text_parts(value: Any) -> list[str]:
    """Flatten supported text containers without introducing missing-value tokens."""
    if isinstance(value, str):
        normalized = " ".join(value.split())
        return [normalized] if normalized else []
    if isinstance(value, np.ndarray):
        # Iterating .flat also handles scalar and multidimensional arrays.
        return [part for entry in value.flat for part in _text_parts(entry)]
    if isinstance(value, (list, tuple)):
        return [part for entry in value for part in _text_parts(entry)]
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return []
    if pd.api.types.is_scalar(value):
        return [str(value)]
    raise ValueError("Product text must be a scalar, list, tuple, or NumPy array")


def build_item_text(row: Any) -> str:
    """Join title, description, and features, including prepared plain strings.

    Raw list/array fields and nested collections are flattened in their original
    order. Missing scalars are omitted rather than becoming 'nan' or '<NA>'.
    """
    return " ".join(part for field in _TEXT_FIELDS for part in _text_parts(row[field]))


def prepare_training_data(item_df: pd.DataFrame) -> pd.DataFrame:
    """Validate product identity/flags and return text in stable product-ID order.

    Sorting makes matrix rows, saved metadata, and learned vocabulary independent
    of the incoming DataFrame row order. Input frames are never modified.
    """
    if not isinstance(item_df, pd.DataFrame):
        raise ValueError("item_df must be a pandas DataFrame")
    missing = _REQUIRED_COLUMNS - set(item_df.columns)
    if missing:
        raise ValueError(f"Missing content-training columns: {', '.join(sorted(missing))}")
    if item_df.empty:
        raise ValueError("Content training requires at least one product")

    ids = item_df["parent_asin"]
    valid_ids = ids.map(
        lambda value: isinstance(value, str) and bool(value.strip()) and value == value.strip()
    )
    if not valid_ids.all():
        raise ValueError("parent_asin must contain nonempty strings without surrounding whitespace")
    if ids.duplicated().any():
        raise ValueError("Content training requires unique parent_asin values")
    if not item_df["is_free"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("is_free must contain boolean values without missing entries")

    prepared = item_df[list(sorted(_REQUIRED_COLUMNS))].copy()
    prepared = prepared.sort_values("parent_asin", kind="stable").reset_index(drop=True)
    for field in (*_TEXT_FIELDS, "main_category"):
        prepared[field] = prepared[field].map(lambda value: " ".join(_text_parts(value)))
    prepared["is_free"] = prepared["is_free"].astype(bool)
    prepared["text"] = prepared.apply(build_item_text, axis=1)
    if not prepared["text"].str.strip().ne("").any():
        raise ValueError("Content training requires a nonempty product-text corpus")
    return prepared


def fit(items: pd.DataFrame, *, max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
        sublinear_tf: bool = TFIDF_SUBLINEAR_TF) -> dict:
    """Fit item text in memory; the vectorizer is not needed by current serving.

    New text requires rebuilding this bundle. Matrix rows and metadata stay
    aligned; float32 stores the item representations with less numeric memory.
    """
    prepared = prepare_training_data(items)
    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=ngram_range,
        sublinear_tf=sublinear_tf, dtype=np.float32,
    )
    try:
        matrix = vectorizer.fit_transform(prepared["text"])
    except ValueError as exc:
        if "empty vocabulary" in str(exc):
            raise ValueError("Product text contains no usable TF-IDF terms") from exc
        raise
    metadata = prepared[["parent_asin", "item_title", "main_category", "is_free"]].copy()
    return {"item_matrix": matrix, "metadata": metadata}
