"""
TF-IDF model building, persistence, and loading for content-based filtering.
"""
from typing import Any
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from models_loader import save_local_artefacts
from .tool_config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_SUBLINEAR_TF,
    MODEL_NAME,
    MODEL_ARTEFACTS
)

logger = logging.getLogger(__name__)

def build_item_text(
    row: Any
) -> str:
    """
    Concatenate item title, description, and features into a single string.

    Args:
    - row (Any): A pandas Series representing one row of the metadata DataFrame.

    Returns:
    - str: A whitespace-joined string of all text fields.
    """
    parts = [str(row["item_title"] or "")]

    desc = row["description"]
    feats = row["features"]

    if isinstance(desc, list):
        parts += [str(d) for d in desc if d]
    if isinstance(feats, list):
        parts += [str(f) for f in feats if f]

    return " ".join(parts)

def build_and_save(
    item_df: pd.DataFrame
) -> None:
    """
    Fit TF-IDF on the item corpus and persist all artifacts to disk.

    Args:
    - item_df (pd.DataFrame): A DataFrame containing the items
    """
    item_meta_df = item_df.copy()
    item_meta_df["text"] = item_meta_df.apply(build_item_text, axis=1)
    item_meta_df = item_meta_df[
        ["parent_asin", "item_title", "main_category", "text", "is_free"]
    ].reset_index(drop=True)

    logger.info("Fitting TF-IDF with max_features=%d", TFIDF_MAX_FEATURES)

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
    )

    item_matrix = tfidf.fit_transform(item_meta_df["text"])

    artefacts = (
        tfidf,
        item_matrix,
        item_meta_df.drop(columns=["text"])
    )

    save_local_artefacts(MODEL_NAME, artefacts, MODEL_ARTEFACTS)
