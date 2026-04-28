"""TF-IDF model building, persistence, and loading for content-based filtering."""

import logging

import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from content_based_filtering.config import (
    MODELS_DIR,
    TFIDF_PATH,
    ITEM_MATRIX_PATH,
    META_PATH,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF
)
from data_loader.main import build_item_text, load_meta

logger = logging.getLogger(__name__)

ArtifactTuple = tuple[
    TfidfVectorizer,
    scipy.sparse.csr_matrix,
    pd.DataFrame,
    dict[str, int],
    dict[int, str],
]


def build_and_save() -> ArtifactTuple:
    """Fit TF-IDF on the item corpus and persist all artifacts to disk.

    Returns:
        Tuple of (tfidf, item_matrix, meta_df, item_to_idx, idx_to_item).

    Raises:
        FileNotFoundError: If source data files are missing.
        OSError: If artifact directory cannot be created or files cannot be written.
    """
    meta_df = load_meta()
    meta_df["text"] = meta_df.apply(build_item_text, axis=1)
    meta_df = meta_df[
        ["parent_asin", "item_title", "main_category", "text", "is_free"]
    ].reset_index(drop=True)

    logger.info("Fitting TF-IDF with max_features=%d", TFIDF_MAX_FEATURES)
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
    )
    item_matrix = tfidf.fit_transform(meta_df["text"])
    logger.info("Item matrix shape: %s", item_matrix.shape)

    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(tfidf, TFIDF_PATH)
        scipy.sparse.save_npz(str(ITEM_MATRIX_PATH), item_matrix)
        meta_df.drop(columns=["text"]).to_parquet(META_PATH, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write model artifacts to {MODELS_DIR}") from exc

    logger.info("Artifacts saved to %s", MODELS_DIR)

    item_to_idx = {asin: i for i, asin in enumerate(meta_df["parent_asin"])}
    idx_to_item = {i: asin for asin, i in item_to_idx.items()}
    return tfidf, item_matrix, meta_df, item_to_idx, idx_to_item


def load_artifacts() -> ArtifactTuple:
    """Load persisted TF-IDF artifacts from disk.

    Returns:
        Tuple of (tfidf, item_matrix, meta_df, item_to_idx, idx_to_item).

    Raises:
        FileNotFoundError: If any artifact file is missing. Run build_and_save() first.
        OSError: If artifact files cannot be read.
    """
    for path in (TFIDF_PATH, ITEM_MATRIX_PATH, META_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Run build_and_save() or: python -m content_based_filtering.main --mode build"
            )

    try:
        logger.info("Loading TF-IDF vectorizer from %s", TFIDF_PATH)
        tfidf: TfidfVectorizer = joblib.load(TFIDF_PATH)

        logger.info("Loading item matrix from %s", ITEM_MATRIX_PATH)
        item_matrix: scipy.sparse.csr_matrix = scipy.sparse.load_npz(
            str(ITEM_MATRIX_PATH)
        )

        logger.info("Loading item metadata from %s", META_PATH)
        meta_df: pd.DataFrame = pd.read_parquet(META_PATH)
    except OSError as exc:
        raise OSError("Failed to read model artifacts from disk.") from exc

    item_to_idx: dict[str, int] = {
        asin: i for i, asin in enumerate(meta_df["parent_asin"])
    }
    idx_to_item: dict[int, str] = {i: asin for asin, i in item_to_idx.items()}

    logger.info("Artifacts loaded — %d items indexed", len(item_to_idx))
    return tfidf, item_matrix, meta_df, item_to_idx, idx_to_item
