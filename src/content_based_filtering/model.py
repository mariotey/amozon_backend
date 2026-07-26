"""TF-IDF model building, persistence, and loading for content-based filtering."""
from typing import Optional, Any
import logging
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from .tool_config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_SUBLINEAR_TF,
    TFIDF_FILENAME, ITEM_MATRIX_FILENAME, META_FILENAME,
)
from config import (
    DATA_OUTPUT_DIR, MODEL_ARTEFACT_DIR,
    ITEM_METADATA_FILENAME
)

logger = logging.getLogger(__name__)

ArtifactTuple = tuple[
    TfidfVectorizer,
    scipy.sparse.csr_matrix,
    pd.DataFrame,
    dict[str, int],
    dict[int, str],
]

def build_item_text(row: Any) -> str:
    """Concatenate item title, description, and features into a single string.

    Args:
        row: A pandas Series representing one row of the metadata DataFrame.

    Returns:
        A whitespace-joined string of all text fields.
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
) -> ArtifactTuple:
    """Fit TF-IDF on the item corpus and persist all artifacts to disk.

    Returns:
        Tuple of (tfidf, item_matrix, item_meta_df, item_to_idx, idx_to_item).

    Raises:
        FileNotFoundError: If source data files are missing.
        OSError: If artifact directory cannot be created or files cannot be written.
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

    logger.info("Item matrix shape: %s", item_matrix.shape)

    try:
        MODEL_ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(tfidf, MODEL_ARTEFACT_DIR / TFIDF_FILENAME)
        scipy.sparse.save_npz(str(MODEL_ARTEFACT_DIR / ITEM_MATRIX_FILENAME), item_matrix)
        item_meta_df.drop(columns=["text"]).to_parquet(MODEL_ARTEFACT_DIR / META_FILENAME, index=False)

    except OSError as exc:
        raise OSError(f"Failed to write model artifacts to {MODEL_ARTEFACT_DIR}") from exc

    logger.info("Artifacts saved to %s", MODEL_ARTEFACT_DIR)

    item_to_idx = {asin: i for i, asin in enumerate(item_meta_df["parent_asin"])}
    idx_to_item = {i: asin for asin, i in item_to_idx.items()}

    return tfidf, item_matrix, item_meta_df, item_to_idx, idx_to_item

def load_artifacts() -> ArtifactTuple:
    """Load persisted TF-IDF artifacts from disk.

    Returns:
        Tuple of (tfidf, item_matrix, item_meta_df, item_to_idx, idx_to_item).

    Raises:
        FileNotFoundError: If any artifact file is missing. Run build_and_save() first.
        OSError: If artifact files cannot be read.
    """
    for path in (
        MODEL_ARTEFACT_DIR / TFIDF_FILENAME,
        MODEL_ARTEFACT_DIR / ITEM_MATRIX_FILENAME,
        MODEL_ARTEFACT_DIR / META_FILENAME
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Run build_and_save() or: python -m content_based_filtering.main --mode build"
            )

    try:
        logger.info("Loading TF-IDF vectorizer from %s", TFIDF_FILENAME)
        tfidf: TfidfVectorizer = joblib.load(MODEL_ARTEFACT_DIR / TFIDF_FILENAME)

        logger.info("Loading item matrix from %s", MODEL_ARTEFACT_DIR / ITEM_MATRIX_FILENAME)
        item_matrix: scipy.sparse.csr_matrix = scipy.sparse.load_npz(
            str(MODEL_ARTEFACT_DIR / ITEM_MATRIX_FILENAME)
        )

        logger.info("Loading item metadata from %s", MODEL_ARTEFACT_DIR / META_FILENAME)
        item_meta_df: pd.DataFrame = pd.read_parquet(MODEL_ARTEFACT_DIR / META_FILENAME)

    except OSError as exc:
        raise OSError("Failed to read model artifacts from disk.") from exc

    item_to_idx: dict[str, int] = {
        asin: i for i, asin in enumerate(item_meta_df["parent_asin"])
    }
    idx_to_item: dict[int, str] = {i: asin for asin, i in item_to_idx.items()}

    logger.info("Artifacts loaded — %d items indexed", len(item_to_idx))

    return tfidf, item_matrix, item_meta_df, item_to_idx, idx_to_item
