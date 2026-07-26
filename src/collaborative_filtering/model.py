"""ALS model building, persistence, and loading for collaborative filtering."""

import logging
import joblib
import json
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.preprocessing import MinMaxScaler
import implicit
from .tool_config import (
    ALS_FACTORS, ALS_REG, ALS_ITERA,
    ALS_MODEL_FILENAME, USER_ITEM_MATRIX_FILENAME, USERID_MAPPING_FILENAME, ITEMPASIN_MAPPING_FILENAME
)
from config import (
    MODEL_ARTEFACT_DIR
)

logger = logging.getLogger(__name__)


def build_and_save(
    item_df: pd.DataFrame,
    user_item_df: pd.DataFrame
) -> tuple:
    """
    Build the ALS collaborative filtering model from raw interaction data and persist all required
    artifacts to disk.

    Returns:
    - tuple: A tuple containing a trained ALS model, user-item interaction matrix (CSR), idx_to_userid
             mapping and idx_to_itempasin mapping
    """
    scaler = MinMaxScaler()

    # Merge user interactions with item features
    merged_df = (
        user_item_df[[
            "user_id", "parent_asin", "helpful_vote", "recency_weight", "review_word_count", "num_review_img", "review_rating"
        ]]
        .merge(
            item_df[["parent_asin", "num_item_img", "num_item_videos", "price"]],
            how="left",
            on="parent_asin"
        )
    )

    # Normalize feature columns (Min-Max scaling)
    cols_to_norm = [
        col for col in merged_df.columns
        if col not in ["user_id", "parent_asin"]
    ]

    scaled = scaler.fit_transform(merged_df[cols_to_norm])
    merged_df[cols_to_norm] = scaled

    # Invert price explicitly inside feature list
    merged_df["price"] = 1 - merged_df["price"]

    # Compute interaction score (unweighted mean, each feature contributes equally)
    merged_df["interaction"] = merged_df[cols_to_norm].mean(axis=1)

    # Aggregate to single score per (user, item)
    merged_df = (
        merged_df
        .groupby(["user_id","parent_asin"], as_index=False)["interaction"].mean()
    )

    # Create categorical codes
    user_id_cats = merged_df["user_id"].astype("category")
    item_pasin_cats = merged_df["parent_asin"].astype("category")

    merged_df["userid_idx"] = user_id_cats.cat.codes
    merged_df["itempasin_idx"] = item_pasin_cats.cat.codes

    # Mapping dictionaries
    idx_to_userid = dict(enumerate(user_id_cats.cat.categories))
    idx_to_itempasin = dict(enumerate(item_pasin_cats.cat.categories))

    # Build the sparse user-item matrix
    interaction_matrix = scipy.sparse.csr_matrix(
        (
            merged_df["interaction"],
            (merged_df["userid_idx"], merged_df["itempasin_idx"])
        ),
        shape=(
            len(user_id_cats.cat.categories),
            len(item_pasin_cats.cat.categories)
        )
    ).astype("float32")

    # Train ALS to learn latent representations for users and items
    model = implicit.als.AlternatingLeastSquares(
        factors = ALS_FACTORS,
        regularization = ALS_REG,
        iterations = ALS_ITERA
    )
    model.fit(interaction_matrix)

    # Persist artifacts to disk
    try:
        MODEL_ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)

        # Save the model typically stores only learned parameters (e.g. latent factors)
        joblib.dump(model, MODEL_ARTEFACT_DIR / ALS_MODEL_FILENAME)

        # Save the user-item matrix, defines the mapping between users and items used during training/inference
        scipy.sparse.save_npz(MODEL_ARTEFACT_DIR / USER_ITEM_MATRIX_FILENAME, interaction_matrix)

        # Save the index mapping for idx to user_id
        with open(MODEL_ARTEFACT_DIR / USERID_MAPPING_FILENAME, "w") as f:
            json.dump({str(k): v for k, v in idx_to_userid.items()}, f)

        # Save the index mapping for idx to items parent_asin
        with open(MODEL_ARTEFACT_DIR / ITEMPASIN_MAPPING_FILENAME, "w") as f:
            json.dump({str(k): v for k, v in idx_to_itempasin.items()}, f)

        print(f"Successfully written model artefacts into {MODEL_ARTEFACT_DIR / MODEL_ARTEFACT_DIR}")

    except OSError as exc:
        raise OSError(f"Failed to write model artefacts to {MODEL_ARTEFACT_DIR / MODEL_ARTEFACT_DIR}") from exc

    return model, interaction_matrix, idx_to_userid, idx_to_itempasin

def load_artifacts() -> tuple:
    """
    Load previously saved model artifacts from disk.

    Returns:
    - tuple: A tuple containing a trained ALS model, user-item interaction matrix (CSR), idx_to_userid
             mapping and idx_to_itempasin mapping

    Raises:
    - FileNotFoundError: if artifact file does not exist
    """
    # Ensure artifact exists
    for path in (
        MODEL_ARTEFACT_DIR  / ALS_MODEL_FILENAME,
        MODEL_ARTEFACT_DIR / USER_ITEM_MATRIX_FILENAME,
        MODEL_ARTEFACT_DIR / USERID_MAPPING_FILENAME,
        MODEL_ARTEFACT_DIR / ITEMPASIN_MAPPING_FILENAME
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Run build_and_save() or: python -m content_based_filtering.main --mode build"
            )

    # Load the CF model
    model = joblib.load(MODEL_ARTEFACT_DIR / ALS_MODEL_FILENAME)

    # Load the user-item matrix
    interaction_matrix = scipy.sparse.load_npz(MODEL_ARTEFACT_DIR / USER_ITEM_MATRIX_FILENAME)

    # Load user index mapping (model index → real user_id)
    with open(MODEL_ARTEFACT_DIR / USERID_MAPPING_FILENAME) as f:
        idx_to_userid = {int(k): v for k, v in json.load(f).items()}

    # Load item index mapping (model index → real item parent_asin)
    with open(MODEL_ARTEFACT_DIR / ITEMPASIN_MAPPING_FILENAME) as f:
        idx_to_itempasin = {int(k): v for k, v in json.load(f).items()}

    print("Successfully loaded model artefacts")

    return model, interaction_matrix, idx_to_userid, idx_to_itempasin