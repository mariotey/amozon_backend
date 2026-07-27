"""ALS model building, persistence, and loading for collaborative filtering."""
import pandas as pd
import scipy.sparse
from sklearn.preprocessing import MinMaxScaler
import implicit
from models_loader import save_local_artefacts
from .tool_config import (
    ALS_FACTORS, ALS_REG, ALS_ITERA,
    MODEL_NAME,
    MODEL_ARTEFACTS
)

def build_and_save(
    item_df: pd.DataFrame,
    user_item_df: pd.DataFrame
) -> None:
    """
    Build the ALS collaborative filtering model from raw interaction data and persist all required
    artifacts to disk.

    Args:
    - item_df (pd.DataFrame): A DataFrame containing the items
    - user_item_df (pd.DataFrame): DataFrame of user-item interactions
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
    artefacts = (
        model,
        interaction_matrix,
        idx_to_userid,
        idx_to_itempasin
    )

    save_local_artefacts(MODEL_NAME, artefacts, MODEL_ARTEFACTS)
