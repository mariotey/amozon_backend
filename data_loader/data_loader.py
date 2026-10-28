"""Fetch source data explicitly; load one current dataset for manual model work."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.ml_config import (
    DATA_OUTPUT_DIR, USER_TABLE_NAME, ITEM_TABLE_NAME, REVIEW_TABLE_NAME,
    TRAIN_FILENAME, VALIDATION_FILENAME, TEST_FILENAME,
)
from utils.review_history import latest_reviews
from .train_test_split import validate_reviews, validate_splits


def load_data(supabase_tablename: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Fetch a source table from Supabase without local fallback or caching."""
    from utils.supabase_utils import extract_table_from_supabase

    order_by = {
        USER_TABLE_NAME: "user_id", ITEM_TABLE_NAME: "parent_asin", REVIEW_TABLE_NAME: "review_id",
    }.get(supabase_tablename)
    return extract_table_from_supabase(supabase_tablename, columns=columns, order_by=order_by)


@dataclass
class RecommendationData:
    """Prepared inputs only; no model fitting, recommendations, or scoring."""
    items: pd.DataFrame
    history: pd.DataFrame
    targets: pd.DataFrame
    excluded_targets: int = 0
    excluded_users: int = 0


class DataLoader:
    """Load the four files saved by train_test_split_data.ipynb; never fetch live data.

    for_collaborative() and for_content_based() prepare independent scenarios.
    phase="train" supplies training inputs without targets; "validation" uses
    train history, and "test" uses train+validation history. No models are run.
    """

    def __init__(self, *, output_dir: Path = DATA_OUTPUT_DIR) -> None:
        output_dir = Path(output_dir)
        frames = []
        for filename in (TRAIN_FILENAME, VALIDATION_FILENAME, TEST_FILENAME, "items.parquet"):
            path = output_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. Run notebook/train_test_split_data.ipynb first.")
            frames.append(pd.read_parquet(path))
        self.train_df, self.validation_df, self.test_df = (validate_reviews(frame) for frame in frames[:3])
        validate_splits(self.train_df, self.validation_df, self.test_df)
        columns = ["parent_asin", "item_title", "description", "features", "main_category", "is_free"]
        if missing := set(columns) - set(frames[3].columns):
            raise ValueError(f"Items are missing columns: {sorted(missing)}")
        self.item_df = frames[3][columns].copy()
        ids = self.item_df.parent_asin
        if self.item_df.empty or not ids.is_unique or not ids.map(
            lambda value: isinstance(value, str) and bool(value.strip()) and value == value.strip()
        ).all():
            raise ValueError("Items require unique, nonempty, trimmed parent_asin values")
        if not self.item_df.is_free.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError("Item is_free values must be booleans")
        all_items = pd.concat([frame.parent_asin for frame in frames[:3]])
        if not all_items.isin(ids).all():
            raise ValueError("Review items are missing from items.parquet; rerun the preparation notebook")

    def for_collaborative(self, phase: str = "validation") -> RecommendationData:
        """Prepare ALS inputs and targets for supported users and observed items."""
        return self._prepare("collaborative", phase)

    def for_content_based(self, phase: str = "validation") -> RecommendationData:
        """Prepare content inputs using all available metadata with nonempty text.

        Snapshot metadata is assumed available at each historical prediction
        boundary. Review timestamps cannot establish metadata availability.
        """
        return self._prepare("content_based", phase)

    def _prepare(self, algorithm: str, phase: str) -> RecommendationData:
        if phase not in {"train", "validation", "test"}:
            raise ValueError("phase must be 'train', 'validation', or 'test'")
        past = (
            pd.concat([self.train_df, self.validation_df], ignore_index=True)
            if phase == "test" else self.train_df
        )
        heldout = self.test_df if phase == "test" else self.validation_df
        # Resolve preferences only within available past; discard source aggregates.
        history = latest_reviews(past[["review_id", "user_id", "parent_asin", "review_date", "review_rating"]])
        age = (heldout.review_date.min() - history.review_date).dt.total_seconds() / 86400
        history["recency_weight"] = np.exp(-np.log(2) * age / 365.25)

        if algorithm == "collaborative":
            counts = past.groupby("parent_asin").agg(
                num_reviews=("review_rating", "size"), avg_rating=("review_rating", "mean"),
            ).reset_index()
            items = self.item_df.merge(counts, on="parent_asin", how="inner", validate="one_to_one")
        else:
            # Content items need not have past interactions. No TF-IDF fitting here.
            has_text = self.item_df[["item_title", "description", "features"]].apply(
                lambda row: any(_has_text(value) for value in row), axis=1,
            )
            items = self.item_df.loc[has_text].copy().reset_index(drop=True)

        if phase == "train":
            return RecommendationData(items, history, heldout.iloc[:0].copy())

        # Score first encounters, not later rating updates of the same product.
        first = heldout.sort_values(["review_date", "review_rating", "review_id"], kind="stable").drop_duplicates(
            ["user_id", "parent_asin"], keep="first",
        )
        seen = pd.MultiIndex.from_frame(history[["user_id", "parent_asin"]])
        unseen = ~pd.MultiIndex.from_frame(first[["user_id", "parent_asin"]]).isin(seen)
        relevant = first.loc[unseen & first.review_rating.ge(4)].copy()
        eligible_users = history.loc[
            history.review_rating.ge(4) & history.parent_asin.isin(items.parent_asin), "user_id"
        ]
        targets = relevant.loc[
            relevant.user_id.isin(eligible_users) & relevant.parent_asin.isin(items.parent_asin)
        ].reset_index(drop=True)
        return RecommendationData(
            items, history, targets,
            excluded_targets=len(relevant) - len(targets),
            excluded_users=relevant.user_id.nunique() - targets.user_id.nunique(),
        )


def _has_text(value) -> bool:
    """Check presence of item text without building features or a vocabulary."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, np.ndarray)):
        return any(_has_text(entry) for entry in np.asarray(value, dtype=object).flat)
    return False if value is None or pd.isna(value) else bool(str(value).strip())
