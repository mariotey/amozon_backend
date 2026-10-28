"""Split review events into shared chronological windows for both models."""
import numpy as np
import pandas as pd


def validate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Validate events, preserving every rating, repeated pair, and extra column."""
    required = {"review_id", "user_id", "parent_asin", "review_date", "review_rating"}
    if missing := required - set(reviews.columns):
        raise ValueError(f"Review is missing columns: {sorted(missing)}")
    if reviews.empty or not reviews.columns.is_unique:
        raise ValueError("Reviews must be nonempty with unique column names")
    frame = reviews.copy()
    for column in ("review_id", "user_id", "parent_asin"):
        if not frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
            and value == value.strip()
        ).all():
            raise ValueError(f"{column} must contain nonempty, trimmed string IDs")
    if not frame.review_id.is_unique:
        raise ValueError("review_id must uniquely identify each event")
    if frame.review_rating.map(
        lambda value: isinstance(value, (bool, np.bool_, complex, np.complexfloating))
    ).any():
        raise ValueError("review_rating must be a real number between 1 and 5")
    frame["review_rating"] = pd.to_numeric(frame.review_rating, errors="raise")
    if frame.review_rating.isna().any() or not (
        np.isfinite(frame.review_rating).all() and frame.review_rating.between(1, 5).all()
    ):
        raise ValueError("review_rating must be finite and between 1 and 5")
    if frame.review_date.map(lambda value: isinstance(value, (int, float, np.number))).any():
        raise ValueError("review_date must contain timestamps, not numeric values")
    frame["review_date"] = pd.to_datetime(frame.review_date, utc=True, format="mixed", errors="raise")
    if frame.review_date.isna().any():
        raise ValueError("review_date must not be missing")
    return frame.sort_values(["review_date", "review_id"], kind="stable").reset_index(drop=True)


def validate_splits(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check event identity and strict global time separation when loading files."""
    train, validation, test = (validate_reviews(frame) for frame in (train, validation, test))
    if not pd.concat([train.review_id, validation.review_id, test.review_id]).is_unique:
        raise ValueError("Review events overlap across splits")
    if not (train.review_date.max() < validation.review_date.min()
            and validation.review_date.max() < test.review_date.min()):
        raise ValueError("Splits must be strictly chronological; equal timestamps stay together")


def train_test_split(
    reviews: pd.DataFrame, *, train_fraction: float = .8, validation_fraction: float = .1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep all events in approximate 80/10/10 time windows; never divide timestamp ties.

    Repeated user/item reviews remain events. Resolve preferences only after
    selecting the history for validation or testing. No users/items are pruned.
    """
    for fraction in (train_fraction, validation_fraction):
        if isinstance(fraction, (bool, np.bool_)) or not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("Split fractions must be numbers between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must leave a nonempty test window")
    frame = validate_reviews(reviews)
    train_end = frame.review_date.iloc[int(len(frame) * train_fraction)]
    validation_end = frame.review_date.iloc[int(len(frame) * (train_fraction + validation_fraction))]
    splits = (
        frame.loc[frame.review_date < train_end].reset_index(drop=True),
        frame.loc[(frame.review_date >= train_end) & (frame.review_date < validation_end)].reset_index(drop=True),
        frame.loc[frame.review_date >= validation_end].reset_index(drop=True),
    )
    if any(part.empty for part in splits):
        raise ValueError("Timestamp ties create an empty split; choose different fractions or use more dated data")
    validate_splits(*splits)
    return splits
