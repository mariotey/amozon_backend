import os
import numpy as np
import pandas as pd
import ast
import json
from config import (
    DATA_INPUT_DIR, DATA_OUTPUT_DIR,
    USER_REVIEW_FILENAME, ITEM_METADATA_FILENAME,
    USER_ITEM_INTERACT_FILENAME, USER_FILENAME, ITEM_FILENAME
)

# Extract metadata features
def safe_len(x):
    """Safely get length of arrays"""
    if isinstance(x, (list, np.ndarray)):
        # Drop duplicated elements in the list
        return len(set(x))
    return 0

def safe_join(x):
    if isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return ''
        return ','.join([str(i) for i in x if i is not None])
    return '' if pd.isna(x) else str(x)

def safe_json_numpy(x):
    if isinstance(x, dict):
        return json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in x.items()})
    elif isinstance(x, np.ndarray):
        return json.dumps(x.tolist())
    elif isinstance(x, (list)):
        return json.dumps(x)
    return '' if pd.isna(x) else str(x)

#################################################################################################
# Import Data
#################################################################################################

review_df = pd.read_parquet(DATA_INPUT_DIR / USER_REVIEW_FILENAME)
meta_df = pd.read_parquet(DATA_INPUT_DIR / ITEM_METADATA_FILENAME)

#################################################################################################
# Reviews Table
#################################################################################################

# Add review_id as column (not index)
modified_review_df = review_df.reset_index(names="review_id")
modified_review_df["review_id"] = modified_review_df["review_id"].astype(str)
modified_review_df["review_date"] = pd.to_datetime(modified_review_df["review_date"])

# Extract text features
modified_review_df["review_text"] = modified_review_df["review_text"].str.replace('\x00', '', regex=False)
modified_review_df["review_text_length"] = modified_review_df["review_text"].str.len()
modified_review_df["review_title_length"] = modified_review_df["review_title"].str.len()
modified_review_df["review_word_count"] = modified_review_df["review_text"].str.split().str.len()

# Sentiment proxy (simple heuristic)
modified_review_df["is_extreme_rating"] = modified_review_df["review_rating"].isin([1.0, 5.0])
modified_review_df["is_positive"] = modified_review_df["review_rating"] >= 4.0
modified_review_df["is_negative"] = modified_review_df["review_rating"] <= 2.0

# Image counts
def count_review_images(image_list):
    """Safely count number of images"""
    if isinstance(image_list, list):
        return len(image_list)
    elif pd.isna(image_list):
        return np.nan
    else:
        # If it"s not a list (maybe a string or other type), try to convert
        try:
            if isinstance(image_list, str):
                parsed = ast.literal_eval(image_list)
                return len(parsed) if isinstance(parsed, list) else 1
            else:
                return 1
        except:
            return 0

modified_review_df["num_review_img"] = modified_review_df["review_images"].apply(count_review_images)

modified_review_df["review_images"] = modified_review_df["review_images"].apply(safe_join)

# -------------------- TEMPORAL FEATURES --------------------
# Calculate days since review (from most recent date in dataset)
max_date = modified_review_df["review_date"].max()
modified_review_df["days_since_review"] = (max_date - modified_review_df["review_date"]).dt.days
modified_review_df["review_year"] = modified_review_df["review_date"].dt.year
modified_review_df["review_month"] = modified_review_df["review_date"].dt.month
modified_review_df["review_date"] = pd.to_datetime(
    modified_review_df["review_date"]
).dt.strftime("%Y-%m-%d")

# Recency weight (exponential decay with 1-year half-life)
modified_review_df["recency_weight"] = np.exp(-modified_review_df["days_since_review"] / 365.25)

#################################################################################################
# User Table
#################################################################################################

user_df = (
    modified_review_df
    .assign(
        review_date = lambda x: pd.to_datetime(x["review_date"])
    )
    .merge(
        meta_df[["parent_asin", "price"]],
        how="left",
        on="parent_asin"
    )
    .assign(
        is_free = lambda row: row["price"] == 0.0
    )
    .groupby("user_id")
    .agg({
        # Basic stats
        "review_id": "count",
        "review_rating": ["mean", "std", "min", "max"],

        # Temporal
        "review_date": ["min", "max"],
        "recency_weight": "sum",

        # Quality signals
        "helpful_vote": ["sum", "mean"],
        "verified_purchase": "sum",

        # Text engagement
        "review_text_length": "mean",
        "review_word_count": "mean",
        "num_review_img": "sum",

        # Rating patterns
        "is_extreme_rating": "mean",
        "is_positive": "mean",
        "is_negative": "mean",

        # Price sensitivity
        "price": "mean",
        "is_free": "mean"
    })
    .reset_index()
)

# Flatten multi-level columns
user_df.columns = ["_".join(col).strip("_") for col in user_df.columns]

# Rename for clarity
user_df = user_df.rename(columns={
    "review_id_count": "num_reviews",
    "review_rating_mean": "avg_rating_given",
    "review_rating_std": "rating_std",
    "review_rating_min": "min_rating_given",
    "review_rating_max": "max_rating_given",
    "review_date_min": "first_review_date",
    "review_date_max": "last_review_date",
    "recency_weight_sum": "total_recency_weight",
    "helpful_vote_sum": "total_helpful_votes_received",
    "helpful_vote_mean": "avg_helpful_votes_per_review",
    "verified_purchase_sum": "num_verified_purchases",
    "review_text_length_mean": "avg_review_length",
    "review_word_count_mean": "avg_review_words",
    "num_review_images_sum": "total_review_images",
    "is_extreme_rating_mean": "extreme_rating_ratio",
    "is_positive_mean": "positive_rating_ratio",
    "is_negative_mean": "negative_rating_ratio",
    "price_mean": "avg_price_purchased",
    "is_free_mean": "free_app_ratio"
})

# Derive additional features
user_df["days_active"] = (
    user_df["last_review_date"] - user_df["first_review_date"]
).dt.days + 1

user_df["reviews_per_day"] = (
    user_df["num_reviews"] / user_df["days_active"]
)

user_df["verified_purchase_ratio"] = (
    user_df["num_verified_purchases"] / user_df["num_reviews"]
)

# User segmentation (from EDA)
user_df["user_segment"] = pd.cut(
    user_df["num_reviews"],
    bins=[0, 1, 10, np.inf],
    labels=["one_time", "occasional", "power_user"]
)

# Rating discriminativeness (higher std = more discriminating)
user_df["is_discriminating"] = user_df["rating_std"] > 1.0

# Handle NaN values
user_df["rating_std"] = user_df["rating_std"].fillna(0)
user_df["avg_price_purchased"] = user_df["avg_price_purchased"].fillna(0)

# print(f"Total users: {len(user_df)}")
# print(f"User segments:\n{user_df['user_segment'].value_counts()}")

user_df["first_review_date"] = pd.to_datetime(
    user_df["first_review_date"]
).dt.strftime("%Y-%m-%d")

user_df["last_review_date"] = pd.to_datetime(
    user_df["last_review_date"]
).dt.strftime("%Y-%m-%d")

#################################################################################################
# Items Table
#################################################################################################

target_item_ids = modified_review_df["parent_asin"].unique().tolist()
item_df = meta_df[meta_df["parent_asin"].isin(target_item_ids)]

item_df["num_item_img"] = item_df["item_images"].apply(
    lambda d: (
        sum(bool(x) for x in d.get("hi_res", [])) +
        sum(bool(x) for x in d.get("large", [])) +
        sum(bool(x) for x in d.get("thumb", []))
    )
)

item_df["num_item_videos"] = item_df["item_videos"].apply(
    lambda d: len(d.get("url", []))
)

# Handle price properly
item_df["price"] = item_df["price"].fillna(-1)  # -1 indicates unknown
item_df["is_free"] = item_df["price"] == 0.0
item_df["has_price_info"] = item_df["price"] >= 0
item_df["price_bucket"] = pd.cut(
    item_df["price"],
    bins=[-1, 0, 1, 10, 25, 50, 100, float("inf")],
    labels=["unknown", "0-1", "1-10", "10-25", "25-50", "50-100", "100+"]
)

item_df["num_categories"] = item_df["categories"].apply(safe_len)

item_df["categories"] = item_df["categories"].apply(safe_join)

item_df["description"] = item_df["description"].apply(safe_join)
item_df["features"] = item_df["features"].apply(safe_join)
item_df["details"] = item_df["details"].apply(safe_join)

item_df["item_images"] = item_df["item_images"].apply(safe_json_numpy)
item_df["item_videos"] = item_df["item_videos"].apply(safe_json_numpy)

item_popularity_df = (
    modified_review_df
    .groupby("parent_asin")
    .agg({
        # Basic stats
        "user_id": "count",
        "review_rating": ["mean", "std", "min", "max"],

        # Temporal
        "review_date": ["min", "max"],
        "recency_weight": "sum",

        # Quality signals
        "helpful_vote": ["sum", "mean"],
        "verified_purchase": "sum",

        # Rating patterns
        "is_positive": "mean",
        "is_negative": "mean",
    })
    .reset_index()
)

# Flatten columns
item_popularity_df.columns = ["_".join(col).strip("_") for col in item_popularity_df.columns]

# Rename
item_popularity_df = item_popularity_df.rename(columns={
    "user_id_count": "num_reviews",
    "review_rating_mean": "avg_rating",
    "review_rating_std": "rating_std",
    "review_rating_min": "min_rating",
    "review_rating_max": "max_rating",
    "review_date_min": "first_review_date",
    "review_date_max": "last_review_date",
    "recency_weight_sum": "total_recency_weight",
    "helpful_vote_sum": "total_helpful_votes",
    "helpful_vote_mean": "avg_helpful_votes",
    "helpful_vote_max": "max_helpful_votes",
    "verified_purchase_sum": "num_verified_reviews",
    "is_positive_mean": "positive_review_ratio",
    "is_negative_mean": "negative_review_ratio"
})

item_df = item_df.merge(
    item_popularity_df,
    how="left",
    on="parent_asin"
)

# Derive features
item_df["last_review_date"] = pd.to_datetime(item_df["last_review_date"])
item_df["first_review_date"] = pd.to_datetime(item_df["first_review_date"])

item_df["days_on_platform"] = (
    item_df["last_review_date"] - item_df["first_review_date"]
).dt.days + 1

item_df["reviews_per_day"] = (
    item_df["num_reviews"] / item_df["days_on_platform"]
)

item_df["verified_review_ratio"] = (
    item_df["num_verified_reviews"] / item_df["num_reviews"]
)

# Item popularity segment (from EDA)
item_df["popularity_segment"] = pd.cut(
    item_df["num_reviews"],
    bins=[0, 1, 10, 100, np.inf],
    labels=["cold_start", "low_coverage", "medium", "popular"]
)

# Quality score (Wilson lower bound for rating confidence)
def wilson_lower_bound(pos, n, confidence=0.95):
    """Wilson score interval for rating confidence"""
    if n == 0:
        return 0
    z = 1.96  # 95% confidence
    phat = pos / n
    return (phat + z*z/(2*n) - z * np.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)

item_df["quality_score"] = item_df.apply(
    lambda row: wilson_lower_bound(
        row["positive_review_ratio"] * row["num_reviews"],
        row["num_reviews"]
    ),
    axis=1
)

# print(f"Total items: {len(item_df)}")
# print(f"Popularity segments:\n{item_df['popularity_segment'].value_counts()}")

item_df["first_review_date"] = pd.to_datetime(
    item_df["first_review_date"]
).dt.strftime("%Y-%m-%d")

item_df["last_review_date"] = pd.to_datetime(
    item_df["last_review_date"]
).dt.strftime("%Y-%m-%d")

#################################################################################################
# Export Data
#################################################################################################

# Create output directory if it doesn't exist
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

modified_review_df.to_parquet(DATA_OUTPUT_DIR / USER_ITEM_INTERACT_FILENAME, index=False)
user_df.to_parquet(DATA_OUTPUT_DIR / USER_FILENAME, index=False)
item_df.to_parquet(DATA_OUTPUT_DIR / ITEM_FILENAME, index=False)
