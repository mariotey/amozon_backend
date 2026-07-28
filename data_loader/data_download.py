"""
Download, preprocess, and export the Amazon Software review datasets.

This module downloads the Amazon Software review and product metadata datasets from the Hugging
Face Hub, performs data cleaning and normalization, and exports the processed datasets as Parquet
files for downstream recommender system pipelines.
"""
import os
import pandas as pd
from datasets import load_dataset
import ast
from tqdm import tqdm

from config import (
    DATA_INPUT_DIR,
    DATASET_NAME, DATASET_CATEGORY,
    USER_REVIEW_FILENAME, ITEM_META_FILENAME,
    DATASET_SPLIT_PERCENTAGE
)

def parse_categories(
    cat_str: str
) -> list[str]:
    """
    Parse and normalize a string representation of a category list.

    Converts a stringified Python list into a sorted list of normalized category names. Category
    names are converted to lowercase, possessive suffixes ("'s") are removed, and known category
    aliases are replaced according to ``category_replace_dict``.

    If the input is missing or represents an empty list, an empty list is returned. If parsing
    fails, the original input is returned as a single-element list.

    Args:
    - cat_str (str): String representation of a list of category names

    Returns:
    - list[str]: Sorted list of normalized category names. Returns an empty list for missing or
                 empty inputs, or a single-element list containing the original input if parsing
                 fails.
    """
    if pd.isna(cat_str) or cat_str.strip() == "[]":
        return []
    try:
        parsed_list = ast.literal_eval(cat_str)

        modified_list = []

        for elem in parsed_list:
            modified_elem = (
                elem
                .lower()
                .replace("\'s", "")
            )

            if modified_elem in category_replace_dict.keys():
                modified_list.append(category_replace_dict[modified_elem])
            else:
                modified_list.append(modified_elem)

        return sorted(modified_list)

    except Exception as e:
        print(f"{e}: {cat_str}")
        return [cat_str]

def parse_videos(
    video_dict: dict[str, list[str]]
) -> dict[str, list[str]]:
    """
    Normalize empty video entries in a video metadata dictionary.

    Replaces placeholder values of ``[""]`` with empty lists so that missing video data is
    represented consistently.

    Args:
    - video_dict (dict[str, list[str]]): Dictionary mapping video metadata fields to lists of
                                         video values

    Returns:
    - dict[str, list[str]]: The same dictionary with any ``[""]`` values replaced by empty lists
    """
    for key, val in video_dict.items():
        if val == [""]:
            video_dict[key] = []

    return video_dict

#######################################################################################################
# Data Fields for User Reviews
#######################################################################################################
#
# - rating (float)             -->   Rating of the product (from 1.0 to 5.0)
# - title (str)                -->   Title of the user review
# - text (str)                 -->   Text body of the user review
# - images (list)              -->   Images that users post after they have received the product
#                                    Each image has different sizes (small, medium, large), represented
#                                    by the small_image_url, medium_image_url, and large_image_url
#                                    respectively
# - asin (str)                 -->   ID of the product
# - parent_asin (str)          -->   Parent ID of the product. Note: Products with different colors,
#                                    styles, sizes usually belong to the same parent ID. The “asin”
#                                    in previous Amazon datasets is actually parent ID. Please use
#                                    parent ID to find product meta
# - user_id (str)              -->   ID of the reviewer
# - timestamp (int)            -->   Time of the review (unix time)
# - verified_purchase (bool)   -->   User purchase verification
# - helpful_vote (int)         -->   Helpful votes of the review
#
#######################################################################################################

review_dataset = load_dataset(
    DATASET_NAME,
    f"raw_review_{DATASET_CATEGORY}",
    split=f"full[:{DATASET_SPLIT_PERCENTAGE}%]"
)

review_df = review_dataset.to_pandas()

review_df["rating"] = pd.to_numeric(review_df["rating"], errors="coerce")
review_df["title"] = review_df["title"].str.strip().str.lower()
review_df["text"] = review_df["text"].str.strip().str.lower()
review_df["images"] = review_df["images"].astype(str)
review_df["asin"] = review_df["asin"].str.lower()
review_df["parent_asin"] = review_df["parent_asin"].str.lower()
review_df["user_id"] = review_df["user_id"].str.lower()
review_df["date"] = pd.to_datetime(review_df["timestamp"], unit="ms").dt.date

# Reorder columns
review_df = review_df[[
    "asin",
    "parent_asin",
    "user_id",
    # "timestamp",
    "date",
    "title",
    "text",
    "images",
    "verified_purchase",
    "helpful_vote",
    "rating"
]]

# Rename columns
review_df = review_df.rename(columns={
    "date": "review_date",
    "title": "review_title",
    "text": "review_text",
    "images": "review_images",
    "rating": "review_rating"
})

#######################################################################################################
# Data Fields for Item Metadata
#######################################################################################################
#
# - main_category (str)	       -->   Main category (i.e., domain) of the product
# - title (str)	               -->   Name of the product
# - average_rating (float)     -->   Rating of the product shown on the product page
# - rating_number (int)        -->   Number of ratings in the product
# - features (list)            -->   Bullet-point format features of the product
# - description (list)         -->   Description of the product
# - price (float)              -->   Price in US dollars (at time of crawling)
# - images (list)              -->   Images of the product. Each image has different sizes (thumb, large,
#                                    hi_res). The “variant” field shows the position of image
# - videos (list)              -->   Videos of the product including title and url
# - store (str)	               -->   Store name of the product
# - categories (list)	       -->   Hierarchical categories of the product
# - details (dict)             -->   Product details, including materials, brand, sizes, etc
# - parent_asin (str)	       -->   Parent ID of the product
# - bought_together (list)     -->   Recommended bundles from the websites
#
#######################################################################################################

meta_dataset = load_dataset(DATASET_NAME,f"raw_meta_{DATASET_CATEGORY}", split="full")

meta_df = pd.DataFrame(meta_dataset)

category_replace_dict = {
    "accounting": "accounting & finance",
    "antivirus": "antivirus & security",
    "education": "education & reference",
    "free one-day shipping i software": "free one-day shipping for software",
    "free one-day shipping on select software with your citi card": "free one-day shipping for software",
    "medicine": "medicine & health sciences",
    "photography": "photography & graphic design",
    "programming": "programming & web development",
    "spreadsheet": "spreadsheet & database",
    "training": "training & tutorials"
}

meta_df["parent_asin"] = meta_df["parent_asin"].str.lower()
meta_df["title"] = meta_df["title"].str.strip().str.lower()
meta_df["main_category"] = meta_df["main_category"].str.strip().str.lower()
meta_df["categories"] = meta_df["categories"].astype(str).apply(parse_categories)
meta_df["videos"] = meta_df["videos"].apply(parse_videos)

# modified_meta_df["average_rating"] = pd.to_numeric(modified_meta_df["average_rating"], errors="coerce")
meta_df["rating_number"] = pd.to_numeric(meta_df["rating_number"], errors="coerce")

meta_df["store"] = meta_df["store"].astype(str).str.strip().str.lower()
meta_df["price"] = pd.to_numeric(meta_df["price"], errors="coerce")

# Reorder columns
meta_df = meta_df[[
    "parent_asin",
    "title",
    "main_category",
    "categories",
    "description",
    "features",
    "details",
    "images",
    "videos",
    # "bought_together", # All rows contain None
    # "average_rating", # Removing since more concern about rating on platform than product page
    "rating_number",
    "store",
    # "subtitle", # All rows contain None
    # "author", # All rows contain None
    "price"
]]

# Rename columns
meta_df = meta_df.rename(columns={
    "title": "item_title",
    "images": "item_images",
    "videos": "item_videos",
    "rating_number": "item_rating"
})

# Create output directory if it doesn't exist
os.makedirs(DATA_INPUT_DIR, exist_ok=True)

# Export Data
review_df.to_parquet(DATA_INPUT_DIR / USER_REVIEW_FILENAME, index=False)
meta_df.to_parquet(DATA_INPUT_DIR/ ITEM_META_FILENAME, index=False)

print("Data Loaded")