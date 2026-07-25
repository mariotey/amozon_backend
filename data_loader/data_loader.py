from typing import Optional, Any
import pandas as pd
from utils.supabase_utils import extract_df_from_supabase
from config import (
    DATA_OUTPUT_DIR,
    USER_FILENAME, ITEM_FILENAME, USER_ITEM_INTERACT_FILENAME, ITEM_METADATA_FILENAME,
    USER_TABLE_NAME, ITEM_TABLE_NAME, REVIEW_TABLE_NAME
)

def load_meta():
    return pd.read_parquet(DATA_OUTPUT_DIR / ITEM_METADATA_FILENAME)

def load_data(
    local_filename,
    supabase_tablename,
    local_read: bool = False
):
    if local_read:
        return pd.read_parquet(DATA_OUTPUT_DIR / local_filename)

    return extract_df_from_supabase(supabase_tablename)

def load_user_item(
    local_read: bool = False
):
    return load_data(
        USER_ITEM_INTERACT_FILENAME,
        REVIEW_TABLE_NAME,
        local_read
    )

def load_user(
    local_read: bool = False
):
    return load_data(
        USER_FILENAME,
        USER_TABLE_NAME,
        local_read
    )

def load_item(
    local_read: bool = False
):
    return load_data(
        ITEM_FILENAME,
        ITEM_TABLE_NAME,
        local_read
    )

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