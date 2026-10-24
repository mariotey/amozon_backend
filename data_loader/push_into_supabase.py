"""
Upload processed recommender system datasets into supabase.

This module loads processed review, item, and user datasets from local Parquet files, performs
data sanitization to ensure compatibility with supabase storage, and uploads the transformed
datasets into their corresponding supabase tables.

Publication step: reads prepared Parquet files, sanitizes missing/non-finite
values, and inserts batches into Review, Item, and User. Executes on import
and writes remotely. Inserts do not replace whole tables; repeated runs may
fail on duplicate keys or duplicate data depending on the database schema.
"""
import pandas as pd
import numpy as np
from utils.supabase_utils import push_table_into_supabase
from config import (
    DATA_OUTPUT_DIR,
    USER_ITEM_INTERACT_FILENAME, ITEM_FILENAME, USER_FILENAME,
    REVIEW_TABLE_NAME, ITEM_TABLE_NAME, USER_TABLE_NAME
)
#################################################################################################
# Import Data
#################################################################################################

reviews_df = pd.read_parquet(DATA_OUTPUT_DIR / USER_ITEM_INTERACT_FILENAME)
item_df = pd.read_parquet(DATA_OUTPUT_DIR / ITEM_FILENAME)
user_df = pd.read_parquet(DATA_OUTPUT_DIR / USER_FILENAME)

#################################################################################################
# Push "Review" Table
#################################################################################################

reviews_df = (
    reviews_df
    .replace([np.nan, np.inf, -np.inf], None)
    .replace({pd.NaT: None})
)

# print(reviews_df.isna().sum())

push_table_into_supabase(reviews_df, REVIEW_TABLE_NAME)

#################################################################################################
# Push "Item" Table
#################################################################################################

item_df = (
    item_df
    .replace([np.nan, np.inf, -np.inf], None)
    .replace({pd.NaT: None})
)

# print(item_df.isna().sum())

push_table_into_supabase(item_df, ITEM_TABLE_NAME)

#################################################################################################
# Push "User" Table
#################################################################################################

user_df = (
    user_df
    .replace([np.nan, np.inf, -np.inf], None)
    .replace({pd.NaT: None})
)

# print(user_df.isna().sum())

push_table_into_supabase(user_df, USER_FILENAME)

print("Successful push of Tabular Data into supabase!")
