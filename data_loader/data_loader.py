"""
Utilities for loading recommender system datasets.

This module provides helper functions and a data loader class for retrieving datasets required by
the recommender system. Datasets are loaded from local Parquet files when available; otherwise,
they are downloaded from Supabase, cached locally, and returned.
"""
import pandas as pd
from utils.supabase_utils import extract_table_from_supabase
from config import (
    DATA_OUTPUT_DIR,
    USER_FILENAME, ITEM_FILENAME, USER_ITEM_INTERACT_FILENAME, ITEM_METADATA_FILENAME,
    USER_TABLE_NAME, ITEM_TABLE_NAME, REVIEW_TABLE_NAME
)

def load_data(
    local_filename: str,
    supabase_tablename: str
) -> pd.DataFrame:
    """
    Load a dataset from a local cache or supabase.

    Attempts to read the dataset from a local Parquet file. If the file does not exist, the dataset
    is downloaded from the specified supabase table, saved locally as a Parquet file for future
    use, and returned.

    Args:
    - local_filename (str): Name of the local Parquet file
    - supabase_tablename (str): Name of the corresponding supabase table

    Returns:
    - pd.DataFrame: The loaded dataset
    """
    parquet_path = DATA_OUTPUT_DIR / local_filename

    try:
        print(f"Reading `{supabase_tablename}` from local drive...")

        return pd.read_parquet(parquet_path)

    except FileNotFoundError:
        print(f"`{supabase_tablename}` cannot be found in local drive.")
        print(f"Fetching `{supabase_tablename}` from supabase and then saving into local drive...")

        df = extract_table_from_supabase(supabase_tablename)
        df.to_parquet(parquet_path, index=False)

        return df

class DataLoader:
    """
    Load and expose datasets required by the recommender system.

    Upon instantiation, the loader retrieves the user, item, and user-item interaction datasets,
    loading them from local cache when available or downloading them from Supabase otherwise.

    Attributes:
    - user_df (pd.DataFrame): User dataset
    - item_df (pd.DataFrame): Item dataset
    - user_item_df (pd.DataFrame): User-item interaction dataset
    """
    def __init__(
            self
        ) -> None:
        """
        Initialize the data loader and load all required datasets.

        Loads the user, item, and user-item interaction datasets into memory. Each dataset is read
        from a local Parquet file if available; otherwise, it is downloaded from supabase and
        cached locally.
        """
        # User dataset
        self.user_df = load_data(
            USER_FILENAME, USER_TABLE_NAME
        )

        # Item dataset
        self.item_df = load_data(
            ITEM_FILENAME, ITEM_TABLE_NAME
        )

        # User-item dataset
        self.user_item_df = load_data(
            USER_ITEM_INTERACT_FILENAME, REVIEW_TABLE_NAME
        )

        print("\nTabular Data successfully loaded!\n")