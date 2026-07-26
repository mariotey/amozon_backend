import os
import pandas as pd
from dotenv import load_dotenv
from utils.supabase_utils import extract_df_from_supabase
from config import (
    DATA_OUTPUT_DIR,
    USER_FILENAME, ITEM_FILENAME, USER_ITEM_INTERACT_FILENAME, ITEM_METADATA_FILENAME,
    USER_TABLE_NAME, ITEM_TABLE_NAME, REVIEW_TABLE_NAME
)

# Load environment variables
load_dotenv()

PRODUCTION_FLAG = os.getenv("PRODUCTION_FLAG", "false").lower() == "true"

def load_data(local_filename, supabase_tablename, prod_flag):

    if prod_flag:
        print("Fetching latest data from supabase...")

        return extract_df_from_supabase(supabase_tablename)

    parquet_path = DATA_OUTPUT_DIR / local_filename

    try:
        print("Reading data from local drive...")

        return pd.read_parquet(parquet_path)

    except FileNotFoundError:
        print("\nFetching from supabase and then saving into local drive...")

        df = extract_df_from_supabase(supabase_tablename)
        df.to_parquet(parquet_path, index=False)

        return df

class DataLoader:
    def __init__(
            self,
            prod_flag=PRODUCTION_FLAG
        ):
        self.user_df = load_data(
            USER_FILENAME, USER_TABLE_NAME, prod_flag
        )

        self.item_df = load_data(
            ITEM_FILENAME, ITEM_TABLE_NAME, prod_flag
        )

        self.user_item_df = load_data(
            USER_ITEM_INTERACT_FILENAME, REVIEW_TABLE_NAME, prod_flag
        )