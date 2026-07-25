from utils.supabase_utils import extract_df_from_supabase
from config import (
    DATA_OUTPUT_DIR,
    USER_TABLE_NAME, REVIEW_TABLE_NAME, ITEM_TABLE_NAME,
    USER_ITEM_INTERACT_FILENAME, USER_FILENAME, ITEM_FILENAME
)

#################################################################################################
# Pull "Review" Table
#################################################################################################

review_df = extract_df_from_supabase(REVIEW_TABLE_NAME)

#################################################################################################
# Pull "User" Table
#################################################################################################

user_df = extract_df_from_supabase(USER_TABLE_NAME)

#################################################################################################
# Pull "Item" Table
#################################################################################################

item_df = extract_df_from_supabase(ITEM_TABLE_NAME)

#################################################################################################
# Export Data
#################################################################################################

review_df.to_parquet(DATA_OUTPUT_DIR / USER_ITEM_INTERACT_FILENAME, index=False)
user_df.to_parquet(DATA_OUTPUT_DIR / USER_FILENAME, index=False)
# item_df.to_parquet(DATA_OUTPUT_DIR / ITEM_FILENAME, index=False)
