import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from tqdm.auto import tqdm
from config import (
    DATA_INPUT_DIR,

)

# Load environment variables
load_dotenv()

SUPABASE_CLIENT = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPBASE_SECRET_KEY")
)

def upload_df_to_supabase(
    dataframe,
    table_name: str,
    batch_size: int = 800,
    show_progress: bool = True,
):
    """
    Upload a pandas DataFrame to a Supabase table in batches.

    Args:
    - df (pd.DataFrame): DataFrame containing records to upload.
    - table_name (str): Name of the destination Supabase table.
    - batch_size (int): Number of rows to upload per batch. Default set to 800.
    - show_progress (bool): Boolean flag to state whether to display a progress bar. Default set
                            to True.
    """
    records = dataframe.to_dict(orient="records")

    iterator = range(0, len(records), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Uploading to {table_name}")

    for start in iterator:
        batch = records[start:start + batch_size]
        SUPABASE_CLIENT.table(table_name).insert(batch).execute()

    print(f"Successfully uploaded {len(records):,} records to '{table_name}'.")

def extract_df_from_supabase(
    table_name,
    batch_size=1000,
    show_progress: bool = True,
):
    """
    Extract all records from a Supabase table in batches and concatenated into a pandas DataFrame.

    Args:
    - table_name (str): Name of the Supabase table to retrieve data from.
    - batch_size (int): Number of records to fetch per request. Default set to 1000.
    - show_progress (bool): Boolean flag to state whether to display a progress bar during data
                            extraction. Default set to True.

    Returns:
    - pd.DataFrame: DataFrame containing all records from the specified Supabase table.
    """
    offset = 0
    all_rows = []

    pbar = tqdm(
        unit="rows",
        desc=f"Fetching from \"{table_name}\" ",
        disable=not show_progress,
    )

    while True:
        res = (
            SUPABASE_CLIENT
            .table(table_name)
            .select("*")
            .range(offset, offset + batch_size - 1)
            .execute()
        )

        data = res.data
        if not data:
            break

        all_rows.extend(data)

        batch_len = len(data)
        offset += batch_size

        pbar.update(batch_len)

    pbar.close()

    return pd.DataFrame(all_rows)
