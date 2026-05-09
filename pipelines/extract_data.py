import os
import pandas as pd
import time
import logging
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm

ITEM_TABLE_NAME = "Item"
REVIEW_TABLE_NAME = "Reviews"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPBASE_SECRET_KEY")

# Create client once
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_table_count(table_name):
    res = (
        supabase_client
        .rpc(
            "get_table_count",
            {"table_name": table_name}
        )
        .execute()
    )

    logger.info(f"Total rows of {table_name}: {res.data}\n")

    return res.data

def extract_table(table_name, batch_size=1000, timesleep=0.75):
    all_data = []
    start = 0

    # 🔥 get total rows once
    total_rows = get_table_count(table_name)

    pbar = tqdm(
        total=total_rows,
        desc=f"Extracting {table_name}",
        unit="rows"
    )

    while True:
        res = (
            supabase_client
            .table(table_name)
            .select("*")
            .range(start, start + batch_size - 1)
            .execute()
        )

        data = res.data
        if not data:
            break

        all_data.extend(data)

        batch_len = len(data)
        start += batch_size

        pbar.update(batch_len)

        time.sleep(timesleep)

    pbar.close()
    return pd.DataFrame(all_data)

def extract_item():
    return extract_table(ITEM_TABLE_NAME, batch_size=1000)

def extract_review():
    return extract_table(REVIEW_TABLE_NAME, batch_size=600)