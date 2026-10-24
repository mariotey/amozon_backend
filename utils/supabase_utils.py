"""
Utility helper functions for interfacing with Supabase functionalities.

Import loads .env and creates the shared Supabase client/storage handle. Read
helpers query tables or download artifacts; write helpers insert, upload, or
reconcile registry/storage contents. Exceptions propagate. Reconciliation can
delete stale registry records and orphan files and is not read-only.
"""
from contextlib import contextmanager
import logging
import os
from time import perf_counter
import pandas as pd
import uuid
from threading import Event
from supabase import create_client
from dotenv import load_dotenv
from tqdm.auto import tqdm
from config import (
    DATA_INPUT_DIR,
    ITEM_TABLE_NAME, REVIEW_TABLE_NAME, USER_TABLE_NAME,
    MODELREGISTRY_TABLE_NAME,
    MODEL_ARTEFACT_BUCKET,
    MODEL_ARTEFACT_DIR
)

# Load environment variables
load_dotenv()

SUPABASE_CLIENT = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPBASE_SECRET_KEY")
)

ARTEFACTS_STORAGE = SUPABASE_CLIENT.storage.from_(MODEL_ARTEFACT_BUCKET)

def push_table_into_supabase(
    dataframe: pd.DataFrame,
    table_name: str,
    batch_size: int = 800,
    show_progress: bool = True,
) -> None:
    """
    Upload a pandas DataFrame to a Supabase table in batches.

    Args:
    - dataframe (pd.DataFrame): DataFrame containing records to upload
    - table_name (str): Name of the destination Supabase table
    - batch_size (int): Number of rows to upload per batch. Default set to 800.
    - show_progress (bool): Boolean flag to state whether to display a progress bar. Default set
                            to True.
    """
    records = dataframe.to_dict(orient="records")

    iterator = range(0, len(records), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Uploading to `{table_name}`")

    for start in iterator:
        batch = records[start:start + batch_size]
        SUPABASE_CLIENT.table(table_name).insert(batch).execute()

    print(f"Successfully uploaded {len(records):,} records to `{table_name}`.")

def extract_table_from_supabase(
    table_name: str,
    batch_size: int = 1000,
    show_progress: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract all records from a Supabase table in batches and concatenated into a pandas DataFrame.

    Args:
    - table_name (str): Name of the Supabase table to retrieve data from
    - batch_size (int): Number of records to fetch per request. Default set to 1000.
    - show_progress (bool): Boolean flag to state whether to display a progress bar during data
                            extraction. Default set to True.

    Returns:
    - pd.DataFrame: DataFrame containing all records from the specified Supabase table
    """
    offset = 0
    all_rows = []

    pbar = tqdm(
        unit="rows",
        desc=f"Fetching from `{table_name}` ",
        disable=not show_progress,
    )

    while True:
        res = (
            SUPABASE_CLIENT
            .table(table_name)
            .select(",".join(columns) if columns else "*")
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

def reconcile_registry(
    tool_name: str
) -> None:
    """Synchronize the model registry with the Supabase storage bucket.

    Removes stale registry entries that no longer have corresponding model artefacts in storage,
    and deletes orphaned storage directories that do not have corresponding registry entries.

    Args:
    - tool_name (str): Name of the recommender tool whose registry and storage should be
                       reconciled
    """
    # Retrieve all registered model IDs.
    response = (
        SUPABASE_CLIENT.table(MODELREGISTRY_TABLE_NAME)
        .select("model_id")
        .eq("tool", tool_name)
        .execute()
    )

    model_ids = {
        row["model_id"]
        for row in response.data
    }

    bucket_objects = ARTEFACTS_STORAGE.list(tool_name)

    bucket_ids = {
        obj["name"]
        for obj in bucket_objects
        if obj["metadata"] is None      # Directories only
    }

    # Registry rows without corresponding storage.
    stale_registry_ids = model_ids - bucket_ids

    for model_id in stale_registry_ids:
        (
            SUPABASE_CLIENT.table(MODELREGISTRY_TABLE_NAME)
            .delete()
            .eq("tool", tool_name)
            .eq("model_id", model_id)
            .execute()
        )

    # Storage directories without corresponding registry rows.
    orphaned_storage_ids = bucket_ids - model_ids

    for model_id in orphaned_storage_ids:
        files = ARTEFACTS_STORAGE.list(f"{tool_name}/{model_id}")

        ARTEFACTS_STORAGE.remove(
            [
                f"{tool_name}/{model_id}/{file['name']}"
                for file in files
            ]
        )

def push_artefacts_into_registry(
    tool_name: str
) -> tuple[str, str]:
    """Create a new model registry entry for a set of model artefacts.

    Reconciles the registry with storage before generating a new model ID and inserting a
    corresponding registry record into the model registry table.

    Args:
    - tool_name (str): Name of the recommender tool that owns the model artefacts

    Returns:
    - tuple:
        - str: Newly generated UUID identifying the model version
        - str: Storage directory where the model artefacts should be uploaded
    """
    reconcile_registry(tool_name)

    model_id = str(uuid.uuid4())
    storage_path = f"{tool_name}/{model_id}"

    (
        SUPABASE_CLIENT.table(MODELREGISTRY_TABLE_NAME)
        .insert(
            {
                "model_id": model_id,
                "tool": tool_name,
                "storage_path": storage_path,
            }
        )
        .execute()
    )

    return model_id, storage_path

def push_artefacts_into_supabase(
    tool_dir: str,
    model_artefacts_dict: dict[str, str],
    storage_path: str
) -> None:
    """Upload model artefacts to Supabase Storage.

    Uploads each model artefact from the local tool directory into the specified storage path in
    the Supabase storage bucket.

    Args:
    - tool_dir (str): Local directory containing the model artefact files
    - model_artefacts_dict (dict[str, str]): Mapping of artefact names to their filenames
    - storage_path (str): Destination directory in the Supabase storage bucket
    """
    for filename in model_artefacts_dict.values():
        filepath = tool_dir / filename

        with open(filepath , "rb") as f:
            ARTEFACTS_STORAGE.upload(
                path=f"{storage_path}/{filename}",
                file=f,
                file_options={"upsert": "true"},
            )

def download_artefacts_from_supabase(
    tool_name: str,
    model_id: str,
    model_artefacts_dict: dict[str, str],
    stop_event: Event | None = None,
) -> None:
    """Download model artefacts from Supabase Storage.

    Attempts to download the specified model version. If the requested model ID cannot be found,
    falls back to downloading the latest available model version for the tool.

    Args:
    - tool_name (str): Name of the recommender tool whose artefacts should be downloaded
    - model_id (str): Identifier of the model version to retrieve
    - model_artefacts_dict (dict[str, str]): Mapping of artefact names to their filenames
    """
    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Model loading stopped")
    try:
        print(f"Fetching `{model_id}` models for `{tool_name}` from supabase...\n")

        response = (
            SUPABASE_CLIENT.table(MODELREGISTRY_TABLE_NAME)
            .select("storage_path")
            .eq("tool", tool_name)
            .eq("model_id", model_id)
            .limit(1)
            .execute()
        )

    except Exception as e:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Model loading stopped") from None
        print(f"{model_id} models for `{tool_name}` cannot be loaded ({type(e).__name__})\n")
        print(f"Fetching latest models for `{tool_name}` from supabase...\n")

        response = (
            SUPABASE_CLIENT.table(MODELREGISTRY_TABLE_NAME)
            .select("storage_path")
            .eq("tool", tool_name)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

    if not response.data:
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Model loading stopped")
        response = (
            SUPABASE_CLIENT.table(MODELREGISTRY_TABLE_NAME)
            .select("storage_path")
            .eq("tool", tool_name)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    if not response.data:
        raise FileNotFoundError(f"No model artefacts registered for {tool_name}")

    storage_path = response.data[0]["storage_path"]

    for filename in model_artefacts_dict.values():
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Model loading stopped")
        remote_path = f"{storage_path}/{filename}"
        local_path = MODEL_ARTEFACT_DIR / tool_name / filename

        print(f"Downloading \"{remote_path}\"...")

        file_bytes = ARTEFACTS_STORAGE.download(remote_path)
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Model loading stopped")

        with open(local_path, "wb") as f:
            f.write(file_bytes)


# Targeted API queries; bulk extraction above remains available to offline workflows.
logger = logging.getLogger(__name__)
CARD_FIELDS = ["parent_asin", "item_title", "main_category", "price", "is_free", "avg_rating", "num_reviews"]
DETAIL_FIELDS = CARD_FIELDS + ["store", "popularity_segment", "quality_score", "description"]
HISTORY_FIELDS = ["user_id", "parent_asin", "review_rating", "recency_weight"]
PROFILE_FIELDS = (
    "user_id,num_reviews,avg_rating_given,min_rating_given,max_rating_given,"
    "first_review_date,last_review_date,days_active,user_segment,"
    "num_verified_purchases,verified_purchase_ratio,avg_price_purchased,"
    "free_app_ratio,total_helpful_votes_received"
)
REVIEW_FIELDS = (
    "review_id,user_id,parent_asin,review_title,review_text,review_rating,"
    "review_date,verified_purchase,helpful_vote"
)


class DependencyUnavailable(RuntimeError):
    """An upstream failure, distinct from a successful query returning no rows."""


@contextmanager
def timed(operation):
    started = perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.3fs", operation, perf_counter() - started)


class ServingData:
    def __init__(self, client=None):
        self.client = SUPABASE_CLIENT if client is None else client
        self.history_page_size = int(os.getenv("HISTORY_PAGE_SIZE", "1000"))
        if not 1 <= self.history_page_size <= 1000:
            raise ValueError("HISTORY_PAGE_SIZE must be between 1 and 1000")

    def _execute(self, query):
        try:
            response = query.execute()
            if not isinstance(response.data, list):
                raise ValueError("Expected a list of database rows")
            return response
        except Exception as exc:
            logger.warning("Supabase serving query failed (%s)", type(exc).__name__)
            raise DependencyUnavailable("Supabase request failed") from None

    def random_users(self, n):
        with timed("Shopper selection"):
            rows = self._execute(self.client.rpc("random_shoppers", {"sample_size": n})).data
        # SQL selects distinct IDs before random ordering/limiting, never client-side sampling.
        if len(rows) > n or any(row.get("user_id") is None for row in rows):
            raise DependencyUnavailable("Invalid shopper selection response")
        if len({row["user_id"] for row in rows}) != len(rows):
            raise DependencyUnavailable("Duplicate shopper selection response")
        return [{"user_id": str(row["user_id"])} for row in rows]

    def popular_items(self, limit):
        with timed("Popular products"):
            return self._execute(
                self.client.table(ITEM_TABLE_NAME).select(",".join(CARD_FIELDS))
                .order("num_reviews", desc=True, nullsfirst=False)
                .order("avg_rating", desc=True, nullsfirst=False)
                .order("parent_asin", nullsfirst=False).limit(limit)
            ).data

    def product(self, parent_asin, *, fields=DETAIL_FIELDS):
        rows = self._execute(
            self.client.table(ITEM_TABLE_NAME).select(",".join(fields))
            .eq("parent_asin", parent_asin).limit(1)
        ).data
        return rows[0] if rows else None

    def product_exists(self, parent_asin):
        return self.product(parent_asin, fields=["parent_asin"]) is not None

    def profile(self, user_id):
        rows = self._execute(
            self.client.table(USER_TABLE_NAME).select(PROFILE_FIELDS).eq("user_id", user_id).limit(1)
        ).data
        return rows[0] if rows else None

    def reviews(self, parent_asin, page, page_size):
        start = (page - 1) * page_size
        response = self._execute(
            self.client.table(REVIEW_TABLE_NAME).select(REVIEW_FIELDS, count="exact")
            .eq("parent_asin", parent_asin)
            .order("review_date", desc=True, nullsfirst=False).order("review_id")
            .range(start, start + page_size - 1)
        )
        total = response.count or 0
        return {"items": response.data, "page": page, "page_size": page_size,
                "total_count": total, "page_count": (total + page_size - 1) // page_size}

    def cards(self, ids):
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        if len(ids) > 100:
            raise ValueError("At most 100 recommendation card IDs are supported")
        rows = []
        # Count is restricted to the requested IDs; pagination handles a lower server row cap.
        while len(rows) < len(ids):
            response = self._execute(
                self.client.table(ITEM_TABLE_NAME).select(",".join(CARD_FIELDS), count="exact")
                .in_("parent_asin", ids).order("parent_asin")
                .range(len(rows), len(ids) - 1)
            )
            if not response.data:
                break
            rows.extend(response.data)
            if response.count is not None and len(rows) >= response.count:
                break
        return rows

    def history(self, user_id):
        """All of one shopper's inference history, including low ratings and seen products."""
        rows, cursor, pages = [], None, 0
        with timed("Shopper history retrieval"):
            while True:
                query = (self.client.table(REVIEW_TABLE_NAME)
                         .select("review_id," + ",".join(HISTORY_FIELDS))
                         .eq("user_id", user_id).order("review_id").limit(self.history_page_size))
                if cursor is not None:
                    query = query.gt("review_id", cursor)
                batch = self._execute(query).data
                pages += 1
                if not batch:
                    break
                if any(row.get("user_id") != user_id or row.get("review_id") is None for row in batch):
                    raise DependencyUnavailable("Invalid shopper history response")
                next_cursor = batch[-1]["review_id"]
                if next_cursor == cursor:
                    raise DependencyUnavailable("Shopper history pagination did not advance")
                rows.extend(batch)
                cursor = next_cursor
                # Never stop on a short page: the server may impose a lower response cap.
        logger.info("Shopper history retrieved %s rows across %s requests", len(rows), pages)
        return pd.DataFrame(rows, columns=HISTORY_FIELDS)
