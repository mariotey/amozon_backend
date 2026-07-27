"""
Utility helper functions for interfacing with Supabase functionalities.
"""
import os
import pandas as pd
import uuid
from supabase import create_client
from dotenv import load_dotenv
from tqdm.auto import tqdm
from config import (
    DATA_INPUT_DIR,
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
    - df (pd.DataFrame): DataFrame containing records to upload
    - table_name (str): Name of the destination Supabase table
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

def extract_table_from_supabase(
    table_name: str,
    batch_size: int = 1000,
    show_progress: bool = True,
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
    model_artefacts_dict: dict[str, str]
) -> None:
    """Download model artefacts from Supabase Storage.

    Attempts to download the specified model version. If the requested model ID cannot be found,
    falls back to downloading the latest available model version for the tool.

    Args:
    - tool_name (str): Name of the recommender tool whose artefacts should be downloaded
    - model_id (str): Identifier of the model version to retrieve
    - model_artefacts_dict (dict[str, str]): Mapping of artefact names to their filenames
    """
    try:
        print(f"Fetching {model_id} models for `{tool_name}` from supabase...\n")

        response = (
            SUPABASE_CLIENT.table(MODELREGISTRY_TABLE_NAME)
            .select("storage_path")
            .eq("tool", tool_name)
            .eq("model_id", model_id)
            .limit(1)
            .execute()
        )

    except Exception as e:
        print(f"{e}: {model_id} models for `{tool_name}` cannot be loaded.")
        print(f"Fetching latest models for `{tool_name}` from supabase...\n")

        response = (
            SUPABASE_CLIENT.table(MODELREGISTRY_TABLE_NAME)
            .select("storage_path")
            .eq("tool", tool_name)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

    storage_path = response.data[0]["storage_path"]

    for filename in model_artefacts_dict.values():
        remote_path = f"{storage_path}/{filename}"
        local_path = MODEL_ARTEFACT_DIR / tool_name / filename

        print(f"Downloading '{remote_path}'...")

        file_bytes = ARTEFACTS_STORAGE.download(remote_path)

        with open(local_path, "wb") as f:
            f.write(file_bytes)
