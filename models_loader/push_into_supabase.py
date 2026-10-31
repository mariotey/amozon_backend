"""Explicit publication of a selected local model bundle; import has no effects."""

import importlib
from pathlib import Path
from typing import cast
import uuid

from config.ml_config import MODEL_ARTEFACT_DIR, MODELREGISTRY_TABLE_NAME, TOOLS
from .models_loader import load_model, validate_model


def publish_model(tool_name: str, path: str | Path | None = None) -> str:
    """
    Upload a validated bundle, then register it. Return its ID for api_config.MODEL_IDS.

    Call deliberately after selecting a model; no existing model is deleted and no publication
    happens during fitting, notebook evaluation, or app startup.

    Args:
    - tool_name(str): Configured recommendation algorithm name.
    - path(str | Path | None): Local path; None uses the configured algorithm bundle path.

    Returns:
    - str: New registered model ID to copy into api_config.MODEL_IDS.
    """
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown recommendation tool: {tool_name}")
    settings = importlib.import_module(f"src.{tool_name}.tool_config")
    path = cast(
        Path,
        (
            Path(path)
            if path is not None
            else MODEL_ARTEFACT_DIR / tool_name / settings.MODEL_FILENAME
        ),
    )
    validate_model(load_model(path), tool_name)
    from utils.supabase_utils import SUPABASE_CLIENT, ARTEFACTS_STORAGE

    model_id = str(uuid.uuid4())
    storage_path = f"{tool_name}/{model_id}"
    with path.open("rb") as stream:
        ARTEFACTS_STORAGE.upload(path=f"{storage_path}/{settings.MODEL_FILENAME}", file=stream)
    # An incomplete upload must never be advertised in the registry.
    SUPABASE_CLIENT.table(MODELREGISTRY_TABLE_NAME).insert(
        {
            "model_id": model_id,
            "tool": tool_name,
            "storage_path": storage_path,
        }
    ).execute()
    return model_id
