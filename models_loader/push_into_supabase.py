"""
Upload recommender system model artefacts into supabase.

This module discovers available recommender system tools, validates the presence of their locally
generated model artefacts, registers new model versions in the model registry, and uploads the
artefacts into supabase Storage.

The upload workflow includes:
- Discovering tool configurations through each tool's "tool_config.py".
- Validating that all expected model artefacts are available locally.
- Creating a new model registry entry with a unique model identifier.
- Uploading model artefacts to the corresponding Supabase Storage path.

This module assumes that model artefacts have already been generated and saved locally before the
upload process begins.
"""
import importlib
from utils.supabase_utils import push_artefacts_into_registry, push_artefacts_into_supabase
from config import (
    REPO_ROOT, MODEL_ARTEFACT_DIR,
    MODELREGISTRY_TABLE_NAME
)

for tool_dir in MODEL_ARTEFACT_DIR.iterdir():
    tool_name = tool_dir.name

    # Skip directories that do not define a `tool_config.py`.
    try:
        config_module = importlib.import_module(
            f"src.{tool_name}.tool_config"
        )
    except ModuleNotFoundError:
        continue

    model_artefacts_dict = config_module.MODEL_ARTEFACTS

    # Check whether all expected artefacts exist locally.
    missing_files = [
        filename
        for filename in model_artefacts_dict.values()
        if not (tool_dir / filename).exists()
    ]

    if missing_files:
        raise ValueError(
            "Missing Model artefacts! Make sure that artefacts are properly generated before any push is made to supabase"
        )
    else:
        _, storage_path = push_artefacts_into_registry(tool_name)

        push_artefacts_into_supabase(tool_dir, model_artefacts_dict, storage_path)

    print("\nSuccessful push of Model artefacts into supabase!\n")
