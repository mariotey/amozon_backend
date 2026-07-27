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

    print("Successful push of Model artefacts into supabase!")
