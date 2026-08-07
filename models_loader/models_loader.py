"""
Utilities for loading and managing recommender system model artefacts.

This module provides functionality for reading, saving, and loading model artefacts required by
recommender system tools. Model artefacts are loaded from local storage when available; otherwise,
they are downloaded from supabase and cached locally.

The module also provides "ModelsLoader", which dynamically discovers available recommender tools,
loads their configurations, and retrieves the corresponding model artefacts.
"""
import pandas as pd
import joblib
import json
import scipy.sparse
import importlib
from pathlib import Path
from typing import Any
from types import ModuleType
from utils.supabase_utils import download_artefacts_from_supabase
from config import (
    REPO_ROOT,
    MODEL_ARTEFACT_DIR,
    TOOLS
)

def read_local_artefacts(
    filename_dir: Path,
    model_artefacts_dict: dict[str, str],
) -> tuple[Any, ...]:
    """
    Load model artefacts from the local directory.

    Args:
    - filename_dir (Path): Directory containing the model artefacts
    - model_artefacts_dict (dict[str, str]): Mapping of artefact names to filenames

    Returns:
    - tuple: Loaded artefacts in the same order as `model_artefacts_dict.values()`
    """
    model_artefacts = []

    for filename in model_artefacts_dict.values():
        filepath = filename_dir / filename
        suffix = filepath.suffix

        if suffix == ".joblib":
            artefact = joblib.load(filepath)

        elif suffix == ".npz":
            artefact = scipy.sparse.load_npz(filepath)

        elif suffix == ".json":
            with open(filepath, encoding="utf-8") as f:
                artefact = {int(k): v for k, v in json.load(f).items()}

        elif suffix == ".parquet":
            artefact = pd.read_parquet(filepath)

        else:
            raise ValueError(
                f"Unsupported model artefact suffix '{suffix}' for '{filepath.name}'. "
            )

        model_artefacts.append(artefact)

    return tuple(model_artefacts)

def save_local_artefacts(
    tool_name: str,
    model_artefacts: tuple[Any, ...],
    model_artefacts_dict: dict[str, str],
) -> None:
    """
    Save model artefacts to the local directory.

    Args:
    - tool_name (str): Name of the tool
    - model_artefacts (tuple): Loaded artefacts in the same order as `model_artefacts_dict.values()`
    - model_artefacts_dict (dict[str, str]): Mapping of artefact names to filenames

    Return:
    - None
    """
    tool_artefact_dir = MODEL_ARTEFACT_DIR / tool_name
    tool_artefact_dir.mkdir(parents=True, exist_ok=True)

    for artefact, filename in zip(model_artefacts, model_artefacts_dict.values()):
        filepath = tool_artefact_dir / filename
        suffix = filepath.suffix

        if suffix == ".joblib":
            joblib.dump(artefact, filepath)

        elif suffix == ".npz":
             scipy.sparse.save_npz(filepath, artefact)

        elif suffix == ".json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in artefact.items()}, f)

        elif suffix == ".parquet":
            artefact.to_parquet(filepath, index=False)

        else:
            raise ValueError(
                f"Unsupported model artefact suffix '{suffix}' for '{filepath.name}'. "
            )

    print("Artefacts successfully saved in local drive.\n")

def load_artefacts(
    tool_config: ModuleType,
    force_remote: bool = False
) -> tuple[Any, ...]:
    """
    Load model artefacts for a configured recommender tool.

    Retrieves model configuration from the provided "tool_config" module, including the tool name,
    model identifier, and artefact mapping. The function first checks whether all expected
    artefacts are available in the local cache. If any artefacts are missing, they are downloaded
    from supabase and saved locally before being loaded.

    Args:
    - tool_config (module): Tool configuration module containing:
        - MODEL_NAME (str): Name of the recommender tool
        - MODEL_ID (str): Unique identifier of the model stored in Supabase
        - MODEL_ARTEFACTS (dict[str, str]): Mapping of artefact names to their corresponding
                                            filenames
    - force_remote (bool): If True, skip the local cache check and always re-download every
                           artefact from supabase, refreshing the local cache. Default False.

    Returns:
    - tuple[Any, ...]: Loaded model artefacts in the same order as defined by
                       `MODEL_ARTEFACTS.values()`
    """
    tool_name = tool_config.MODEL_NAME
    model_id = tool_config.MODEL_ID
    model_artefacts_dict = tool_config.MODEL_ARTEFACTS

    # Directory containing the tool's model artefacts.
    tool_artefact_dir = MODEL_ARTEFACT_DIR / tool_name
    tool_artefact_dir.mkdir(parents=True, exist_ok=True)

    # Check whether all expected artefacts exist locally.
    missing_files = [
        filename
        for filename in model_artefacts_dict.values()
        if not (tool_artefact_dir / filename).exists()
    ]

    if force_remote or missing_files:
        if force_remote:
            print(f"Fetching latest artefacts for `{tool_name}` from supabase...")
        else:
            print(f"Missing {len(missing_files)} artefacts for `{tool_name}`. ")

        # Download the latest artefacts and cache them locally.
        download_artefacts_from_supabase(tool_name, model_id, model_artefacts_dict)
        model_artefacts = read_local_artefacts(tool_artefact_dir, model_artefacts_dict)

    else:
        print(f"Reading models for `{tool_name}` from local drive...")

        # Load previously cached artefacts.
        model_artefacts = read_local_artefacts(tool_artefact_dir, model_artefacts_dict)

    return model_artefacts

class ModelsLoader:
    """
    Dynamically discover and load model artefacts for selected recommender tools.

    Attributes:
    - model_artefacts:
        Dictionary where keys are tool names and values are tuples containing
        the loaded model artefacts in the order defined by each tool's
        MODEL_ARTEFACTS configuration.
    """

    def __init__(
        self,
        tools: list[str] | None = None,
        force_remote: bool = False
    ) -> None:
        """
        Initialise the model loader and load artefacts for selected tools.

        Args:
        - tools (list[str] | None): List of tool names to load. If None, all configured tools with
                                    a valid tool_config.py file will be loaded.
        - force_remote (bool): If True, skip the local cache check and always re-download every
                               tool's artefacts from supabase. Default False.

        Raises:
        - ImportError: If a selected tool does not contain a valid tool_config module.
        """
        self.model_artefacts = {}

        # If no tools specified, load all available tools
        selected_tools = tools if tools is not None else TOOLS

        for tool in selected_tools:
            tool_config_path = REPO_ROOT / "src" / tool / "tool_config.py"

            if not tool_config_path.is_file():
                raise FileNotFoundError(
                    f"No tool_config.py found for tool '{tool}'"
                )

            tool_config = importlib.import_module(
                f"src.{tool}.tool_config"
            )

            self.model_artefacts[tool] = load_artefacts(
                tool_config=tool_config,
                force_remote=force_remote
            )

            print(f"\nArtefacts successfully loaded for `{tool}`!\n")
