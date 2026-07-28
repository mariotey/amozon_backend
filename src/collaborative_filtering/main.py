"""
Inference handlers for the collaborative filtering module.

Exposes importable functions designed to be called directly by an Azure Function handler (or any
other serverless runtime). Resource-intensive objects such as datasets and model artefacts are
loaded lazily and cached in memory so that warm invocations avoid repeated disk I/O and model
deserialization overhead.

Azure Function usage example:
    from collaborative_filtering.main import get_user_recommendations

CLI usage (from repo root or src/):
    python -m src.collaborative_filtering.main --mode build
    python -m src.collaborative_filtering.main --mode user --user_id <id> --n 10
"""
import argparse
import logging
import sys
from typing import Any
import pandas as pd
from data_loader import DataLoader
from models_loader import ModelsLoader
from .model import build_and_save
from .recommender import recommend_for_user
from .tool_config import DEFAULT_TOP_N, MODEL_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_data_loader_obj: DataLoader | None = None
_models_loader_obj: ModelsLoader | None = None

def get_data_loader() -> DataLoader:
    """"
    Retrieve the cached data loader instance.

    The data loader is initialized lazily on first invocation and reused across subsequent calls.
    This avoids repeatedly loading datasets from local storage or external sources such as supabase
    during warm runtime executions.

    The cached instance can be invalidated by resetting the module-level cache variable, for
    example after rebuilding model artefacts.

    Returns:
    - DataLoader: Cached data loader instance.
    """
    global _data_loader_obj

    if _data_loader_obj is None:
        _data_loader_obj = DataLoader()

    return _data_loader_obj

def get_models_loader() -> ModelsLoader:
    """
    Retrieve the cached model artefact loader instance.

    The model loader is initialized lazily on first invocation and reused across subsequent calls
    to avoid repeated model artefact loading and deserialization.

    The cached instance can be invalidated after rebuilding model artefacts to ensure subsequent
    inference requests load the latest versions.

    Returns:
    - ModelsLoader: Cached model artefact loader instance.
    """
    global _models_loader_obj

    if _models_loader_obj is None:
        _models_loader_obj = ModelsLoader()

    return _models_loader_obj

# ── Public handler functions (Azure Function entry points) ────────────────────

def get_user_recommendations(
    user_id: str,
    n: int = DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """
    Generate top-N item recommendations for a given user.

    It delegates the recommendation logic to the recommender module and enriches results with item
    metadata.

    Args:
    - user_id (str): The id of the targeted user
    - n (int): The number of recommended products to make for the targeted user

    Returns:
    - list[dict[str, Any]]: List of recommended items with columns "parent_asin", "item_title",
                            "main_category" and "is_free"

    Raises:
    - ValueError: If user_id is empty or n is invalid
    - FileNotFoundError: If required data or artifacts are missing
    """
    # Input Validation
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    logger.info("Fetching recommendations for user: %s (n=%d)", user_id, n)

    data_loader = get_data_loader()
    artefact_loader = get_models_loader()

    # Core recommendation logic, returns a list of item parent_asin
    recs_ids: list[str] = recommend_for_user(
        user_id=user_id,
        artefacts=artefact_loader.model_artefacts[MODEL_NAME],
        n=n
    )

    # Load item metadata and filter for targeted items
    item_df = data_loader.item_df
    recs = item_df[item_df["parent_asin"].isin(recs_ids)][
        [
            "parent_asin",
            "item_title",
            "main_category",
            "is_free"
        ]
    ]

    # Convert to list of dictionaries (API-friendly format)
    return recs.to_dict(orient="records")

# ── CLI wrapper (thin shell around the handler functions) ─────────────────────

def build_model() -> None:
    """
    Train the ALS collaborative filtering model and persist model artefacts.

    Loads the required training datasets, fits the recommendation model, and
    saves the generated artefacts to the configured storage location.

    After rebuilding, cached data and model loader instances are cleared so that
    future inference requests reload the latest model artefacts.

    Raises:
    - FileNotFoundError: If required training datasets are unavailable.
    - OSError: If model artefacts cannot be written to storage.
    """
    global _data_loader_obj, _models_loader_obj

    logger.info("Starting model build")

    data_loader = get_data_loader()

    build_and_save(
        data_loader.item_df,
        data_loader.user_item_df
    )

    _data_loader_obj = None
    _models_loader_obj = None

    logger.info("Model build complete")

def _parse_args(
    argv: list[str] | None = None
) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Args:
    - argv (list[str] | None): Argument list (defaults to sys.argv when None)

    Returns:
    - argparse.Namespace: Parsed arguments object containing "mode" (build/user), "user_id"
                          (optional) and "n" (number of recommendations)
    """
    parser = argparse.ArgumentParser(
        description="Collaborative Filtering Recommender — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection: build model or get recommendations
    parser.add_argument(
        "--mode",
        choices=["build", "user"],
        required=True,
        help="build: fit & save model | user: user recs",
    )

    # Required only for user mode
    parser.add_argument("--user_id", type=str, default=None, help="User ID (mode=user)")

    # Number of recommendations to return
    parser.add_argument("--n", type=int, default=DEFAULT_TOP_N, help="Number of results")

    return parser.parse_args(argv)

def run_cli(
    argv: list[str] | None = None
) -> None:
    """
    Parse CLI arguments and dispatch to the appropriate handler.

    Args:
    - argv (list[str] | None): Argument list (defaults to sys.argv when None)

    Raises:
    - SystemExit: On argument errors or unrecoverable runtime errors
    """
    args = _parse_args(argv)

    try:
        # Build Mode
        if args.mode == "build":
            build_model()
            return

        # Recommendation Mode
        elif args.mode == "user":
            if not args.user_id:
                raise ValueError("--user_id is required for mode=user")

            recs = get_user_recommendations(
                user_id = args.user_id,
                n = args.n
            )

            # Pretty print results as a table
            print(pd.DataFrame(recs).to_string(index=False))

    # Error Handling
    except FileNotFoundError as exc:
        logger.error("Data or artifact file not found: %s", exc)
        sys.exit(1)
    except OSError as exc:
        logger.error("I/O error: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        sys.exit(1)

# Script entry point
if __name__ == "__main__":
    run_cli()
