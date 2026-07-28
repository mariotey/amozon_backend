"""
Inference handlers for the content-based filtering module.

Exposes importable functions designed to be called directly by an Azure Function handler (or any
other serverless runtime). Resource-intensive objects such as datasets and model artefacts are
loaded lazily and cached in memory so that warm invocations avoid repeated disk I/O and
deserialization overhead.

Azure Function usage example:
    from content_based_filtering.main import get_user_recommendations, get_similar_items

CLI usage (from repo root or src/):
    python -m src.content_based_filtering.main --mode build
    python -m src.content_based_filtering.main --mode user --user_id <id> --n 10
    python -m src.content_based_filtering.main --mode item --asin <asin> --n 10
"""
import argparse
import logging
import sys
from typing import Any
import pandas as pd
from data_loader import DataLoader
from models_loader import ModelsLoader
from .model import build_and_save
from .recommender import recommend_for_user, similar_items
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
        _models_loader_obj = ModelsLoader(
            tools=[MODEL_NAME]
        )

    return _models_loader_obj

# ── Public handler functions (Azure Function entry points) ────────────────────

def get_user_recommendations(
    user_id: str,
    n: int = DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """
    Return top-N content-based recommendations for a user.

    Designed to be called directly by an Azure Function HTTP trigger.

    Args:
    - user_id (str): The user identifier
    - n (int): Number of recommendations to return (default: DEFAULT_TOP_N)

    Returns:
    - list[dict[str, Any]]: List of dicts with keys ["parent_asin", "score", "item_title",
                            "main_category", "is_free"]. Returns an empty list if the user has no
                            qualifying history.

    Raises:
    - ValueError: If user_id is empty or n is not a positive integer
    - FileNotFoundError: If model artifacts are not found on disk
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    logger.info("Fetching recommendations for user: %s (n=%d)", user_id, n)

    data_loader = get_data_loader()
    artefact_loader = get_models_loader()

    recs: pd.DataFrame = recommend_for_user(
        user_id=user_id,
        user_item_df=data_loader.user_item_df,
        artefacts=artefact_loader.model_artefacts[MODEL_NAME],
        n=n,
    )
    return recs.to_dict(orient="records")

def get_similar_items(
    parent_asin: str,
    n: int = DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """
    Return top-N items most similar to the given ASIN.

    Designed to be called directly by an Azure Function HTTP trigger.

    Args:
    - parent_asin (str): The ASIN of the seed item
    - n (int): Number of similar items to return (default: DEFAULT_TOP_N)

    Returns:
    - list[dict[str, Any]]: List of dicts with keys ["parent_asin", "score", "item_title",
                            "main_category"]. Returns an empty list if the ASIN is unknown.

    Raises:
    - ValueError: If parent_asin is empty or n is not a positive integer
    - FileNotFoundError: If model artifacts are not found on disk
    """
    if not parent_asin or not parent_asin.strip():
        raise ValueError("parent_asin must be a non-empty string")
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    logger.info("Fetching similar items for ASIN: %s (n=%d)", parent_asin, n)

    artefact_loader = get_models_loader()

    result: pd.DataFrame = similar_items(
        parent_asin=parent_asin,
        artefacts=artefact_loader.model_artefacts[MODEL_NAME],
        n=n,
    )
    return result.to_dict(orient="records")

def build_model() -> None:
    """
    Build and persist the TF-IDF recommendation model.

    Loads the required item metadata, trains the content-based filtering model, and saves all
    generated model artefacts to the configured storage location.

    After rebuilding, cached data and model loader instances are cleared so that subsequent
    inference requests reload the latest artefacts.

    Raises:
    - FileNotFoundError: If required input datasets are unavailable.
    - OSError: If model artefacts cannot be written to storage.
    """
    global _data_loader_obj, _models_loader_obj

    logger.info("Starting model build")

    data_loader = get_data_loader()

    build_and_save(
        data_loader.item_df
    )

    _data_loader_obj = None
    _models_loader_obj = None

    logger.info("Model build complete")

# ── CLI wrapper (thin shell around the handler functions) ─────────────────────

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
        description="Content-Based Recommender — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["build", "user", "item"],
        required=True,
        help="build: fit & save model | user: user recs | item: similar items",
    )
    parser.add_argument("--user_id", type=str, default=None, help="User ID (mode=user)")
    parser.add_argument("--asin", type=str, default=None, help="Item ASIN (mode=item)")
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
        if args.mode == "build":
            build_model()

        elif args.mode == "user":
            if not args.user_id:
                logger.error("--user_id is required for mode=user")
                sys.exit(1)
            recs = get_user_recommendations(args.user_id, n=args.n)
            print(pd.DataFrame(recs).to_string(index=False))

        elif args.mode == "item":
            if not args.asin:
                logger.error("--asin is required for mode=item")
                sys.exit(1)
            recs = get_similar_items(args.asin, n=args.n)
            print(pd.DataFrame(recs).to_string(index=False))

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
