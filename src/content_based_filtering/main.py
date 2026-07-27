"""
Inference handlers for the content-based filtering module.

Exposes importable functions designed to be called directly by an Azure Function
handler (or any other serverless runtime). Each handler is stateless — artifacts
are loaded once at module import time so that warm invocations skip disk I/O.

Azure Function usage example:
    from content_based_filtering.main import get_user_recommendations, get_similar_items

CLI usage (from repo root or src/):
    python -m content_based_filtering.main --mode build
    python -m content_based_filtering.main --mode user --user_id <id> --n 10
    python -m content_based_filtering.main --mode item --asin <asin> --n 10
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
from .tool_config import DEFAULT_TOP_N, MODEL_NAME, MODEL_ARTEFACTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Module-level artifact cache (loaded once per warm container) ──────────────

data_obj = DataLoader()
artefact_obj = ModelsLoader()

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

    recs: pd.DataFrame = recommend_for_user(
        user_id=user_id,
        user_item_df=data_obj.user_item_df,
        artefacts=artefact_obj.model_artefacts[MODEL_NAME],
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

    result: pd.DataFrame = similar_items(
        parent_asin=parent_asin,
        artefacts=artefact_obj.model_artefacts[MODEL_NAME],
        n=n,
    )
    return result.to_dict(orient="records")

def build_model() -> None:
    """
    Fit the TF-IDF model and persist all artifacts to disk.

    Raises:
    - FileNotFoundError: If source data files are missing
    - OSError: If artifacts cannot be written to disk
    """
    logger.info("Starting model build")

    build_and_save(
        data_obj.item_df
    )

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
