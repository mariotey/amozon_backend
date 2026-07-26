"""
Inference handlers for the collaborative filtering module.

Exposes importable functions designed to be called directly by an Azure Function
handler (or any other serverless runtime). Each handler is stateless — artifacts
are loaded once at module import time so that warm invocations skip disk I/O.

Azure Function usage example:
    from collaborative_filtering.main import get_user_recommendations

CLI usage (from repo root or src/):
    python -m collaborative_filtering.main --mode build
    python -m collaborative_filtering.main --mode user --user_id <id> --n 10
"""

import argparse
import logging
import sys
from typing import Any
import pandas as pd
from data_loader import DataLoader
from .model import build_and_save, load_artifacts
from .recommender import recommend_for_user
from .tool_config import DEFAULT_TOP_N

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Module-level artifact cache (loaded once per warm container) ──────────────

_item_df: pd.DataFrame | None = None
_artifacts: tuple | None = None
data_obj = DataLoader()

def _get_artifacts() -> tuple:
    """Return cached artifacts, loading from disk on first call.

    Returns:
        Tuple of (tfidf, item_matrix, meta_df, item_to_idx, idx_to_item).

    Raises:
        FileNotFoundError: If artifacts have not been built yet.
        OSError: If artifact files cannot be read.
    """
    global _artifacts  # pylint: disable=global-statement
    if _artifacts is None:
        logger.info("Cold start — loading artifacts from disk")
        _artifacts = load_artifacts()
    return _artifacts

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
    - ValueError: If user_id is empty or n is invalid.
    - FileNotFoundError: If required data or artifacts are missing.
    """
    # Input Validation
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    logger.info("Fetching recommendations for user: %s (n=%d)", user_id, n)

    # Core recommendation logic, returns a list of item parent_asin
    recs_ids: list[str] = recommend_for_user(
        user_id=user_id,
        n=n,
        artifacts=_get_artifacts()
    )

    # Load item metadata and filter for targeted items
    item_df = data_obj.item_df
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
    """Fit the ALS model and persist all artifacts to disk.

    Raises:
    - FileNotFoundError: If source data files are missing.
    - OSError: If artifacts cannot be written to disk.
    """
    logger.info("Starting model build")

    build_and_save(
        data_obj.item_df,
        data_obj.user_item_df
    )

    logger.info("Model build complete")

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Args:
    - argv (list[str] | None): Argument list (defaults to sys.argv when None).

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

def run_cli(argv: list[str] | None = None) -> None:
    """Entry point for CLI execution.

    Parse CLI arguments and dispatch to the appropriate handler.

    Args:
    - argv (list[str] | None): Argument list (defaults to sys.argv when None).

    Raises:
    - SystemExit: On argument errors or unrecoverable runtime errors.
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
