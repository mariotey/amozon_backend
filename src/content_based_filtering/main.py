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

from content_based_filtering.config import DEFAULT_TOP_N
from content_based_filtering.model import build_and_save, load_artifacts
from content_based_filtering.recommender import recommend_for_user, similar_items
from data_loader.main import load_user_item

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Module-level artifact cache (loaded once per warm container) ──────────────
_artifacts: tuple | None = None


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
    """Return top-N content-based recommendations for a user.

    Designed to be called directly by an Azure Function HTTP trigger.

    Args:
        user_id: The user identifier.
        n: Number of recommendations to return (default: DEFAULT_TOP_N).

    Returns:
        List of dicts with keys: parent_asin, score, item_title,
        main_category, is_free. Returns an empty list if the user
        has no qualifying history.

    Raises:
        ValueError: If user_id is empty or n is not a positive integer.
        FileNotFoundError: If model artifacts are not found on disk.
        OSError: If artifacts cannot be read.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    _, item_matrix, meta_df, item_to_idx, idx_to_item = _get_artifacts()

    logger.info("Fetching recommendations for user: %s (n=%d)", user_id, n)
    user_item_df = load_user_item()

    recs: pd.DataFrame = recommend_for_user(
        user_id=user_id,
        user_item_df=user_item_df,
        item_matrix=item_matrix,
        meta_df=meta_df,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        n=n,
    )
    return recs.to_dict(orient="records")


def get_similar_items(
    parent_asin: str,
    n: int = DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """Return top-N items most similar to the given ASIN.

    Designed to be called directly by an Azure Function HTTP trigger.

    Args:
        parent_asin: The ASIN of the seed item.
        n: Number of similar items to return (default: DEFAULT_TOP_N).

    Returns:
        List of dicts with keys: parent_asin, score, item_title,
        main_category. Returns an empty list if the ASIN is unknown.

    Raises:
        ValueError: If parent_asin is empty or n is not a positive integer.
        FileNotFoundError: If model artifacts are not found on disk.
        OSError: If artifacts cannot be read.
    """
    if not parent_asin or not parent_asin.strip():
        raise ValueError("parent_asin must be a non-empty string")
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    _, item_matrix, meta_df, item_to_idx, idx_to_item = _get_artifacts()

    logger.info("Fetching similar items for ASIN: %s (n=%d)", parent_asin, n)
    result: pd.DataFrame = similar_items(
        parent_asin=parent_asin,
        item_matrix=item_matrix,
        meta_df=meta_df,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        n=n,
    )
    return result.to_dict(orient="records")


def build_model() -> None:
    """Fit the TF-IDF model and persist all artifacts to disk.

    Raises:
        FileNotFoundError: If source data files are missing.
        OSError: If artifacts cannot be written to disk.
    """
    logger.info("Starting model build")
    build_and_save()
    logger.info("Model build complete")


# ── CLI wrapper (thin shell around the handler functions) ─────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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


def run_cli(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the appropriate handler.

    Args:
        argv: Argument list (defaults to sys.argv when None).

    Raises:
        SystemExit: On argument errors or unrecoverable runtime errors.
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


if __name__ == "__main__":
    run_cli()
