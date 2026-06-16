"""
Deterministic recommendation router (Approach 1).

Routing logic:
    1. Attempt Collaborative Filtering (ALS) for the given user.
    2. On ValueError (unknown / cold-start user) fall back to
       Content-Based Filtering (TF-IDF).
    3. Any other exception (FileNotFoundError, OSError) propagates to
       the caller — both pipelines must have their artifacts built
       before this module can serve requests.

CLI usage (from src/):
    python -m deterministic_router.main --user_id <id> --n 10
"""

import argparse
import logging
import sys
from typing import Any

import pandas as pd

from collaborative_filtering.main import get_user_recommendations as _cf_recommend
from content_based_filtering.main import get_user_recommendations as _cb_recommend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_TOP_N = 10


# ── Public API ────────────────────────────────────────────────────────────────


def get_recommendations(
    user_id: str,
    n: int = _DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """Return top-N recommendations for a user via deterministic routing.

    Tries Collaborative Filtering first; falls back to Content-Based
    Filtering when the user is absent from the CF training set.

    Args:
        user_id: Identifier of the target user.
        n: Number of recommendations to return (default: 10).

    Returns:
        List of recommendation dicts. Keys vary by pipeline:
            CF — parent_asin, item_title, main_category, is_free
            CB — parent_asin, item_title, main_category, is_free, score
        Returns an empty list when CB finds no qualifying history.

    Raises:
        ValueError: If user_id is empty or n is not a positive integer.
        FileNotFoundError: If required model artifacts are missing.
        OSError: If artifact files cannot be read.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    logger.info("Routing recommendations for user: %s (n=%d)", user_id, n)

    try:
        recs = _cf_recommend(user_id, n)
        logger.info("CF pipeline served %d recommendations for user: %s", len(recs), user_id)
        return recs
    except ValueError:
        logger.info(
            "User %s not in CF training set — falling back to CB pipeline",
            user_id,
        )

    recs = _cb_recommend(user_id, n)
    logger.info("CB pipeline served %d recommendations for user: %s", len(recs), user_id)
    return recs


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv when None).

    Returns:
        Parsed namespace with attributes: user_id, n.
    """
    parser = argparse.ArgumentParser(
        description="Deterministic Recommendation Router (Approach 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m deterministic_router.main --user_id AH2IFH762VY5HG373JSJ7BQVPUA3Q\n"
            "  python -m deterministic_router.main --user_id <id> --n 5\n"
        ),
    )
    parser.add_argument(
        "--user_id",
        type=str,
        required=True,
        help="User ID to generate recommendations for",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=_DEFAULT_TOP_N,
        help=f"Number of recommendations to return (default: {_DEFAULT_TOP_N})",
    )
    return parser.parse_args(argv)


def run_cli(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and print recommendations as a table.

    Args:
        argv: Argument list (defaults to sys.argv when None).

    Raises:
        SystemExit: On argument errors or unrecoverable runtime errors.
    """
    args = _parse_args(argv)

    try:
        recs = get_recommendations(user_id=args.user_id, n=args.n)

        if not recs:
            print("No recommendations found for this user.")
            return

        print(pd.DataFrame(recs).to_string(index=False))

    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("Model artifacts not found: %s", exc)
        sys.exit(1)
    except OSError as exc:
        logger.error("I/O error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    run_cli()
