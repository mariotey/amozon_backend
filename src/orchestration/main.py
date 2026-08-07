"""
Entry points for the orchestration layer.

Exposes the natural-language handler for serverless runtimes, plus a CLI mirroring the style of
the recommender tools.

Serverless usage example:
    from src.orchestration.main import recommend

CLI usage (from repo root):
    python -m src.orchestration.main --mode query --query "give me 5 recommendations for user <id>"
    python -m src.orchestration.main --mode analyze --user_id <id>
    python -m src.orchestration.main --mode popular --n 10

Only `--mode query` contacts an LLM provider and therefore requires an API key. The `analyze` and
`popular` modes run entirely offline against the local parquet files and model artefacts, so the
routing substrate can be verified without one.
"""
import argparse
import logging
import sys
import pandas as pd
from .agents import recommend
from .analysis import analyze_user_profile
from .models import AgentResponse, RecommendedItem
from .orchestration_config import DEFAULT_TOP_N
from .popular import get_popular_items

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Re-exported so callers can import the handler from this module, matching the other tools.
__all__ = ["recommend", "run_cli"]

# ── CLI wrapper (thin shell around the handler functions) ─────────────────────

def _print_items(
    items: list[RecommendedItem]
) -> None:
    """
    Pretty-print recommended items as a table.

    Args:
    - items (list[RecommendedItem]): Items to display

    Returns:
    - None
    """
    if not items:
        print("(no items)")
        return

    frame = pd.DataFrame([item.model_dump() for item in items])

    # Every engine populates "score" except the ones that produce no relevance score at all;
    # drop the column entirely in that case rather than printing a column of nulls.
    if frame["score"].isna().all():
        frame = frame.drop(columns=["score"])

    print(frame.to_string(index=False))

def _print_response(
    response: AgentResponse
) -> None:
    """
    Pretty-print an agent response.

    Args:
    - response (AgentResponse): The agent's structured output

    Returns:
    - None
    """
    print(f"\n{response.answer}\n")
    print(f"user_id     : {response.user_id}")
    print(f"engine_used : {response.engine_used}\n")

    _print_items(response.items)

def _parse_args(
    argv: list[str] | None = None
) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Args:
    - argv (list[str] | None): Argument list (defaults to sys.argv when None)

    Returns:
    - argparse.Namespace: Parsed arguments object containing "mode", "query", "user_id" and "n"
    """
    parser = argparse.ArgumentParser(
        description="Recommendation Orchestration Agent — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["query", "analyze", "popular"],
        default="query",
        help=(
            "query: run the agent on a natural-language request (needs an LLM API key) | "
            "analyze: engine availability for a user (offline) | "
            "popular: popularity fallback (offline)"
        ),
    )
    parser.add_argument("--query", type=str, default=None, help="Request text (mode=query)")
    parser.add_argument("--user_id", type=str, default=None, help="User ID (mode=analyze)")
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
        if args.mode == "query":
            if not args.query:
                logger.error("--query is required for mode=query")
                sys.exit(1)

            _print_response(recommend(args.query))

        elif args.mode == "analyze":
            if not args.user_id:
                logger.error("--user_id is required for mode=analyze")
                sys.exit(1)

            analysis = analyze_user_profile(args.user_id)
            print(analysis.model_dump_json(indent=2))

        elif args.mode == "popular":
            _print_items(get_popular_items(n=args.n))

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
