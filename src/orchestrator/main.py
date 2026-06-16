"""
Context-based recommendation router (Approach 2).

Delegates routing decisions to a LangGraph ReAct agent that reasons over
the request context and selects between Collaborative Filtering (ALS) and
Content-Based Filtering (TF-IDF) tools.

Note: The LLM call adds ~500ms–2s of latency per request.  For
high-throughput or sub-100ms SLA use cases prefer deterministic_router.

CLI usage (from src/):
    python -m orchestrator.main --user_id <id> --n 10
"""

import argparse
import json
import logging
import sys
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage

from orchestrator.graph import agent
from orchestrator.nodes import format_request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_TOP_N = 10


# ── Result extraction ─────────────────────────────────────────────────────────


def _extract_recommendations(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the tool output from the agent's final state.

    Iterates the message list in reverse and returns the content of the
    last ToolMessage, which holds the raw pipeline output.

    Args:
        result: The dict returned by agent.invoke().

    Returns:
        List of recommendation dicts, or an empty list if no tool was called.

    Raises:
        ValueError: If the tool output cannot be parsed as a list.
    """
    messages = result.get("messages", [])

    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            content = message.content
            # ToolMessage content is already a Python object when tools return
            # structured data, but may be a JSON string in some LangGraph versions.
            if isinstance(content, list):
                return content
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Tool output is not valid JSON: {content!r}"
                    ) from exc

    logger.warning("No ToolMessage found in agent output — returning empty list")
    return []


# ── Public API ────────────────────────────────────────────────────────────────


def get_recommendations(
    user_id: str,
    n: int = _DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """Return top-N recommendations for a user via context-based routing.

    The LangGraph ReAct agent selects and calls the appropriate pipeline
    tool based on the request context.

    Args:
        user_id: Identifier of the target user.
        n: Number of recommendations to return (default: 10).

    Returns:
        List of recommendation dicts. Keys vary by pipeline:
            CF — parent_asin, item_title, main_category, is_free
            CB — parent_asin, item_title, main_category, is_free, score
        Returns an empty list when CB finds no qualifying history.

    Raises:
        ValueError: If user_id is empty, n is not a positive integer,
            or the agent output cannot be parsed.
        FileNotFoundError: If required model artifacts are missing.
        OSError: If artifact files cannot be read.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if n < 1:
        raise ValueError(f"n must be a positive integer, got {n}")

    logger.info("Agent routing recommendations for user: %s (n=%d)", user_id, n)

    initial_state = format_request(user_id, n)
    result = agent.invoke(initial_state)

    recs = _extract_recommendations(result)
    logger.info("Agent returned %d recommendations for user: %s", len(recs), user_id)
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
        description="Context-Based Recommendation Router (Approach 2 — LangGraph Agent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m orchestrator.main --user_id AH2IFH762VY5HG373JSJ7BQVPUA3Q\n"
            "  python -m orchestrator.main --user_id <id> --n 5\n"
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
        logger.error("Invalid input or parse error: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("Model artifacts not found: %s", exc)
        sys.exit(1)
    except OSError as exc:
        logger.error("I/O error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    run_cli()
