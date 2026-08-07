"""
Configuration constants for the orchestration module.

NOTE: This module is deliberately NOT named `tool_config.py`. That filename is reserved by the
      tool-config contract (`AGENTS.md`) for recommender tools carrying ML artefacts, which
      `ModelsLoader` loads via `importlib.import_module(f"src.{tool}.tool_config")`. The
      orchestration package owns no artefacts and must never be registered in `config.TOOLS`.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Agent model ────────────────────────────────────────────────────────────────────────────────
# The agent is served through OpenRouter, which fronts many providers behind one key. Both values
# come from the environment so the model can be swapped without touching source.

# OpenRouter model slug, in "<vendor>/<model>" form.
DEFAULT_AGENT_MODEL: str = "tencent/hy3"

AGENT_MODEL: str = os.getenv("MODEL_NAME", DEFAULT_AGENT_MODEL)

# OpenRouter API key. Absent only when running the offline modes, which never build the agent.
AGENT_API_KEY: str | None = os.getenv("API_KEY")

# Number of times the agent may retry a tool call that raises ModelRetry before failing the run.
AGENT_RETRIES: int = 10

# ── Recommendation defaults ────────────────────────────────────────────────────────────────────

DEFAULT_TOP_N: int = 10

# Upper bound on `n`, so a malformed query cannot request an unbounded result set.
MAX_TOP_N: int = 50

# ── Tool names ─────────────────────────────────────────────────────────────────────────────────
# Referenced by `analyze_user` when reporting engine availability, and by the system prompt.

COLLABORATIVE_TOOL: str = "collaborative_recommendations"
CONTENT_BASED_TOOL: str = "content_based_recommendations"
POPULAR_TOOL: str = "popular_items"

# ── popular_items ranking ──────────────────────────────────────────────────────────────────────

# Items in these popularity segments are excluded from the popularity fallback: they carry too
# few reviews for `quality_score` (a Wilson lower bound) to be meaningful.
POPULAR_EXCLUDED_SEGMENTS: tuple[str, ...] = ("cold_start",)

# Ranking columns, applied in order, all descending.
POPULAR_RANK_COLUMNS: tuple[str, ...] = ("quality_score", "num_reviews")
