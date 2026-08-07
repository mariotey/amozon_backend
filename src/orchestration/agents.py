"""
Construction of the orchestration agent.

The agent owns the routing decision — which recommendation engine to run for a given request —
but makes it from the verdict `analyze_user` returns rather than by guessing from the query text.
Data-state questions ("is this user in the trained matrix?") stay in Python; language questions
("what did they ask for?") stay with the model.

The tools are registered as ordinary synchronous functions. Pydantic AI runs sync tools in a
worker thread, which keeps the heavy inference calls — content-based filtering computes cosine
similarity over the full item matrix — off the event loop without any explicit offloading here.
"""
import logging
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from .models import AgentResponse
from .orchestration_config import AGENT_API_KEY, AGENT_MODEL, AGENT_RETRIES
from .prompt import SYSTEM_PROMPT
from .tools import (
    analyze_user,
    collaborative_recommendations,
    content_based_recommendations,
    popular_items,
)

logger = logging.getLogger(__name__)

_agent_obj: Agent[None, AgentResponse] | None = None

def build_model() -> OpenRouterModel:
    """
    Construct the OpenRouter-backed model the agent runs on.

    Returns:
    - OpenRouterModel: Model configured from the MODEL_NAME and API_KEY environment variables.

    Raises:
    - ValueError: If API_KEY is not set. Only the `query` path needs it; the analysis and
                  popularity paths run offline and never reach here.
    """
    if not AGENT_API_KEY:
        raise ValueError(
            "API_KEY is not set. Add your OpenRouter key to .env as API_KEY=... "
            "(offline modes such as `--mode analyze` do not require it)."
        )

    return OpenRouterModel(
        AGENT_MODEL,
        provider=OpenRouterProvider(api_key=AGENT_API_KEY),
    )

def build_agent(
    model: Model | str | None = None
) -> Agent[None, AgentResponse]:
    """
    Construct the orchestration agent with its tools and instructions.

    Args:
    - model (Model | str | None): Model to run on. Defaults to the configured OpenRouter model.
                                  Pass `TestModel()` here to exercise tool routing with no API
                                  calls.

    Returns:
    - Agent[None, AgentResponse]: A configured agent producing a validated AgentResponse.

    Raises:
    - ValueError: If no model is supplied and API_KEY is not set.
    """
    return Agent(
        model or build_model(),
        output_type=AgentResponse,
        instructions=SYSTEM_PROMPT,
        tools=[
            analyze_user,
            collaborative_recommendations,
            content_based_recommendations,
            popular_items,
        ],
        retries=AGENT_RETRIES,
    )

def get_agent() -> Agent[None, AgentResponse]:
    """
    Retrieve the cached orchestration agent instance.

    The agent is constructed lazily on first invocation and reused across subsequent calls, so a
    warm serverless runtime avoids rebuilding the tool schemas on every request.

    Returns:
    - Agent[None, AgentResponse]: Cached agent instance.
    """
    global _agent_obj

    if _agent_obj is None:
        logger.info("Building orchestration agent with model: %s", AGENT_MODEL)
        _agent_obj = build_agent()

    return _agent_obj

# ── Public handler function (serverless entry point) ───────────────────────────────────────────

def recommend(
    query: str
) -> AgentResponse:
    """
    Answer a natural-language recommendation request.

    Designed to be called directly by an HTTP trigger, mirroring the handler style of the
    recommender tools.

    Args:
    - query (str): The natural-language request, containing the user id

    Returns:
    - AgentResponse: The chosen engine, the recommended items, and a short written answer.

    Raises:
    - ValueError: If query is empty
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    logger.info("Running orchestration agent for query: %s", query)

    result = get_agent().run_sync(query)

    return result.output
