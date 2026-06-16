"""
Node functions for the LangGraph ReAct recommendation orchestrator.

The LLM is initialised at module level (bound with tools) so agent_node
is a plain function — no factory or wrapper needed.

Nodes:
    format_request  — builds the initial AgentState from (user_id, n).
    agent_node      — calls the tool-bound LLM using the ReAct prompt.
    should_continue — conditional edge function; routes to "tools", loops
                      back to "agent", or terminates at END.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from orchestrator.prompts import REACT_PROMPT
from orchestrator.state import AgentState
from orchestrator.tools import TOOLS

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 10
_FINAL_ANSWER_PHRASES = ("final answer:", "in conclusion:", "to summarize:")

# Initialised once at import time — reused across all warm invocations
llm_with_tools = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
).bind_tools(TOOLS)


# ── Entry helper ──────────────────────────────────────────────────────────────


def format_request(
    user_id: str,
    n: int,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> dict[str, Any]:
    """Build the initial AgentState from (user_id, n).

    Args:
        user_id: Identifier of the target user.
        n: Number of recommendations requested.
        max_iterations: Hard cap on agent loop iterations (default: 10).

    Returns:
        Partial AgentState dict with the initial HumanMessage and
        iteration counters.
    """
    logger.info("Formatting request for user: %s (n=%d)", user_id, n)
    content = (
        f"Get {n} product recommendations for user '{user_id}'. "
        "Use collaborative filtering if the user has an established history, "
        "otherwise use content-based filtering."
    )
    return {
        "messages": [HumanMessage(content=content)],
        "iterations": 0,
        "max_iterations": max_iterations,
    }


# ── Agent node ────────────────────────────────────────────────────────────────


def agent_node(
    state: AgentState,
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """Call the tool-bound LLM and return its response.

    Uses REACT_PROMPT (ChatPromptTemplate + MessagesPlaceholder) to
    compose the system prompt with the current message history before
    invoking the LLM.

    An iteration guard triggers a graceful early-exit message instead of
    calling the LLM when max_iterations is already reached.

    Args:
        state: Current AgentState.
        config: Optional LangGraph RunnableConfig (passed through to LLM).

    Returns:
        AgentState patch dict with the new AIMessage and updated iteration count.
    """
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", _DEFAULT_MAX_ITERATIONS)

    if iterations >= max_iterations:
        logger.warning("Max iterations (%d) reached — forcing termination", max_iterations)
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Final Answer: Maximum iterations reached. "
                        "Please retry with a valid user ID or ensure model artifacts are built."
                    )
                )
            ],
            "iterations": iterations + 1,
            "max_iterations": max_iterations,
        }

    formatted = REACT_PROMPT.format_messages(messages=state["messages"])
    logger.info("Calling LLM (iteration %d/%d)", iterations + 1, max_iterations)
    response = llm_with_tools.invoke(formatted, config)

    return {
        "messages": [response],
        "iterations": iterations + 1,
        "max_iterations": max_iterations,
    }


# ── Routing function ──────────────────────────────────────────────────────────


def should_continue(state: AgentState) -> str:
    """Decide the next graph step after the agent node.

    Routing rules (in priority order):
        1. Max iterations hit → END
        2. Last message is AIMessage with tool_calls → "tools"
        3. Last message is AIMessage with a final-answer phrase → END
        4. Last message is AIMessage still reasoning → "agent"
        5. Last message is ToolMessage (observation received) → "agent"
        6. Fallback → "agent"

    Args:
        state: Current AgentState after the agent node.

    Returns:
        One of "tools", "agent", or END.
    """
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", _DEFAULT_MAX_ITERATIONS)
    messages: list[BaseMessage] = state["messages"]

    if iterations >= max_iterations:
        logger.info("Routing → END (max iterations reached)")
        return END

    if not messages:
        return "agent"

    last = messages[-1]

    if isinstance(last, AIMessage):
        if last.tool_calls:
            logger.info("Routing → tools (tool call requested)")
            return "tools"

        if any(phrase in last.content.lower() for phrase in _FINAL_ANSWER_PHRASES):
            logger.info("Routing → END (final answer detected)")
            return END

        logger.info("Routing → agent (LLM still reasoning)")
        return "agent"

    if isinstance(last, ToolMessage):
        logger.info("Routing → agent (tool observation received)")
        return "agent"

    return "agent"
