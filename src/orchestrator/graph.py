"""
LangGraph StateGraph assembly for the context-based recommendation router.

Graph layout:
    START
      │
      ▼
    agent  ◄──────────────┐
      │                    │
      ▼ (should_continue)  │
    tools ─────────────────┘
      │
      ▼ (no tool_calls / final answer)
     END

Nodes:
    agent — LLM call with tool binding (defined in nodes.py)
    tools — executes the requested tool via ToolNode

Edges:
    START → agent                         (unconditional)
    agent → tools | agent | END           (conditional via should_continue)
    tools → agent                         (unconditional loop-back)
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from orchestrator.nodes import agent_node, should_continue
from orchestrator.state import AgentState
from orchestrator.tools import TOOLS

logger = logging.getLogger(__name__)


def build_agent() -> StateGraph:
    """Compile and return the recommendation StateGraph.

    Returns:
        A compiled LangGraph CompiledGraph ready to invoke.
    """
    logger.info("Building recommendation agent graph")

    tool_node = ToolNode(TOOLS)

    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "agent": "agent",
            END: END,
        },
    )
    builder.add_edge("tools", "agent")

    return builder.compile()


# Module-level singleton — compiled once per process / warm container.
agent = build_agent()
