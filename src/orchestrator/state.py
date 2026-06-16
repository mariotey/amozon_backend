"""
Agent state schema for the LangGraph ReAct orchestrator.
"""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Conversation state threaded through every node in the graph.

    Attributes:
        messages: Cumulative message list.  add_messages appends rather
            than overwrites on every node return.
        iterations: Number of llm_call node invocations so far.
        max_iterations: Hard cap on llm_call invocations to prevent
            runaway loops.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    iterations: int
    max_iterations: int
