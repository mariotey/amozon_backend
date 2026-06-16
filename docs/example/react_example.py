import os
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ReActState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iterations: int
    max_iterations: int

# Define the search tool using @tool decorator like example.py
@tool
def search_web(query: str) -> str:
    """
    Search the web for information using Tavily.
    Use this to find current information, facts, or answers to questions.

    Args:
        query: The search query string

    Returns:
        Search results as a string
    """
    try:
        search_tool = TavilySearchResults(max_results=3)
        results = search_tool.run(query)
        return str(results)
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize the LLM with tools like example.py
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
tools = [search_web]
llm_with_tools = llm.bind_tools(tools)

# ReAct system prompt enforcing strict one-action-at-a-time format
react_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a ReAct agent. Follow this EXACT format with ONE action at a time:

Thought: [Your reasoning about what to do next]
Action: [ONLY ONE action - describe what you want to search for]

Then STOP and wait for:
Observation: [Tool results will be provided]

Then continue with:
Thought: [Analyze the observation]
Action: [Next single action] OR Final Answer: [Complete response]

CRITICAL RULES:
1. ALWAYS start with "Thought:"
2. ONLY ONE action per response - never multiple searches
3. For multi-part questions: do ONE search, wait for observation, then next search
4. Use this exact format: "Thought:" then "Action:"
5. Never say "I need to use a tool" - use the format above

Example for "What's NYC weather and India's capital?":

Response 1:
Thought: I need to answer two questions. Let me start with NYC weather first.
Action: Search for current weather in New York City.

[Wait for observation]

Response 2:
Thought: Got the weather info. Now I need India's capital.
Action: Search for the capital of India.

[Wait for observation]

Response 3:
Thought: Now I have both pieces of information.
Final Answer: [Combined answer with weather and capital]

Remember: ONE action per response, follow exact format."""),
    MessagesPlaceholder(variable_name="messages"),
]) # This creates a slot in the prompt template that will be filled with whatever is passed as the messages parameter to format_messages().

'''  Example:
  If you have:
  - Template: [system_message, MessagesPlaceholder("messages")]
  - Call: template.format_messages(messages=[HumanMessage("Hello"), AIMessage("Hi")])
  
    The result becomes:
  [system_message, HumanMessage("Hello"), AIMessage("Hi")]
  '''

def react_agent(state: ReActState, config: RunnableConfig = None):
    """
    ReAct agent with proper prompting and tool binding
    """
    messages = state["messages"]
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 5)

    # Check if we've exceeded max iterations
    if iterations >= max_iterations:
        final_msg = AIMessage(content="I've reached the maximum number of iterations. Let me provide a final answer based on the information I've gathered so far.")
        return {
            "messages": [final_msg],
            "iterations": iterations + 1,
            "max_iterations": max_iterations
        }

    # Use the ReAct prompt with tool binding
    formatted_messages = react_prompt.format_messages(messages=messages)

    ''' So MessagesPlaceholder takes a list of message objects and inserts them into that position in the template. It's different from regular string     
  placeholders because it handles actual message objects, not just text.'''
    
    response = llm_with_tools.invoke(formatted_messages)

    return {
        "messages": [response],
        "iterations": iterations + 1,
        "max_iterations": max_iterations
    }

# Create the tool node like example.py
tool_node = ToolNode(tools)

def should_continue(state: ReActState) -> str:
    """Determine whether to continue to tools, keep reasoning, or end"""
    messages = state["messages"]
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 5)

    # Check max iterations
    if iterations >= max_iterations:
        return END

    if not messages:
        return "agent"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        # If there are tool calls, continue to tools
        if last_message.tool_calls:
            return "tools"

        # Check if this is a final answer
        content = last_message.content.lower()
        if any(phrase in content for phrase in ["final answer:", "in conclusion:", "to summarize:", "based on all the information gathered"]):
            return END

        # If the AI is still thinking/reasoning (contains "Thought:" or "Action:"), continue
        if any(phrase in content for phrase in ["thought:", "action:"]):
            return "agent"

    # Check if we just got tool results - continue reasoning
    elif isinstance(last_message, ToolMessage):
        return "agent"

    # Continue reasoning - removed the premature end condition
    return "agent"

# Create the ReAct workflow
workflow = StateGraph(ReActState)

# Add nodes like example.py
workflow.add_node("agent", react_agent)
workflow.add_node("tools", tool_node)

# Add edges like example.py
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "agent": "agent",
        END: END,
    },
)
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

def run_react_agent(user_input: str, max_iterations: int = 5):
    """Run the ReAct agent with tool binding like example.py"""
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "iterations": 0,
        "max_iterations": max_iterations
    }

    print(f"User: {user_input}")
    print("-" * 50)

    result = app.invoke(initial_state)

    # Print the conversation like example.py
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                print(f"AI: I need to use a tool to search for this information.")
                for tool_call in message.tool_calls:
                    print(f"Tool Call: {tool_call['name']}({tool_call['args']})")
            else:
                print(f"AI: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"Tool Result: {message.content}")

    print("-" * 50)
    return result

# Test the agent with multi-part questions
if __name__ == "__main__":
    # Example 1: Multi-part question requiring consolidation
    # run_react_agent("What is the current weather in New York and what is the capital of India?")

    # print("\n")

    # # Example 2: Single search question
    # run_react_agent("What is the current population of Tokyo?")

    # print("\n")

    # # Example 3: Complex multi-step question
    run_react_agent("What is the GDP of France and what is its current president?")
