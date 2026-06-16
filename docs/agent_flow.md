# Agent Flow — Recommendation Orchestrator

## Input

`get_recommendations(user_id, n)` in `main.py` builds a single `HumanMessage`:

```
"Get {n} product recommendations for user '{user_id}'.
Use collaborative filtering if the user has an established history,
otherwise use content-based filtering."
```

This is the only input the agent receives — no pre-fetched user profile or history flag.

## How the LLM Calls Tools

Tool signatures in `tools.py` are bound to the LLM via `.bind_tools(TOOLS)` in `nodes.py`.  
LangChain auto-generates a JSON schema from the function signatures and sends it alongside the message.  
The LLM reads the natural language message and maps values to tool arguments itself — no explicit extraction prompt needed.

```json
{ "user_id": "USER_ID", "n": 10 }
```

## Routing Logic

The agent always tries CF first. Routing is **reactive, not predictive**:

1. Call `collaborative_filtering_recommend(user_id, n)`
2. If the user exists in the CF training set → return results → done
3. If CF raises `ValueError` (user not in training set) → fall back to `content_based_recommend(user_id, n)`
4. If CB returns empty list → user has no qualifying interaction history → report that to caller

## Graph Loop (`nodes.py` — `should_continue`)

```
agent_node → tool_calls? → ToolNode → agent_node → final answer? → END
                                                  ↑ ValueError fallback loops here
```

- Max iterations cap: 10 (configurable via `max_iterations`)
- Terminates when LLM response contains `"Final Answer:"`, `"In conclusion:"`, or `"To summarize:"`
