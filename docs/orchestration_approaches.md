# Recommendation Orchestration — Approaches

This document covers two strategies for orchestrating the recommendation pipelines.
Both pipelines expose `get_user_recommendations(user_id, n)` with the same signature;
the orchestration layer decides which one to invoke.

---

## Table of Contents

1. [Shared Interface](#1-shared-interface)
2. [Approach 1 — Deterministic Routing](#2-approach-1--deterministic-routing)
   - [Logic](#21-logic)
   - [Implementation](#22-implementation)
   - [Edge cases](#23-edge-cases)
3. [Approach 2 — Context-Based Routing (LangGraph)](#3-approach-2--context-based-routing-langgraph)
   - [When to use](#31-when-to-use)
   - [Tool definitions](#32-tool-definitions)
   - [Graph definition](#33-graph-definition)
   - [Calling the agent](#34-calling-the-agent)
   - [Routing behaviour](#35-routing-behaviour)
   - [Edge cases](#36-edge-cases)
4. [Comparison](#4-comparison)
5. [Quick Reference](#5-quick-reference)

---

## 1. Shared Interface

Both pipelines expose the same function signature:

```python
get_user_recommendations(user_id: str, n: int = 10) -> list[dict]
```

| Parameter | Type  | Required | Description                             |
|-----------|-------|----------|-----------------------------------------|
| `user_id` | `str` | yes      | Identifier of the target user           |
| `n`       | `int` | no       | Number of recommendations (default: 10) |

**Response shape differs slightly between pipelines:**

| Key             | CF response | CB response | Description              |
|-----------------|-------------|-------------|--------------------------|
| `parent_asin`   | yes         | yes         | ASIN of recommended item |
| `item_title`    | yes         | yes         | Product title            |
| `main_category` | yes         | yes         | Product category         |
| `is_free`       | yes         | yes         | Whether the item is free |
| `score`         | no          | yes         | Cosine similarity score  |

---

## 2. Approach 1 — Deterministic Routing

### 2.1 Logic

Route based on whether the user exists in the CF training set.
The CF system raises `ValueError` for unknown users; the CB system handles them gracefully by returning an empty list.

```
get_recommendations(user_id, n)
    │
    Try CF (ALS) → known user → return CF results
    │
    ValueError (unknown user)
    │
    Fall back to CB (TF-IDF) → return CB results
```

### 2.2 Implementation

```python
from collaborative_filtering.main import get_user_recommendations as cf_recommend
from content_based_filtering.main import get_user_recommendations as cb_recommend

def get_recommendations(user_id: str, n: int = 10) -> list[dict]:
    try:
        return cf_recommend(user_id, n)
    except ValueError:
        return cb_recommend(user_id, n)
```

### 2.3 Edge cases

| Scenario                                    | Behaviour                                              |
|---------------------------------------------|--------------------------------------------------------|
| User exists in CF training set              | CF result returned                                     |
| User not in CF training set                 | Falls back to CB                                       |
| User in CB but no ratings >= 3              | CB returns empty list — no further fallback            |
| Truly cold user (no history at all)         | CB returns empty list                                  |
| Artifacts not built                         | Both raise `FileNotFoundError` — handle upstream       |

---

## 3. Approach 2 — Context-Based Routing (LangGraph)

### 3.1 When to use

Use this approach when the caller can supply request context beyond just a `user_id`
(e.g. a natural-language query, a session type, or explicit intent). The agent reasons
over the context to pick the appropriate pipeline.

**Not suitable for sub-100ms SLAs** — the LLM call adds ~500ms–2s of latency.

### 3.2 Tool definitions

```python
from langchain_core.tools import tool

@tool
def collaborative_filtering_recommend(user_id: str, n: int = 10) -> list[dict]:
    """
    Recommend items using Collaborative Filtering (ALS).
    Best for users with an established rating or purchase history.
    Raises ValueError if the user is not in the training set.
    """
    from collaborative_filtering.main import get_user_recommendations
    return get_user_recommendations(user_id, n)

@tool
def content_based_recommend(user_id: str, n: int = 10) -> list[dict]:
    """
    Recommend items using Content-Based Filtering (TF-IDF).
    Works for new or cold-start users. Returns an empty list
    if the user has no qualifying interaction history (ratings >= 3).
    """
    from content_based_filtering.main import get_user_recommendations
    return get_user_recommendations(user_id, n)
```

### 3.3 Graph definition

```python
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6")
tools = [collaborative_filtering_recommend, content_based_recommend]

agent = create_react_agent(llm, tools)
```

### 3.4 Calling the agent

```python
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                f"Get {n} recommendations for user {user_id}. "
                "Use collaborative filtering if the user has an established history, "
                "otherwise use content-based filtering."
            )
        }
    ]
})
```

### 3.5 Routing behaviour

| Request context                               | Pipeline chosen                  |
|-----------------------------------------------|----------------------------------|
| "Personalised recommendations for me"         | CF (if known user)               |
| "I just signed up, suggest something"         | CB                               |
| "Based on items I rated highly"               | CF                               |
| "Based on item descriptions / what I browsed" | CB                               |
| Ambiguous or unknown user                     | CB (safe default)                |

### 3.6 Edge cases

| Scenario                                    | Behaviour                                                       |
|---------------------------------------------|-----------------------------------------------------------------|
| Agent calls CF for an unknown user          | Tool raises `ValueError` — handle in the tool wrapper or system prompt |
| Ambiguous context                           | Agent defaults to CB as the safer option                        |
| Artifacts not built                         | Tool raises `FileNotFoundError` — propagates to agent           |

---

## 4. Comparison

| Dimension          | Approach 1 — Deterministic     | Approach 2 — LangGraph          |
|--------------------|--------------------------------|---------------------------------|
| Routing signal     | user_id in CF training set     | Request context + LLM reasoning |
| Latency overhead   | Near-zero                      | ~500ms–2s (LLM call)            |
| Flexibility        | Fixed rule                     | Adapts to request intent        |
| Complexity         | Single try/except              | Agent + tool definitions        |
| Dependencies       | None beyond the two pipelines  | `langgraph`, `langchain-anthropic` |
| Best for           | High-throughput APIs           | Conversational / agentic UX     |

---

## 5. Quick Reference

```python
# --- Approach 1: deterministic ---
from collaborative_filtering.main import get_user_recommendations as cf_recommend
from content_based_filtering.main import get_user_recommendations as cb_recommend

def get_recommendations(user_id: str, n: int = 10) -> list[dict]:
    try:
        return cf_recommend(user_id, n)
    except ValueError:
        return cb_recommend(user_id, n)

# --- Approach 2: LangGraph agent ---
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

agent = create_react_agent(
    ChatAnthropic(model="claude-sonnet-4-6"),
    [collaborative_filtering_recommend, content_based_recommend]
)

result = agent.invoke({
    "messages": [{"role": "user", "content": f"Recommend {n} items for user {user_id}."}]
})
```