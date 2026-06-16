"""
Orchestrator — Approach 2: Context-Based Routing via LangGraph ReAct agent.

The agent reasons over the request context (user_id + optional intent) and
selects between Collaborative Filtering (ALS) and Content-Based Filtering
(TF-IDF) tools.
"""
