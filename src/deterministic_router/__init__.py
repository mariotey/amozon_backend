"""
Deterministic Router — Approach 1 recommendation orchestration.

Routes recommendation requests to the Collaborative Filtering (ALS) pipeline
for known users and falls back to Content-Based Filtering (TF-IDF) for unknown
or cold-start users.
"""
