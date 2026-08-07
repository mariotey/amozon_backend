# Agentic Orchestration Layer — Design Strategy

> Design for a natural-language orchestration layer on top of the two existing recommender
> tools (`src/collaborative_filtering/`, `src/content_based_filtering/`).
> Prerequisites: the build/serve lifecycle in `docs/BUILD_TEST.md` is complete; algorithm
> details are in `docs/CB_CF_FILTERING.md`.
>
> **Status: implemented in `src/orchestration/`.** See `docs/ORCHESTRATION_IMPLEMENTATION.md` for
> what was built, the decisions that closed §8, and how to run it. This document remains the
> record of *why* the design is shaped this way; where the two disagree, the implementation notes
> are authoritative.

---

## 1. Scope

One use case: a user submits a natural-language query containing their `user_id`, and the
agent returns recommendations. Nothing else is in scope — no conversational memory, no
item-title search, no authenticated session layer.

---

## 2. The problem the agent has to solve

A query like *"give recommendations for user ag6hllx…"* carries **no information about which
engine to use**. The two engines have asymmetric requirements and asymmetric failure modes
(`docs/CB_CF_FILTERING.md`, "Cold-start asymmetry"):

| Engine | Requires | Fails by |
|---|---|---|
| Collaborative (ALS) | `user_id` present in the trained matrix | **raises** `ValueError` (`src/collaborative_filtering/recommender.py:48-49`) |
| Content-based (TF-IDF) | ≥ 1 review with `review_rating >= 3` | returns `[]` (`src/content_based_filtering/recommender.py:111-112`) |

The same sentence should route differently for different users:

- `ag6hllxrsby3efcfgqgjxvjabvfq` — 176 positive reviews, in the matrix → collaborative works
- `agci7fah4gl5fi65hylkwtmfz2cq` — 1 review rated 1.0 → collaborative **raises**, content-based returns `[]`

So the agent needs to **look the user up before choosing**. That lookup is itself a tool.

---

## 3. Design — analysis tool first, then routing

Four tools. The LLM calls the analysis tool, reads the result, then picks an engine.

```python
analyze_user(user_id)                       # tool 1 — always called first
collaborative_recommendations(user_id, n)   # tool 2
content_based_recommendations(user_id, n)   # tool 3
popular_items(n)                            # tool 4 — cold-start escape hatch
```

The LLM makes the routing decision. It just makes it **informed** rather than by guessing.

### 3.1 Flow

```
"give me 5 recommendations for user ag6hllx…"
        │
   [LLM]  extracts user_id + n=5, calls analyze_user
        │
   analyze_user("ag6hllx…")
        └─► { available_engines: ["collaborative_recommendations",
                                  "content_based_recommendations"],
              num_positive_reviews: 176, top_category: "software", … }
        │
   [LLM]  picks collaborative_recommendations(user_id, 5)
        │
   [LLM]  writes the answer over the returned rows
```

---

## 4. `analyze_user` must return a verdict, not raw statistics

This is the decision that makes or breaks the design.

**Avoid** returning only raw numbers:

```json
{"user_id": "agci7fa…", "num_reviews": 1, "avg_rating": 1.0, "in_matrix": false}
```

That forces the LLM to know that `MIN_RATING_THRESHOLD` is `3`, that `in_matrix: false`
means collaborative *raises* rather than returns empty, and that one review is too thin a
profile. Those thresholds would live in a system prompt, where they can be ignored under
paraphrase.

**Return an explicit verdict instead:**

```json
{
  "user_id": "agci7fah4gl5fi65hylkwtmfz2cq",
  "num_reviews": 1,
  "num_positive_reviews": 0,
  "top_category": null,
  "prefers_free": false,
  "available_engines": ["popular_items"],
  "unavailable": {
    "collaborative_recommendations": "user not in the trained ALS matrix",
    "content_based_recommendations": "no reviews rated 3 or above"
  }
}
```

Threshold logic stays in Python where it is testable. The LLM still chooses — including
between multiple available engines, informed by the query wording — but it cannot choose one
that will fail.

`top_category` and `prefers_free` are included because they let the model write a better
final answer (*"based on your interest in X…"*), not because they affect routing.

### 4.1 How the fields are derived

| Field | Source |
|---|---|
| `num_reviews` | `user_item_df` filtered to `user_id` |
| `num_positive_reviews` | same, `review_rating >= MIN_RATING_THRESHOLD` (`content_based_filtering/tool_config.py`) |
| `in ALS matrix` | `user_id` present in `idx_to_userid_mapping.json` values |
| `top_category` | mode of `main_category` over positive history |
| `prefers_free` | `is_free` ratio over positive history `> FREE_PREFERENCE_THRESHOLD` (0.5) |

### 4.2 Engine availability rules

| Condition | Engine offered |
|---|---|
| `user_id` in ALS matrix | `collaborative_recommendations` |
| `num_positive_reviews >= 1` | `content_based_recommendations` |
| always | `popular_items` |

`popular_items` is always listed so `available_engines` is never empty.

---

## 5. Implementation notes

### 5.1 Recommendation tools must not raise

`collaborative_filtering.recommender.recommend_for_user` raises `ValueError` for unknown
users. Inside a tool, an uncaught exception aborts the agent run. Wrap it:

```python
{"status": "unavailable", "reason": "user not in the trained ALS matrix"}
```

so the model can recover by calling a different tool. The same applies to the
`FileNotFoundError` / `OSError` paths the existing CLIs already catch
(`docs/CB_CF_FILTERING.md` §A.5, §B.5).

### 5.2 Pin tool order in the system prompt

> Always call `analyze_user` first. Only call a recommendation tool that appears in its
> `available_engines` list.

Without this the model will sometimes skip straight to a recommendation call.

### 5.3 `popular_items` is the only new recommendation logic

`item.parquet` already carries `quality_score` (Wilson lower bound) and `popularity_segment`
(`docs/ARCHITECTURE.md` §3.2), so this is a short sort-and-slice — but it does not exist
today, and it is the only thing standing between a brand-new user and an empty response.

### 5.4 Normalize the output schema

Collaborative returns `parent_asin, item_title, main_category, is_free` (no score);
content-based returns those plus `score`. Emit one shared envelope from all three
recommendation tools, or the model will phrase answers inconsistently depending on which
engine fired.

### 5.5 Blocking, CPU-heavy calls

`cosine_similarity(profile, item_matrix)` runs over 89,251 × 10,000. Do not run it on the
event loop — either define tools `async def` and use `anyio.to_thread.run_sync`, or define
them `def` and let the framework thread them. Decide deliberately. Carry over the
`OPENBLAS_NUM_THREADS=1` note (`docs/CB_CF_FILTERING.md` §A.7) for the ALS path.

---

## 6. Why Pydantic AI

| Feature | Fit here |
|---|---|
| `@agent.tool` | Derives the JSON schema from type hints + docstring; tool descriptions live next to the code. |
| `output_type=<PydanticModel>` | Validated response envelope instead of free text — this is a backend, not a chatbot. |
| `ModelRetry` | Lets a tool ask for a corrected call instead of failing the run. |
| `TestModel` / `FunctionModel` | Routing tests with **zero API calls** — matters given the repo's offline-CLI testing culture (`docs/BUILD_TEST.md`). |

Model-agnostic and Python 3.13-compatible, so it does not constrain the existing stack.

**Caveat — this breaks the fully-offline story.** Everything in `docs/BUILD_TEST.md` runs
with dummy Supabase credentials and no network; the agent layer needs a live LLM API key.
Keep `analyze_user`'s underlying logic importable and testable without the agent, and keep
`python -m src.<tool>.main --mode user` working offline as before.

**Verify the API before implementing** — Pydantic AI's surface has moved across releases;
pull current docs rather than coding from memory.

---

## 7. Repo placement

> **Retracted.** This section originally argued against `src/`, claiming `ModelsLoader` would try
> to import a nonexistent `tool_config.py` for an agent package placed there. That is wrong:
> `ModelsLoader` iterates the explicit `config.TOOLS` list, not the `src/` directory listing
> (`models_loader/models_loader.py:191-203`), so an unregistered package under `src/` is never
> discovered. The layer was built at **`src/orchestration/`**; see
> `docs/ORCHESTRATION_IMPLEMENTATION.md` §1-2 for the final layout.
>
> The one real constraint survives: the package must stay out of `config.TOOLS`, and its config
> module is named `orchestration_config.py` so it cannot be mistaken for a registrable tool.

`user_id` arrives inside the query, so it is an ordinary tool parameter — no dependency
injection needed.

> **Later, if this moves behind an authenticated HTTP API:** `user_id` should come from the
> session rather than from the query text, otherwise one user can request another's
> recommendations. Not a concern for CLI/demo use.

---

## 8. Open decisions — all closed

| # | Decision | Resolution |
|---|---|---|
| 1 | Model provider / model id | OpenRouter, `MODEL_NAME` + `API_KEY` from `.env` |
| 2 | `popular_items` ranking | `quality_score` desc, tie-broken by `num_reviews`, `cold_start` segment excluded |
| 3 | Response shape | Structured `output_type=AgentResponse` |

Rationale for each is in `docs/ORCHESTRATION_IMPLEMENTATION.md` §5.

---

## 9. Build order

1. `analyze_user` + its underlying lookups — pure Python, offline-testable.
2. `popular_items` — the one missing recommendation path.
3. `schemas.py` — unified output envelope (§5.4).
4. Tool wrappers around the two existing handlers, with exceptions converted to status
   objects (§5.1).
5. Pydantic AI wiring + system prompt.
6. CLI (`agent/main.py`), mirroring the `python -m src.<tool>.main` style.
