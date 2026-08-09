# Orchestration Layer — Request Flow, End to End

> Walks a single query through `src/orchestration/` from the moment it arrives to the moment a
> reply goes back, with file:line references. The *why* behind these choices is in
> `docs/AGENTIC_ORCHESTRATION.md`; what was built and the open questions are in
> `docs/ORCHESTRATION_IMPLEMENTATION.md`. This document is the *how it runs*.

---

## 0. Shape of the system

There is no custom graph/state-machine here. It's a **single LLM agent with four tools**,
built on Pydantic AI, running one query at a time with no memory between calls:

```
                         recommend(query: str)
                                 │
                     [Agent.run_sync(query)]
                                 │
                    ┌────────────▼────────────┐
                    │   LLM (router/decider)   │◄────────────────┐
                    │  guided by SYSTEM_PROMPT │                  │
                    └────────────┬─────────────┘                  │
                                 │ tool call                       │ tool result
                                 ▼                                 │
                        ┌────────────────┐                        │
                        │  analyze_user   │────────────────────────┘
                        └────────────────┘
                                 │ available_engines: [...]
                                 ▼
                   LLM picks exactly one tool:
        ┌─────────────────────┐ ┌───────────────────────┐ ┌───────────────┐
        │ collaborative_      │ │ content_based_         │ │ popular_items  │
        │ recommendations     │ │ recommendations        │ │ (fallback,     │
        │ (ALS model)         │ │ (TF-IDF cosine sim)     │ │  always works) │
        └─────────────────────┘ └───────────────────────┘ └───────────────┘
                                 │ status="unavailable" → LLM retries a different engine
                                 ▼
                    LLM emits validated AgentResponse
                    (answer, user_id, engine_used, items)
```

The routing "decision" is made twice, in two different places, deliberately:

1. **Python decides what's *possible*** — `analyze_user_profile()` computes a hard verdict
   (`available_engines`) from real data, not vibes.
2. **The LLM decides what's *best*** — given that verdict plus the wording of the query, it
   picks one engine and can retry another if the first fails.

> **In plain terms:** think of it like a restaurant. Python is the waiter who checks the
> kitchen and tells the customer "we're out of the fish today, but we have chicken and pasta."
> The LLM is the customer who then decides, based on what's actually available and what they
> asked for, what to order. The waiter never lets the customer order something the kitchen
> can't make — but within what's possible, the customer (LLM) still makes the real choice.

---

## 1. Entry point

`recommend(query: str) -> AgentResponse` — `src/orchestration/agents.py:104-129`.

This is the one function meant to sit behind an HTTP handler eventually (there is no web
framework wired up yet). It is also reachable from the CLI:

```powershell
python -m src.orchestration.main --mode query --query "give me 5 recommendations for user ag6hllx…"
```

via `run_cli()` → `src/orchestration/main.py:115-160`, which dispatches to `recommend()` for
`--mode query` (other CLI modes bypass the agent entirely — see §6).

### 1.1 What happens inside `recommend()`

1. **Validate.** Reject an empty/whitespace `query` (`agents.py:122-123`).
2. **Get the agent.** `get_agent()` (`agents.py:84-100`) returns a lazily-built, cached
   `pydantic_ai.Agent` singleton (`_agent_obj`, `agents.py:30`) — built once per warm process,
   reused across calls.
3. **Run it.** `get_agent().run_sync(query)` (`agents.py:127`) — the raw query string is handed
   straight to the LLM. There is no separate intent-classification or NLU step in Python; the
   model itself parses out things like the `user_id` and the desired count.
4. **Return.** `result.output`, already validated against the `AgentResponse` schema, goes back
   to the caller.

> **In plain terms:** the query is just a plain sentence, e.g. *"give me 5 recommendations for
> user ag6hllx…"*. There's no code that manually pulls the user id out with regex or string
> splitting — the LLM reads the sentence itself and figures out the user id and how many items
> are wanted, the same way a person reading the message would.

### 1.2 What the agent is built from

`build_agent()` (`agents.py:54-82`) wires together:

| Piece | Source |
|---|---|
| Model | `build_model()` (`agents.py:32-52`) → OpenRouter, model id from `MODEL_NAME` env var (default `tencent/hy3`), key from `API_KEY` |
| Output schema | `output_type=AgentResponse` — forces structured JSON, not prose |
| Instructions | `SYSTEM_PROMPT` (`prompt.py:16-48`) — the routing rulebook, see §3 |
| Tools | `[analyze_user, collaborative_recommendations, content_based_recommendations, popular_items]` (`tools.py`) |
| Retries | `AGENT_RETRIES = 10` — how many times a tool may raise `ModelRetry` before the run fails (`orchestration_config.py:27`) |

Nothing here contacts the network until `recommend()` actually runs — importing the module
resolves no API key, which is what keeps `--mode analyze` / `--mode popular` usable offline.

---

## 2. Step 1 — the analysis tool always fires first

The system prompt hard-codes this ordering: **`analyze_user` before any recommendation tool,
every time.** The LLM extracts `user_id` (and `n`, default 10) from the free-text query, then
calls it.

`analyze_user` tool wrapper — `tools.py:61-82` — calls `analyze_user_profile()` in
`analysis.py:54-133`:

1. Validates `user_id` is non-empty; a malformed id becomes `ModelRetry`
   (`tools.py:80-82`) so the model can re-ask instead of aborting the run.
2. Loads the interaction table and item metadata via `resources.get_user_item_df()` /
   `resources.get_item_df()` (`resources.py:34-51`), which delegate to the collaborative-
   filtering module's cached `DataLoader`.
3. Filters to this user's *positively rated* history (`review_rating >= MIN_RATING_THRESHOLD`)
   joined with item metadata (`analysis.py:28-52`).
4. Computes `top_category` (mode of category) and `prefers_free` (share of free items >
   `FREE_PREFERENCE_THRESHOLD`) — used only to make the final answer sound informed, not for
   routing.
5. Computes `available_engines` — the actual decision input:

   | Condition | Engine offered |
   |---|---|
   | `user_id` present in the trained ALS matrix | `collaborative_recommendations` |
   | ≥ 1 positively-rated review | `content_based_recommendations` |
   | always | `popular_items` |

6. Returns a `UserAnalysis` object (`models.py:12-39`) with `available_engines` and an
   `unavailable` map explaining what was ruled out and why.

This is why decision-making is split across languages: the thresholds that determine whether
an engine will actually work live in tested Python, not in a prompt the model could
paraphrase past.

> **In plain terms:** before the LLM is allowed to pick a recommendation strategy, the system
> first does a quick background check on the user, like a librarian checking your library card
> before recommending books: *"Have you borrowed enough books before that I can tell what you
> like? Have you rated anything recently? Are you even in our records at all?"*
>
> The answer to that check is not "yes/no, use your judgement" — it's a hard, pre-computed list
> of which recommendation engines are actually *capable* of working for this specific user, e.g.
> `["collaborative_recommendations", "content_based_recommendations", "popular_items"]` for an
> active user, or just `["popular_items"]` for someone brand new with almost no history. The LLM
> never sees raw numbers like "1 review, rating 1.0" and has to decide for itself whether that's
> "enough" — Python already decided that and handed over the verdict.
>
> **Important: `analyze_user` does not choose a recommendation engine.** Its job stops at
> producing `available_engines` — the menu. It has no opinion on which item off that menu is
> the *best* one for this query. That choice — the actual "which recommendation do we run"
> decision — happens one step later, entirely inside the LLM, in §3 below.

---

## 3. Step 2 — the LLM picks an engine

> **This is where the actual decision happens.** `analyze_user` (§2) only narrowed the field down
> to a list of *safe* options — it made no recommendation of its own. Everything from here on —
> which one of those safe options to actually use — is decided by the LLM, guided by
> `SYSTEM_PROMPT`, not by any Python `if/else` logic.

`SYSTEM_PROMPT` (`prompt.py:16-48`) tells the model, in order:

1. Never call a recommendation tool before `analyze_user`.
2. Choose exactly one engine from `available_engines`:
   - prefer `collaborative_recommendations` when available and the user has real history;
   - prefer `content_based_recommendations` for "similar to X"-style asks, or when it's the
     only personalized option;
   - fall back to `popular_items` when neither personalized engine is available.
3. If the chosen tool comes back `status="unavailable"`, try the next engine in
   `available_engines` — `popular_items` is always in that list, so there's always a landing
   spot.
4. Write a 1–2 sentence answer over the returned items **verbatim** — no reordering, no
   inventing items — referencing `top_category` / `prefers_free` for color, and say
   explicitly if it fell back to popularity.

> **In plain terms — how the "which recommendation?" decision actually gets made:**
>
> There is no `if/else` block anywhere in the Python code that says "if user has 176 reviews,
> use collaborative filtering." That decision genuinely happens *inside the LLM's head*, guided
> by plain-English instructions in `SYSTEM_PROMPT`. It works like this:
>
> 1. The LLM already has the `available_engines` list from `analyze_user` (§2) — its menu of
>    safe choices.
> 2. It reads the instructions, which amount to a simple priority order:
>    - "If collaborative filtering is on the menu and this user has a real history → prefer it."
>      (Collaborative filtering looks at *other users who behaved similarly* — "people who
>      liked what you liked also liked this" — so it needs a user with enough history to find
>      those similar people.)
>    - "If the user is asking for something 'similar to X', or collaborative isn't available
>      but content-based is → use content-based." (Content-based looks at *the actual attributes
>      of items the user liked* — category, description, etc. — and finds more items like those.)
>    - "If neither personalized option is available → use popularity." (Just show what's
>      generally well-rated and widely reviewed — the safe, generic answer for a user we know
>      almost nothing about.)
> 3. It calls that one tool.
> 4. If the tool comes back saying `"unavailable"` (e.g. an artefact file is missing, or a
>    last-minute edge case slipped through), the LLM doesn't give up — it just tries the next
>    engine still left in `available_engines`. Because `popular_items` is always in that list and
>    always works, the user is guaranteed to get *some* answer.
>
> **Worked example:** query = *"recommend 5 items for user agci7fah4gl5fi65hylkwtmfz2cq"*.
> `analyze_user` reports this user has only 1 review, rated 1.0 (below the "positive" threshold),
> and isn't in the trained collaborative-filtering matrix. So `available_engines` comes back as
> just `["popular_items"]`. The LLM has no real choice to make here — collaborative and
> content-based were never offered — so it calls `popular_items(5)` and tells the user, honestly,
> that these are popular picks rather than personalized ones because there isn't enough history
> yet.

---

## 4. Step 3 — the recommendation is triggered

Whichever tool the LLM calls, none of the three recommendation tools ever raise — failures
become `RecommendationResult(status="unavailable", reason=...)` so the model can recover
(`tools.py:1-22`).

**`collaborative_recommendations(user_id, n)`** — `tools.py:84-137`
- Clamps `n` to `[1, 50]`.
- Calls `get_user_recommendations()` in `src/collaborative_filtering/main.py:92-144`, which
  loads the cached `DataLoader`/`ModelsLoader` and calls
  `recommend_for_user()` in `src/collaborative_filtering/recommender.py:15-59` — this maps
  `user_id` to a matrix index and asks the trained **implicit-ALS** model for `N`
  recommendations.
- Unknown user → `ValueError` → wrapped to `status="unavailable"`. Missing artefacts →
  `FileNotFoundError`/`OSError` → same. No relevance `score` in the output.

**`content_based_recommendations(user_id, n)`** — `tools.py:139-194`
- Same clamp/wrap pattern, calls `get_user_recommendations()` in
  `src/content_based_filtering/main.py:95-145` → `recommend_for_user()` in
  `src/content_based_filtering/recommender.py:79-170`:
  - builds a weighted **TF-IDF** profile vector from the user's positive history,
  - runs **cosine similarity** against the full item matrix,
  - masks seen items, boosts the user's top category, optionally filters to free items, and
    returns the top `N` with a `score` column.
- Empty history or empty result → `status="unavailable"` (an empty success would be
  misleading).

**`popular_items(n)`** — `tools.py:196-229`
- Calls `get_popular_items()` in `popular.py:24-68` — sorts `item_df` by `quality_score` then
  `num_reviews`, descending, excluding the `cold_start` segment (falls back to the unfiltered
  ranking if that leaves too few items). No model artefacts involved. Always succeeds — this is
  the guaranteed floor under every query.

All three responses are normalized to the same `RecommendationResult` / `RecommendedItem`
envelope (`models.py:42-93`) before going back to the LLM, so the model doesn't have to reason
about schema differences between engines.

> **In plain terms:** each engine is a different way of answering "what should this person see
> next?" —
> - **Collaborative filtering** = "people similar to you liked these things."
> - **Content-based filtering** = "these items are similar, in category/description, to things
>   you already liked."
> - **Popularity** = "here's what's broadly well-liked," used when we don't know enough about
>   the person to personalize.
>
> Whichever one runs, it's not allowed to crash the whole request — if something goes wrong
> (unknown user, missing file, empty result), it just reports "couldn't do it" and hands control
> back to the LLM instead of blowing up the whole answer.

---

## 5. Step 4 — the reply is assembled

The LLM itself produces the final object; Pydantic AI validates it against `AgentResponse`
(`models.py:115-131`) before `recommend()` returns it:

```python
class AgentResponse:
    answer: str                     # 1-2 sentence natural-language summary
    user_id: str | None
    engine_used: str | None         # which tool's result this reply is based on
    items: list[RecommendedItem]    # copied verbatim from the tool result
```

`agents.py:129` pulls `result.output` out of the Pydantic AI run result and hands it back to
whatever called `recommend()` — today, that's the CLI's `_print_response`; later, an HTTP
handler.

---

## 6. What's *not* in this flow

- **No conversation memory.** Every `recommend(query)` call is independent — no chat history is
  threaded between calls, and `user_id` must be present in the query text itself (there's no
  session/auth layer yet).
- **No custom routing code beyond `analyze_user`.** There's no separate classifier or graph —
  the engine choice is made by the LLM reading `SYSTEM_PROMPT` + the tool result, and that's the
  whole "graph."
- **Two CLI modes skip the agent entirely** — useful for offline testing since they need no API
  key: `--mode analyze` calls `analyze_user_profile()` directly (`main.py:142`), `--mode popular`
  calls `get_popular_items()` directly (`main.py:146`). Only `--mode query` touches the LLM.
- **Caching, not state.** The only thing that persists across calls is expensive-resource
  caching (the built agent, the loaded dataframes, the model artefacts) — not anything about the
  conversation. See `resources.py:1-32` and the `DataLoader`/`ModelsLoader` singletons in the
  collaborative/content-based `main.py` modules.
