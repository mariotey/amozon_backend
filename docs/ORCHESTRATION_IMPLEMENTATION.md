# Orchestration Layer — Implementation Notes

> What was actually built for the agentic orchestration layer, the decisions taken along the way,
> and how to run it. The design rationale lives in `docs/AGENTIC_ORCHESTRATION.md`; this document
> records the implementation.
>
> **Status: code written, not yet executed.** Nothing in this document has been verified by a
> run — see §7.

---

## 1. Placement

The package lives at `src/orchestration/`, alongside the two recommender tools.

`docs/AGENTIC_ORCHESTRATION.md` §7 argued against `src/` on the grounds that `ModelsLoader`
discovers packages there. **That concern was wrong and is retracted.** `ModelsLoader` iterates the
explicit `config.TOOLS` list, not the `src/` directory listing
(`models_loader/models_loader.py:191-203`), so an unregistered package under `src/` is never
touched. `src/orchestration/` is safe for exactly as long as it stays out of `config.TOOLS`.

The tool-config contract in `AGENTS.md` reserves the filename `tool_config.py` for tools carrying
ML artefacts. The orchestration package owns no artefacts, so its config module is deliberately
named `orchestration_config.py` — it must not look like a registrable tool.

---

## 2. Module layout

| File | Responsibility |
|---|---|
| `orchestration_config.py` | Model id, API key, result-count bounds, tool names, ranking config |
| `models.py` | Pydantic schemas + `to_recommended_items()` normaliser |
| `prompt.py` | System instructions |
| `resources.py` | Lazily-cached datasets and the ALS user-id set |
| `analysis.py` | `analyze_user_profile()` — the routing verdict |
| `popular.py` | `get_popular_items()` — the popularity fallback |
| `tools.py` | The four agent-facing tools |
| `agents.py` | Model + agent construction, `recommend()` handler |
| `main.py` | CLI dispatcher |

Dependency direction is strictly one-way:
`main → agents → tools → {analysis, popular} → resources → data_loader / recommender tools`.
Nothing below `tools.py` imports Pydantic AI, so the entire recommendation substrate stays
importable and testable without an LLM.

---

## 3. The four tools

```python
analyze_user(user_id)                       # always called first
collaborative_recommendations(user_id, n)
content_based_recommendations(user_id, n)
popular_items(n)
```

Registered on the agent via the `tools=[...]` keyword rather than `@agent.tool` decorators, which
keeps `tools.py` free of any agent import and independently callable.

They are registered as **synchronous** functions. Pydantic AI runs sync tools in a worker thread,
which resolves §5.5 of the design doc — the content-based cosine similarity over the full item
matrix never touches the event loop — without explicit `anyio.to_thread` offloading.

### 3.1 Wrappers, not logic

`collaborative_recommendations` and `content_based_recommendations` add no recommendation logic.
They call the existing handlers (`src/<tool>/main.py: get_user_recommendations`) and exist to do
three things those handlers deliberately do not:

1. **Convert failures into data.** An uncaught exception inside a tool aborts the agent run. CF
   raises `ValueError` for an unknown user; both engines raise `FileNotFoundError`/`OSError` for
   missing artefacts. All become `status="unavailable"` with a `reason`, so the agent can fall
   back to another engine.
2. **Normalise the schema.** CF emits no `score`, CBF does. Both are returned through one
   `RecommendationResult` envelope.
3. **Bound `n`** to `[1, MAX_TOP_N]`, so a malformed query cannot request an unbounded result set.

CBF signals "no qualifying history" by returning `[]` rather than raising, so an empty list is
mapped to `unavailable` too — otherwise the agent would report success with zero items.

`analyze_user` is the one tool that can raise: a malformed `user_id` becomes `ModelRetry`, which
asks the model for a corrected call instead of failing the run.

---

## 4. `analyze_user` returns a verdict

As specified in the design doc, `analyze_user_profile()` returns `available_engines` rather than
raw statistics, so the rating threshold never has to be described to the model in a prompt.

Thresholds are **imported from `src/content_based_filtering/tool_config.py`**
(`MIN_RATING_THRESHOLD`, `FREE_PREFERENCE_THRESHOLD`) rather than redefined, so a change there
cannot silently desynchronise routing from the engine's real behaviour.

Availability rules as implemented:

| Condition | Result |
|---|---|
| `user_id` in `idx_to_userid_mapping.json` values | offer `collaborative_recommendations` |
| ≥ 1 review rated `>= MIN_RATING_THRESHOLD` | offer `content_based_recommendations` |
| always | offer `popular_items` |

**Known imprecision.** The content-based rule is necessary but not strictly sufficient: the engine
also requires the rated item to be present in the TF-IDF index (`cb_meta.parquet`). Checking that
here would mean loading the full content-based artefacts just to run an analysis, so the residual
case is absorbed downstream — the wrapper reports an empty result as `unavailable` and the agent
falls back. This is a deliberate trade, documented in the function's docstring.

The ALS membership check reads the artefact tuple by a position derived from
`MODEL_ARTEFACTS` key order rather than a hard-coded index, so reordering that dict cannot
silently break routing.

---

## 5. Decisions taken

The design doc left three decisions open (§8). All three were settled to keep the build moving;
each is cheap to revisit.

### 5.1 Model provider — OpenRouter

Configured from two environment variables, read in `orchestration_config.py`:

```
MODEL_NAME=tencent/hy3
API_KEY=<your OpenRouter key>
```

Both live in `.env`, which is gitignored (`.gitignore:9`). Pydantic AI has first-class OpenRouter
support, so this uses `OpenRouterModel` + `OpenRouterProvider` rather than an OpenAI base-URL
override.

> **Note on naming:** `MODEL_NAME` and `API_KEY` are generic for a repo that also holds Supabase
> credentials, and `MODEL_NAME` collides conceptually with the `MODEL_NAME` constant in each
> tool's `tool_config.py`. Worth renaming to `ORCHESTRATION_MODEL` / `OPENROUTER_API_KEY` if this
> grows.

The model is constructed **lazily** inside `get_agent()`. Importing any orchestration module —
including `main.py` — resolves no model and reads no key, which is what keeps the offline modes
in §6 usable with no credentials at all.

### 5.2 `popular_items` ranking

Sort by `quality_score` descending, tie-broken by `num_reviews` descending, after excluding the
`cold_start` popularity segment — `quality_score` is a Wilson lower bound and is not meaningful
for products with too few reviews. If that filter leaves fewer than `n` items, the unfiltered
ranking is used so the caller always gets a full result set.

No new artefacts were needed: `item.parquet` already carries both columns.

### 5.3 Response shape — structured

`output_type=AgentResponse`, carrying `answer`, `user_id`, `engine_used` and `items`. This is a
backend, so a validated envelope beats prose; `engine_used` also makes the routing decision
observable from the response alone.

---

## 6. Running it

```powershell
# Full agent — requires API_KEY
python -m src.orchestration.main --mode query --query "give me 5 recommendations for user ag6hllxrsby3efcfgqgjxvjabvfq"

# Routing verdict for one user — fully offline, no API key
python -m src.orchestration.main --mode analyze --user_id ag6hllxrsby3efcfgqgjxvjabvfq

# Popularity fallback — fully offline, no API key
python -m src.orchestration.main --mode popular --n 10
```

`--mode analyze` and `--mode popular` exist specifically to preserve the repo's offline testing
culture (`docs/BUILD_TEST.md`). They exercise the analysis, availability and fallback logic — the
whole routing substrate — with no LLM and no network. Only `--mode query` contacts a provider.

Prerequisite: both models must already be built, since `analyze_user` reads the CF user mapping.

```powershell
python -m src.collaborative_filtering.main --mode build
python -m src.content_based_filtering.main --mode build
```

### Demo users

| user_id | Expected `available_engines` |
|---|---|
| `ag6hllxrsby3efcfgqgjxvjabvfq` | collaborative + content-based + popular |
| `agci7fah4gl5fi65hylkwtmfz2cq` | popular only (1 review rated 1.0) |

The second is the cold-start case and the reason `popular_items` exists.

---

## 7. Not yet verified

No module in `src/orchestration/` has been executed. Specifically unverified:

- Every import resolves, and `pydantic_ai.models.openrouter` / `pydantic_ai.providers.openrouter`
  exist at the installed version.
- `to_recommended_items()` correctly unboxes the NumPy scalars and missing values that come out of
  `DataFrame.to_dict(orient="records")` for both engines.
- The `_UID_MAPPING_POSITION` lookup returns the mapping and not another artefact.
- `popular_items` output — whether the `cold_start` exclusion leaves a sensible catalogue.
- Whether `tencent/hy3` reliably calls `analyze_user` first and honours `available_engines`.

The offline modes are the cheapest way to close most of this; the last item needs a live run.

---

## 8. Follow-ups

- Add a `tests/` harness using Pydantic AI's `TestModel` / `FunctionModel` to assert tool-call
  order and routing with zero API calls.
- Rotate the OpenRouter key if it has been pasted anywhere outside `.env`.
- Register nothing in `config.TOOLS` for this package — see §1.
- If this moves behind an authenticated HTTP API, take `user_id` from the session rather than
  from the query text (`docs/AGENTIC_ORCHESTRATION.md` §7).
