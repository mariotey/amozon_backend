# Deployment Strategy

## 1. Decision

Deploy the orchestration layer (`src/orchestration`) — and by extension the two recommender
tools it wraps — as an **Azure Container App**, not an Azure Function. The existing `main.py`
handlers in `src/collaborative_filtering`, `src/content_based_filtering`, and
`src/orchestration` remain plain importable Python functions; only the hosting shell around them
changes (a FastAPI process instead of a Functions worker).

## 2. Why not Azure Functions

The choice is driven by the in-process singleton caching design documented in
`docs/ARCHITECTURE.md#10-inference-time-loading-into-cache`, not by a generic platform preference.

| Concern | Azure Functions (Consumption) | Azure Functions (Premium) | Container Apps |
|---|---|---|---|
| Cache lifetime | Tied to worker process; scales to zero aggressively → frequent cold starts | Pre-warmed instances keep at least one worker alive | `minReplicas: 1` keeps at least one instance alive indefinitely |
| Cold-start cost | Full forced Supabase reload of `user_item_df` (244K rows) + both tools' model artefacts, every cold start | Same reload logic, but rare in practice | Same reload logic, but only on redeploy/restart |
| Memory ceiling | ~1.5 GB per instance | Larger, but still capped per plan | Set explicitly per container (2–4 GB+), sized to fit both tools' caches simultaneously |
| Orchestration loads both tools | Both singleton caches + LLM client resident at once — tight on Consumption | Workable | Comfortable |
| Native deps (`implicit`/OpenBLAS, scikit-learn) | Functions Python worker environment fights manual thread-limiting (`OPENBLAS_NUM_THREADS`, per `docs/CB_CF_FILTERING.md#a7-deployment-note`) | Same constraint | Full Dockerfile control — set env vars, base image, BLAS backend directly |
| Execution timeout | 5 min default / 10 min max | Configurable, higher | None |
| LLM agent call (multi-step tool-calling loop) | Risk of hitting timeout under Consumption | Lower risk | No timeout risk |
| Fits existing declared-but-unwired FastAPI/uvicorn deps | No | No | Yes — this is exactly what they're for |

**Bottom line**: the whole point of the singleton cache is to avoid re-loading ~250K rows of
review data and two sets of model artefacts on every request. A platform that scales to zero
between requests defeats that design. Container Apps with `minReplicas: 1` (or more) keeps the
cache warm continuously, matching how the code was actually built to behave.

Azure Functions Premium is a legitimate fallback if invocation volume is low/spiky enough that
per-invocation billing matters more than warm-cache guarantees — pre-warmed Premium instances
give a similar "always-ready" effect. It is not the primary recommendation because it still
inherits the Functions packaging/timeout/native-dependency constraints for no offsetting benefit
here.

## 3. Target architecture

```
                         ┌─────────────────────────────┐
                         │      Azure Container App     │
                         │  (minReplicas: 1, autoscale)  │
                         │                               │
   HTTP request  ──────► │  FastAPI app (uvicorn)        │
                         │    └─ src/orchestration/main   │
                         │         ├─ agents.recommend()  │
                         │         ├─ analysis.py         │
                         │         └─ popular.py          │
                         │              │                 │
                         │   in-process singleton caches  │
                         │   (DataLoader, ModelsLoader ×2) │
                         └───────────────┬───────────────┘
                                         │  cold start / build only
                                         ▼
                         ┌─────────────────────────────┐
                         │           Supabase            │
                         │  Postgres (User/Item/Review)  │
                         │  Storage (ModelArtefacts)      │
                         └─────────────────────────────┘
```

- One container image, one FastAPI app, thin routes around the existing handler functions
  (`get_user_recommendations`, `get_similar_items`, `recommend`, `analyze_user_profile`,
  `get_popular_items`).
- No code changes required inside `src/collaborative_filtering`, `src/content_based_filtering`,
  or `src/orchestration` — the handlers are already plain functions designed to be called by
  "an Azure Function handler (or any other serverless runtime)."

## 4. Work required

1. **FastAPI wrapper** — new module (e.g. `src/api/main.py`) exposing routes that call the
   existing handlers directly; no business logic lives in the wrapper.
2. **Dockerfile** — base Python image, install via `uv`/`pyproject.toml`, set
   `OPENBLAS_NUM_THREADS=1` (per `docs/CB_CF_FILTERING.md#a7-deployment-note`), `CMD` runs
   `uvicorn`.
3. **Container Apps config** — `minReplicas: 1`, memory sized to comfortably hold both tools'
   deserialized caches (see `docs/ARCHITECTURE.md#105-rough-size-of-what-ends-up-in-memory-per-warm-instance`
   as a lower-bound reference), env vars for Supabase credentials via secrets, health probe
   hitting a lightweight route that doesn't force a cache load.
4. **Build/deploy pipeline** — image build → push to Azure Container Registry → `az containerapp
   update`. Model rebuilds (`build_model()`) stay an offline step; a new image/restart is what
   invalidates the in-process cache in production (mirrors the existing "`build_model()` resets
   both globals" behavior, just at the process level instead of within a single long-lived
   process).

## 5. Open questions

- Whether `analyze`/`popular` routes should be split into a separate, lighter-weight service
  (they don't need the LLM client) — deferred until real traffic/latency data exists.
- Autoscaling thresholds beyond `minReplicas: 1` (CPU/concurrency-based) — deferred until load
  characteristics are known.
