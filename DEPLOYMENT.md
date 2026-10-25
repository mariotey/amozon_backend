# Technical Guide

For the business overview, see [README.md](README.md).

## Local setup

Use Python 3.13 and run commands from the repository root:

```bash
pip install uv
uv sync --frozen
```

Create a local `.env` file with these values, or set them in your environment:

```env
SUPABASE_URL=https://YOUR-PROJECT.supabase.co
SUPBASE_SECRET_KEY=YOUR_SERVER_SIDE_SECRET_KEY
```

Keep the existing spelling `SUPBASE_SECRET_KEY`; this is what the code reads.
`SUPABASE_PUBLIC_KEY` is not used by the backend. Keep credentials server-side
and out of Git.

```bash
uv run uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs once the server starts, including while models load. `/redoc` and
`/openapi.json` provide alternative documentation and the API specification.

## Required data and models

Supabase must contain the prepared `User`, `Item`, and `Review` tables, the
`ModelRegistry` table, and the `ModelArtefacts` storage bucket. Table schemas
must match the outputs of `data_loader/data_transform.py`, including shopper
aggregates, product metadata, review text, and review dates.

Each engine's `tool_config.py` defines its model ID and required artifact files.
The registry supplies storage paths; missing configured versions fall back to
the latest registered version. Keep datasets and models compatible.

Startup schedules exactly one background model initializer. It loads trained
artifacts from local caches, downloading missing model files from Supabase.
It does not instantiate the bulk `DataLoader` or extract complete User, Item,
or Review tables. Browsing handlers query Supabase directly, independently of
models. `data_loader`, `setup.py`, training commands, and default standalone
engine handlers keep their full-data/offline behavior; do not run `setup.py`
as an API startup command.

The saved ALS model, complete user-item matrix, and both ID maps remain together.
The content model keeps its complete TF-IDF feature matrix, vectorizer, and
row-aligned metadata for all candidates, including products not reviewed by the
selected shopper. Startup validates matrix/factor/mapping/vocabulary dimensions,
identifier uniqueness, schemas, and finite weights without querying serving
tables. This detects structural mismatch, not training provenance; publish
compatible artifact sets and prepared table updates through your existing process.
No model retraining, image packaging, or automatic release refresh is added.

## Required database setup (manual; not applied)

Review [`sql/001_on_demand_serving.sql`](sql/001_on_demand_serving.sql) against
your actual Supabase schema and existing indexes, then apply it yourself as the
database owner before using the random-shopper endpoint. This implementation
has not modified any live resource. The SQL file was reviewed against PostgreSQL
syntax and Supabase's documented function permissions, but was not executed
against a PostgreSQL server in this workspace.

The file adds `public.random_shoppers(sample_size integer default 5)`. It first
selects **distinct non-null Review.user_id values**, orders them by `random()`,
then limits the result to 1–100 requested IDs. This preserves the current checkout's
eligibility rule: at least one review, regardless of rating, model membership, or
presence in User aggregates. It does not download every ID into Python, sample
from only the first page, or treat `LIMIT` alone as random sampling. Sampling
still requires database-side deduplication and random ordering; it is not a
constant-cost operation. Measure its query plan on your actual dataset.

The function is `SECURITY INVOKER`, uses an empty search path with qualified
relations/functions, validates the limit, revokes execute from PUBLIC, anon,
and authenticated, and grants execute only to `service_role`. Use the existing
server-side Supabase credential. It also needs SELECT on User/Item/Review and
ModelRegistry plus read access to ModelArtefacts. If you use a dedicated backend
role, explicitly grant RPC execute and the corresponding table/RLS access to
that role instead; do not expose server credentials or open the RPC to browsers.

The proposed indexes match the query patterns:

| Index | Purpose |
| --- | --- |
| Unique Review `(user_id, review_id)` including `parent_asin, review_rating, recency_weight` | Selected-shopper history keyset paging and eligible-ID lookup. |
| Review `(parent_asin, review_date DESC NULLS LAST, review_id ASC)` | Filtered review pages with the existing ordering. |
| Item `(num_reviews DESC NULLS LAST, avg_rating DESC NULLS LAST, parent_asin ASC NULLS LAST)` | Ordered popular-product limits. |
| Unique Item `(parent_asin)` and User `(user_id)` | Direct links, profiles, and bounded recommendation-card batches. |

Keep equivalent existing primary-key/unique indexes rather than creating
duplicates under different names. Review IDs must be non-null and unique within
each shopper for complete keyset paging; the SQL aborts on null IDs and its
unique index fails on duplicates. The existing transformation generates unique
text review IDs. The normal index builds are transactional and can block writes:
review them and choose a maintenance window. On larger live datasets, have your
database operator adapt index creation to a separately reviewed concurrent build
outside a transaction. No new infrastructure is required.

Ensure the Supabase Data API maximum response rows setting is at least `100`
(the normal default is higher), so bounded popular-product, random-shopper, and
review-page requests support their existing maximum limits. `HISTORY_PAGE_SIZE`
defaults to `1000` and accepts `1`–`1000`. History/card paging also handles a
server cap below the requested batch size. Do not raise the response cap to
download entire tables. The RPC signature cache is reloaded by the SQL file.

References: [Supabase database functions and permissions](https://supabase.com/docs/guides/database/functions),
[Python ordering](https://supabase.com/docs/reference/python/order), and
[PostgreSQL index ordering](https://www.postgresql.org/docs/current/indexes-ordering.html).

## Request flow, readiness, and errors

The existing frontend may request five random shoppers and five popular products
independently, select a shopper, and call the recommendation endpoint. No session,
initial-sample allowlist, or global selected shopper is stored by the backend.
Profiles and direct product URLs also work without making those initial calls.
Frontend automatic selection/retry behavior is outside this backend-only change.

For recommendations, both models must be ready; a missing/loading/failed model
does not become a successful empty result or trigger a different engine. The
existing CF-first decision then applies:

1. A mapped user is offered collaborative results from the saved ALS factors,
   saved interaction row, and product mapping. Seen exclusion uses that compatible
   saved row. Successful CF results do not fetch any raw Review history.
2. If CF is ineligible or successfully returns no usable products, retrieve only
   this shopper's history for content eligibility/fallback. Select `review_id`
   for paging and `user_id,parent_asin,review_rating,recency_weight` for inference.
   Keyset queries are filtered by user_id, ordered by review_id, and continue
   after the last returned ID until an empty page, even if an earlier page was
   shorter than requested. History is never truncated at the API response limit.
3. Pass the complete request-local history explicitly to content inference.
   Low ratings are retained for seen-item exclusion. Existing positive-rating
   eligibility, rating/recency weighting, category boost, free preference, and
   full candidate coverage are unchanged. There is no shared history swap.
4. Fetch card metadata for returned IDs in a bounded `.in_()` batch, with paging
   if necessary. Preserve engine order and scores regardless of database row order.
   CF filters unavailable catalogue IDs as before; content cards retain engine
   fields and null metadata when a product is absent, as the previous enrichment did.

Supabase operations and inference run in FastAPI's synchronous handler worker
threads. A single managed `asyncio.to_thread` task initializes the models off the
event loop. An inference lock limits concurrent engine computations; it is
released before collaborative metadata queries. Shopper-history queries can
overlap without sharing their DataFrames. There are no process-wide
shopper/history/product caches or unbounded memoization. Request data is released
after the request; a prolific shopper can still require a large history and many
queries. Matrices/mappings/content metadata remain shared model resources.
Model validation also builds a fixed user-ID lookup and content-item index maps
once per loaded model set. Requests reuse these bounded dictionaries instead of
scanning user mappings or rebuilding full item indexes. They share existing
identifier strings and do not grow with requests or cache live shopper/product data.
Separate database queries do not constitute one transaction-wide snapshot, so
concurrent edits can affect results. Keep live metadata/history compatible with
published model coverage; models do not learn from new data automatically.

`/health` keeps `{"status":"ok"}` as process liveness while models load or fail.
`/ready` returns HTTP 503 until models are ready, then HTTP 200, with a response
such as `{"status":"ready","resources":{"models":"ready","serving_data":"on_demand"}}`.
Neither performs a database health probe or validates that the RPC is installed.
Model states are `queued`, `loading`, `ready`, `error`, or `stopped`; overall
readiness is `loading`, `ready`, `degraded`, or `stopping`.

Loading recommendation requests return 503 with `Retry-After: 5`. Failed model
initialization returns 503 with `Required resources failed to load`; browsing
remains usable and a restart is required after fixing the model issue. A failed
on-demand Supabase query, including a missing/unauthorized RPC, returns 503 with
`{"detail":"Supabase request failed"}` and `Retry-After: 5`; it never masquerades
as an unknown product or an empty recommendation. Existing 404s, validation 422s,
successful empty results, and unexpected inference 500s remain distinct. Do not
retry 404/422 responses as loading failures.

Shutdown signals and awaits the model task before dropping references. It checks
between artifact downloads/loads; in-flight SDK requests or native deserialization
finish first. SDK timeouts bound individual network waits, not total shutdown.
No job retry loop, automatic publication, or model version refresh is introduced.

## API endpoints

All endpoints use `GET`. Replace path placeholders with actual IDs.

| Path (with example parameters) | Returns |
| --- | --- |
| `/health` | `{ "status": "ok" }` once the server starts; liveness only. |
| `/ready` | Model readiness; 200 when ready, otherwise 503. Serving data is `on_demand`. |
| `/api/v1/randomusers?n=5` | At most n distinct random `{ "user_id": "..." }` shoppers selected in Supabase from review history. |
| `/api/v1/users/{user_id}` | Prepared shopper profile: rating/review statistics, activity, segment, and purchase statistics. |
| `/api/v1/users/{user_id}/history?verified_only=true&page=1&page_size=20` | Paginated verified purchase reviews with product metadata; works independently of models. |
| `/api/v1/users/{user_id}/recommendations?n=8` | `user_id`, `engine_used`, `reason`, and `items`. Default `n` is 10. |
| `/api/v1/popular_items?limit=5` | Product cards ordered by review count, then average rating, descending; product ID breaks ties. |
| `/api/v1/items/{parent_asin}` | Product card plus store, popularity segment, quality score, and description. |
| `/api/v1/items/{parent_asin}/reviews?page=1&page_size=10` | Newest reviews first; `items`, `page`, `page_size`, `total_count`, and `page_count`. |

Product cards contain `parent_asin`, `item_title`, `main_category`, `price`,
`is_free`, `avg_rating`, `num_reviews`, and `item_images`. Recommendation items may also include
`score`. Unknown prices use `-1`; missing metadata uses JSON `null`.

`item_images` is returned as an object preserving Amazon's parallel arrays
(`variant`, `large`, `hi_res`, `thumb`). Stored JSON strings are decoded only
for the API response; existing objects are preserved. Missing, malformed, or
non-object values return `null`. Product details, popular products, both
recommendation paths, and each history entry's nested `item` include this field.
No new database columns, upload changes, or data migration are needed. Frontend
rendering must select an image URL and handle missing images separately.

Recommendations try collaborative filtering first, then content-based filtering
when a review rated at least 3 overlaps the content model. If neither produces
items, HTTP 200 contains `items: null` and `engine_used: null`. Unexpected
model/data errors propagate as server errors.

Shopper history defaults to `verified_only=true`; use `false` to include all
reviews. Results are filtered in Supabase by `user_id` and, when requested,
`verified_purchase=true`, ordered by `review_date` descending (null dates last)
then `review_id` ascending. It returns `user_id`, `verified_only`, `items`,
`page`, `page_size`, `total_count`, and `page_count`. Each entry contains the
review fields and an `item` object with product card fields (or `null` if the
product is missing). Counts refer to matching reviews, not distinct products
or orders. Repeated reviews of a product remain separate records. Unknown users,
users without verified reviews, and pages beyond the end return HTTP 200 with
empty `items`; no User aggregate or model membership is required. Database
failures return 503, including failures fetching product metadata.

This is review-based history from the configured dataset slice, not complete
order history. `review_date` is not a purchase date, and Item prices are not
historical prices paid. No order IDs, quantities, or unreviewed purchases are
available. The endpoint follows the existing public demo shopper-ID access
model. No database migration or live database update is required by this route.

`n`, `limit`, and `page_size` accept 1–100; `page` starts at 1. Invalid inputs
return 422. Unknown shopper profiles and products return 404. Review pages beyond
the last page return an empty array with the actual totals. Random shoppers may
return fewer than requested when the available population is smaller.

The popular-products URL is `/api/v1/popular_items`, replacing
`/api/v1/items/popular`. Update frontend callers accordingly.

FastAPI also exposes GET `/openapi.json`, `/docs`, `/redoc`, and the Swagger helper
`/docs/oauth2-redirect`. No engine-specific HTTP routes or homepage are added.
Defaults are `n=5` for random shoppers, `limit=5` for popular products, `n=10`
for recommendations, and `page=1,page_size=10` for reviews. All existing business
endpoint URLs, supported bounds, and successful response fields are retained.

## Render deployment

1. Connect the repository as a Docker web service using the root `Dockerfile`.
2. Add `SUPABASE_URL` and `SUPBASE_SECRET_KEY` as service environment variables.
3. Manually review/apply the SQL setup above and confirm RPC/table permissions.
4. Keep the health-check path at `/health`; use the Dockerfile's default command
   with one Uvicorn process. Using `/ready` as the traffic gate would delay early
   browsing until models finish.
5. Check `/health`, `/docs`, `/api/v1/randomusers?n=5`, and
   `/api/v1/popular_items?limit=5` independently of `/ready`. Recommendation
   requests may temporarily return 503 while models load.

The server binds to `0.0.0.0` using `PORT`, defaulting to `10000`. The Docker
image excludes `.env`, local datasets, and models, so fresh instances need
Supabase access for models and on-demand queries. Runtime model caches can be
lost on spin-down, restart, or redeploy; a fresh instance may download models
again. Python imports, model downloads/deserialization/alignment validation,
and Render provisioning remain startup costs. No measured speedup is claimed.

Logs record shopper selection, popular-product queries, model readiness,
history retrieval (including row/page counts), and recommendation generation
timings. They omit credentials and upstream exception bodies. Uvicorn's
`Application startup complete` now means background loading was scheduled;
use `Model readiness ready after ...` for recommendation availability. These
timers exclude earlier Python imports/container provisioning. Monitor database
query costs, concurrent history memory, and instance RAM on your deployment.

For a local Docker check:

```bash
docker build -t amozon-backend .
docker run --rm --env-file .env -p 10000:10000 amozon-backend
```

## Maintenance and tests

| Location | Responsibility |
| --- | --- |
| `app.py` | On-demand HTTP routes, shared model initialization, readiness, and shutdown. |
| `utils/supabase_utils.py` | Supabase helpers, bounded serving queries, history paging, metadata batches, and timings. |
| `models_loader/models_loader.py` | Artifact loading, alignment checks, and reusable model lookup indexes. |
| `sql/001_on_demand_serving.sql` | Manual random-shopper RPC, permissions, and proposed query indexes. |
| `src/routing.py` | Recommendation selection and null fallback. |
| `src/collaborative_filtering/`, `src/content_based_filtering/` | Model training and recommendation logic. |
| `data_loader/` | Offline data preparation and full-data caches for training workflows. |
| `config.py` | Paths, resource names, and enabled engines. |
| `tests/` | Routing, API contracts, and engine regression checks. |
| `setup.py` | Local directory/cache bootstrap; executing it downloads missing resources. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependency declarations, locked versions, and Python version. |
| `Dockerfile`, `.dockerignore` | Container build/start command and excluded build inputs. |
| `.gitignore` | Excludes local secrets, datasets, models, bytecode, tool caches, and build output. |
| `notebook/` | Exploratory notebooks; separate from the deployed API. |

Generated `data/`, `models/`, virtual environments, Python bytecode, notebook
checkpoints, and common test/type-check/lint caches are ignored by Git. Keep
`uv.lock` tracked for reproducible dependency installation. `.gitignore` does
not remove files already tracked; use `git ls-files -ci --exclude-standard` to
check for tracked files that now match ignore rules.

Run the automated tests:

```bash
uv run python -m unittest discover -s tests -v
```

Tests use fake bounded database responses and real engine computations with small
matrices. They cover model-only startup, browsing during loading/failure, scoped
and fully paginated histories under lower server limits, concurrent shopper
isolation, ranked metadata enrichment, existing rating/recency/category/free
rules, endpoint contracts, and error distinctions. No tests contact live Supabase.

Training commands (write model files locally):

```bash
uv run python -m src.collaborative_filtering.main --mode build
uv run python -m src.content_based_filtering.main --mode build
```

Training does not publish models automatically. Data preparation/upload scripts
and `models_loader/push_into_supabase.py` handle publication separately. Refresh
compatible model caches and restart to use updated trained artifacts; restarting
alone reuses existing model files. Serving queries read prepared table changes
on subsequent requests without a bulk-data cache refresh. Offline training/data
tools retain their original full-cache policy.

For model initialization failures, inspect `/ready` and model logs, then check
Supabase access and registered artifact alignment. For browsing failures, check
table/RPC schemas, permissions, indexes, and query errors locally. For a 404,
compare the frontend URL with `/openapi.json` and confirm the
latest revision was deployed. A 404 at `/` is expected: there is no homepage.
