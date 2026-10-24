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

Open http://127.0.0.1:8000/docs after startup completes. `/redoc` and
`/openapi.json` provide alternative documentation and the API specification.

## Required data and models

Supabase must contain the prepared `User`, `Item`, and `Review` tables, the
`ModelRegistry` table, and the `ModelArtefacts` storage bucket. Table schemas
must match the outputs of `data_loader/data_transform.py`, including shopper
aggregates, product metadata, review text, and review dates.

Each engine's `tool_config.py` defines its model ID and required artifact files.
The registry supplies storage paths; missing configured versions fall back to
the latest registered version. Keep datasets and models compatible.

Startup loads product metadata, review interactions, and both models from local
caches, downloading missing files from Supabase. Missing resources prevent
startup. User profiles and paginated reviews query Supabase on each request.

## API endpoints

All endpoints use `GET`. Replace path placeholders with actual IDs.

| Path (with example parameters) | Returns |
| --- | --- |
| `/health` | `{ "status": "ok" }` after startup succeeds. |
| `/api/v1/randomusers?n=5` | Array of unique random `{ "user_id": "..." }` shoppers from loaded review history. |
| `/api/v1/users/{user_id}` | Prepared shopper profile: rating/review statistics, activity, segment, and purchase statistics. |
| `/api/v1/users/{user_id}/recommendations?n=8` | `user_id`, `engine_used`, `reason`, and `items`. Default `n` is 10. |
| `/api/v1/popular_items?limit=5` | Product cards ordered by review count, then average rating, descending; product ID breaks ties. |
| `/api/v1/items/{parent_asin}` | Product card plus store, popularity segment, quality score, and description. |
| `/api/v1/items/{parent_asin}/reviews?page=1&page_size=10` | Newest reviews first; `items`, `page`, `page_size`, `total_count`, and `page_count`. |

Product cards contain `parent_asin`, `item_title`, `main_category`, `price`,
`is_free`, `avg_rating`, and `num_reviews`. Recommendation items may also include
`score`. Unknown prices use `-1`; missing metadata uses JSON `null`.

Recommendations try collaborative filtering first, then content-based filtering
when a review rated at least 3 overlaps the content model. If neither produces
items, HTTP 200 contains `items: null` and `engine_used: null`. Unexpected
model/data errors propagate as server errors.

`n`, `limit`, and `page_size` accept 1?100; `page` starts at 1. Invalid inputs
return 422. Unknown shopper profiles and products return 404. Review pages beyond
the last page return an empty array with the actual totals. Random shoppers may
return fewer than requested when the available population is smaller.

The popular-products URL is `/api/v1/popular_items`, replacing
`/api/v1/items/popular`. Update frontend callers accordingly.

## Render deployment

1. Connect the repository as a Docker web service using the root `Dockerfile`.
2. Add `SUPABASE_URL` and `SUPBASE_SECRET_KEY` as service environment variables.
3. Set the health-check path to `/health`; use the Dockerfile's default command.
4. Deploy, wait for startup to complete, then check `/health`, `/docs`, and
   `/api/v1/popular_items?limit=5` on the deployed URL.

The server binds to `0.0.0.0` using `PORT`, defaulting to `10000`. The Docker
image excludes `.env`, local datasets, and models, so fresh instances need
Supabase access. `/health` checks responsiveness, not live database connectivity.

For a local Docker check:

```bash
docker build -t amozon-backend .
docker run --rm --env-file .env -p 10000:10000 amozon-backend
```

## Maintenance and tests

| Location | Responsibility |
| --- | --- |
| `app.py` | HTTP routes and shared startup resources. |
| `src/routing.py` | Recommendation selection and null fallback. |
| `src/collaborative_filtering/`, `src/content_based_filtering/` | Model training and recommendation logic. |
| `data_loader/`, `models_loader/`, `utils/supabase_utils.py` | Data preparation, caching, uploads, and Supabase access. |
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

Current test caveat: `tests/test_routing.py` still references the previous
`/api/v1/items/popular` path. Those assertions need updating to match the renamed
endpoint; documentation updates do not change test code.

Training commands (write model files locally):

```bash
uv run python -m src.collaborative_filtering.main --mode build
uv run python -m src.content_based_filtering.main --mode build
```

Training does not publish models automatically. Data preparation/upload scripts
and `models_loader/push_into_supabase.py` handle publication separately. Refresh
compatible caches and restart the service to serve updated data/models; restarting
alone reuses existing local caches.

For startup failures, check Supabase access, table schemas, and registered model
files. For a 404, compare the frontend URL with `/openapi.json` and confirm the
latest revision was deployed. A 404 at `/` is expected: there is no homepage.
