# Technical Setup, Deployment, and Testing

For the project overview, see [README.md](README.md).

The service exposes one recommendation endpoint and one health check. FastAPI's
`/docs`, `/redoc`, and `/openapi.json` provide generated API documentation.
Individual engines remain callable in Python and through their CLI commands, but
are not separate HTTP endpoints.

## Recommendation routing

`GET /api/v1/users/{user_id}/recommendations?n=10` uses this fixed order:

1. Try collaborative filtering if the user exists in the trained ALS mapping.
2. If no items were returned or the user is absent from ALS, try content-based
   filtering when at least one review rated 3 or above overlaps the TF-IDF index.
3. Otherwise, return popular unseen items ranked by `quality_score` descending,
   `num_reviews` descending, and `parent_asin` ascending to break ties.

The first eligible engine producing items wins. There is no LLM, blended ranking,
or arbitrary minimum-review threshold. Unexpected data/model failures propagate
as server errors instead of silently triggering fallback. Routing and results
use the data/model snapshot loaded at startup under the existing cache policy.

## Project files

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI routes and shared startup resources |
| `src/routing.py` | Deterministic CF-first routing and popularity fallback |
| `Dockerfile` | Python 3.13 environment, dependencies, and server startup |
| `.dockerignore` | Excludes credentials, local environments, datasets, and models from the image |
| `src/collaborative_filtering/` | ALS training and recommendations |
| `src/content_based_filtering/` | TF-IDF training, recommendations, and item similarity |
| `data_loader/`, `models_loader/` | Local caches and Supabase downloads |
| `utils/supabase_utils.py` | Supabase queries and storage helpers |
| `config.py` | Paths, table names, bucket name, and enabled tools |
| `pyproject.toml`, `uv.lock` | Dependencies and locked versions |

## Before you start

You need a Supabase project containing the processed datasets and trained model
artefacts. Local `data/` and `models/` folders are excluded from the Docker image,
so Render must be able to download their equivalents from Supabase.

| Supabase resource | Required contents |
| --- | --- |
| `Item` table | `parent_asin`, `item_title`, `main_category`, `is_free`, `quality_score`, `num_reviews` |
| `Review` table | `user_id`, `parent_asin`, `review_rating`, `recency_weight` |
| `ModelRegistry` table | Model entries with `tool`, `model_id`, `storage_path`, and `created_at` for latest-version fallback |
| `ModelArtefacts` bucket | All files listed in each tool's `tool_config.py`, under the corresponding registry storage path |

The tools are named `collaborative_filtering` and `content_based_filtering`.
Use matching model artefacts and datasets. Each tool requests its configured
`MODEL_ID`; if the registry has no matching entry, the loader tries the latest
registered version for that tool. Missing or inaccessible resources prevent startup.

The application uses two environment variables:

| Variable | Value |
| --- | --- |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPBASE_SECRET_KEY` | Your server-side Supabase service-role key |

`SUPBASE_SECRET_KEY` intentionally matches the existing spelling in the code.
`SUPABASE_PUBLIC_KEY` is not used. Keep the service-role key private.

## Run locally

Use Python 3.13 and run commands from the repository root.

1. Install uv and the locked dependencies:

   ```bash
   pip install uv
   uv sync --frozen
   ```

2. For local development, create a `.env` file in the repository root:

   ```env
   SUPABASE_URL=https://YOUR-PROJECT.supabase.co
   SUPBASE_SECRET_KEY=YOUR_SERVICE_ROLE_KEY
   ```

   Alternatively, set these variables in your terminal environment. `.env` is
   optional and is excluded from Git and Docker. Render uses dashboard settings.

3. Start the API:

   ```bash
   uv run uvicorn app:app --reload
   ```

4. Wait for `Application startup complete`, then open http://127.0.0.1:8000/docs.
   Missing data and models are downloaded during startup.

To test the Docker image locally instead, start Docker Desktop and run:

```bash
docker build -t amozon-backend .
docker run --rm --memory=512m --cpus=0.1 --env-file .env -p 10000:10000 amozon-backend
```

Open http://localhost:10000/docs after startup completes. This uses Supabase
downloads rather than your local dataset/model folders, like a fresh Render
instance. The limits approximate Render Free's memory and CPU allocation.
Stop the container with Ctrl+C.

## Deploy on Render

1. Commit and push all application changes to your Git repository, including
   `app.py`, `Dockerfile`, `.dockerignore`, the loader changes, `pyproject.toml`,
   and `uv.lock`. Do not commit `.env`, local datasets, or model artefacts.
2. Sign in to [Render](https://dashboard.render.com) and select **New > Web Service**.
3. Connect your Git provider, select this repository, and choose the branch
   containing your changes.
4. Configure the service:

   | Setting | Value |
   | --- | --- |
   | Name | For example, `amozon-backend` |
   | Language | **Docker** |
   | Root Directory | Leave empty |
   | Dockerfile Path | `./Dockerfile` |
   | Instance Type | **Free** |
   | Docker Command | Leave empty to use the Dockerfile's CMD |
   | Health Check Path | `/health` (under Advanced) |

5. Add `SUPABASE_URL` and `SUPBASE_SECRET_KEY` under **Environment Variables**.
   Use the values from your Supabase project; do not upload your `.env` file.
6. Select **Create Web Service**. Do not attach a paid disk or add another service.
   Render builds the image from the Dockerfile; no separate build/start commands
   or manual `PORT` setting are needed. The Dockerfile binds to `0.0.0.0` and
   uses Render's `PORT`, defaulting to 10000 locally.
7. Watch the deployment logs. During the build, expect dependency installation.
   During startup, expect Supabase downloads, followed by:

   ```text
   Application startup complete.
   Uvicorn running on http://0.0.0.0:10000
   ```

   The displayed port may differ if Render sets a different `PORT`.
8. When the service becomes **Live**, copy its assigned `https://...onrender.com`
   URL. Use that URL as `BASE_URL` in the tests below.

See [Render's Docker instructions](https://render.com/docs/docker) for the
platform's deployment settings.

## Test the deployed API

### 1. Check health and API documentation

Replace `BASE_URL` with your assigned service URL, including `https://`.
Open `BASE_URL/health`. Expected HTTP status: **200**.

```json
{"status": "ok"}
```

Then open `BASE_URL/docs`. Swagger UI should list only `/health` and
`/api/v1/users/{user_id}/recommendations`. Health becomes available after startup
loading succeeds; it does not continually check Supabase connectivity. Visiting
`/` returns 404 because no homepage is defined.

### 2. Choose test users

- **Collaborative:** a user from the deployed ALS model's `idx_to_userid_mapping.json`.
- **Content-based:** a user absent from that mapping, with at least one review
  rated 3 or above for an item present in the content model's `cb_meta.parquet`.
- **Popularity:** an unknown user ID, such as `__new_test_user__`.

If local and deployed models match, print a known ALS user with:

```bash
uv run python -c "import json; from pathlib import Path; print(next(iter(json.loads(Path('models/collaborative_filtering/idx_to_userid_mapping.json').read_text()).values())))"
```

An ALS user with positive reviews still gets collaborative results first. A user
merely existing in the Review table is not necessarily present in the ALS model.

### 3. Request recommendations

In `/docs`, expand **GET /api/v1/users/{user_id}/recommendations**, select
**Try it out**, enter a test user ID and `n=3`, then select **Execute**.
Alternatively, open this URL after replacing the hostname and `USER_ID`:

```text
https://YOUR-SERVICE.onrender.com/api/v1/users/USER_ID/recommendations?n=3
```

Expected HTTP status: **200**. Example response (illustrative values):

```json
{
  "user_id": "EXAMPLE_USER",
  "engine_used": "collaborative",
  "reason": "User is present in the trained ALS model",
  "items": [
    {
      "parent_asin": "EXAMPLE_ASIN",
      "item_title": "Example software",
      "main_category": "Software",
      "is_free": true
    }
  ]
}
```

| Test user | Expected engine when that engine produces items |
| --- | --- |
| Known ALS user | `collaborative` |
| User absent from ALS with qualifying indexed positive reviews | `content_based` |
| Unknown user or one without an eligible personalised profile | `popular` |

The response always contains `user_id`, `engine_used`, `reason`, and `items`.
Item fields are `parent_asin`, `item_title`, `main_category`, and `is_free`;
content-based items also include `score`. Missing metadata is JSON `null`.
Check the `reason` if an empty personalised result led to a later engine.

Without `n`, the default is 10; the range is 1 through 100. Fewer items can be
returned when candidates are limited. Popularity excludes previously reviewed
items; if no unseen candidates remain, `items` is empty. These checks establish
serving behavior, not recommendation quality, which needs separate evaluation.

### 4. Check validation and error behavior

| Test | Expected response |
| --- | --- |
| `n=0`, `n=101`, or `n=abc` | 422; JSON validation error |
| Blank user ID (for example `%20`) | 422; `{"detail": "ID must not be blank"}` |
| Unknown user | 200; popularity fallback, not 404 |
| No eligible unseen catalogue candidates | 200; `items: []` inside the response object |
| Unexpected data/model failure | Server error; no silent fallback |
| `/` or a removed direct tool endpoint | 404; `{"detail": "Not Found"}` |

The old collaborative, content-user, and item-similarity HTTP routes were removed.
Clients must use the single user recommendation endpoint.

## Automated routing tests

Run from the repository root:

```bash
uv run python -m unittest discover -s tests -v
```

Tests use small datasets and mocked inference, without downloading resources or
calling real models. They cover CF priority, content eligibility, empty-result
fallback, popularity ranking/exclusions, validation, and the public HTTP contract.

## Render Free behavior and troubleshooting

Render Free has 512 MB RAM and limited CPU. The app shares its models and selected
dataset columns, uses one worker, and runs one recommendation calculation at a
time to reduce memory use. Confirm actual container usage during deployment;
larger datasets can exceed the free instance's memory.

Free services sleep after 15 minutes without traffic. Waking up takes about a
minute before accounting for this app's loading time. Files written at runtime
are lost on sleep, restart, or redeploy, so data/model downloads repeat. The
workspace's 750 monthly free instance hours are shared across free services.
Builds and bandwidth have separate limits; Supabase has its own limits. Stay
within included allowances to keep costs at zero. See
[Render's free service limits](https://render.com/docs/free).

| Symptom | What to check |
| --- | --- |
| Build fails | Read the first build error; dependency/native-library installation happens in the Dockerfile |
| Startup fails | Verify both environment variables, exact Supabase resource names, registry entries, and stored artefacts |
| Process is killed or runs out of memory | Check Render logs/metrics; reduce the served dataset or model size to remain on Free |
| First request is slow | Allow the service to wake and download its resources; inspect startup logs |
| 404 at `/` | Expected; open `/docs` or `/health` |
| Unexpected popularity fallback | Inspect `reason`, ALS membership, and qualifying reviews in the content index |

The API currently has no authentication. Supabase credentials stay on the server;
the recommendation routes accept IDs supplied by the caller.

## Optional: work with models locally

Serving existing models does not require `setup.py` or rebuilding either model.
To download the full local datasets and artefacts for offline work, run:

```bash
uv run python -m setup
```

That full-data setup also requires the `User` table. To rebuild either model,
provide the complete training datasets and run from the repository root:

```bash
uv run python -m src.collaborative_filtering.main --mode build
uv run python -m src.content_based_filtering.main --mode build
```

These commands save local artefacts; they do not publish them to Supabase.
