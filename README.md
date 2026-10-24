# Amozon Recommendation API

A FastAPI service that exposes two software recommendation tools:

- **Collaborative filtering:** recommends items for users known to the trained ALS model.
- **Content-based filtering:** recommends items from a user's positive review history and finds similar items using TF-IDF features.

The API loads existing data and model artefacts from local caches or Supabase. It
does not train models or upload data when it starts. Render runs the app using the
repository's Dockerfile; no `render.yaml` is required.

## Project files

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI routes and shared startup resources |
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
| `Item` table | `parent_asin`, `item_title`, `main_category`, `is_free` |
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

Open `BASE_URL/health` in your browser. Expected status: **200**.

```json
{"status": "ok"}
```

Then open `BASE_URL/docs`. You should see Swagger UI with the health route and
three recommendation routes. `/health` becomes available after data/model loading
succeeds; it does not continually check Supabase connectivity. Visiting `/` returns
404 because no homepage route is defined.

### 2. Choose valid test IDs

For collaborative filtering, use a user ID from the deployed model's
`idx_to_userid_mapping.json`. A user merely existing in the Review table does not
mean the ALS model knows that user.

For content recommendations, use a user with reviews rated at least 3 for items
present in the content model. For similar items, use a `parent_asin` from that
model's `cb_meta.parquet`.

If your local models match the deployed versions, these commands print a known
ALS user and a known content-model item:

```bash
uv run python -c "import json; from pathlib import Path; print(next(iter(json.loads(Path('models/collaborative_filtering/idx_to_userid_mapping.json').read_text()).values())))"
uv run python -c "import pandas as pd; print(pd.read_parquet('models/content_based_filtering/cb_meta.parquet')['parent_asin'].iloc[0])"
```

A known ALS user may still have no qualifying content-based history.

### 3. Call each recommendation route

In `BASE_URL/docs`, expand a route, select **Try it out**, enter its ID, set
`n` to `3`, and select **Execute**. Check both the response code and response body.
You can also paste the following paths after your service URL in a browser,
replacing `USER_ID` and `ASIN` with your test IDs:

| Request path | Expected successful response |
| --- | --- |
| `/api/v1/collaborative/users/USER_ID/recommendations?n=3` | 200; an array of up to 3 items with `parent_asin`, `item_title`, `main_category`, `is_free` |
| `/api/v1/content/users/USER_ID/recommendations?n=3` | 200; an array of up to 3 items with the same fields plus `score` |
| `/api/v1/content/items/ASIN/similar?n=3` | 200; an array of up to 3 items with `parent_asin`, `item_title`, `main_category`, `score` |

For example, a collaborative response has this shape. Values below are illustrative:

```json
[
  {
    "parent_asin": "EXAMPLE_ASIN",
    "item_title": "Example software",
    "main_category": "Software",
    "is_free": true
  }
]
```

Missing metadata is returned as JSON `null`. Results depend on the model and data;
not every request can produce the requested number of recommendations. Without
`n`, the default is 10; the allowed range is 1 through 100.

### 4. Verify error and empty-result behavior

| Test | Expected response |
| --- | --- |
| `n=0`, `n=101`, or `n=abc` on any recommendation route | 422; JSON validation error |
| Unknown collaborative user | 404; `{"detail": "Unknown user_id: ..."}` |
| Content user with no qualifying history | 200; `[]` |
| Unknown similar-item ID | 200; `[]` |
| `/` | 404; `{"detail": "Not Found"}` |

A 200 with `[]` in the documented cases is valid. For a positive-result test, use
an ID with known history/model membership and confirm the returned item fields.
These checks verify serving behavior; evaluating recommendation quality requires
separate model evaluation.

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
| 404 for a collaborative user | Use an ID from the deployed ALS model mapping |
| Empty content recommendations | Check qualifying reviews and whether reviewed items exist in the model |

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
