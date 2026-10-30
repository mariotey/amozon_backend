# Application setup and deployment

For a local fork, install Python 3.13 and run `python setup.py` from the repository
root (`py -3.13 setup.py` on Windows also works). This installs the locked
dependencies into `.venv`, registers its notebook kernel, creates the configured
data/model folders, and writes `.env` placeholders only if the file is missing.
Existing credentials and artifacts are preserved. It does not contact Supabase
or download models. Use `--skip-install` to create folders and the template only.

Set server-side credentials in `.env` or the deployment environment:

```env
SUPABASE_URL=https://YOUR-PROJECT.supabase.co
SUPBASE_SECRET_KEY=YOUR_SERVER_SIDE_SECRET_KEY
```

Keep the existing spelling SUPBASE_SECRET_KEY. Never put the secret in a
notebook output or browser application.

## Database prerequisites

Supabase must have the prepared User, Item, Review, and ModelRegistry tables,
and the ModelArtefacts storage bucket. Prepare schemas to match
`data_loader/data_transform.py`. Use `notebook/publish_dataset.ipynb` for the
intentional full replacement of User/Item/Review; it does not export intermediate
local files or alter model records. Run it outside serving use: batched deletion
and insertion are not one transaction. Manually review/apply
[sql/001_serving_indexes.sql](sql/001_serving_indexes.sql): these indexes support
the active history, product, profile, and popularity queries. Demo shopper IDs
come from application configuration; no shopper-sampling RPC is required.
This repository does not apply SQL or change database permissions automatically.

## Model bundles and publication

The application loads `models/<algorithm>/model.joblib`. Missing files are
fetched from the exact configured tool/model ID in ModelRegistry and the bucket.
It does not silently fall back to a different model version. Each bundle is
validated for finite values, identifier uniqueness, dimensions, and row alignment.

The repository does not include local model files. Configured IDs must refer to
compatible single-file bundles in your Supabase project; older multi-file models
are not converted on download. To publish models for a clean deployment:

1. Complete the notebook workflow and deliberately save the selected bundles.
2. Explicitly publish each required bundle from a trusted local environment:

   ```python
   from models_loader.push_into_supabase import publish_model
   model_id = publish_model("collaborative_filtering")
   print(model_id)
   # Repeat for content_based_filtering when its selected bundle is ready.
   ```

3. Put each returned ID under its algorithm name in
   `config/api_config.py` MODEL_IDS. Publication uploads before inserting the registry row and does not
   delete older models. It never happens during app startup or evaluation.
4. Deploy/restart with matching configuration. Existing local bundle caches are
   reused; replace or deliberately remove a cache to fetch a newly configured ID.

The notebook final-test model uses train+validation history. Saving/publishing is
explicit and does not automatically refit on the test set or all source history.

## Run and API behavior

```powershell
uv run uvicorn app:app --reload
```

Docker uses the existing Dockerfile and PORT (default 10000). Build and run from
the repository root:

```powershell
docker build -t amozon-backend .
docker run --rm --env-file .env -p 10000:10000 amozon-backend
```

The image installs dependencies with `uv sync --frozen --no-install-project`;
it does not need the local setup script or notebook kernel. Models and data
are excluded from the image, so a clean container needs published compatible
bundles. One background worker loads models; browsing does not wait for it.
One inference lock bounds numeric inference memory, without holding database
metadata queries. The API never loads experiment split files or entire tables.

| Endpoint | Purpose |
| --- | --- |
| `/health` | Liveness. |
| `/ready` | Model readiness; does not guarantee database connectivity. |
| `/docs`, `/redoc`, `/openapi.json` | API documentation/schema. |
| `/api/v1/users/{user_id}/recommendations?n=10` | CF, then content, then unseen-popular fallback. |
| `/api/v1/demo_user` | Exactly three fixed shopper IDs keyed by recommendation scenario; no database or model dependency. |
| `/api/v1/items/{parent_asin}` | Product details. |
| `/api/v1/items/{parent_asin}/reviews` | Paginated reviews. |
| `/api/v1/users/{user_id}` | Shopper profile. |
| `/api/v1/users/{user_id}/history` | Paginated verified history by default; verified_only=false includes all reviews. |

There is no root homepage and no standalone popular-items endpoint. Unknown
shopper IDs remain valid recommendation requests. Empty engines permit fallback;
model/database failures propagate as errors. Recommendation routes return 503
until models are ready. Content history stays local to the request; CF uses its
saved interaction matrix. Seen dislikes and neutral products remain excluded.

The demo response contains only three user IDs:

```json
{
  "collaborative": "ae24aljoiqjoibv3ycrbis4ekx2a",
  "content_based": "aeayitldccp4fwimhv7tgr67ysnq",
  "popular": "aeqwyrohwdka275azy6mrqlu33vq"
}
```

`config/api_config.py` is the single source for these examples. They are verified
against the published models; the recommendation endpoint still chooses the
actual engine. After publishing new models or replacing source data, request
recommendations for each ID and confirm three different engines with nonempty
results. Update these IDs if needed. The old `/api/v1/randomusers` route is removed.

Deploy the backend before the updated frontend, which now loads `/api/v1/demo_user`.
When opened on localhost, Flask defaults to `http://127.0.0.1:8000`, so run this
updated backend locally alongside the frontend. Public storefront hosts default
to the hosted API. Set `API_BASE_URL` in the frontend environment to override
either default (for example, when using a different backend port). Restart Flask
after changing that environment variable. Existing database functions/indexes
are not dropped or changed by this code update.

## Verification and navigation

```powershell
uv run python -m unittest discover -s tests -v
```

The three regression modules cover first-run setup, data/chronology, real tiny model fits and
merged notebooks, bundle persistence/metrics, and API startup/fallback/isolation.
They use temporary local artifacts and fake database responses, not live Supabase.

See [README.md](README.md) for the folder map and [TRAINING.md](TRAINING.md) for
notebook instructions. `setup.py` is a local first-run helper, not a packaging,
model-building, or deployment entry point. Start the API with `uvicorn app:app`;
its router calls the recommendation functions directly. Notebook split files
are not required for serving.
