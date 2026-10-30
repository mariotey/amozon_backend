# Amozon Recommendation Backend

A FastAPI backend for software discovery using collaborative filtering,
content-based recommendations, and an unseen-popular-products fallback.
Supabase supplies catalogue, review, shopper, and published-model data.

## Start here

Install Python 3.13, fork/clone the repository, and run from its root:

```powershell
python setup.py
```

On Windows, `py -3.13 setup.py` also works. On macOS/Linux, use
`python3.13 setup.py` if needed. Setup installs `uv` through pip when missing,
installs the dependencies in `uv.lock` into `.venv`, and registers the
`Amozon Backend (.venv)` notebook kernel inside that environment. It creates:

```text
data/output/
models/collaborative_filtering/
models/content_based_filtering/
.env                         # only when missing
```

Edit `.env` with your server-side Supabase credentials:

```env
SUPABASE_URL=https://YOUR-PROJECT.supabase.co
SUPBASE_SECRET_KEY=YOUR_SERVER_SIDE_SECRET_KEY
```

Keep the existing spelling `SUPBASE_SECRET_KEY`. Setup preserves existing files,
credentials, datasets, and models. It does not provision Supabase, download data
or models, train, or publish. `python setup.py --skip-install` creates only the
folders and missing `.env` template. Dependency installation requires internet;
if `implicit` needs compiling, a C/C++ toolchain, CMake, and OpenBLAS development
libraries may be needed (the Dockerfile supplies these for Linux).

Start notebooks with:

```powershell
uv run jupyter lab
```

Choose the `Amozon Backend (.venv)` kernel. In an IDE, select the repository's
`.venv` interpreter. If pip installed uv outside your PATH, replace `uv` with
`python -m uv` in the commands below.

The manual workflow is:

1. `notebook/publish_dataset.ipynb`: download and transform data in memory, then
   intentionally replace all rows in Supabase User, Item, and Review. No raw or
   transformed local files are exported; the final cell is the publishing action.
2. `notebook/train_test_split_data.ipynb`: fetch Supabase data once and save
   shared chronological splits plus item metadata in `data/output`.
3. `notebook/cf_recommend.ipynb`: load CF data, fit candidates, inspect validation
   results, select settings, then deliberately run the final-test and save cells.
4. `notebook/cb_recommend.ipynb`: the same sequence for content-based filtering,
   using its own supported population and item catalogue.

The preparation and recommendation examples are merged into these two algorithm
notebooks. They are independent experiments, not a competition between models.
Run cells in order and stop after validation until settings are selected. The
final publishing cell in either notebook uploads to Supabase when executed. See
[TRAINING.md](TRAINING.md) for inputs, metrics, and publishing instructions.

## Where the code lives

| Location | Purpose |
| --- | --- |
| `setup.py` | First-run dependency installation, notebook kernel, folders, and missing `.env` template. |
| `data_loader/` | Download/transform helpers, explicit Supabase reads, chronological splitting, algorithm-specific data views. |
| `src/collaborative_filtering/model.py` | Build the signed rating matrix and fit ALS in memory. |
| `src/content_based_filtering/model.py` | Fit item-text TF-IDF and return precomputed representations. |
| Each `recommender.py` | Recommendations and independent algorithm evaluation. |
| Each `tool_config.py` | Algorithm defaults, identity, and bundle filename. |
| `config/ml_config.py` | Dataset settings, split filenames, local data paths, and shared `REPO_ROOT`. |
| `config/api_config.py` | Serving paths, shared Supabase names, published `MODEL_IDS`, and three fixed `DEMO_USERS`. |
| `src/metrics.py` | Shared Recall@K and NDCG@K calculations; populations remain algorithm-specific. |
| `src/routing.py` | Calls the recommenders directly with loaded bundles; CF -> content -> popular fallback. |
| `models_loader/` | Validate, save, load, and explicitly publish single-file model bundles. |
| `utils/review_history.py` | Shared latest-review preference resolution. |
| `utils/supabase_utils.py` | Manual source reads, targeted serving queries, and model download. |
| `sql/001_serving_indexes.sql` | Manual query indexes for history, product, profile, and popularity lookups. |
| `tests/` | Three small regression modules for data, models/notebooks, and API behavior. |

The obsolete evaluation/training runners and unused inference/scripts/release
folders are removed. Dataset preparation, experiments, and API serving have one
active path each. No dataset versioning or training manifests are generated.

## Serving artifacts

Each algorithm saves one file:

```text
models/collaborative_filtering/model.joblib
models/content_based_filtering/model.joblib
```

CF bundles contain the ALS model, signed interaction matrix, ordered user IDs,
and ordered item IDs. Content bundles contain the item matrix and aligned
metadata. The vectorizer is used during fitting but not persisted; adding new
product text requires rebuilding the content bundle. Inverse lookup maps are
constructed once at application model loading.

Data and model files are not committed. For a clean fork, either prepare and save
models through the notebooks or configure IDs for compatible published bundles
in your own Supabase project. The API downloads missing model files using
`config/api_config.py` `MODEL_IDS`; it does not select the latest registry row.
See [DEPLOYMENT.md](DEPLOYMENT.md) for publication and cache behavior.

## Run the API

After configuring Supabase tables, storage, indexes, and models as described in
[DEPLOYMENT.md](DEPLOYMENT.md), run:

```powershell
uv run uvicorn app:app --reload
```

Open `/docs` for endpoints. `/health` is liveness; `/ready` is model readiness.
Browsing uses targeted Supabase queries and works independently of model loading.
CF uses saved factors/history; content fallback loads only the requested
shopper's history. Model and database failures remain errors, not empty results.

`GET /api/v1/demo_user` returns exactly three fixed IDs keyed by `collaborative`,
`content_based`, and `popular`. It does not query Supabase or wait for models.
The frontend uses these examples instead of a random-shopper endpoint; keep their
IDs in `config/api_config.py` and recheck their routes after model/data updates.

Run the offline regression suite with:

```powershell
uv run python -m unittest discover -s tests -v
```

The tests use temporary files and fake Supabase responses; they do not publish
data or models. See [TRAINING.md](TRAINING.md) for the ML workflow and
[DEPLOYMENT.md](DEPLOYMENT.md) for API deployment.
