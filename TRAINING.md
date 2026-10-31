# Notebook workflow

Run `python setup.py` once from the repository root, fill in `.env`, then start
`uv run jupyter lab`. Select `Amozon Backend (.venv)` or the `.venv` interpreter
in your IDE. Setup installs dependencies and creates folders; it does not run
notebooks or create Supabase tables. See [DEPLOYMENT.md](DEPLOYMENT.md) for the
required database and storage resources.

Use `notebook/publish_dataset.ipynb` to download/transform in memory and intentionally replace
Supabase User, Item, and Review. There are no raw/transformed local exports; the datasets library
may maintain its own download cache. Publication is batched and not atomic, so failures require
rerunning the replacement. ModelRegistry and Storage are separate. Then run
`notebook/train_test_split_data.ipynb`. This writes one current set of train.parquet,
validation.parquet, test.parquet, and items.parquet under `data/output` (the shared
config.ml_config.DATA_OUTPUT_DIR). Re-running replaces those files; no dataset versions or split
manifests are created.

## One notebook per algorithm

Open `notebook/cf_recommend.ipynb` or `notebook/cb_recommend.ipynb` and follow its
numbered sections. The two former data-preparation notebooks are merged here.

1. Defaults and the recommendation count come from each algorithm's
   tool_config.py. CANDIDATES initially references those settings; add overrides
   only when deliberately tuning.
2. Load `training`, `validation`, and `test` via DataLoader. Inspect shapes,
   historical date ranges, and supported/excluded target counts.
3. Fit each candidate on training data using src/<algorithm>/model.fit().
   Evaluate it on validation inputs using recommender.evaluate().
4. Select parameters by validation NDCG, with Recall as a tie-breaker.
   Review the per-user table; adjust candidates and rerun validation if needed.
5. Run the final-test cell only after freezing settings. This fits a fresh
   model on train+validation history and scores the held-out test targets.
   It does not reuse train-only ALS factors. Do not tune against test scores.
6. Run the save cell only when you want to replace the current local
   serving bundle with that tested model. Saving does not upload it.

Both notebooks execute final testing and saving when those cells are run; there
are no RUN_FINAL_TEST or SAVE_MODEL switches. Stop after validation until ready
to run the remaining cells deliberately. The CF publishing cell also executes
its upload when run, so use individual cells rather than Run All when inspecting
or tuning. Empty supported
target populations raise a clear error instead of inventing a score. Fitting
functions do not read local files, download anything, or write artifacts.

## Data boundaries and populations

The shared chronological split is approximately 80/10/10. Equal UTC timestamps
stay together. All ratings and repeated events remain in the saved files.
Invalid IDs/dates/ratings, overlapping events, empty windows, and missing item
metadata fail explicitly. No minimum-interaction or warm-start pruning changes
the event windows.

For validation, past history is train. For test, it is train+validation.
Repeated user/item preferences are resolved within that past only, preserving
all seen products including dislikes and neutral ratings. Content recency uses
a 365.25-day half-life referenced to the next window's start. Historical source
popularity and recency aggregates are not reused.

CF uses previously observed items and users with positive historical preferences.
Its signed confidence is sign * (1 + 20 * abs(rating - 3) / 2): ratings >=4 are
positive, <=2 negative, and intermediate ratings are stored as explicit zeros.
Those zeros must remain in the matrix for seen-item filtering.

Content uses all saved metadata items with nonempty text, even without historical
interactions. It fits title+description+features. Positive historical reviews
create rating/recency-weighted profiles; category boost and free-only preference
rules remain part of the recommender. Metadata is an undated snapshot assumed
available at each boundary. New items/text require rebuilding the saved matrix.

Targets are first encounters with previously unseen products rated >=4. Later
rating changes cannot replace the first encounter. Each algorithm evaluates its
own supported users/items. excluded_targets counts removed relevant user/item
targets; excluded_users counts users left with no supported targets. These are
coverage counts, not model-quality scores.

## Scores and persistence

`recommender.evaluate(bundle, phase, k)` returns `summary` and `per_user`.
Recall@K measures retrieval and binary NDCG@K measures ordering. Both are
macro-averaged over users with supported relevant targets. Eligible users with
empty recommendations score zero. Empty target populations have no score.
Algorithms share calculations in src/metrics.py, not populations or rankings.

`model.fit(...)` returns a named dictionary bundle. `save_model(bundle, path)`
validates and atomically replaces one compressed joblib file; `load_model(path)`
validates on read. CF stores model/interactions/user_ids/item_ids; content stores
item_matrix/metadata. No positional tuple ordering or training provenance package
is required. Published IDs identify serving models, not versions of datasets.

Both notebooks implement publication directly in their final "Push Saved Model
into Supabase" cell. Run it deliberately after saving. It prints the new ID for
config/api_config.py MODEL_IDS and does not call the publishing helper. The helper
remains available when publishing outside the notebooks:

```python
from models_loader.push_into_supabase import publish_model
model_id = publish_model("content_based_filtering")
print(model_id)
```

Both publication paths generate the model ID locally and upload before inserting
the registry row. Copy the returned ID into `config/api_config.py` `MODEL_IDS`.
See [DEPLOYMENT.md](DEPLOYMENT.md) for application cache/deployment behavior. Regression tests
use temporary datasets and artifacts and never contact live Supabase.

## Configuration ownership

Use config/ml_config.py for dataset settings, local experiment paths, and split
filenames. It exposes REPO_ROOT and imports shared Supabase/model resource names
from config/api_config.py. Algorithm hyperparameters remain in each tool_config.py.
Published serving versions are selected by MODEL_IDS in config/api_config.py.

Both recommendation notebooks import their workflow paths from config.ml_config.
A running kernel retains loaded configuration; restart it when ready to pick up
configuration changes.
