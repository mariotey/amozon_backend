# Complete Architecture

## 1. Overview

`amozon_backend` is a layered recommender backend built around the Amazon Reviews 2023 (Software category) dataset. Two recommender tools — **Collaborative Filtering (ALS)** and **Content-Based Filtering (TF-IDF)** — are trained locally, with all processed tabular data and model artefacts persisted to a Supabase backend (Postgres tables + a Storage bucket). Inference handlers are exposed as importable functions intended for serverless runtimes (Azure Functions), with thin CLI wrappers and Jupyter notebooks for development.

### Layered pipeline (end to end)
```
[HuggingFace Hub] ──data_download.py──► data/input/*.parquet
                                              │
                                     data_transform.py (feature engineering)
                                              │
                                       data/output/*.parquet
                                              │
                                     push_into_supabase.py ──► Supabase Tables
                                              │                          (User / Item / Review)
[Local training]                              │
   src/<tool>/model.py ──► models/<tool>/*.joblib/.npz/.json/.parquet
                                              │
                                     models_loader/push_into_supabase.py
                                              │ ──► Supabase ModelRegistry (table) +
                                              │     ModelArtefacts (storage bucket)
                                              │
[Inference runtime]                           ▼
   setup.py ──► DataLoader (cache tables) + ModelsLoader (cache artefacts)
                                              │
   src/<tool>/main.py  ── get_data_loader()/get_models_loader() (lazy cached singletons)
        │                                     │
        ├── recommender.py ── pure recommendation logic
        └── exposed as importable handler fns ──► Azure Function / CLI / notebook
```

---

## 2. Layers & responsibilities

### 2.1 `utils/supabase_utils.py` — shared Supabase client
Creates two module-level singletons at import time and exposes six free functions wrapping them:

- `SUPABASE_CLIENT = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPBASE_SECRET_KEY"))` — single `supabase-py` client (note the deliberate typo `SUPBASE_SECRET_KEY`).
- `ARTEFACTS_STORAGE = SUPABASE_CLIENT.storage.from_(MODEL_ARTEFACT_BUCKET)` — handle bound to the `"ModelArtefacts"` bucket.

| Function | `utils/supabase_utils.py` | Purpose |
|---|---|---|
| `push_table_into_supabase(df, table_name, batch_size=800, show_progress=True)` | `:27-53` | Batched `insert` of DataFrame records into a Supabase table. |
| `extract_table_from_supabase(table_name, batch_size=1000, show_progress=True) -> pd.DataFrame` | `:55-103` | Paginated full-table extract via `.range(offset, offset+batch_size-1)`. |
| `reconcile_registry(tool_name)` | `:105-161` | Bidirectional sync of `ModelRegistry` rows vs `ModelArtefacts` bucket folders for a tool. |
| `push_artefacts_into_registry(tool_name) -> (model_id, storage_path)` | `:163-196` | Reconciles, generates a fresh `uuid4`, inserts a `ModelRegistry` row. |
| `push_artefacts_into_supabase(tool_dir, model_artefacts_dict, storage_path)` | `:198-221` | Uploads each artefact file to `<storage_path>/<filename>` with `upsert=true`. |
| `download_artefacts_from_supabase(tool_name, model_id, model_artefacts_dict)` | `:223-274` | Looks up `storage_path` by `(tool, model_id)`, falls back to latest by `created_at`, downloads files to local cache. |

Reused across: `data_loader/data_loader.py:9`, `data_loader/push_into_supabase.py:10`, `models_loader/models_loader.py:19`, `models_loader/push_into_supabase.py:18`.

### 2.2 `data_loader/` — data ETL pipeline
| File | Responsibility |
|---|---|
| `data_download.py` | Pull `McAuley-Lab/Amazon-Reviews-2023` (Software) from HF Hub, normalize/clean, write `data/input/review_data.parquet` + `meta_data.parquet`. |
| `data_transform.py` | Feature-engineer into 3 tables: `user-item-interaction.parquet`, `user.parquet`, `item.parquet` under `data/output/`. |
| `push_into_supabase.py` | Sanitize NaN/Inf/NaT → `None`, push the 3 tables to Supabase `Review`/`Item`/`User`. |
| `data_loader.py` | `DataLoader` class + `load_data()` — **read path**: local Parquet cache first, fetch from Supabase on miss. |

### 2.3 `models_loader/` — model artefact pipeline
| File | Responsibility |
|---|---|
| `models_loader.py` | Suffix-aware `read_local_artefacts` / `save_local_artefacts`, cache-or-download `load_artefacts`, and `ModelsLoader` orchestrator class. |
| `push_into_supabase.py` | Walk each tool dir under `models/`, validate, register + upload to Supabase. |

### 2.4 `src/<tool>/` — recommender tools
Each tool is a self-contained package following the **tool-config contract** (see §5):
```
src/<tool>/
├── tool_config.py   # MODEL_NAME, MODEL_ID, MODEL_ARTEFACTS (+ hyperparameters)
├── model.py         # build_and_save(): training pipeline
├── recommender.py   # pure inference logic
└── main.py          # lazy-cached handlers + CLI entry point
```

### 2.5 Top-level glue
- `config.py` — central constants (paths, table names, dataset config, `TOOLS`).
- `setup.py` — one-time bootstrap: create dirs → `DataLoader()` → `ModelsLoader(tools=TOOLS)`.
- `pyproject.toml` / `uv.lock` / `.python-version` — dependency management with `uv`.

---

## 3. Data pipeline (end to end)

### 3.1 Download (`data_download.py`)
- `load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Software", split="full[:5%]")` → reviews.
- `load_dataset(..., "raw_meta_Software", split="full")` → metadata.
- Normalization: lowercase `asin`/`parent_asin`/`user_id`/`title`/`text`; derive `review_date` from `timestamp` (ms); `parse_categories` (lowercase, strip `'s`, alias via `category_replace_dict`, sorted); `parse_videos` (`[""]` → `[]`); numeric coercion of `rating`/`rating_number`/`price`.
- Output: `data/input/review_data.parquet`, `data/input/meta_data.parquet`.

### 3.2 Transform (`data_transform.py`)
Produces three engineered tables under `data/output/`:

**`user-item-interaction.parquet`** (22 columns; ~244k rows) — the `Review` table:
- Adds `review_id` (from index), text features (`review_text_length`, `review_word_count`), sentiment proxies (`is_extreme_rating`, `is_positive`, `is_negative`), `num_review_img`, temporal features (`days_since_review`, `review_year`, `review_month`), and `recency_weight = exp(-days_since_review/365.25)`.

**`user.parquet`** — the `User` table:
- Per-`user_id` aggregates: `num_reviews`, `avg_rating_given`, `rating_std`, `total_helpful_votes_received`, `num_verified_purchases`, `avg_review_length`, `avg_review_words`, `total_review_images`, `extreme_rating_ratio`, `positive_rating_ratio`, `negative_rating_ratio`, `avg_price_purchased`, `free_app_ratio`.
- Derived: `days_active`, `reviews_per_day`, `verified_purchase_ratio`.
- Segmentations: `user_segment` ∈ {one_time, occasional, power_user} (bins `[0,1,10,inf]`), `is_discriminating = rating_std > 1.0`.

**`item.parquet`** — the `Item` table:
- Filtered to items referenced in reviews only.
- `num_item_img`, `num_item_videos`, `price.fillna(-1)`, `is_free`, `has_price_info`, `price_bucket` (bins `[-1,0,1,10,25,50,100,inf]`), `num_categories`.
- Popularity aggregation (groupby `parent_asin`): `num_reviews`, `avg_rating`, `rating_std`, `total_recency_weight`, `total_helpful_votes`, `num_verified_reviews`, `positive_review_ratio`, `negative_review_ratio`.
- Derived: `days_on_platform`, `reviews_per_day`, `verified_review_ratio`, `popularity_segment` ∈ {cold_start, low_coverage, medium, popular} (bins `[0,1,10,100,inf]`), `quality_score` (Wilson lower bound, z=1.96).

### 3.3 Push to Supabase (`push_into_supabase.py`)
- Replace `np.nan`/`np.inf`/`pd.NaT` → `None`, then `push_table_into_supabase` into `Review`, `Item`, and `User` tables.
- ⚠️ Known bug: line 64 passes `USER_FILENAME` (`"user.parquet"`) instead of `USER_TABLE_NAME` (`"User"`).

### 3.4 Pull / read path (`data_loader.py`)
`setup.py` instantiates `DataLoader()`. For each table, `load_data(local_filename, supabase_tablename, force_remote=False)`:
1. If `not force_remote`, try `pd.read_parquet(DATA_OUTPUT_DIR / local_filename)`.
2. On `FileNotFoundError`, or if `force_remote=True` (local read skipped), call `extract_table_from_supabase(supabase_tablename)`, save to local parquet, return.

`force_remote=True` is how inference forces a fresh Supabase pull instead of trusting the local cache — see §10.

Mappings (from `config.py`):
| Local filename | Supabase table |
|---|---|
| `user.parquet` | `User` |
| `item.parquet` | `Item` |
| `user-item-interaction.parquet` | `Review` |

The cache is persistent: subsequent runs reuse local Parquet without re-downloading.

---

## 4. Model pipeline (end to end)

### 4.1 Build (train + persist locally)
Triggered by `python -m src.<tool>.main --mode build`:
- **CF** (`src/collaborative_filtering/model.py:build_and_save`): merge `user_item_df` with `item_df` numeric features → MinMax-scale 8 features → invert `price` (`1 - price`) → mean into `interaction` → groupby `(user, item).mean()` → `csr_matrix` (float32) → `implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20).fit(...)`.
- **CBF** (`src/content_based_filtering/model.py:build_and_save`): concatenate `item_title + description + features` → `TfidfVectorizer(max_features=10_000, ngram_range=(1,2), sublinear_tf=True).fit_transform(...)`.

Both call `save_local_artefacts(MODEL_NAME, artefacts, MODEL_ARTEFACTS)` which writes to `models/<tool_name>/<filename>`.

### 4.2 Register + upload to Supabase (`models_loader/push_into_supabase.py`)
For each tool directory under `models/`:
1. Dynamically import `src.<tool>.tool_config`.
2. Validate all `MODEL_ARTEFACTS` files exist locally (else `ValueError`).
3. `push_artefacts_into_registry(tool_name)` → reconciles, generates `model_id` (uuid4), inserts `ModelRegistry` row, returns `storage_path = f"{tool_name}/{model_id}"`.
4. `push_artefacts_into_supabase(...)` uploads each file to `ModelArtefacts/<tool_name>/<model_id>/<filename>` with `upsert=true`.

### 4.3 Bootstrap (`setup.py`)
1. Create `DATA_DIR`, `DATA_INPUT_DIR`, `DATA_OUTPUT_DIR`, `MODEL_ARTEFACT_DIR`.
2. `DataLoader()` — pull/cache the 3 tables.
3. `ModelsLoader(tools=TOOLS)` — for each tool, check local `models/<tool>/`; if any file missing, `download_artefacts_from_supabase(tool, MODEL_ID, MODEL_ARTEFACTS)` (queries `ModelRegistry` by `MODEL_ID`, falls back to latest by `created_at`), then `read_local_artefacts`.

### 4.4 Inference (`src/<tool>/main.py`)
- Lazy-cached singletons `get_data_loader()` / `get_models_loader()` survive warm serverless invocations.
- Public handlers return JSON-serializable `list[dict]`:
  - CF: `get_user_recommendations(user_id, n=10)`.
  - CBF: `get_user_recommendations(user_id, n=10)` and `get_similar_items(parent_asin, n=10)`.
- `build_model()` resets both singletons so the next call reloads fresh artefacts.
- The handlers call `get_data_loader(force_remote=True)` / `get_models_loader(force_remote=True)`, so the call that creates the cache pulls straight from Supabase rather than trusting local disk. See §10 for the complete loading-into-cache flow.

---

## 5. The tool-config contract (pluggable tools)

Each tool lives under `src/<tool_name>/` and must ship a `tool_config.py` exposing:
- `MODEL_NAME` (derived from the directory name, e.g. `Path(__file__).parent.name`).
- `MODEL_ID` (UUID pointing to the Supabase registry entry to download).
- `MODEL_ARTEFACTS` (ordered dict of artefact-key → filename; order determines the positional tuple returned by `read_local_artefacts`).

`ModelsLoader` discovers tools via `importlib.import_module(f"src.{tool}.tool_config")` (`models_loader/models_loader.py:201-203`). Adding a new tool only requires creating a new `src/<tool>/` package conforming to this contract and registering it in `config.TOOLS`.

### Current tools
| Tool | `MODEL_NAME` | `MODEL_ID` | Artefacts |
|---|---|---|---|
| Collaborative Filtering | `collaborative_filtering` | `9139d794-f9b3-4188-9370-33ceb92111fd` | `als_model.joblib`, `cf_user_item_matrix.npz`, `idx_to_userid_mapping.json`, `idx_to_itempasin_mapping.json` |
| Content-Based Filtering | `content_based_filtering` | `fefdee34-e3fe-4314-acab-f911c02680d3` | `cb_tfidf.joblib`, `cb_item_matrix.npz`, `cb_meta.parquet` |

### Artefact I/O dispatch (suffix-aware)
| Suffix | Reader | Writer |
|---|---|---|
| `.joblib` | `joblib.load` | `joblib.dump` |
| `.npz` | `scipy.sparse.load_npz` | `scipy.sparse.save_npz` |
| `.json` | `json.load` (keys → `int`) | `json.dump` (keys → `str`) |
| `.parquet` | `pd.read_parquet` | `df.to_parquet(index=False)` |

---

## 6. Supabase backend schema

### Postgres tables
| Table | Source | Key columns |
|---|---|---|
| `User` | `user.parquet` | `user_id`, `num_reviews`, `avg_rating_given`, `rating_std`, `user_segment`, `is_discriminating`, `verified_purchase_ratio`, `avg_price_purchased`, `free_app_ratio`, ... |
| `Item` | `item.parquet` | `parent_asin`, `item_title`, `main_category`, `categories`, `description`, `features`, `price`, `is_free`, `price_bucket`, `num_reviews`, `avg_rating`, `quality_score`, `popularity_segment`, ... |
| `Review` | `user-item-interaction.parquet` | `review_id`, `user_id`, `parent_asin`, `review_rating`, `recency_weight`, `helpful_vote`, `verified_purchase`, `is_positive`, `is_negative`, `review_word_count`, `days_since_review`, ... (22 cols) |
| `ModelRegistry` | `push_artefacts_into_registry` | `model_id` (uuid), `tool` (str), `storage_path` (str `"{tool}/{model_id}"`), `created_at` (timestamp, db default) |

### Storage bucket
- `ModelArtefacts/` with layout:
  ```
  ModelArtefacts/
  ├── collaborative_filtering/<uuid4>/{als_model.joblib, cf_user_item_matrix.npz, idx_to_userid_mapping.json, idx_to_itempasin_mapping.json}
  └── content_based_filtering/<uuid4>/{cb_tfidf.joblib, cb_item_matrix.npz, cb_meta.parquet}
  ```
- `reconcile_registry(tool_name)` keeps the table and bucket in sync per tool: deletes stale registry rows whose storage folder is gone, removes orphaned storage folders without a registry row.

---

## 7. Configuration constants (`config.py`)

| Constant | Value | Purpose |
|---|---|---|
| `REPO_ROOT` | repo dir | Locate `src/` tool packages. |
| `DATA_DIR` / `DATA_INPUT_DIR` / `DATA_OUTPUT_DIR` | `data/[input\|output]` | Raw download + processed/cache parquets. |
| `MODEL_ARTEFACT_DIR` | `REPO_ROOT/models` | Local artefact cache + training output. |
| `MODEL_ARTEFACT_BUCKET` | `"ModelArtefacts"` | Supabase storage bucket. |
| `USER_TABLE_NAME` / `REVIEW_TABLE_NAME` / `ITEM_TABLE_NAME` | `"User"` / `"Review"` / `"Item"` | Supabase tables. |
| `MODELREGISTRY_TABLE_NAME` | `"ModelRegistry"` | Model version registry. |
| `DATASET_NAME` | `"McAuley-Lab/Amazon-Reviews-2023"` | HF dataset id. |
| `DATASET_CATEGORY` | `"Software"` | Dataset subset. |
| `DATASET_SPLIT_PERCENTAGE` | `5` | Review split percentage. |
| `USER_REVIEW_FILENAME` / `ITEM_METADATA_FILENAME` | `review_data.parquet` / `meta_data.parquet` | Raw input filenames. |
| `USER_FILENAME` / `ITEM_FILENAME` / `USER_ITEM_INTERACT_FILENAME` | `user.parquet` / `item.parquet` / `user-item-interaction.parquet` | Processed output filenames. |
| `TOOLS` | `["collaborative_filtering", "content_based_filtering"]` | Canonical tool list. |

---

## 8. Dependency stack

| Dependency (`pyproject.toml`) | Role |
|---|---|
| `datasets>=4.8.5` | HF dataset download. |
| `dotenv>=0.9.9` | `.env` loading. |
| `fastapi>=0.129.0` | Declared, **not yet referenced** — reserved for future HTTP layer. |
| `fastparquet>=2025.12.0` | Parquet backend. |
| `implicit>=0.7.2` | ALS (CF). |
| `ipykernel>=7.2.0`, `jupyter>=1.1.1` | Notebooks. |
| `numpy>=2.4.2` | Numeric core. |
| `pandas>=3.0.0` | Tabular processing. |
| `pyarrow>=23.0.0` | Parquet I/O. |
| `scikit-learn>=1.8.0` | `MinMaxScaler`, `TfidfVectorizer`, `cosine_similarity`. |
| `scipy>=1.17.0` | Sparse matrices. |
| `supabase>=2.30.0` | Supabase client. |
| `tqdm>=4.67.3` | Progress bars. |
| `uvicorn>=0.40.0` | Declared, **not yet referenced** — reserved for future ASGI server. |
| `joblib` | Used by `models_loader`; **transitive** via scikit-learn (not explicit in pyproject). |

Python ≥ 3.13; managed with `uv` (lockfile `uv.lock`). Not intended for Anaconda/Miniconda.

---

## 9. Environment variables

| Variable | Used at | Purpose |
|---|---|---|
| `SUPABASE_URL` | `utils/supabase_utils.py:21` | Project URL. |
| `SUPBASE_SECRET_KEY` | `utils/supabase_utils.py:22` | Service-role secret key (deliberate typo `SUPBASE`). |
| `SUPABASE_PUBLIC_KEY` | README only | Documented but **not read by any source file** — reserved for future client-side layer. |

Loaded via `dotenv.load_dotenv()` at import time of `utils/supabase_utils.py`.

---

## 10. Inference-time loading into cache

- Current serving: importable handlers optimized for warm serverless invocations. Both `main.py` files document that their public functions are *"designed to be called directly by an Azure Function handler (or any other serverless runtime)"*.
- CLI fallback mirrors the handlers for local testing.
- FastAPI/uvicorn are declared dependencies for an obvious next step (thin `FastAPI()` wrapper around the handlers); not yet wired.

### 10.1 Three layers of caching

Loading data/artefacts for inference passes through three stacked layers, each with a different scope and lifetime:

1. **Local disk cache** (`data/output/*.parquet`, `models/<tool>/*`) — persists across process restarts, shared by every runtime on the same machine/container image.
2. **Supabase** (`User`/`Item`/`Review` tables, `ModelArtefacts` storage bucket) — the durable source of truth, always available regardless of local state.
3. **In-process singleton cache** (`_data_loader_obj`, `_models_loader_obj` module globals in each `src/<tool>/main.py`) — lives only as long as the Python process is alive; scoped to a single warm serverless worker instance.

### 10.2 What actually happens on an inference call

`get_user_recommendations` / `get_similar_items` call `get_data_loader(force_remote=True)` and/or `get_models_loader(force_remote=True)` (CF: `main.py:122-123`; CBF: `main.py:136-137,175`). Both accessors follow the same pattern:

```python
def get_data_loader(force_remote: bool = False) -> DataLoader:
    global _data_loader_obj
    if _data_loader_obj is None:
        _data_loader_obj = DataLoader(force_remote=force_remote)
    return _data_loader_obj
```

- **First call in a fresh process** (`_data_loader_obj is None`): `force_remote=True` reaches `DataLoader.__init__` → `load_data(...)`, which **skips the local Parquet read entirely** and fetches straight from Supabase (`extract_table_from_supabase`), then overwrites the local Parquet cache with the fresh result. Same pattern for `ModelsLoader` → `load_artefacts` against the `ModelArtefacts` storage bucket. The full DataFrames and deserialized model artefacts are built and stored as attributes on the new instance — held **entirely in memory**, not paginated or streamed.
- **Every subsequent call in that same warm process**: `_data_loader_obj`/`_models_loader_obj` are no longer `None`, so the `if ... is None:` check short-circuits and the cached instance is returned immediately. `force_remote=True` is passed again on every call but is **ignored** — it only ever mattered on the call that created the object. No disk I/O, no Supabase round-trip; the request is served from the objects already resident in memory.
- **`build_model()`** resets both globals to `None`, so the *next* inference call after a rebuild re-triggers step 1 (a fresh forced Supabase pull) rather than serving stale cached artefacts.

### 10.3 Why inference forces Supabase but build/setup don't

`build_model()` and `setup.py` call `get_data_loader()`/`DataLoader()` with the default `force_remote=False` (local-cache-first) — appropriate for training and bootstrap, where the freshly-transformed local Parquet files *are* the authoritative input. Inference deliberately reverses that preference on first load, trading a slower cold start for a guarantee that a cold worker never serves recommendations off a stale/incomplete local cache. This is scoped to the two inference entry points only (`get_data_loader`/`get_models_loader` calls inside `get_user_recommendations`/`get_similar_items`); the shared `DataLoader`/`ModelsLoader` classes and `load_data`/`load_artefacts` functions keep local-cache-first as their default (`force_remote=False`) for every other caller.

### 10.4 Serverless process lifecycle implications

Because the singleton cache is a plain Python module global, its lifetime is tied 1:1 to the OS process:

- **Warm invocation** (Azure Functions reuses an already-running worker): globals persist, memory is not cleared, the cached objects from the previous invocation are reused instantly — no Supabase or disk access.
- **Cold start** (first request ever, after scale-to-zero idle, or after the platform recycles/replaces the worker): a fresh process imports the module, both globals start as `None` again, and the full forced-Supabase load path in §10.2 runs from scratch.
- **Scale-out**: each concurrent worker instance is its own process with its own independent copy of the cache — there is no cross-instance sharing, so N concurrently-scaled instances means N separate in-memory copies of the same data/artefacts, each loaded independently on its own first request.
- The hosting plan controls how often this happens: a Consumption-style scale-to-zero plan produces frequent cold starts (cache rarely stays warm); an always-on plan keeps at least one instance running so the cache persists much longer. Nothing in this codebase itself schedules cache eviction — it is purely a side effect of process lifecycle.

### 10.5 Rough size of what ends up in memory per warm instance

Based on the live Supabase state recorded in `docs/SUPABASE_STATE.md` (verified 2026-08-08):

| Cached object | Contents | Approx. driver of size |
|---|---|---|
| `DataLoader.user_df` | 92,558 rows | `User` table columns (user aggregates) |
| `DataLoader.item_df` | 27,797 rows | `Item` table columns (item metadata + text) |
| `DataLoader.user_item_df` | 244,009 rows, 22 cols | `Review` table — the largest tabular object |
| `ModelsLoader` (`collaborative_filtering`) | ALS model + interaction matrix + 2 id-mapping dicts | `als_model.joblib` (16.4 MB on disk), `cf_user_item_matrix.npz` (0.9 MB), `idx_to_userid_mapping.json` (2.4 MB), `idx_to_itempasin_mapping.json` (0.5 MB) |
| `ModelsLoader` (`content_based_filtering`) | TF-IDF vectorizer + item matrix + meta DataFrame | `cb_tfidf.joblib` (0.4 MB), `cb_item_matrix.npz` (1.5 MB), `cb_meta.parquet` (1.0 MB) |

These are on-disk serialized sizes (a reasonable lower bound); deserialized in-memory footprint is typically larger than the on-disk size for pandas DataFrames and scikit-learn/`implicit` objects (pandas object-dtype columns and Python-object overhead in particular inflate this). If both tools are loaded in the same process (e.g. via the orchestration layer, which imports both `main.py` modules and therefore creates both sets of singletons), all of the above is resident simultaneously.

---

## 11. Notebooks

| Notebook | Role |
|---|---|
| `notebook/test.ipynb` | Smoke test: instantiates `ModelsLoader()` + `DataLoader()`, verifies `user_item_df` (244,009 rows × 22 cols). |
| `notebook/cf_recommend.ipynb` | CF prototype — the reference implementation refactored into `src/collaborative_filtering/`. Shows matrix shape `(2,589,466 × 89,246)`. |
| `notebook/cb_recommend.ipynb` | CBF prototype — refactored into `src/content_based_filtering/`. Shows matrix shape `(89,251 × 10,000)`. |

Notebooks use `../data/...` / `../models/...` paths and must be run from `notebook/`; production code uses `config.py` absolute `Path`s.

---

## 12. Known caveats & asymmetries

- **`push_into_supabase.py:64` bug**: passes `USER_FILENAME` where `USER_TABLE_NAME` is intended.
- **`wilson_lower_bound`**: `confidence` param unused; z hardcoded to `1.96`.
- **CF cold start**: raises `ValueError` for unknown `user_id` (`recommender.py:48-49`) — no fallback. **CBF cold start**: returns empty result (`recommender.py:111-112`).
- **CF `recommender.py` docstring** claims category boost + free filtering, but the implementation does pure ALS scoring only.
- **`MODEL_ID` lifecycle**: pinned in `tool_config.py` for downloads; `push_artefacts_into_registry` generates a **new** UUID at upload — operator must manually update the pinned `MODEL_ID`, or the "latest by `created_at`" fallback is used.
- **`joblib`** not explicit in `pyproject.toml` (transitive via scikit-learn).
