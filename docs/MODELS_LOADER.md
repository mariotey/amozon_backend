# Models Loader — Complete Documentation

The `models_loader` package is the **model artefact management layer**. It sits between the training pipelines (`src/<tool>/model.py`) and the inference layer (`src/<tool>/main.py` + `recommender.py`), and is responsible for the entire read/write lifecycle of model artefacts — both locally and in Supabase Storage.

## 1. Module map

| File | Role |
|---|---|
| `__init__.py` | Re-exports `from .models_loader import *` (`read_local_artefacts`, `save_local_artefacts`, `load_artefacts`, `ModelsLoader`). |
| `models_loader.py` | Suffix-aware I/O, cache-or-download orchestration, `ModelsLoader` orchestrator class. |
| `push_into_supabase.py` | Walk each tool dir under `models/`, validate, register + upload to Supabase. (script) |

Imported by: `setup.py` (bootstrap), each `src/<tool>/main.py` (lazy warm-cache loading), each `src/<tool>/model.py` (for `save_local_artefacts`).

---

## 2. Local artefact storage

**Root** (`config.py:17`): `MODEL_ARTEFACT_DIR = REPO_ROOT / "models"`.

**Per-tool layout** (created by `save_local_artefacts` at `models_loader.py:84-85`):
```
<repo_root>/models/
├── collaborative_filtering/
│   ├── als_model.joblib                 (implicit.als.AlternatingLeastSquares)
│   ├── cf_user_item_matrix.npz          (scipy.sparse.csr_matrix, float32)
│   ├── idx_to_userid_mapping.json       ({int idx: str user_id})
│   └── idx_to_itempasin_mapping.json     ({int idx: str parent_asin})
└── content_based_filtering/
    ├── cb_tfidf.joblib                  (sklearn TfidfVectorizer)
    ├── cb_item_matrix.npz               (scipy.sparse, TF-IDF features)
    └── cb_meta.parquet                  (pandas DataFrame)
```

Filenames are static (no version/timestamp in the name). Versioning lives in the Supabase `ModelRegistry` table.

### Filename mapping (from each `tool_config.py`)
**Collaborative filtering** (`src/collaborative_filtering/tool_config.py:11-16`):
```python
MODEL_ARTEFACTS = {
    "als_model_filename":          "als_model.joblib",
    "user_item_matrix_filename":   "cf_user_item_matrix.npz",
    "uid_mapping_filename":        "idx_to_userid_mapping.json",
    "itempasin_mapping_filename":  "idx_to_itempasin_mapping.json"
}
```
**Content-based filtering** (`src/content_based_filtering/tool_config.py:11-15`):
```python
MODEL_ARTEFACTS = {
    "tfidf_filename":       "cb_tfidf.joblib",
    "item_matrix_filename": "cb_item_matrix.npz",
    "meta_filename":        "cb_meta.parquet"
}
```
The **order** of `MODEL_ARTEFACTS.values()` dictates the positional order of the artefact tuple saved/loaded.

---

## 3. Suffix-aware artefact I/O (`models_loader.py`)

### `read_local_artefacts(filename_dir, model_artefacts_dict) -> tuple` (`:26-66`)
For each `filename` in `model_artefacts_dict.values()`, dispatch on suffix:
| Suffix | Reader |
|---|---|
| `.joblib` | `joblib.load(filepath)` |
| `.npz` | `scipy.sparse.load_npz(filepath)` |
| `.json` | `{int(k): v for k,v in json.load(f).items()}` (keys cast to `int` — critical for CF mappings) |
| `.parquet` | `pd.read_parquet(filepath)` |
| else | `raise ValueError` |

Returns a tuple in the order of `model_artefacts_dict.values()`.

### `save_local_artefacts(tool_name, model_artefacts, model_artefacts_dict) -> None` (`:68-109`)
1. `tool_artefact_dir = MODEL_ARTEFACT_DIR / tool_name`; `mkdir(parents=True, exist_ok=True)`.
2. `zip(model_artefacts, model_artefacts_dict.values())` pairs each artefact with its filename **positionally** — order discipline is the caller's responsibility.
3. Dispatch on suffix:
| Suffix | Writer |
|---|---|
| `.joblib` | `joblib.dump(artefact, filepath)` |
| `.npz` | `scipy.sparse.save_npz(filepath, artefact)` |
| `.json` | `json.dump({str(k): v for k,v in artefact.items()}, f)` (keys stringified) |
| `.parquet` | `artefact.to_parquet(filepath, index=False)` |
| else | `raise ValueError` |

Called by both `src/collaborative_filtering/model.py:101` and `src/content_based_filtering/model.py:72` after training.

### `load_artefacts(tool_config, force_remote=False) -> tuple` (`:111-167`)
1. Unpack `MODEL_NAME`, `MODEL_ID`, `MODEL_ARTEFACTS` from `tool_config`.
2. `tool_artefact_dir = MODEL_ARTEFACT_DIR / tool_name`; ensure it exists.
3. Build `missing_files` (any expected filename not on disk).
4. Branch: `if force_remote or missing_files:`
   - **`force_remote=True`** (regardless of what's cached) or **files missing** → `download_artefacts_from_supabase(tool_name, model_id, model_artefacts_dict)`, then `read_local_artefacts(...)`. (Log message differs: "Fetching latest artefacts..." for `force_remote`, "Missing N artefacts..." for a genuine cache miss.)
   - **`force_remote=False` and all cached** → `read_local_artefacts(...)` directly, no Supabase contact.
5. Return the loaded tuple.

Always materialises files on disk first (even under `force_remote`, the download still writes to `tool_artefact_dir` before reading back) — there is no in-memory-only remote path.

### `class ModelsLoader` (`:169-220`)
**`__init__(self, tools: list[str] | None = None, force_remote: bool = False)`**
1. `self.model_artefacts = {}`.
2. `selected_tools = tools if tools is not None else TOOLS` (fallback to `config.TOOLS`).
3. For each `tool`:
   - `tool_config_path = REPO_ROOT / "src" / tool / "tool_config.py"`; raise `FileNotFoundError` if absent.
   - `tool_config = importlib.import_module(f"src.{tool}.tool_config")`.
   - `self.model_artefacts[tool] = load_artefacts(tool_config=tool_config, force_remote=force_remote)`.

Every loaded artefact (the fitted model object, sparse matrices, id-mapping dicts, metadata DataFrame) is deserialized in full and held **entirely in memory** in `self.model_artefacts[tool]` — no lazy/partial loading — for the lifetime of the `ModelsLoader` instance.

Entry points: `setup.py:34` (`ModelsLoader(tools=TOOLS)`, default `force_remote=False`), `src/collaborative_filtering/main.py` and `src/content_based_filtering/main.py` (both `ModelsLoader(tools=[MODEL_NAME], force_remote=...)` via `get_models_loader()` — `force_remote=True` only at inference, see `docs/CB_CF_FILTERING.md` and `docs/ARCHITECTURE.md#10-inference-time-loading-into-cache`).

Recommenders access `artefact_loader.model_artefacts[MODEL_NAME]` (CF: `main.py:117`; CBF: `main.py:117,154`).

---

## 4. Push to Supabase Storage bucket

**Bucket** (`config.py:22`): `MODEL_ARTEFACT_BUCKET = "ModelArtefacts"`.
**Storage handle** (`utils/supabase_utils.py:25`): `ARTEFACTS_STORAGE = SUPABASE_CLIENT.storage.from_(MODEL_ARTEFACT_BUCKET)`.

### `push_artefacts_into_registry(tool_name) -> (model_id, storage_path)` (`utils/supabase_utils.py:163-196`)
1. `reconcile_registry(tool_name)` — cleans up stale/orphaned entries first.
2. `model_id = str(uuid.uuid4())` — fresh UUID4 per push.
3. `storage_path = f"{tool_name}/{model_id}"`.
4. Insert `ModelRegistry` row `{"model_id", "tool", "storage_path"}`.
5. Return `(model_id, storage_path)`.

### `push_artefacts_into_supabase(tool_dir, model_artefacts_dict, storage_path) -> None` (`:198-221`)
For each `filename` in `model_artefacts_dict.values()`: open `tool_dir / filename` in binary, `ARTEFACTS_STORAGE.upload(path=f"{storage_path}/{filename}", file=f, file_options={"upsert": "true"})`.

**Resulting storage layout:**
```
ModelArtefacts/
├── collaborative_filtering/<uuid4>/
│   ├── als_model.joblib
│   ├── cf_user_item_matrix.npz
│   ├── idx_to_userid_mapping.json
│   └── idx_to_itempasin_mapping.json
└── content_based_filtering/<uuid4>/
    ├── cb_tfidf.joblib
    ├── cb_item_matrix.npz
    └── cb_meta.parquet
```

---

## 5. Pull path (read at setup / inference)

`ModelsLoader.__init__` → `load_artefacts` → `download_artefacts_from_supabase` (when local files missing).

### `download_artefacts_from_supabase(tool_name, model_id, model_artefacts_dict) -> None` (`utils/supabase_utils.py:223-274`)
1. **Primary** (`:238-248`): query `ModelRegistry` for `tool == tool_name AND model_id == model_id`, select `storage_path`, limit 1.
2. **Fallback** (`:250-261`): on any exception, re-query the **latest** version (`order("created_at", desc=True).limit(1)`).
3. `storage_path = response.data[0]["storage_path"]`.
4. For each `filename`:
   - `remote_path = f"{storage_path}/{filename}"`.
   - `local_path = MODEL_ARTEFACT_DIR / tool_name / filename` (model_id NOT preserved locally — cache is always the latest pulled version, flattened).
   - `ARTEFACTS_STORAGE.download(remote_path)` → write bytes to `local_path`.

After download, `load_artefacts` calls `read_local_artefacts` to deserialize.

### Where `MODEL_ID` comes from
Each tool's `tool_config.py` hardcodes a pinned `MODEL_ID`:
- CF: `9139d794-f9b3-4188-9370-33ceb92111fd`
- CBF: `fefdee34-e3fe-4314-acab-f911c02680d3`

If the pinned ID isn't found in the registry, the fallback grabs the most recent by `created_at`.

---

## 6. `ModelRegistry` table

| Column | Type (inferred) | Notes |
|---|---|---|
| `model_id` | str (UUID4) | Inserted at `push_artefacts_into_registry:188`. |
| `tool` | str | Inserted at `:189`. |
| `storage_path` | str (`"{tool_name}/{model_id}"`) | Inserted at `:190`. |
| `created_at` | timestamp | Used for `order("created_at", desc=True)` fallback; db default `now()`. |

### Operations
1. **Insert** — `push_artefacts_into_registry`.
2. **Lookup by ID** — `download_artefacts_from_supabase` primary path.
3. **Lookup latest** — fallback path.
4. **Reconcile** (`reconcile_registry`, `:105-161`): per tool, fetch all `model_id`s from registry; list bucket at prefix `tool_name` (treat entries with `metadata is None` as directories); compute `stale_registry_ids = model_ids - bucket_ids` → delete from registry; `orphaned_storage_ids = bucket_ids - model_ids` → remove their files from storage. Invoked at the start of every push.

### Model ID lifecycle
- **Created**: UUID4 in `push_artefacts_into_registry`.
- **Pinned**: hardcoded in each `tool_config.py:MODEL_ID` for inference reproducibility.
- **Retrieved**: by exact match (preferred) or latest by `created_at` (fallback).
- **Deleted**: by `reconcile_registry` if the matching storage folder has been removed.

---

## 7. Full artefact lifecycle

```
BUILD TIME                                  |  RUNTIME
                                            |
src/<tool>/main.py build_model()            |  src/<tool>/main.py get_user_recommendations()
      │                                     |        │
      ▼                                     |        ▼
src/<tool>/model.py build_and_save()        |  ModelsLoader(tools=[MODEL_NAME])
      │  (fits ALS / TF-IDF)                |        │
      ▼                                     |        ▼
save_local_artefacts(MODEL_NAME, ...)        |  load_artefacts(tool_config)
      │  (models_loader.py:68)              |        │
      ▼                                     |        ├── All artefacts cached locally?
Serialise by suffix into                    |        │   YES → read_local_artefacts()
models/<tool_name>/<filename>               |        │   NO  → download_artefacts_from_supabase()
(.joblib / .npz / .json / .parquet)         |        │           │
      │                                     |        │           ▼
      ▼                                     |        │     Query ModelRegistry for MODEL_ID
[Optional] python -m models_loader.push_into_supabase  │     (fallback: latest by created_at)
      │                                     |        │           │
      ▼                                     |        │           ▼
for tool_dir in models/*:                   |        │     ARTEFACTS_STORAGE.download(remote_path)
 ├─ import src.<tool>.tool_config           |        │     → write to models/<tool>/<filename>
 ├─ validate all files exist (else ValueErr)|        │           │
 ├─ push_artefacts_into_registry(tool_name) |        │           ▼
 │   ├─ reconcile_registry(tool_name)       |        │     read_local_artefacts() → deserialise
 │   ├─ model_id = uuid.uuid4()             |        │           │
 │   └─ INSERT INTO ModelRegistry(...)      |        │           ▼
 └─ push_artefacts_into_supabase(...)       |        │     self.model_artefacts[tool] = tuple
      └─ ARTEFACTS_STORAGE.upload(...)      |        │           │
                                            |        │           ▼
                                            |     recommender.recommend_for_user(...)
```

**Narrative:**
1. **Build** — `src/<tool>/main.py:build_model` fetches `DataLoader`, calls `build_and_save(...)` in `src/<tool>/model.py`.
2. **Save locally** — `save_local_artefacts` writes each artefact to `models/<tool_name>/<filename>` via suffix dispatch.
3. **(Optional) Push to Supabase** — `models_loader.push_into_supabase` (script): validate, reconcile + register, upload.
4. **Pin version** (manual) — operator edits `src/<tool>/tool_config.py:MODEL_ID` to the new UUID.
5. **Pull at setup/run** — `ModelsLoader` → `load_artefacts` → `download_artefacts_from_supabase` if cache incomplete.
6. **Load into memory** — `read_local_artefacts` (suffix-aware deserialisation).
7. **Inference** — `src/<tool>/main.py` accesses `artefact_loader.model_artefacts[MODEL_NAME]` and calls the recommender.
8. **Cache invalidation** — `main.py:build_model` resets `_data_loader_obj` and `_models_loader_obj` to `None` (CF: `main.py:162-163`; CBF: `main.py:183-184`).

> **`force_remote` caveat**: `get_models_loader(force_remote=True)` (used by the inference handlers) only has an effect on the call that **creates** `_models_loader_obj` — i.e. the first call after cold start or after `build_model()` clears the cache. Once `_models_loader_obj` is set, every later call reuses that cached object regardless of the `force_remote` value passed in, since the `if _models_loader_obj is None:` check short-circuits before `force_remote` is ever consulted again. See `docs/ARCHITECTURE.md#10-inference-time-loading-into-cache` for the full warm/cold walkthrough.

---

## 8. Config constants & env vars

| Constant | Value | Used by |
|---|---|---|
| `REPO_ROOT` (`config.py:9`) | repo dir | `ModelsLoader.__init__:194` to locate `src/<tool>/tool_config.py`. |
| `MODEL_ARTEFACT_DIR` (`config.py:17`) | `REPO_ROOT/models` | `save_local_artefacts:84`, `load_artefacts:138`, `push_into_supabase.py:24`, `download_artefacts_from_supabase:267`. |
| `TOOLS` (`config.py:59-62`) | `["collaborative_filtering","content_based_filtering"]` | `ModelsLoader.__init__:191` fallback; `setup.py:34`. |
| `MODEL_ARTEFACT_BUCKET` (`config.py:22`) | `"ModelArtefacts"` | `utils/supabase_utils.py:13,25`. |
| `MODELREGISTRY_TABLE_NAME` (`config.py:28`) | `"ModelRegistry"` | `utils/supabase_utils.py:12,119,143,185,242,255`. |

Env vars (loaded in `utils/supabase_utils.py:18`): `SUPABASE_URL`, `SUPBASE_SECRET_KEY` (deliberate typo).

---

## 9. Dependencies
- **Third-party**: `pandas`, `joblib` (serialisation), `scipy.sparse`, `supabase`, `python-dotenv`, `tqdm`, `implicit` (CF), `scikit-learn` (CF MinMaxScaler, CBF TfidfVectorizer).
- **Standard library**: `json`, `importlib`, `pathlib`, `typing`, `types`, `os`, `uuid`, `argparse`/`logging`/`sys`.
- **Internal**: `utils.supabase_utils`, `config`, each `src/<tool>/tool_config`.
- **Note**: `joblib` is not an explicit dependency in `pyproject.toml` — supplied transitively by scikit-learn.
