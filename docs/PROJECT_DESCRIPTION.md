# Amazon Software Recommender — Project Description

> A Python 3.13 recommender-system backend that builds two complementary recommendation engines (Collaborative Filtering and Content-Based Filtering) over the **Amazon Reviews 2023 — Software** dataset, persisting processed data and model artefacts to a Supabase backend and serving recommendations through importable handlers designed for serverless runtimes (Azure Functions).

---

## 1. What the project does

Given the Amazon Software reviews/metadata dataset (McAuley-Lab/Amazon-Reviews-2023, `Software` category, 5% review split), the backend:

1. **Ingests & cleans** raw reviews and item metadata from the Hugging Face Hub.
2. **Engineers features** into three processed tables: `User`, `Item`, and `Review` (user–item interactions).
3. **Persists** those tables to Supabase Postgres and local Parquet caches.
4. **Trains two recommender models**:
   - **Collaborative Filtering (CF)** — implicit-feedback **ALS matrix factorization** (via the `implicit` library) over a user×item interaction matrix built from normalized review + item features.
   - **Content-Based Filtering (CBF)** — **TF-IDF + cosine similarity** over item textual metadata (title, description, features), with category-boost and free-item-preference re-ranking.
5. **Registers and stores** model artefacts in Supabase (`ModelRegistry` table + `ModelArtefacts` storage bucket) with version reconciliation.
6. **Serves recommendations** via importable, lazily-cached Python handlers (`get_user_recommendations`, `get_similar_items`) and thin CLIs, intended to be wrapped by an Azure Function HTTP trigger.

---

## 2. The two recommendation systems at a glance

| Aspect | Collaborative Filtering (CF) | Content-Based Filtering (CBF) |
|---|---|---|
| Signal source | Similar users' behavior (user–item interactions) | Item features + a user's own history |
| Algorithm | ALS matrix factorization (`implicit.als`) | TF-IDF vectorization + cosine similarity |
| Inputs | User–item interaction matrix (8 normalized features) | Item text (title + description + features) |
| Needs item metadata | Only item numeric features for the interaction score | Yes (textual content) |
| Cold start (new user) | Raises `ValueError` — no fallback | Returns empty list (graceful) |
| Cold start (new item) | Poor (must be in training matrix) | Better (works if item text is in the matrix) |
| Personalization | High | Medium–High |
| Novelty | Higher (peer behavior) | Lower (limited to known preferences) |
| Output | Top-N `parent_asin` enriched with item metadata | Top-N items with similarity `score` + metadata |
| Extra surfaces | `get_user_recommendations` only | `get_user_recommendations` **and** `get_similar_items` |

---

## 3. Technology stack

- **Language**: Python ≥ 3.13 (managed with `uv`).
- **Data**: pandas, numpy, pyarrow, fastparquet, Hugging Face `datasets`.
- **ML**: `implicit` (ALS), scikit-learn (`TfidfVectorizer`, `MinMaxScaler`, `cosine_similarity`), scipy.sparse.
- **Backend / persistence**: Supabase (Postgres tables + Storage bucket), `supabase-py`.
- **Serving**: importable handlers optimized for warm serverless invocations (FastAPI + uvicorn are declared as dependencies but not yet wired).
- **Dev**: Jupyter notebooks for prototyping.

---

## 4. Repository layout

```
amozon_backend/
├── config.py              # global constants (paths, table names, dataset config, TOOLS)
├── setup.py               # one-time bootstrap: mkdirs + DataLoader + ModelsLoader
├── pyproject.toml         # PEP 621 metadata + dependencies (uv-managed)
├── utils/supabase_utils.py    # shared Supabase client + 6 helper functions
├── data_loader/          # ETL: download → transform → push to / pull from Supabase
├── models_loader/        # model artefact push/pull + suffix-aware I/O
├── src/
│   ├── collaborative_filtering/   # ALS tool (tool_config, model, recommender, main)
│   └── content_based_filtering/  # TF-IDF tool (tool_config, model, recommender, main)
├── notebook/             # prototype notebooks (cf_recommend, cb_recommend, test)
├── data/    (gitignored) # input/ raw parquets, output/ processed parquets (DataLoader cache)
└── models/  (gitignored) # per-tool artefact cache
```

---

## 5. End-to-end workflows

### Data pipeline (refresh)
```
HuggingFace Hub ──data_download.py──► data/input/{review_data,meta_data}.parquet
                                            │
                                   data_transform.py (feature engineering)
                                            ▼
                                  data/output/{user,item,user-item-interaction}.parquet
                                            │
                                  push_into_supabase.py ──► Supabase Tables (User/Item/Review)
```

### Model pipeline (build → register → serve)
```
src/<tool>/model.py build_and_save() ──► models/<tool>/*.joblib|.npz|.json|.parquet
                                                  │
                                  models_loader/push_into_supabase.py
                                                  │ ──► Supabase ModelRegistry (table) +
                                                  │     ModelArtefacts (storage bucket)
[Inference runtime]
setup.py ──► DataLoader (cache tables) + ModelsLoader (cache artefacts)
                                                  │
src/<tool>/main.py  ── get_data_loader()/get_models_loader() (lazy cached singletons)
     ├── recommender.py (pure recommendation logic)
     └── exposed as importable handler fns ──► Azure Function / CLI / notebook
```

---

## 6. Setup & usage

### Prerequisites
- Python 3.13+ (not Anaconda/Miniconda).
- A `.env` file in the repo root with:
  ```env
  SUPABASE_URL=<your-supabase-url>
  SUPABASE_PUBLIC_KEY=<your-key>   # documented but not currently read by source
  SUPBASE_SECRET_KEY=<your-service-role-key>   # NOTE: deliberate typo "SUPBASE" — must match exactly
  ```
- A Supabase project with tables `User`, `Item`, `Review`, `ModelRegistry` and a Storage bucket named `ModelArtefacts`.

### Bootstrap a fresh environment
```bash
pip install uv
uv sync
python -m setup          # creates dirs, pulls tables + artefacts from Supabase
```

### Build a model
```bash
python -m src.collaborative_filtering.main --mode build
python -m src.content_based_filtering.main --mode build
```

### Get recommendations
```bash
# Collaborative filtering
python -m src.collaborative_filtering.main --mode user --user_id <id> --n 10

# Content-based filtering
python -m src.content_based_filtering.main --mode user --user_id <id> --n 10
python -m src.content_based_filtering.main --mode item --asin <parent_asin> --n 10
```

### Publish built artefacts to Supabase (after a build)
```bash
python -m models_loader.push_into_supabase
```

---

## 7. Documentation index

| Document | Contents |
|---|---|
| `docs/PROJECT_DESCRIPTION.md` | This file — high-level overview. |
| `docs/ARCHITECTURE.md` | Complete architecture, layers, data/model flows, tool contract, Supabase schema. |
| `docs/DATA_LOADER.md` | Complete data loader: download, transform, push, pull, schemas, transformations. |
| `docs/MODELS_LOADER.md` | Complete models loader: artefact I/O, registry, push/pull lifecycle. |
| `docs/CB_CF_FILTERING.md` | Both recommendation systems in detail: algorithms, training, inference. |
| `AGENTS.md` | Agent operating guide for this repository. |

---

## 8. Known caveats

- The `tool_config.py` files are **constants modules**, not FastAPI/MCP tool definitions — the "tool" abstraction is the ModelsLoader's discovery contract.
- `data_loader/push_into_supabase.py:64` passes `USER_FILENAME` (`"user.parquet"`) where `USER_TABLE_NAME` (`"User"`) is intended — a latent bug.
- `joblib` is used by `models_loader` but is not an explicit dependency in `pyproject.toml` (supplied transitively by scikit-learn).
- The `confidence` parameter of `wilson_lower_bound` (in `data_transform.py`) is unused; the z-score is hardcoded to `1.96`.
- CF raises `ValueError` for unknown users; CBF returns an empty result — an asymmetry callers must handle.
