# Recommendation System — Complete Flow Documentation

This document covers both recommendation pipelines end-to-end: the **Collaborative Filtering (CF)** system
and the **Content-Based Filtering (CB)** system. It is intended as the authoritative reference for
orchestration, Azure Function wiring, and any code that needs to call into either pipeline.

---

## Table of Contents

1. [Repository Layout](#1-repository-layout)
2. [Shared Data Layer](#2-shared-data-layer)
3. [Content-Based Filtering](#3-content-based-filtering)
   - [Module structure](#31-module-structure)
   - [Build phase](#32-build-phase)
   - [Inference entry points](#33-inference-entry-points)  ← **start here for orchestration**
   - [End-to-end inference flow](#34-end-to-end-inference-flow)
   - [Artifacts on disk](#35-artifacts-on-disk)
   - [Hyperparameters](#36-hyperparameters)
4. [Collaborative Filtering](#4-collaborative-filtering)
   - [Module structure](#41-module-structure)
   - [Build phase](#42-build-phase)
   - [Inference entry point](#43-inference-entry-point)  ← **start here for orchestration**
   - [End-to-end inference flow](#44-end-to-end-inference-flow)
   - [Artifacts on disk](#45-artifacts-on-disk)
   - [Hyperparameters](#46-hyperparameters)
5. [Shared Utilities and Dependencies](#5-shared-utilities-and-dependencies)
6. [Data Paths and Environment Variables](#6-data-paths-and-environment-variables)
7. [CLI Reference](#7-cli-reference)
8. [Errors and Edge Cases](#8-errors-and-edge-cases)

---

## 1. Repository Layout

```
amozon_backend/
├── src/
│   ├── collaborative_filtering/
│   │   ├── __init__.py          # empty
│   │   ├── config.py            # paths + hyperparameters
│   │   ├── model.py             # ALS build / load
│   │   ├── recommender.py       # inference logic
│   │   └── main.py              # PUBLIC ENTRY POINTS + CLI
│   │
│   ├── content_based_filtering/
│   │   ├── __init__.py          # re-exports public API
│   │   ├── config.py            # paths + hyperparameters
│   │   ├── model.py             # TF-IDF build / load
│   │   ├── recommender.py       # inference logic
│   │   └── main.py              # PUBLIC ENTRY POINTS + CLI
│   │
│   └── data_loader/
│       ├── __init__.py          # empty
│       ├── config.py            # parquet file paths
│       └── main.py              # load_item, load_user_item, load_meta, build_item_text
│
├── pipelines/
│   └── extract_data.py          # Supabase extraction (ETL only, not used at inference)
│
├── backend/
│   └── models/                  # all trained artifacts land here (created at build time)
│
├── etl/
│   └── data/
│       └── output/              # source parquet files consumed by both pipelines
│
└── pyproject.toml
```

---

## 2. Shared Data Layer

Both pipelines share a single `data_loader` package (`src/data_loader/`). All paths resolve relative
to the repository root via `Path(__file__).resolve().parents[N]`.

### Source parquet files

| Loader function    | File path (from repo root)                        | Required columns                                                                        |
|--------------------|---------------------------------------------------|-----------------------------------------------------------------------------------------|
| `load_meta()`      | `etl/data/output/meta_data.parquet`               | none enforced                                                                           |
| `load_user()`      | `etl/data/output/user.parquet`                    | none enforced                                                                           |
| `load_item()`      | `etl/data/output/item.parquet`                    | `parent_asin`, `is_free` (plus `item_title`, `description`, `features`, `num_item_img`, `num_item_videos`, `price`, `main_category` used downstream) |
| `load_user_item()` | `etl/data/output/user-item-interaction.parquet`   | `user_id`, `parent_asin`, `review_rating`, `recency_weight` (plus `helpful_vote`, `recency_weight`, `review_word_count`, `num_review_img` used by CF build) |

### Utility function

`build_item_text(row)` — concatenates `item_title`, `description[]`, and `features[]` into a single
whitespace-joined string. Used only during the CB build phase by `content_based_filtering/model.py`.

### How parquet files are produced

The raw data originates in Supabase (tables `Item` and `Reviews`). `pipelines/extract_data.py` pulls
them via the Supabase Python client in paginated batches and returns `pd.DataFrame` objects. The
resulting DataFrames must be written to the parquet paths above before either model can be built.
`pipelines/extract_data.py` requires two environment variables:

| Variable              | Usage                     |
|-----------------------|---------------------------|
| `SUPABASE_URL`        | Supabase project URL      |
| `SUPBASE_SECRET_KEY`  | Supabase service-role key (note: typo in source — `SUPBASE`, not `SUPABASE`) |

---

## 3. Content-Based Filtering

### 3.1 Module structure

```
src/content_based_filtering/
├── config.py      — constants: artifact paths, TF-IDF hyperparams, scoring params
├── model.py       — build_and_save(), load_artifacts()
├── recommender.py — build_user_profile(), recommend_for_user(), similar_items()
├── main.py        — get_user_recommendations(), get_similar_items(), build_model(), run_cli()
└── __init__.py    — re-exports: get_user_recommendations, get_similar_items, build_model,
                                  build_and_save, load_artifacts
```

### 3.2 Build phase

Run once (or whenever the item catalog changes) to fit the TF-IDF model and persist artifacts.

**Trigger:**
```bash
python -m content_based_filtering.main --mode build
# or from Python:
from content_based_filtering.main import build_model
build_model()
```

**What happens internally (`model.build_and_save`):**

```
load_item()   →  item.parquet
     │
     build_item_text(row) per row
     Concatenates: item_title + description[] + features[]  →  "text" column
     │
     TfidfVectorizer.fit_transform(text corpus)
       max_features = 10,000
       ngram_range  = (1, 2)   ← unigrams + bigrams
       sublinear_tf = True     ← log-scaled TF
     │
     Persist to backend/models/:
       cb_tfidf.joblib       ← fitted TfidfVectorizer
       cb_item_matrix.npz    ← sparse CSR matrix  (n_items × 10,000)
       cb_meta.parquet       ← parent_asin, item_title, main_category, is_free
```

### 3.3 Inference entry points

> **This is the section to read first when wiring up an Azure Function or orchestrator.**

Both functions live in `src/content_based_filtering/main.py` and are also exported from
`src/content_based_filtering/__init__.py`.

Artifacts are cached at module level (`_artifacts`). The first call within a process triggers a cold
load from disk; all subsequent calls reuse the in-memory objects.

---

#### Entry point A — `get_user_recommendations`

```python
from content_based_filtering.main import get_user_recommendations

results = get_user_recommendations(user_id: str, n: int = 10) -> list[dict]
```

| Parameter | Type  | Required | Description                              |
|-----------|-------|----------|------------------------------------------|
| `user_id` | `str` | yes      | Identifier of the target user            |
| `n`       | `int` | no       | Number of recommendations (default: 10) |

**Returns:** `list[dict]` — one dict per recommended item, with these keys:

| Key             | Type    | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `parent_asin`   | `str`   | Amazon ASIN of the recommended product           |
| `score`         | `float` | Cosine similarity score (post category-boost)    |
| `item_title`    | `str`   | Product title                                    |
| `main_category` | `str`   | Product category                                 |
| `is_free`       | `bool`  | Whether the product is free                      |

Returns an **empty list** if the user has no interaction history with a rating >= 3.

**Raises:**
- `ValueError` — if `user_id` is empty/whitespace, or `n < 1`
- `FileNotFoundError` — if model artifacts have not been built yet
- `OSError` — if artifact files cannot be read

---

#### Entry point B — `get_similar_items`

```python
from content_based_filtering.main import get_similar_items

results = get_similar_items(parent_asin: str, n: int = 10) -> list[dict]
```

| Parameter      | Type  | Required | Description                              |
|----------------|-------|----------|------------------------------------------|
| `parent_asin`  | `str` | yes      | ASIN of the seed item                    |
| `n`            | `int` | no       | Number of similar items (default: 10)   |

**Returns:** `list[dict]` — one dict per similar item, with these keys:

| Key             | Type    | Description                              |
|-----------------|---------|------------------------------------------|
| `parent_asin`   | `str`   | ASIN of the similar product              |
| `score`         | `float` | Cosine similarity score                  |
| `item_title`    | `str`   | Product title                            |
| `main_category` | `str`   | Product category                         |

Note: `is_free` is **not** included in the similar-items response.

Returns an **empty list** if the ASIN is not present in the index.

**Raises:**
- `ValueError` — if `parent_asin` is empty/whitespace, or `n < 1`
- `FileNotFoundError` — if model artifacts have not been built yet
- `OSError` — if artifact files cannot be read

---

### 3.4 End-to-end inference flow

#### User recommendations flow

```
get_user_recommendations(user_id, n)
    │
    _get_artifacts()  [cold: reads disk; warm: returns cached tuple]
    Returns: (tfidf, item_matrix, meta_df, item_to_idx, idx_to_item)
    │
    load_user_item()
    Reads: etl/data/output/user-item-interaction.parquet  ← loaded on EVERY call
    │
    recommender.recommend_for_user(user_id, user_item_df, item_matrix,
                                   meta_df, item_to_idx, idx_to_item, n)
        │
        build_user_profile(user_id, user_item_df, item_matrix, item_to_idx)
            Filter rows:  user_id matches  AND  review_rating >= 3  AND  asin in index
            weight[i]  =  (review_rating[i] / 5)  ×  recency_weight[i]
            profile    =  weighted sum of item vectors  /  sum(weights)
            Shape: (1 × 10,000)
            Returns None if no qualifying history
        │
        If profile is None → return empty DataFrame
        │
        cosine_similarity(profile, item_matrix)  →  scores[n_items]
        │
        Set scores[seen_item_indices] = -1   ← exclude already-interacted items
        │
        Derive user preferences from positive history (rating >= 3):
            top_category  = most frequent main_category
            prefers_free  = (fraction of free liked items > 0.5)
        │
        Take top (n × 5) candidates by score  →  pool DataFrame
        Merge pool with meta_df (item_title, main_category, is_free)
        │
        Apply category boost:  score += 0.1  where main_category == top_category
        Apply free filter:     drop paid items  if prefers_free == True
        │
        Sort by score descending → head(n)
        Return DataFrame[parent_asin, score, item_title, main_category, is_free]
    │
    .to_dict(orient="records")
    Return list[dict]
```

#### Similar items flow

```
get_similar_items(parent_asin, n)
    │
    _get_artifacts()  [cold: reads disk; warm: returns cached tuple]
    │
    recommender.similar_items(parent_asin, item_matrix, meta_df,
                              item_to_idx, idx_to_item, n)
        │
        If parent_asin not in item_to_idx → return empty DataFrame
        │
        idx = item_to_idx[parent_asin]
        cosine_similarity(item_matrix[idx], item_matrix)  →  scores[n_items]
        scores[idx] = -1   ← exclude the seed item itself
        │
        top_idx = argsort(scores)[::-1][:n]
        Build DataFrame[parent_asin, score]
        Merge with meta_df (item_title, main_category)
        Return DataFrame
    │
    .to_dict(orient="records")
    Return list[dict]
```

### 3.5 Artifacts on disk

All artifacts land in `backend/models/` (relative to repo root).

| File                      | Format        | Contents                                       | Used at inference |
|---------------------------|---------------|------------------------------------------------|-------------------|
| `cb_tfidf.joblib`         | joblib        | Fitted `TfidfVectorizer` (10k-word vocabulary) | No (saved for reuse/re-transform) |
| `cb_item_matrix.npz`      | scipy NPZ     | Sparse CSR matrix `(n_items × 10,000)`         | Yes               |
| `cb_meta.parquet`         | Parquet       | `parent_asin`, `item_title`, `main_category`, `is_free` | Yes  |

The artifact tuple returned by `load_artifacts()` is:

```python
(tfidf, item_matrix, meta_df, item_to_idx, idx_to_item)
#  [0]       [1]        [2]       [3]          [4]
```

### 3.6 Hyperparameters

All in `src/content_based_filtering/config.py`.

| Constant                    | Value   | Effect                                                                    |
|-----------------------------|---------|---------------------------------------------------------------------------|
| `TFIDF_MAX_FEATURES`        | 10,000  | Vocabulary size cap for TF-IDF                                            |
| `TFIDF_NGRAM_RANGE`         | (1, 2)  | Unigrams and bigrams                                                      |
| `TFIDF_SUBLINEAR_TF`        | `True`  | Log-scales term frequencies to reduce dominance of high-frequency terms   |
| `DEFAULT_TOP_N`             | 10      | Default `n` when not specified by the caller                              |
| `CANDIDATE_POOL_MULTIPLIER` | 5       | Oversample `n × 5` candidates before post-filters (category boost, free) |
| `MIN_RATING_THRESHOLD`      | 3       | Minimum review rating to count as a "liked" item for profile building     |
| `CATEGORY_BOOST`            | 0.1     | Score bonus applied to items in the user's top category                   |
| `FREE_PREFERENCE_THRESHOLD` | 0.5     | If > 50% of a user's liked items are free, apply free-only filter         |

---

## 4. Collaborative Filtering

### 4.1 Module structure

```
src/collaborative_filtering/
├── config.py      — constants: artifact paths, ALS hyperparams, default n
├── model.py       — build_and_save(), load_artifacts()
├── recommender.py — recommend_for_user()
├── main.py        — get_user_recommendations(), build_model(), run_cli()
└── __init__.py    — empty
```

### 4.2 Build phase

Run once (or whenever interaction data changes substantially) to fit the ALS model.

**Trigger:**
```bash
python -m collaborative_filtering.main --mode build
# or from Python:
from collaborative_filtering.main import build_model
build_model()
```

**What happens internally (`model.build_and_save`):**

```
load_item()       →  item.parquet
load_user_item()  →  user-item-interaction.parquet
     │
     Merge on parent_asin, keeping columns:
       user_id, parent_asin, helpful_vote, recency_weight,
       review_word_count, num_review_img, review_rating,
       num_item_img, num_item_videos, price
     │
     MinMaxScaler on all feature columns (excludes user_id, parent_asin)
     Invert price:  price = 1 - price   ← cheaper is better
     │
     interaction = row-wise mean of all normalised features
     Aggregate to one score per (user, item) via mean
     │
     Encode user_id → userid_idx  (category codes)
     Encode parent_asin → itempasin_idx  (category codes)
     │
     Build sparse CSR matrix: rows=users, cols=items, values=interaction score
     Shape: (n_users × n_items), dtype float32
     │
     implicit.als.AlternatingLeastSquares(
         factors=50, regularization=0.01, iterations=20
     ).fit(interaction_matrix)
     │
     Persist to backend/models/:
       als_model.joblib              ← trained ALS model (latent factors)
       cf_user_item_matrix.npz       ← sparse interaction matrix
       idx_to_userid_mapping.json    ← {str(idx): user_id}
       idx_to_itempasin_mapping.json ← {str(idx): parent_asin}
```

### 4.3 Inference entry point

> **This is the section to read first when wiring up an Azure Function or orchestrator.**

The single inference function lives in `src/collaborative_filtering/main.py`.

Artifacts are cached at module level (`_artifacts`). The first call within a process triggers a cold
load from disk; all subsequent calls reuse the in-memory objects.

---

#### Entry point — `get_user_recommendations`

```python
from collaborative_filtering.main import get_user_recommendations

results = get_user_recommendations(user_id: str, n: int = 10) -> list[dict]
```

| Parameter | Type  | Required | Description                              |
|-----------|-------|----------|------------------------------------------|
| `user_id` | `str` | yes      | Identifier of the target user            |
| `n`       | `int` | no       | Number of recommendations (default: 10) |

**Returns:** `list[dict]` — one dict per recommended item, with these keys:

| Key             | Type   | Description                              |
|-----------------|--------|------------------------------------------|
| `parent_asin`   | `str`  | ASIN of the recommended product          |
| `item_title`    | `str`  | Product title                            |
| `main_category` | `str`  | Product category                         |
| `is_free`       | `bool` | Whether the product is free              |

Note: unlike the CB system, the CF response does **not** include a `score` field. The ALS model's
scores are used internally for ranking but are not surfaced in the output.

**Raises:**
- `ValueError` — if `user_id` is empty/whitespace, `n < 1`, or `user_id` is not in the training set
- `FileNotFoundError` — if model artifacts have not been built yet
- `OSError` — if artifact files cannot be read

**Important:** the CF system raises `ValueError` for an unknown `user_id` (cold-start users are not
handled). The CB system handles unknown users gracefully by returning an empty list. Plan accordingly
in the orchestration layer.

---

### 4.4 End-to-end inference flow

```
get_user_recommendations(user_id, n)
    │
    Input validation: user_id non-empty, n >= 1
    │
    _get_artifacts()  [cold: reads disk; warm: returns cached tuple]
    Returns: (model, user_item_matrix, idx_to_userid, idx_to_itempasin)
    │
    recommender.recommend_for_user(user_id, n, artifacts)
        │
        userid_to_idx = {v: k for k, v in idx_to_userid.items()}  (inverse map)
        user_idx = userid_to_idx.get(user_id)
        If user_idx is None → raise ValueError("Unknown user_id")
        │
        model.recommend(
            userid     = np.int32(user_idx),
            user_items = user_item_matrix[user_idx],   ← row for this user
            N          = n
        )
        Returns: (recomm_idx[], recomm_scores[])
        │
        Map recomm_idx → parent_asin via idx_to_itempasin
        Return list[str]  ← list of parent_asin values
    │
    load_item()
    Reads: etl/data/output/item.parquet  ← loaded on EVERY call
    Filter to items whose parent_asin is in the recommendation list
    Select columns: parent_asin, item_title, main_category, is_free
    │
    .to_dict(orient="records")
    Return list[dict]
```

**Note on implicit.recommend:** The ALS `model.recommend()` call automatically excludes items the
user has already interacted with (via the `user_items` argument). No manual masking is needed.

### 4.5 Artifacts on disk

All artifacts land in `backend/models/` (relative to repo root).

| File                            | Format    | Contents                                                    |
|---------------------------------|-----------|-------------------------------------------------------------|
| `als_model.joblib`              | joblib    | Trained `implicit.als.AlternatingLeastSquares` model        |
| `cf_user_item_matrix.npz`       | scipy NPZ | Sparse CSR interaction matrix `(n_users × n_items)`, float32 |
| `idx_to_userid_mapping.json`    | JSON      | `{"0": "user_id_string", ...}` — integer index to user_id  |
| `idx_to_itempasin_mapping.json` | JSON      | `{"0": "B00XXXXX", ...}` — integer index to parent_asin    |

The artifact tuple returned by `load_artifacts()` is:

```python
(model, interaction_matrix, idx_to_userid, idx_to_itempasin)
#  [0]         [1]               [2]              [3]
```

### 4.6 Hyperparameters

All in `src/collaborative_filtering/config.py`.

| Constant       | Value | Effect                                               |
|----------------|-------|------------------------------------------------------|
| `ALS_FACTORS`  | 50    | Dimensionality of latent factor space                |
| `ALS_REG`      | 0.01  | L2 regularisation strength                          |
| `ALS_ITERA`    | 20    | Number of ALS training iterations                   |
| `DEFAULT_TOP_N`| 10    | Default `n` when not specified by the caller         |

---

## 5. Shared Utilities and Dependencies

| Component              | Used by CF | Used by CB | Notes                                                  |
|------------------------|-----------|-----------|--------------------------------------------------------|
| `data_loader.load_item()`      | Build + inference | Build + inference | CF calls it at inference to enrich output; CB calls it at build only |
| `data_loader.load_user_item()` | Build only        | Inference (every call) | CB re-reads on every inference call — not cached |
| `data_loader.load_meta()`      | Not used          | Build only (via `model.py`) | |
| `data_loader.build_item_text()`| Not used          | Build only | Constructs the text corpus for TF-IDF |
| `backend/models/` directory    | Yes               | Yes        | Both systems write to and read from the same directory |

**Performance note:** `load_user_item()` in the CB system and `load_item()` in the CF system are
called on **every inference request**, not cached. For high-throughput deployments, consider caching
these DataFrames at the module level, analogous to how model artifacts are cached.

---

## 6. Data Paths and Environment Variables

### Artifact paths (auto-resolved from `__file__`)

Both config modules resolve paths relative to the repository root:

```
REPO_ROOT = Path(__file__).resolve().parents[3]
```

| Path                                           | System | Role                   |
|------------------------------------------------|--------|------------------------|
| `{REPO_ROOT}/backend/models/`                  | Both   | All trained artifacts  |
| `{REPO_ROOT}/etl/data/output/item.parquet`     | Both   | Item feature data      |
| `{REPO_ROOT}/etl/data/output/user-item-interaction.parquet` | Both | Interaction data |
| `{REPO_ROOT}/etl/data/output/meta_data.parquet`| CB     | Raw item metadata      |
| `{REPO_ROOT}/etl/data/output/user.parquet`     | Neither at inference | User data (available, not actively used in inference) |

### Environment variables

Only the ETL pipeline (`pipelines/extract_data.py`) requires environment variables. Neither
recommendation system reads environment variables at build or inference time.

| Variable              | Required by       | Description                             |
|-----------------------|-------------------|-----------------------------------------|
| `SUPABASE_URL`        | `extract_data.py` | Supabase project URL                    |
| `SUPBASE_SECRET_KEY`  | `extract_data.py` | Supabase service-role key (**note typo**: `SUPBASE`, not `SUPABASE`) |

These should be set in a `.env` file at the repo root; `extract_data.py` calls `load_dotenv()`.

---

## 7. CLI Reference

Both modules support a CLI via `python -m <module>.main`.

### Content-Based Filtering

```bash
# Fit model and save artifacts
python -m content_based_filtering.main --mode build

# Recommend for a user
python -m content_based_filtering.main --mode user --user_id <user_id> --n 10

# Find similar items
python -m content_based_filtering.main --mode item --asin <parent_asin> --n 10
```

### Collaborative Filtering

```bash
# Fit model and save artifacts
python -m collaborative_filtering.main --mode build

# Recommend for a user
python -m collaborative_filtering.main --mode user --user_id <user_id> --n 10
```

CLI output is printed as a tabular string via `pd.DataFrame.to_string(index=False)`.

---

## 8. Errors and Edge Cases

| Scenario                                        | CB system behaviour                  | CF system behaviour              |
|-------------------------------------------------|--------------------------------------|----------------------------------|
| Unknown `user_id` (not in training data)        | Returns empty list (no profile built) | Raises `ValueError`             |
| User has no ratings >= 3                        | Returns empty list (profile is None) | N/A — ALS still has the user    |
| Unknown `parent_asin` in `get_similar_items`    | Returns empty list                   | N/A — CF has no similar-items function |
| Artifacts not yet built                         | Raises `FileNotFoundError`           | Raises `FileNotFoundError`       |
| `n < 1`                                         | Raises `ValueError`                  | Raises `ValueError`              |
| Empty/whitespace `user_id`                      | Raises `ValueError`                  | Raises `ValueError`              |
| `prefers_free=True` but no free items in pool   | Returns empty list (pool filtered to empty) | N/A                        |

---

## Quick-reference: calling inference from an orchestrator

```python
# --- Content-Based: user recommendations ---
from content_based_filtering.main import get_user_recommendations

recs = get_user_recommendations(user_id="AEHX3IYYFXEA2", n=10)
# Returns: [{"parent_asin": ..., "score": ..., "item_title": ..., "main_category": ..., "is_free": ...}, ...]

# --- Content-Based: similar items ---
from content_based_filtering.main import get_similar_items

similar = get_similar_items(parent_asin="B00ABCDEFG", n=10)
# Returns: [{"parent_asin": ..., "score": ..., "item_title": ..., "main_category": ...}, ...]

# --- Collaborative Filtering: user recommendations ---
from collaborative_filtering.main import get_user_recommendations as cf_get_user_recommendations

recs = cf_get_user_recommendations(user_id="AEHX3IYYFXEA2", n=10)
# Returns: [{"parent_asin": ..., "item_title": ..., "main_category": ..., "is_free": ...}, ...]
# Note: no "score" key in the CF response
```
