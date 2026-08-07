# Collaborative & Content-Based Filtering — Complete Documentation

The backend ships two recommender "tools" under `src/`, each a self-contained package following the project's tool-config contract (`tool_config.py` → `model.py` → `recommender.py` → `main.py`). Both are trained over the Amazon Reviews 2023 Software subset and served through lazily-cached handlers designed for serverless runtimes.

---

# Part A — Collaborative Filtering (ALS)

`src/collaborative_filtering/` — implicit-feedback **ALS matrix factorization** (via the `implicit` library) over a user×item interaction matrix built from normalized review + item features.

## A.1 Module map

| File | Role |
|---|---|
| `__init__.py` | One-line docstring package marker. |
| `tool_config.py` | Constants: `MODEL_NAME`, `MODEL_ID`, `MODEL_ARTEFACTS`, ALS hyperparams, `DEFAULT_TOP_N`. |
| `model.py` | Training: build CSR user–item matrix, fit ALS, save artefacts. |
| `recommender.py` | Inference: map `user_id` → internal index, `model.recommend`, return ASINs. |
| `main.py` | Lazy/cached loaders, public handler, CLI. |

## A.2 `tool_config.py` constants

| Constant | Value | Meaning |
|---|---|---|
| `MODEL_NAME` | `"collaborative_filtering"` (`Path(__file__).parent.name`) | Tool dir + storage prefix. |
| `MODEL_ID` | `"9139d794-f9b3-4188-9370-33ceb92111fd"` | Pinned Supabase registry version to download. |
| `MODEL_ARTEFACTS` | 4-entry dict (see below) | Logical name → filename; order defines the artefact tuple. |
| `ALS_FACTORS` | `50` | Latent dimensionality. |
| `ALS_REG` | `0.01` | L2 regularization. |
| `ALS_ITERA` | `20` | ALS iterations. |
| `DEFAULT_TOP_N` | `10` | Default recommendation count. |

```python
MODEL_ARTEFACTS = {
    "als_model_filename":          "als_model.joblib",
    "user_item_matrix_filename":   "cf_user_item_matrix.npz",
    "uid_mapping_filename":        "idx_to_userid_mapping.json",
    "itempasin_mapping_filename":  "idx_to_itempasin_mapping.json"
}
```

## A.3 Model architecture

**Algorithm**: `implicit.als.AlternatingLeastSquares` — factorizes `R ≈ U · Vᵀ` by alternating ridge-regression subproblems. **No explicit alpha confidence scaling** is applied; the CSR data values themselves are the preference/confidence weights.

**User–item interaction matrix construction** (`model.py:15-101` `build_and_save(item_df, user_item_df)`):
1. Merge `user_item_df` with `item_df[["parent_asin","num_item_img","num_item_videos","price"]]` on `parent_asin`.
2. Identify 8 feature columns (excluding `user_id`, `parent_asin`):
   - From `user_item_df`: `helpful_vote`, `recency_weight`, `review_word_count`, `num_review_img`, `review_rating`.
   - From `item_df`: `num_item_img`, `num_item_videos`, `price`.
3. `MinMaxScaler` fit-transform to `[0,1]`.
4. Invert price: `price = 1 - price` (cheaper → higher value).
5. `interaction = mean` across all 8 post-inversion columns (unweighted).
6. `groupby(["user_id","parent_asin"]).mean()` — aggregate multiple reviews of same (user, item).
7. Cast `user_id`/`parent_asin` to `category`; assign integer codes; build reverse maps `idx_to_userid = {idx: user_id}`, `idx_to_itempasin = {idx: parent_asin}`.
8. `scipy.sparse.csr_matrix((interaction, (userid_idx, itempasin_idx)), shape=(num_users, num_items))`, dtype `float32`.
   - Real-world shape (per notebook): **(2,589,466 users × 89,246 items)**, ~4,828,480 non-zeros.
9. `AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20).fit(interaction_matrix)`.

**Artefact tuple** (order matches `MODEL_ARTEFACTS.values()`):
`(model, interaction_matrix, idx_to_userid, idx_to_itempasin)`.

## A.4 Inference (`recommender.py:recommend_for_user`)

`recommend_for_user(user_id, artefacts, n=10) -> list[str]`:
1. Validate `n >= 1`.
2. Unpack `(model, user_item_matrix, idx_to_userid, idx_to_itempasin)`.
3. Invert `idx_to_userid` → `userid_to_idx = {user_id: idx}` (rebuilt per call).
4. `user_idx = userid_to_idx.get(user_id)`. If `None` → `raise ValueError("Unknown user_id: ...")`. **No cold-start fallback.**
5. `model.recommend(userid=np.int32(user_idx), user_items=user_item_matrix[np.int32(user_idx)], N=n)` → `(recomm_idx, recomm_scores)`. `implicit` automatically excludes items already present in `user_items`.
6. Return `[idx_to_itempasin[i] for i in recomm_idx]`.

> ⚠️ The `recommender.py` docstring (lines 23-25) claims category boost + free filtering, but the implementation does **pure ALS scoring only**.

## A.5 `main.py` handlers

| Function | Behavior |
|---|---|
| `get_data_loader(force_remote=False)` (`:36-60`) | Lazily instantiate + cache `DataLoader`. `force_remote` only takes effect on the call that creates the cache. |
| `get_models_loader(force_remote=False)` (`:62-88`) | Lazily instantiate + cache `ModelsLoader(tools=[MODEL_NAME])`. Same first-call-only caveat. |
| `get_user_recommendations(user_id, n=10)` (`:92-144`) | Calls `get_data_loader(force_remote=True)` / `get_models_loader(force_remote=True)` → validate inputs → call `recommend_for_user` → enrich ASINs with `item_df[["parent_asin","item_title","main_category","is_free"]]` → return `list[dict]`. |
| `build_model()` (`:148-176`) | Calls `get_data_loader()` (default `force_remote=False`) → `build_and_save(data_loader.item_df, data_loader.user_item_df)` → reset both cached singletons. |

> Loading-into-cache mechanics (local-cache-vs-Supabase precedence, singleton reuse across warm calls, cold-start/scale-out behavior, approximate in-memory footprint) are covered in full in `docs/ARCHITECTURE.md#10-inference-time-loading-into-cache`.

### CLI (`python -m src.collaborative_filtering.main`)
- `--mode build` → `build_model()`.
- `--mode user --user_id <id> --n <int>` → `get_user_recommendations` → print pandas table.
- Catches `FileNotFoundError`/`OSError`/`ValueError` → `sys.exit(1)`.

## A.6 ID mapping
- Supabase `User.user_id` ↔ ALS row index via `userid_to_idx` (built from `idx_to_userid_mapping.json`).
- ALS column index ↔ Supabase `Item.parent_asin` via `idx_to_itempasin` (`idx_to_itempasin_mapping.json`).
- JSON keys are stringified on save; `read_local_artefacts` casts back to `int`.

## A.7 Deployment note
The notebook shows an OpenBLAS threading warning from `implicit.cpu.als`; set `OPENBLAS_NUM_THREADS=1` (or `threadpoolctl.threadpool_limits`) to avoid severe ALS slowdown in production.

---

# Part B — Content-Based Filtering (TF-IDF + Cosine Similarity)

`src/content_based_filtering/` — TF-IDF over item textual metadata + cosine similarity against a user's weighted TF-IDF profile, with category-boost and free-item-preference re-ranking.

## B.1 Module map

| File | Role |
|---|---|
| `__init__.py` | Re-exports `build_model, get_similar_items, get_user_recommendations, build_and_save`. |
| `tool_config.py` | Constants: `MODEL_NAME`, `MODEL_ID`, `MODEL_ARTEFACTS`, TF-IDF hyperparams, recommendation hyperparams. |
| `model.py` | Training: build text column, fit TF-IDF, save artefacts. |
| `recommender.py` | Inference: `build_user_profile`, `recommend_for_user`, `similar_items`, index maps. |
| `main.py` | Lazy/cached loaders, public handlers, CLI. |

## B.2 `tool_config.py` constants

| Constant | Value | Meaning |
|---|---|---|
| `MODEL_NAME` | `"content_based_filtering"` | Tool dir + storage prefix. |
| `MODEL_ID` | `"fefdee34-e3fe-4314-acab-f911c02680d3"` | Pinned registry version. |
| `MODEL_ARTEFACTS` | 3-entry dict (see below) | Logical name → filename; order defines the artefact tuple. |
| `TFIDF_MAX_FEATURES` | `10_000` | Vocab size. |
| `TFIDF_NGRAM_RANGE` | `(1, 2)` | Unigrams + bigrams. |
| `TFIDF_SUBLINEAR_TF` | `True` | `tf := 1 + log(tf)`. |
| `DEFAULT_TOP_N` | `10` | Default recommendation count. |
| `CANDIDATE_POOL_MULTIPLIER` | `5` | Pool = `n * 5` before re-ranking. |
| `MIN_RATING_THRESHOLD` | `3` | Only reviews ≥3 count as positive history. |
| `CATEGORY_BOOST` | `0.1` | Additive bonus for items in user's top category. |
| `FREE_PREFERENCE_THRESHOLD` | `0.5` | If >50% of positive-history items are free, user "prefers free". |

```python
MODEL_ARTEFACTS = {
    "tfidf_filename":       "cb_tfidf.joblib",
    "item_matrix_filename": "cb_item_matrix.npz",
    "meta_filename":        "cb_meta.parquet"
}
```

## B.3 Model architecture

**Features used**: only item-side textual metadata — no user features, no collaborative signals.

Per-item text corpus (`model.py:17-39` `build_item_text(row)`):
- Start with `item_title`.
- Append each truthy element of `description` (list) and `features` (list).
- Whitespace-join into one string. No extra normalization — delegated to `TfidfVectorizer` defaults.

**Algorithm** (`model.py:41-72` `build_and_save(item_df)`):
1. Deep-copy input; apply `build_item_text` row-wise into `"text"` column.
2. Keep `["parent_asin","item_title","main_category","text","is_free"]`, reset index.
3. `TfidfVectorizer(max_features=10_000, ngram_range=(1,2), sublinear_tf=True).fit_transform(text)` → `item_matrix` (scipy CSR, shape `(n_items, 10_000)`).
   - Per notebook: **89,251 items × 10,000** features.
4. Artefact tuple (order matches `MODEL_ARTEFACTS.values()`):
   `(tfidf, item_matrix, item_meta_df.drop(columns=["text"]))` — the saved `cb_meta.parquet` carries only `parent_asin, item_title, main_category, is_free`.

**Similarity**: `sklearn.metrics.pairwise.cosine_similarity` (TfidfVectorizer L2-normalizes rows by default).

## B.4 Inference (`recommender.py`)

### `build_user_profile(user_id, user_item_df, item_matrix, item_to_idx) -> np.ndarray | None` (`:20-52`)
1. Filter `user_item_df` to the user's rows.
2. Keep rows where `parent_asin` is known to the model **and** `review_rating >= MIN_RATING_THRESHOLD (3)`.
3. If empty → return `None`.
4. Per-item weights = `(review_rating / 5) * recency_weight`.
5. Profile = `(weights @ item_matrix[indices]) / (weights.sum() + 1e-9)` — weighted average of item TF-IDF rows.
6. Return reshaped `(1, n_features)`.

### `build_item_index_maps(meta_df) -> (item_to_idx, idx_to_item)` (`:54-77`)
Bidirectional `parent_asin ↔ row_index` from `meta_df["parent_asin"]` order.

### `recommend_for_user(user_id, user_item_df, artefacts, n=10) -> pd.DataFrame` (`:79-170`)
1. Unpack `(_, item_matrix, meta_df) = artefacts` — the tfidf vectorizer is **not used at inference** (matrix already materialized).
2. Build index maps.
3. Validate `n >= 1`.
4. Build user profile; if `None` → return empty DataFrame (**graceful cold start**).
5. `scores = cosine_similarity(profile, item_matrix).flatten()`.
6. **Mask seen items**: set `scores[seen_idx] = -1` for all items the user interacted with (any rating).
7. **Derive preferences** from positive history (`review_rating >= 3`):
   - `top_category` = mode of `main_category` over positive history, or `None`.
   - `prefers_free` = `hist_meta["is_free"].mean() > 0.5`.
8. **Candidate pool**: top `n * CANDIDATE_POOL_MULTIPLIER` (= 5n) by descending score; join with `item_title`, `main_category`, `is_free`.
9. **Category boost**: add `0.1` to score for pool items whose `main_category == top_category`.
10. **Free filter**: if `prefers_free`, drop non-free rows.
11. Sort by (boosted) score desc, `.head(n)`.

Return schema: `["parent_asin","score","item_title","main_category","is_free"]`.

### `similar_items(parent_asin, artefacts, n=10) -> pd.DataFrame` (`:172-217`)
1. Unpack artefacts, build index maps.
2. Validate `n >= 1`.
3. If ASIN unknown → return empty DataFrame.
4. `idx = item_to_idx[parent_asin]`.
5. `scores = cosine_similarity(item_matrix[idx], item_matrix).flatten()`.
6. Self-mask: `scores[idx] = -1`.
7. Top-N via `np.argsort(scores)[::-1][:n]`.
8. Join metadata (`item_title`, `main_category` — **no** `is_free`).

Return schema: `["parent_asin","score","item_title","main_category"]`.

## B.5 `main.py` handlers

| Function | Behavior |
|---|---|
| `get_data_loader(force_remote=False)` (`:39-63`) | Lazily instantiate + cache `DataLoader`. `force_remote` only takes effect on the call that creates the cache. |
| `get_models_loader(force_remote=False)` (`:65-91`) | Lazily instantiate + cache `ModelsLoader(tools=[MODEL_NAME])`. Same first-call-only caveat. |
| `get_user_recommendations(user_id, n=10)` (`:95-145`) | Calls `get_data_loader(force_remote=True)` / `get_models_loader(force_remote=True)` → validate → `recommend_for_user(user_id, data_loader.user_item_df, artefacts, n)` → `to_dict(orient="records")`. |
| `get_similar_items(parent_asin, n=10)` (`:147-182`) | Calls `get_models_loader(force_remote=True)` → validate → `similar_items(parent_asin, artefacts, n)` → `to_dict(orient="records")`. **No data loader needed** (similarity is artefact-only). |
| `build_model()` (`:184-211`) | Calls `get_data_loader()` (default `force_remote=False`) → `build_and_save(data_loader.item_df)` → reset both cached singletons. |

> Loading-into-cache mechanics (local-cache-vs-Supabase precedence, singleton reuse across warm calls, cold-start/scale-out behavior, approximate in-memory footprint) are covered in full in `docs/ARCHITECTURE.md#10-inference-time-loading-into-cache`.

### CLI (`python -m src.content_based_filtering.main`)
- `--mode build` → `build_model()`.
- `--mode user --user_id <id> --n <int>` → `get_user_recommendations`.
- `--mode item --asin <parent_asin> --n <int>` → `get_similar_items`.
- Catches `FileNotFoundError`/`OSError`/`ValueError` → `sys.exit(1)`.

## B.6 Data columns relied upon
- **`Item` table / `item.parquet`** (consumed by `build_and_save`): `parent_asin`, `item_title`, `main_category`, `description` (list), `features` (list), `is_free`.
- **`Review` table / `user-item-interaction.parquet`** (consumed by `recommend_for_user`): `user_id`, `parent_asin`, `review_rating`, `recency_weight`.
- **Saved `cb_meta.parquet`**: `parent_asin`, `item_title`, `main_category`, `is_free` (text/description/features dropped at save).

---

# Part C — Side-by-side comparison

| Aspect | Collaborative Filtering | Content-Based Filtering |
|---|---|---|
| Signal source | Similar users' behavior | Item features + user's own history |
| Algorithm | ALS matrix factorization (`implicit.als`) | TF-IDF + cosine similarity |
| Library | `implicit` | scikit-learn (`TfidfVectorizer`, `cosine_similarity`) |
| Inputs | user–item interaction matrix (8 normalized features) | item text (title + description + features) |
| Matrix shape (notebook) | (2,589,466 × 89,246), ~4.83M non-zeros | (89,251 × 10,000) |
| Hyperparams | factors=50, reg=0.01, iter=20 | max_features=10k, ngram=(1,2), sublinear_tf |
| Needs item metadata | Only numeric features for interaction score | Yes (textual content) |
| Cold start (new user) | `ValueError` — no fallback | Empty result — graceful |
| Cold start (new item) | Poor (must be in training matrix) | Better (works if item text is in matrix) |
| Re-ranking | None (pure ALS scoring) | Category boost (+0.1) + free-item filter |
| Inference surfaces | `get_user_recommendations` | `get_user_recommendations` **and** `get_similar_items` |
| Return shape | `list[dict]` with `parent_asin`, `item_title`, `main_category`, `is_free` | `list[dict]` with `parent_asin`, `score`, `item_title`, `main_category`, (`is_free`) |
| Artefacts | `als_model.joblib`, `cf_user_item_matrix.npz`, 2× `*.json` | `cb_tfidf.joblib`, `cb_item_matrix.npz`, `cb_meta.parquet` |

## Cold-start asymmetry
- **CF** raises `ValueError("Unknown user_id: ...")` for users not in the training matrix (`recommender.py:48-49`). Callers must catch this and fall back (e.g., to CBF).
- **CBF** returns an empty DataFrame/list for users with no positive history (`recommender.py:111-112`) or unknown ASINs (`recommender.py:198-200`).

### CBF "no positive history" — worked example
A user returns `[]` from `get_user_recommendations` whenever they have **no qualifying history**,
i.e. no reviews with `review_rating >= MIN_RATING_THRESHOLD` (default `3`, see `tool_config.py`).
`build_user_profile` (`recommender.py:20-52`) builds the user's taste vector from positively-rated
items only; with none it returns `None`, `recommend_for_user` logs
`No positive history found for user: <id>` and returns an empty DataFrame (`recommender.py:111-112`).

Concrete demo `user_id`s against the local 5% Software subset:

| `user_id` | # reviews | positive? | Result |
|---|---|---|---|
| `agci7fah4gl5fi65hylkwtmfz2cq` | 1, rating 1.0 | No (`is_positive == False`) | `[]` — empty, with warning |
| `ag6hllxrsby3efcfgqgjxvjabvfq` | 176 (rating ≥ 4) | Yes | top-N list returned |

So when picking a demo `user_id` for `--mode user`, choose one with at least one review rated ≥ 3
(otherwise the call succeeds but returns nothing). A quick way to find candidates:

```python
import pandas as pd
df = pd.read_parquet("data/output/user-item-interaction.parquet")
pos = df[df["is_positive"] == True]["user_id"].value_counts()
print(pos.head())
```

## Notebooks
- `notebook/cf_recommend.ipynb` — CF prototype, the reference implementation refactored into `src/collaborative_filtering/`.
- `notebook/cb_recommend.ipynb` — CBF prototype, refactored into `src/content_based_filtering/`. The production modules are cleaned-up, configurable, artefact-persisted refactors of these notebooks.

---

# Part D — Build / Save / Load / Serve lifecycle (both tools)

### Build & save
- `build_model()` (`main.py`) → `build_and_save(...)` (`model.py`) → `save_local_artefacts(MODEL_NAME, artefacts, MODEL_ARTEFACTS)` (`models_loader/models_loader.py:68-109`).
- Files written to `REPO_ROOT / "models" / <tool_name> / <filename>` via suffix dispatch.
- **Note**: build only writes locally. Uploading to Supabase requires separate `models_loader.push_into_supabase` (offline step).

### Load (lazy, cached)
- `get_models_loader(force_remote=...)` → `ModelsLoader(tools=[MODEL_NAME], force_remote=...)` → `load_artefacts(tool_config, force_remote=...)`.
- If `force_remote=False` (default, used by `build_model()`): checks local cache; if any artefact missing → `download_artefacts_from_supabase(tool, MODEL_ID, MODEL_ARTEFACTS)` → `read_local_artefacts`.
- If `force_remote=True` (used by the inference handlers, on the call that creates the cached singleton only): skips the local-cache check and always re-downloads from Supabase before reading.
- Stored in `model_artefacts[MODEL_NAME]`, fully in memory for the life of the `ModelsLoader` instance.

### Serve
- Handlers `get_user_recommendations` / `get_similar_items` are the serving surface, importable with module-scope cached loaders for warm serverless reuse.
- These handlers call the loaders with `force_remote=True`, so a cold worker's first inference call always pulls fresh data/artefacts from Supabase rather than trusting whatever happens to be on local disk; every subsequent warm call reuses the in-memory singleton regardless of the flag (see `docs/ARCHITECTURE.md#10-inference-time-loading-into-cache` for the full mechanics).
- `build_model()` invalidates caches so a rebuild is observable by subsequent calls.

### CLI
- `python -m src.<tool>.main --mode {build|user|item} ...`
