# Building & Testing the Recommenders (Offline, CLI)

This guide walks through building both recommender models and testing them via the CLI, fully
offline (no live Supabase connection required). Prerequisites:

- `.env` populated with dummy Supabase credentials (so import-time client construction in
  `utils/supabase_utils.py:20` succeeds — see `docs/MODELS_LOADER.md` §8):
  ```
  SUPABASE_URL=https://example.supabase.co
  SUPBASE_SECRET_KEY=dummy
  ```
- `data/output/*.parquet` already present (produced by `data_download` + `data_transform`; see
  `docs/DATA_LOADER.md`). These parquets feed both training and inference.

---

## 1. Build models (works offline)

```powershell
python -m src.collaborative_filtering.main --mode build
python -m src.content_based_filtering.main --mode build
```

Flow: `build_model()` (`main.py`) → `build_and_save(...)` (`model.py`) → `save_local_artefacts`
(`models_loader/models_loader.py:68-109`) writes artefacts to `models/<tool>/` locally via
suffix dispatch (`.joblib` / `.npz` / `.json` / `.parquet`).

**No Supabase call is made** during build — the dummy credentials are never used. Build only
writes to disk; uploading to Supabase is a separate optional step (`models_loader.push_into_supabase`)
that you should NOT run offline.

Resulting artefact layout (see `docs/MODELS_LOADER.md` §2):

```
models/
├── collaborative_filtering/
│   ├── als_model.joblib
│   ├── cf_user_item_matrix.npz
│   ├── idx_to_userid_mapping.json
│   └── idx_to_itempasin_mapping.json
└── content_based_filtering/
    ├── cb_tfidf.joblib
    ├── cb_item_matrix.npz
    └── cb_meta.parquet
```

`build_model()` also resets the lazy-cached `_data_loader_obj` / `_models_loader_obj` singletons
so subsequent inference reloads fresh artefacts (CF: `main.py:162-163`; CBF: `main.py:183-184`).

---

## 2. Inference (works offline once built)

```powershell
# CF — user recommendations
python -m src.collaborative_filtering.main --mode user --user_id ag6hllxrsby3efcfgqgjxvjabvfq --n 10

# CBF — user recommendations
python -m src.content_based_filtering.main --mode user --user_id ag6hllxrsby3efcfgqgjxvjabvfq --n 10

# CBF — similar items
python -m src.content_based_filtering.main --mode item --asin <parent_asin> --n 10
```

Flow: `get_user_recommendations` / `get_similar_items` → `get_models_loader()` →
`ModelsLoader(tools=[MODEL_NAME])` → `load_artefacts(tool_config)`
(`models_loader/models_loader.py:111-161`):

1. Check `models/<tool>/` for all expected artefacts.
2. If **all present** → `read_local_artefacts(...)` directly. **No Supabase.**
3. If any missing → falls back to `download_artefacts_from_supabase` (will fail with dummy
   creds). So ensure the build step completed before running inference.

---

## 3. Picking valid demo IDs

### CF (`--mode user`)
The user must exist in the training matrix. CF raises `ValueError("Unknown user_id: ...")` for
unknown users (`src/collaborative_filtering/recommender.py:48-49`) — no cold-start fallback.

Find a valid `user_id` from the interaction table:

```python
import pandas as pd
df = pd.read_parquet("data/output/user-item-interaction.parquet")
print(df["user_id"].value_counts().head())
```

### CBF (`--mode user`)
The user must have **qualifying history**: at least one review with `review_rating >= 3`
(`MIN_RATING_THRESHOLD` in `tool_config.py`). Otherwise `build_user_profile` returns `None`
and the call logs `No positive history found for user: <id>` and returns `[]`
(`src/content_based_filtering/recommender.py:111-112`). See `docs/CB_CF_FILTERING.md` "Cold-start
asymmetry" for a worked example.

Find candidates with positive reviews:

```python
import pandas as pd
df = pd.read_parquet("data/output/user-item-interaction.parquet")
pos = df[df["is_positive"] == True]["user_id"].value_counts()
print(pos.head())
```

### CBF (`--mode item`)
The `parent_asin` must be present in the saved `cb_meta.parquet`. Find one:

```python
import pandas as pd
meta = pd.read_parquet("models/content_based_filtering/cb_meta.parquet")
print(meta["parent_asin"].head())
```

---

## 4. Expected offline behavior summary

| Step | Supabase? | Reads | Writes |
|---|---|---|---|
| `data_download` | No | HuggingFace Hub | `data/input/*.parquet` |
| `data_transform` | No | `data/input/*.parquet` | `data/output/*.parquet` |
| `--mode build` (CF + CBF) | No | `data/output/*.parquet` | `models/<tool>/*` |
| `--mode user` / `--mode item` | No (if artefacts cached) | `data/output/*.parquet` + `models/<tool>/*` | none |

Do **NOT** run these offline (they require live Supabase):
- `python -m data_loader.push_into_supabase`
- `python -m models_loader.push_into_supabase`
- `python -m setup`
