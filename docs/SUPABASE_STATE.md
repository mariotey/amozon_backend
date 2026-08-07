# Supabase Live State

Verified 2026-08-08 by querying the live Supabase project directly (not just reading code). Re-run the checks below if this file goes stale — don't trust it blindly.

## Connection

Root `.env` already has the exact env vars the code reads:
- `SUPABASE_URL`
- `SUPBASE_SECRET_KEY` (deliberate typo — matches `utils/supabase_utils.py:22`, do not "fix" it)
- `SUPABASE_PUBLIC_KEY` is present but **unused** by any script — only the secret key is read.

`utils/supabase_utils.py` calls `load_dotenv()` and builds `SUPABASE_CLIENT` from those two vars at import time. No code changes are needed to connect `data_loader`, `models_loader`, or inference (`src/<tool>/main.py`) to Supabase — it already works as-is.

## Tables (row counts as of verification date)

| Table | Rows |
|---|---|
| `User` | 92,558 |
| `Item` | 27,797 |
| `Review` | 244,009 |
| `ModelRegistry` | 2 |

## Storage bucket `ModelArtefacts`

`ModelRegistry` contains exactly one row per tool, and both `model_id`s match the pinned `MODEL_ID` hardcoded in each tool's `tool_config.py` — so a cold-cache inference run will resolve the pinned IDs successfully, not fall back to "latest".

| tool | model_id | storage_path | created_at |
|---|---|---|---|
| `collaborative_filtering` | `9139d794-f9b3-4188-9370-33ceb92111fd` | `collaborative_filtering/9139d794-f9b3-4188-9370-33ceb92111fd` | 2026-07-28T17:04:55Z |
| `content_based_filtering` | `fefdee34-e3fe-4314-acab-f911c02680d3` | `content_based_filtering/fefdee34-e3fe-4314-acab-f911c02680d3` | 2026-07-28T17:05:03Z |

Files present at each storage path:

**`collaborative_filtering/9139d794-f9b3-4188-9370-33ceb92111fd/`**
- `als_model.joblib` (16,407,612 bytes)
- `cf_user_item_matrix.npz` (930,457 bytes)
- `idx_to_itempasin_mapping.json` (497,098 bytes)
- `idx_to_userid_mapping.json` (2,446,389 bytes)

**`content_based_filtering/fefdee34-e3fe-4314-acab-f911c02680d3/`**
- `cb_tfidf.joblib` (390,755 bytes)
- `cb_item_matrix.npz` (1,533,346 bytes)
- `cb_meta.parquet` (1,022,212 bytes)

## Bottom line

Data tables, model registry, and model artefact files are all live and consistent with the pinned `MODEL_ID`s in code. Deleting local caches (`data/output/*.parquet`, `models/*`) and running inference cold should pull everything from Supabase successfully with zero code changes.

## Known bug (not yet fixed, doesn't block reads)

`data_loader/push_into_supabase.py:64` passes `USER_FILENAME` (`"user.parquet"`) instead of `USER_TABLE_NAME` (`"User"`) when pushing the user table. Only affects re-pushing fresh data, not reading what's already live. See `docs/DATA_LOADER.md` anomaly #1.

## How to re-verify

```python
from dotenv import load_dotenv
load_dotenv()
from utils.supabase_utils import SUPABASE_CLIENT, ARTEFACTS_STORAGE

for t in ["User", "Item", "Review", "ModelRegistry"]:
    print(t, SUPABASE_CLIENT.table(t).select("*", count="exact").limit(1).execute().count)

for row in SUPABASE_CLIENT.table("ModelRegistry").select("*").execute().data:
    print(row)
    print(ARTEFACTS_STORAGE.list(row["storage_path"]))
```
