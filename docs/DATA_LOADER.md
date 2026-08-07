# Data Loader — Complete Documentation

The `data_loader` package is the **data ingestion, transformation, and persistence layer**. It closes the data loop:

```
HuggingFace Hub → local raw parquet → engineered parquet → Supabase tables → local cached parquet → recommender models
```

## 1. Module map

| File | Role | Run as |
|---|---|---|
| `__init__.py` | Re-exports `from .data_loader import *` (`DataLoader`, `load_data`). | imported |
| `data_download.py` | Pull raw Amazon Software dataset from HF Hub, clean, write `data/input/*.parquet`. | script |
| `data_transform.py` | Feature-engineer into 3 tables under `data/output/`. | script |
| `push_into_supabase.py` | Sanitize nulls, push 3 tables to Supabase. | script |
| `data_loader.py` | `DataLoader` class + `load_data()` — local-cache-or-Supabase **read path**. | imported |

> ⚠️ `data_download.py`, `data_transform.py`, and `push_into_supabase.py` execute module-level code on import (no `if __name__ == "__main__":` guard) — they are designed to be run as standalone scripts, **not imported**. Only `data_loader.py` is import-safe (and is the only thing `__init__.py` re-exports).

---

## 2. `data_download.py` — Raw download & cleaning

### Imports / config
`pandas`, `datasets.load_dataset`, `ast`, `tqdm.tqdm`; from `config`: `DATA_INPUT_DIR`, `DATASET_NAME`, `DATASET_CATEGORY`, `USER_REVIEW_FILENAME`, `ITEM_METADATA_FILENAME`, `DATASET_SPLIT_PERCENTAGE`.

### Helper functions

**`parse_categories(cat_str: str) -> list[str]`** (`:21-65`)
1. Return `[]` if NaN or `"[]"`.
2. `ast.literal_eval` the stringified list.
3. For each element: lowercase, strip `'s` possessive suffix.
4. If the result is a key in `category_replace_dict` (`:179-190`), replace with canonical alias; otherwise keep.
5. Return the **sorted** list.

**`parse_videos(video_dict) -> dict`** (`:67-87`)
- For any value equal to `[""]`, replace with `[]`. Mutates in place.

### Review dataset (`:112-151`)
- `load_dataset(DATASET_NAME, "raw_review_Software", split="full[:5%]")` — first 5% of the full Software review split.
- Column normalization:
  - `rating` → numeric (`errors="coerce"`).
  - `title`, `text` → stripped + lowercased.
  - `images` → string repr.
  - `asin`, `parent_asin`, `user_id` → lowercased.
  - `date` derived from `timestamp` via `pd.to_datetime(..., unit="ms").dt.date`; `timestamp` dropped.
- Reorder columns: `[asin, parent_asin, user_id, date, title, text, images, verified_purchase, helpful_vote, rating]`.
- Rename: `date→review_date`, `title→review_title`, `text→review_text`, `images→review_images`, `rating→review_rating`.

### Meta dataset (`:175-230`)
- `load_dataset(DATASET_NAME, "raw_meta_Software", split="full")`.
- `category_replace_dict` (`:179-190`) maps 10 aliases to canonical category names (e.g. `"accounting"` → `"accounting & finance"`).
- Column normalization:
  - `parent_asin`, `title`, `main_category`, `store` → stripped + lowercased.
  - `categories` → string-cast then `parse_categories`.
  - `videos` → `parse_videos`.
  - `rating_number`, `price` → numeric.
- Reorder columns, dropping `bought_together`, `average_rating`, `subtitle`, `author`. Final: `[parent_asin, title, main_category, categories, description, features, details, images, videos, rating_number, store, price]`.
- Rename: `title→item_title`, `images→item_images`, `videos→item_videos`, `rating_number→item_rating`.

### Export (`:233-237`)
- `os.makedirs(DATA_INPUT_DIR, exist_ok=True)`.
- `review_df.to_parquet(DATA_INPUT_DIR / "review_data.parquet", index=False)`.
- `meta_df.to_parquet(DATA_INPUT_DIR / "meta_data.parquet", index=False)`.

---

## 3. `data_transform.py` — Feature engineering

### Helper functions

**`safe_len(x) -> int`** (`:20-38`) — `len(set(x))` for lists/arrays, else `0`.
**`safe_join(x) -> str`** (`:40-60`) — `','.join(...)` for lists/arrays; `""` for NaN, else `str(x)`.
**`safe_json_numpy(x) -> str`** (`:62-83`) — `json.dumps` for dicts/arrays/lists (numpy → `.tolist()`); `""` for NaN, else `str(x)`.
**`count_review_images(image_list) -> int|float`** (`:85-114`) — `len` for lists; `np.nan` for NaN; `ast.literal_eval` for strings; `1` otherwise; `0` on parse failure.
**`wilson_lower_bound(pos, n, confidence=0.95) -> float`** (`:116-140`) — Wilson score lower bound with hardcoded `z=1.96` (the `confidence` arg is accepted but **unused**); returns `0` if `n == 0`.

### Reviews / user-item-interaction table (`:153-184`) → `data/output/user-item-interaction.parquet`
- `review_id` from `reset_index` (cast to `str`).
- `review_date` → datetime.
- Strip `\x00` from `review_text`.
- Derive `review_text_length`, `review_title_length`, `review_word_count`.
- Sentiment proxies: `is_extreme_rating` (rating ∈ {1,5}), `is_positive` (≥4), `is_negative` (≤2).
- `num_review_img` via `count_review_images`; `review_images` → comma-string via `safe_join`.
- Temporal: `days_since_review`, `review_year`, `review_month`; `review_date` → `"%Y-%m-%d"`.
- `recency_weight = exp(-days_since_review / 365.25)` — exponential decay (~1-year half-life).

### User table (`:190-296`) → `data/output/user.parquet`
- Left-merge reviews with `meta_df[["parent_asin","price"]]` to attach item price; `is_free = (price == 0.0)`.
- `groupby("user_id")` aggregations (multi-level columns flattened):
  - `review_id` → count; `review_rating` → mean/std/min/max; `review_date` → min/max; `recency_weight` → sum; `helpful_vote` → sum/mean; `verified_purchase` → sum; `review_text_length` → mean; `review_word_count` → mean; `num_review_img` → sum; `is_extreme_rating`/`is_positive`/`is_negative` → mean; `price` → mean; `is_free` → mean.
- Renames (flattened → semantic): `review_id_count→num_reviews`, `review_rating_mean→avg_rating_given`, `review_rating_std→rating_std`, `review_rating_min→min_rating_given`, `review_rating_max→max_rating_given`, `review_date_min→first_review_date`, `review_date_max→last_review_date`, `recency_weight_sum→total_recency_weight`, `helpful_vote_sum→total_helpful_votes_received`, `helpful_vote_mean→avg_helpful_votes_per_review`, `verified_purchase_sum→num_verified_purchases`, `review_text_length_mean→avg_review_length`, `review_word_count_mean→avg_review_words`, `num_review_img_sum→total_review_images`, `is_extreme_rating_mean→extreme_rating_ratio`, `is_positive_mean→positive_rating_ratio`, `is_negative_mean→negative_rating_ratio`, `price_mean→avg_price_purchased`, `is_free_mean→free_app_ratio`.
- Derived: `days_active = (last - first).days + 1`; `reviews_per_day = num_reviews / days_active`; `verified_purchase_ratio = num_verified_purchases / num_reviews`.
- `user_segment = pd.cut(num_reviews, bins=[0,1,10,inf], labels=["one_time","occasional","power_user"])`.
- `is_discriminating = rating_std > 1.0`; fill NaN `rating_std→0`, `avg_price_purchased→0`.
- Reformat `first_review_date`/`last_review_date` → `"%Y-%m-%d"`.

> ⚠️ Anomaly: the rename maps `num_review_images_sum→total_review_images` but the flattened column is actually `num_review_img_sum`, so this rename is a no-op and the column stays `num_review_img_sum`.

### Item table (`:302-429`) → `data/output/item.parquet`
- Filter `meta_df` to items referenced in reviews (`target_item_ids` = unique `parent_asin` from reviews).
- `num_item_img` = count of truthy entries across `hi_res`/`large`/`thumb` of `item_images` dict.
- `num_item_videos` = length of `url` list inside `item_videos` dict.
- Price: `price.fillna(-1)` (unknown sentinel); `is_free = (price==0.0)`; `has_price_info = (price>=0)`; `price_bucket = pd.cut(bins=[-1,0,1,10,25,50,100,inf], labels=["unknown","0-1","1-10","10-25","25-50","50-100","100+"])`.
- `num_categories = safe_len(categories)`; `categories`/`description`/`features`/`details` → comma-strings via `safe_join`; `item_images`/`item_videos` → JSON via `safe_json_numpy`.
- Popularity aggregation (groupby `parent_asin` on reviews): `user_id→count`, `review_rating→mean/std/min/max`, `review_date→min/max`, `recency_weight→sum`, `helpful_vote→sum/mean`, `verified_purchase→sum`, `is_positive/is_negative→mean`.
- Renames: `user_id_count→num_reviews`, `review_rating_mean→avg_rating`, `review_rating_std→rating_std`, `review_rating_min→min_rating`, `review_rating_max→max_rating`, `review_date_min→first_review_date`, `review_date_max→last_review_date`, `recency_weight_sum→total_recency_weight`, `helpful_vote_sum→total_helpful_votes`, `helpful_vote_mean→avg_helpful_votes`, `verified_purchase_sum→num_verified_reviews`, `is_positive_mean→positive_review_ratio`, `is_negative_mean→negative_review_ratio`.
- Left-merge item features with popularity on `parent_asin`.
- Derived: `days_on_platform = (last - first).days + 1`; `reviews_per_day = num_reviews / days_on_platform`; `verified_review_ratio = num_verified_reviews / num_reviews`.
- `popularity_segment = pd.cut(num_reviews, bins=[0,1,10,100,inf], labels=["cold_start","low_coverage","medium","popular"])`.
- `quality_score = wilson_lower_bound(positive_review_ratio * num_reviews, num_reviews)` per row.
- Reformat `first_review_date`/`last_review_date` → `"%Y-%m-%d"`.

### Export (`:436-440`)
- `os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)`.
- Write `user-item-interaction.parquet`, `user.parquet`, `item.parquet` (all `index=False`).

---

## 4. `push_into_supabase.py` — Upload to Supabase

- Imports `push_table_into_supabase` from `utils.supabase_utils`.
- Read the 3 output parquets.
- For each: replace `np.nan`/`np.inf`/`-np.inf` → `None` and `pd.NaT` → `None`, then push:
  - `reviews_df` → `Review` table (`REVIEW_TABLE_NAME`).
  - `item_df` → `Item` table (`ITEM_TABLE_NAME`).
  - `user_df` → ⚠️ **bug**: `push_table_into_supabase(user_df, USER_FILENAME)` passes `"user.parquet"` where `USER_TABLE_NAME` (`"User"`) is intended (`:64`).

---

## 5. `data_loader.py` — Read path (local-cache-first)

### `load_data(local_filename, supabase_tablename, force_remote=False) -> pd.DataFrame` (`:16-55`)
1. `parquet_path = DATA_OUTPUT_DIR / local_filename`.
2. If `not force_remote`: try `pd.read_parquet(parquet_path)`. On success, return immediately (no Supabase contact).
3. On `FileNotFoundError`, **or if `force_remote=True`** (local read skipped entirely): call `extract_table_from_supabase(supabase_tablename)`, save to `parquet_path` (`to_parquet(index=False)`, refreshing the local cache file), return.

### `class DataLoader` (`:57-99`)
**`__init__(self, force_remote: bool = False)`** loads three datasets and stores them as attributes, forwarding `force_remote` to every `load_data(...)` call:
| Attribute | Local filename | Supabase table |
|---|---|---|
| `self.user_df` | `user.parquet` | `User` |
| `self.item_df` | `item.parquet` | `Item` |
| `self.user_item_df` | `user-item-interaction.parquet` | `Review` |

Prints `"Tabular Data successfully loaded!"`. With the default `force_remote=False` the cache is persistent — subsequent runs reuse local Parquet without re-downloading. With `force_remote=True`, every dataset is always re-pulled from Supabase (used by inference — see [`docs/ARCHITECTURE.md#inference-time-loading-into-cache`](ARCHITECTURE.md#10-inference-time-loading-into-cache)).

All three loaded DataFrames are held **fully in memory** as attributes on the `DataLoader` instance — not streamed or lazily paginated — for the lifetime of the instance.

Consumers: `setup.py:30` (`DataLoader()`, default `force_remote=False`), `src/collaborative_filtering/main.py` and `src/content_based_filtering/main.py` (via `get_data_loader()`, which passes `force_remote=True` only at inference — see `docs/CB_CF_FILTERING.md`).

---

## 6. Implied Supabase table schemas

### `Review` table (from `user-item-interaction.parquet`, 22 cols)
`review_id` (str), `asin` (str), `parent_asin` (str), `user_id` (str), `review_date` (str `YYYY-MM-DD`), `review_title` (str), `review_text` (str), `review_images` (str), `verified_purchase` (bool), `helpful_vote` (int), `review_rating` (float), `review_text_length` (int), `review_title_length` (int), `review_word_count` (int), `is_extreme_rating` (bool), `is_positive` (bool), `is_negative` (bool), `num_review_img` (int/float), `days_since_review` (int), `review_year` (int), `review_month` (int), `recency_weight` (float).

### `Item` table (from `item.parquet`)
`parent_asin`, `item_title`, `main_category`, `categories` (comma-str), `description` (comma-str), `features` (comma-str), `details` (comma-str), `item_images` (JSON), `item_videos` (JSON), `item_rating`, `store`, `price` (−1 unknown), `num_item_img`, `num_item_videos`, `is_free`, `has_price_info`, `price_bucket`, `num_categories`, `num_reviews`, `avg_rating`, `rating_std`, `min_rating`, `max_rating`, `first_review_date`, `last_review_date`, `total_recency_weight`, `total_helpful_votes`, `avg_helpful_votes`, `num_verified_reviews`, `positive_review_ratio`, `negative_review_ratio`, `days_on_platform`, `reviews_per_day`, `verified_review_ratio`, `popularity_segment`, `quality_score`.

### `User` table (from `user.parquet`)
`user_id`, `num_reviews`, `avg_rating_given`, `rating_std` (0 for NaN), `min_rating_given`, `max_rating_given`, `first_review_date`, `last_review_date`, `total_recency_weight`, `total_helpful_votes_received`, `avg_helpful_votes_per_review`, `num_verified_purchases`, `avg_review_length`, `avg_review_words`, `total_review_images` (note: actually `num_review_img_sum` due to rename bug), `extreme_rating_ratio`, `positive_rating_ratio`, `negative_rating_ratio`, `avg_price_purchased` (0 for NaN), `free_app_ratio`, `days_active`, `reviews_per_day`, `verified_purchase_ratio`, `user_segment`, `is_discriminating`.

---

## 7. Config constants & env vars used

From `config.py`: `DATA_INPUT_DIR`, `DATA_OUTPUT_DIR`, `USER_REVIEW_FILENAME`, `ITEM_METADATA_FILENAME`, `USER_FILENAME`, `ITEM_FILENAME`, `USER_ITEM_INTERACT_FILENAME`, `REVIEW_TABLE_NAME`, `ITEM_TABLE_NAME`, `USER_TABLE_NAME`, `DATASET_NAME`, `DATASET_CATEGORY`, `DATASET_SPLIT_PERCENTAGE`.

From `utils/supabase_utils.py`: env vars `SUPABASE_URL`, `SUPBASE_SECRET_KEY` (deliberate typo).

Hardcoded in code: `category_replace_dict` (10 alias→canonical mappings), `z = 1.96` (Wilson), `365.25` (recency decay divisor).

---

## 8. Dependencies
- **Third-party**: `pandas`, `numpy`, `datasets` (HF), `tqdm`, `supabase`, `python-dotenv`, `fastparquet`/`pyarrow`.
- **Standard library**: `os`, `ast`, `json`, `uuid`, `typing`.
- **Internal**: `config`, `utils.supabase_utils`.

---

## 9. Anomalies & notes
1. **`push_into_supabase.py:64` bug** — passes `USER_FILENAME` instead of `USER_TABLE_NAME`.
2. **`wilson_lower_bound` `confidence` unused** — z hardcoded to `1.96`.
3. **`num_review_images_sum` rename is a no-op** — actual column stays `num_review_img_sum`.
4. **`helpful_vote_max` rename is dead** — not produced by the aggregation.
5. **`SUPBASE_SECRET_KEY`** env var typo — `.env` must match exactly.
6. **Script files execute on import** — `data_download.py`, `data_transform.py`, `push_into_supabase.py` have no `__main__` guard.
7. **`tqdm` imported in `data_download.py:12` but unused** in the module body.
8. **`extract_table_from_supabase` pagination** relies on empty-page termination; no total-count check (works for ~244k rows but is fragile).
