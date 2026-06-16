# Content-Based Filtering — Process Flow

## Overview

The content-based filtering module lives in `src/content_based_filtering/` and operates in two distinct phases:

1. **Build phase** — fit a TF-IDF model on the product corpus and save all artifacts to disk
2. **Inference phase** — load the saved artifacts and serve recommendations on request

---

## File Layout

```
src/content_based_filtering/
├── config.py        # All paths and hyperparameters in one place
├── data_loader.py   # Reads raw parquet files, builds item text corpus
├── model.py         # Fits TF-IDF, saves/loads artifacts
├── recommender.py   # User profile builder and scoring logic
└── main.py          # Entry points — CLI and Azure Function handlers
```

---

## Phase 1: Build (Offline, Run Once)

Run via CLI from the repo root or `src/`:

```bash
python -m content_based_filtering.main --mode build
```

### Step-by-step

```
data/output/meta_data.parquet          data/output/item.parquet
           │                                      │
           └──────────── load_meta() ─────────────┘
                               │
                    Merge on parent_asin
                    Fill missing is_free → False
                               │
                    build_item_text() per row
                    Concatenates: item_title + description[] + features[]
                    into a single whitespace-joined string
                               │
                    TfidfVectorizer.fit_transform()
                    max_features=10,000
                    ngram_range=(1,2)   ← unigrams + bigrams
                    sublinear_tf=True   ← log-scaled term frequencies
                               │
              ┌────────────────┼────────────────────┐
              │                │                    │
    cb_tfidf.joblib   cb_item_matrix.npz      cb_meta.parquet
    (vectorizer)      (sparse TF-IDF matrix)  (item metadata)
              └────────────────┴────────────────────┘
                        models/ directory
```

### What gets saved

| Artifact | Path | Format | Contents |
|---|---|---|---|
| TF-IDF vectorizer | `models/cb_tfidf.joblib` | joblib | Fitted `TfidfVectorizer` with vocabulary |
| Item TF-IDF matrix | `models/cb_item_matrix.npz` | scipy sparse NPZ | Shape: `(n_items × 10,000)` — one row per product |
| Item metadata | `models/cb_meta.parquet` | Parquet | `parent_asin`, `item_title`, `main_category`, `is_free` |

> The vectorizer itself is **not used at inference time** for user recommendations — only the pre-computed `item_matrix` is needed. It is saved so the same vocabulary can be reused later (e.g., to transform query text or new items without re-fitting).

---

## Phase 2: Inference (Online, Per Request)

### Artifact loading

`main.py` keeps a **module-level cache** (`_artifacts`). On the very first request (cold start), all three artifact files are read from disk. Every subsequent call reuses the in-memory objects — no repeated disk I/O.

```python
_artifacts: tuple | None = None   # None → cold start

def _get_artifacts():
    global _artifacts
    if _artifacts is None:                 # first call only
        _artifacts = load_artifacts()      # reads .joblib + .npz + .parquet
    return _artifacts                      # warm calls return immediately
```

`load_artifacts()` in `model.py` returns a five-tuple:

```
(tfidf, item_matrix, meta_df, item_to_idx, idx_to_item)
```

| Object | Type | Description |
|---|---|---|
| `tfidf` | `TfidfVectorizer` | Fitted vectorizer (vocabulary preserved) |
| `item_matrix` | `csr_matrix` | Sparse item vectors `(n_items × 10,000)` |
| `meta_df` | `DataFrame` | Item metadata keyed by `parent_asin` |
| `item_to_idx` | `dict[str, int]` | ASIN → row index in `item_matrix` |
| `idx_to_item` | `dict[int, str]` | Row index → ASIN (reverse lookup) |

---

## Inference Path A: User Recommendations

Entry point: `get_user_recommendations(user_id, n=10)`

```
user_id
    │
    ├── load_user_item()
    │   Reads data/output/user-item-interaction.parquet on every call
    │   Columns: user_id, parent_asin, review_rating, recency_weight
    │
    └── build_user_profile()
            │
            Filter interactions: review_rating >= 3 (MIN_RATING_THRESHOLD)
            Filter to items that exist in item_to_idx
            │
            weight = (rating / 5) × recency_weight   ← normalise + decay
            │
            profile = weighted sum of item row vectors
                      ÷ total weight
            Shape: (1 × 10,000)   ← single dense vector representing the user
            │
    cosine_similarity(profile, item_matrix)
    Scores: flat array of length n_items
            │
    Mask seen items → score = -1
            │
    Derive user preferences from positive history:
        top_category  ← most frequent main_category in liked items
        prefers_free  ← True if >50% of liked items are free
            │
    Take top (n × 5) candidate pool by score
    Merge with meta_df for category and is_free columns
            │
    Apply category boost: score += 0.1 if main_category == top_category
    Apply free filter: drop paid items if prefers_free == True
            │
    Sort by score descending → top n rows
            │
    Return: List[dict] with keys:
        parent_asin, score, item_title, main_category, is_free
```

---

## Inference Path B: Similar Items

Entry point: `get_similar_items(parent_asin, n=10)`

```
parent_asin
    │
    Lookup idx = item_to_idx[parent_asin]
    │
    cosine_similarity(item_matrix[idx], item_matrix)
    Scores: flat array of length n_items
    │
    Set self-score to -1 (exclude seed item)
    │
    Top n by score
    Merge with meta_df
    │
    Return: List[dict] with keys:
        parent_asin, score, item_title, main_category
```

No user history needed — purely item-to-item similarity in TF-IDF space.

---

## CLI Usage

```bash
# Build / re-fit the model
python -m content_based_filtering.main --mode build

# Get recommendations for a user
python -m content_based_filtering.main --mode user --user_id <id> --n 10

# Get items similar to a product
python -m content_based_filtering.main --mode item --asin <asin> --n 10
```

---

## Key Hyperparameters (config.py)

| Parameter | Value | Effect |
|---|---|---|
| `TFIDF_MAX_FEATURES` | 10,000 | Vocabulary size cap |
| `TFIDF_NGRAM_RANGE` | (1, 2) | Unigrams + bigrams |
| `TFIDF_SUBLINEAR_TF` | True | Log-scaled TF, reduces dominance of very frequent terms |
| `MIN_RATING_THRESHOLD` | 3 | Minimum rating to count as a "liked" item in user profile |
| `CANDIDATE_POOL_MULTIPLIER` | 5 | Oversample `n × 5` candidates before applying post-filters |
| `CATEGORY_BOOST` | 0.1 | Score bonus for matching the user's top category |
| `FREE_PREFERENCE_THRESHOLD` | 0.5 | Fraction of free-liked items above which free-only filter activates |

---

## Data Flow Summary

```
Raw parquet files (data/output/)
        │
        │  BUILD (once)
        ▼
TF-IDF fit → models/cb_tfidf.joblib
           → models/cb_item_matrix.npz
           → models/cb_meta.parquet
        │
        │  INFERENCE (per request)
        ▼
Load artifacts (once, cached in-process)
        │
        ├── user request → build user profile → cosine similarity → rank + filter → recs
        └── item request → item vector lookup → cosine similarity → rank → similar items
```