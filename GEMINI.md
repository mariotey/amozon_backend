# Recommender System Project - Context for Claude Code

**Project:** Amazon Software Recommendation Engine
**Goal:** Build hybrid recommendation system combining collaborative and content-based filtering
**Last Updated:** 2026-01-06

---

## Project Overview

This project implements a recommendation system for Amazon software products using the Amazon Reviews 2023 dataset. The system combines collaborative filtering (user-item interactions) and content-based filtering (product features) to provide personalized software recommendations.

### Why Hybrid Approach?

Based on our EDA, the dataset exhibits:
- **Extreme sparsity:** 99.9979% sparse user-item matrix
- **Cold start issues:** 67.3% of users have only 1 review, 27.6% of products have only 1 review
- **Rich content:** Review text, product descriptions, and metadata available
- **Recommendation:** Hybrid system essential to handle sparsity and cold start problems

---

## Data Sources

### Primary Dataset Files
Located in `data/output/`:

1. **`review_data.parquet`** (4,880,181 records)
   - User reviews and ratings (1999-2023)
   - 2,589,466 unique users × 89,246 products
   - Includes: rating, review text, verified purchase, helpful votes, timestamp

2. **`meta_data.parquet`** (89,251 records)
   - Product catalog and metadata
   - Includes: title, category, price, features, descriptions, images

### Documentation Files

📋 **[DATA_DICTIONARY.md](data/DATA_DICTIONARY.md)**
- Complete schema reference for both datasets
- Column definitions, data types, examples
- Data model and relationships
- Usage examples and code snippets
- **Use this for:** Understanding data structure, column meanings, join keys

📊 **[AMAZON_SOFTWARE_ANALYSIS_REPORT.md](data/AMAZON_SOFTWARE_ANALYSIS_REPORT.md)**
- Comprehensive EDA with insights
- Rating distributions, user behavior, temporal patterns
- Recommendation strategy based on data characteristics
- **Use this for:** Understanding data patterns, ML strategy decisions

---

## Key Data Characteristics

### Critical Metrics
| Metric | Value | Implication |
|--------|-------|-------------|
| Total Reviews | 4,880,181 | Large dataset, sufficient for training |
| Unique Users | 2,589,466 | User diversity high |
| Unique Products | 89,246 | Product catalog size |
| Sparsity | 99.9979% | Collaborative filtering challenging |
| Date Range | 1999-2023 (24 years) | Need temporal decay weighting |
| Verified Purchases | 95.19% | High trust signal available |
| One-time Users | 67.3% | Cold start is major challenge |

### Rating Distribution
```
5 stars: 54.68% (2,668,636 reviews)
4 stars: 17.56% (857,082 reviews)
3 stars: 8.59%  (419,356 reviews)
2 stars: 4.90%  (239,253 reviews)
1 star:  14.26% (695,854 reviews)
```
**Pattern:** Highly polarized - 68.9% are extreme ratings (1 or 5 stars)

### User Segmentation
- **Power Users** (>10 reviews): 31,676 users (1.2%) → Use collaborative filtering
- **Occasional Users** (2-10 reviews): 814,338 users (31.4%) → Hybrid approach
- **One-time Users** (1 review): 1,743,452 users (67.3%) → Content-based + popularity

### Product Segmentation
- **Popular** (>100 reviews): 4,922 products (5.5%) → Collaborative filtering works well
- **Medium** (11-100 reviews): 16,603 products (18.6%) → Hybrid approach
- **Low Coverage** (2-10 reviews): 43,085 products (48.3%) → Content-based preferred
- **Cold Start** (0-1 reviews): 24,641 products (27.6%) → Pure content-based

---

## Technical Approach

### Recommendation Strategy

```
┌─────────────────────────────────────────────────────────┐
│           HYBRID RECOMMENDATION SYSTEM                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐   │
│  │ Collaborative        │  │ Content-Based        │   │
│  │ Filtering            │  │ Filtering            │   │
│  ├──────────────────────┤  ├──────────────────────┤   │
│  │ • Matrix Factor.     │  │ • TF-IDF on text     │   │
│  │ • User-based CF      │  │ • Category matching  │   │
│  │ • Item-based CF      │  │ • Price filtering    │   │
│  │ • Neural CF          │  │ • Feature similarity │   │
│  └──────────┬───────────┘  └──────────┬───────────┘   │
│             │                          │               │
│             └────────┬─────────────────┘               │
│                      │                                 │
│              ┌───────▼────────┐                        │
│              │ Weighted       │                        │
│              │ Ensemble       │                        │
│              └───────┬────────┘                        │
│                      │                                 │
│              ┌───────▼────────┐                        │
│              │ Recommendations│                        │
│              └────────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

### Dynamic Weighting Strategy
```python
# Weight based on user activity
if user_reviews > 10:
    cf_weight = 0.7, content_weight = 0.3  # Power users
elif user_reviews >= 2:
    cf_weight = 0.5, content_weight = 0.5  # Occasional users
else:
    cf_weight = 0.2, content_weight = 0.8  # One-time/new users
```

### Feature Engineering Priorities

**User Features:**
- Average rating (user leniency bias)
- Rating variance (discriminativeness)
- Review activity level
- Verified purchase ratio
- Temporal patterns

**Item Features:**
- Average rating & variance
- Number of reviews (popularity)
- Price point
- Category hierarchy
- TF-IDF vectors from review text and descriptions
- Helpful vote aggregates

**Interaction Features:**
- Rating (explicit feedback)
- Verified purchase status (trust signal)
- Helpful votes (quality signal)
- Review recency (temporal decay)
- Review text sentiment

---

## Data Relationships

### Primary Join Key
```
review_data.parent_asin ←→ meta_data.parent_asin
```

**Coverage:**
- ✅ 89,246 products have both reviews AND metadata (100% of reviewed products)
- ⚠️ 5 products have metadata but NO reviews (cold start)
- ✅ 0 orphaned reviews (all reviews have corresponding metadata)

### User-Item Matrix
```python
# Matrix dimensions
rows (users): 2,589,466
cols (items): 89,246
density: 0.0021%
filled_cells: 4,880,181
empty_cells: 231,094,602,455
```

---

## Project Structure

```
Recommender/
├── data/
│   ├── output/
│   │   ├── review_data.parquet       # Review dataset
│   │   └── meta_data.parquet         # Product metadata
│   ├── DATA_DICTIONARY.md            # Schema reference
│   ├── AMAZON_SOFTWARE_ANALYSIS_REPORT.md  # EDA report
│   └── (raw data files...)
├── notebooks/
│   ├── EDA_*.ipynb                   # Exploratory analysis notebooks
│   └── ETL_*.ipynb                   # Data processing notebooks
├── src/                              # Source code (to be created)
│   ├── data/                         # Data loading utilities
│   ├── models/                       # Recommendation models
│   ├── features/                     # Feature engineering
│   └── evaluation/                   # Metrics and evaluation
├── tests/                            # Unit tests (to be created)
├── claude.md                         # This context file
└── README.md                         # Project documentation
```

---

## Development Phases

### ✅ Phase 0: Data Understanding (COMPLETED)
- [x] EDA on review dataset
- [x] EDA on metadata dataset
- [x] Combined analysis
- [x] Schema documentation
- [x] Strategy formulation

### 🔄 Phase 1: Data Preparation (IN PROGRESS)
- [ ] Create train/validation/test splits (temporal split recommended)
- [ ] Build user-item matrix (sparse format)
- [ ] Feature engineering pipeline
- [ ] Text preprocessing (TF-IDF, embeddings)
- [ ] Data loaders and utilities

### ⏳ Phase 2: Baseline Models
- [ ] Popularity-based recommender
- [ ] Item-based collaborative filtering
- [ ] Content-based filtering (TF-IDF similarity)
- [ ] Evaluation framework

### ⏳ Phase 3: Advanced Models
- [ ] Matrix Factorization (SVD, ALS)
- [ ] Neural Collaborative Filtering
- [ ] Deep learning content models
- [ ] Hybrid ensemble

### ⏳ Phase 4: Optimization & Deployment
- [ ] Hyperparameter tuning
- [ ] Model selection and ensembling
- [ ] Inference optimization (caching, indexing)
- [ ] API development
- [ ] Documentation

---

## Key Insights from EDA

### 🔴 Critical Challenges

1. **Extreme Sparsity (99.9979%)**
   - Traditional CF methods will struggle
   - Matrix factorization requires careful tuning
   - Consider neural CF or graph-based methods

2. **Cold Start Problem**
   - 67.3% users have 1 review only
   - 27.6% products have 1 review only
   - Content-based filtering is essential fallback

3. **Positive Bias**
   - 72.2% of reviews are 4-5 stars
   - May need to normalize ratings per user
   - Consider implicit feedback (clicks, views) if available

4. **Temporal Drift**
   - 24-year span with declining review quality over time
   - Recent reviews (2020+) have lower average ratings
   - Implement time-decay weighting

### 🟢 Opportunities

1. **Rich Textual Content**
   - Review text (avg 140 chars, 26 words)
   - Product descriptions and features
   - Can extract aspects, sentiments, topics

2. **Trust Signals**
   - 95.19% verified purchases
   - Helpful votes available (42.56% reviews have votes)
   - Can weight by trust/quality

3. **Power Users**
   - 31,676 users with >10 reviews
   - Contribute 12.1% of all reviews
   - Reliable for collaborative filtering

4. **Price Information**
   - 18.16% have price data
   - 61.38% are free software
   - Enable price-aware recommendations

---

## Model Evaluation Strategy

### Offline Metrics
```python
# Ranking Metrics
- Precision@K (K=5, 10, 20)
- Recall@K
- nDCG@K (normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)

# Rating Prediction
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)

# Diversity & Coverage
- Catalog Coverage (% of items recommended)
- Intra-list Diversity (similarity within recommendations)
- Novelty (how different from popular items)
- Serendipity (surprising but relevant)
```

### Train/Test Split Strategy
**Recommended:** Temporal split
- Train: Reviews before 2022-01-01
- Validation: Reviews 2022-01-01 to 2022-12-31
- Test: Reviews 2023-01-01 onwards

**Reasoning:** Mimics production scenario (predict future from past)

---

## Quick Reference Commands

### Load Data
```python
import pandas as pd

reviews = pd.read_parquet('data/output/review_data.parquet')
metadata = pd.read_parquet('data/output/meta_data.parquet')

# Join datasets
data = reviews.merge(metadata, on='parent_asin', how='left')
```

### Common Filters
```python
# Verified reviews only
verified = reviews[reviews['verified_purchase'] == True]

# Active users only (>= 5 reviews)
active_users = reviews.groupby('user_id').size() >= 5
reviews_active = reviews[reviews['user_id'].isin(active_users[active_users].index)]

# Products with sufficient reviews (>= 10)
popular_items = reviews.groupby('parent_asin').size() >= 10
reviews_popular = reviews[reviews['parent_asin'].isin(popular_items[popular_items].index)]

# Recent reviews (2020+)
reviews['date'] = pd.to_datetime(reviews['timestamp'], unit='ms')
recent = reviews[reviews['date'].dt.year >= 2020]
```

### Build User-Item Matrix
```python
from scipy.sparse import csr_matrix

# Create user and item mappings
user_to_idx = {user: idx for idx, user in enumerate(reviews['user_id'].unique())}
item_to_idx = {item: idx for idx, item in enumerate(reviews['parent_asin'].unique())}

# Map to indices
user_indices = reviews['user_id'].map(user_to_idx)
item_indices = reviews['parent_asin'].map(item_to_idx)
ratings = reviews['rating'].values

# Create sparse matrix
user_item_matrix = csr_matrix(
    (ratings, (user_indices, item_indices)),
    shape=(len(user_to_idx), len(item_to_idx))
)
```

---

## Important Considerations

### Data Quality
- ✅ No missing values in critical fields (rating, user_id, parent_asin)
- ⚠️ 174 empty review texts (0.00%) - handle in text processing
- ⚠️ 20.47% products missing price - use NaN handling
- ✅ 100% join coverage between reviews and metadata

### Performance Optimization
- Use sparse matrices for user-item interactions
- Cache computed similarities
- Consider approximate nearest neighbors (ANN) for large-scale retrieval
- Batch inference for production

### Ethical Considerations
- Avoid filter bubbles (balance personalization with diversity)
- Ensure fair representation (don't over-recommend popular items)
- Respect user privacy (anonymized user IDs)
- Transparent recommendations (explainability)

---

## References

📖 **Documentation:**
- [Data Dictionary](data/DATA_DICTIONARY.md) - Schema and data types
- [EDA Report](data/AMAZON_SOFTWARE_ANALYSIS_REPORT.md) - Analysis and insights

📚 **Recommended Reading:**
- Collaborative Filtering: Matrix Factorization techniques (SVD++, ALS)
- Neural CF: "Neural Collaborative Filtering" (He et al., 2017)
- Hybrid Systems: "Hybrid Recommender Systems" (Burke, 2002)
- Cold Start: "Content-Based Recommendations" (Pazzani & Billsus, 2007)

---

## Quick Start for New Contributors

1. **Understand the data:**
   - Read `data/DATA_DICTIONARY.md` for schema
   - Read `data/AMAZON_SOFTWARE_ANALYSIS_REPORT.md` for insights

2. **Load and explore:**
   ```python
   import pandas as pd
   reviews = pd.read_parquet('data/output/review_data.parquet')
   metadata = pd.read_parquet('data/output/meta_data.parquet')
   print(reviews.head())
   print(metadata.head())
   ```

3. **Check current phase:**
   - See "Development Phases" section above
   - Current focus: Phase 1 - Data Preparation

4. **Contribute:**
   - Follow project structure
   - Add tests for new code
   - Update documentation

---

**Last Updated:** 2026-01-06
**Status:** Phase 1 - Data Preparation
**Next Steps:** Create train/test splits, build feature engineering pipeline

---

*This file serves as quick context for Claude Code. For detailed information, refer to the linked documentation files.*