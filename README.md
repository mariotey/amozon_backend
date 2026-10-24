# Amozon Recommendation Backend

Amozon helps shoppers discover relevant software by using past customer reviews
and product information to suggest products. This repository contains the backend
service that a separate frontend, such as the Django application, calls to display
recommendations and catalogue information.

## What it provides

- Personalised product suggestions, with a reason for the recommendation approach used.
- Popular products for general browsing.
- Product details and customer reviews.
- Shopper profiles and random shoppers for exploring the experience.

## How recommendations work

1. **Similar shopping patterns:** use a trained model of customer preferences when
   the shopper is supported by that model.
2. **Product interests:** otherwise, use the characteristics of products the
   shopper previously rated positively.
3. **No suitable result:** return no recommendations if neither approach produces
   suggestions. Popular products remain a separate browsing option.

The selection follows fixed rules. It does not rely on a conversational AI to
choose the recommendation approach.

## Scope and expectations

The project uses the Software subset of the Amazon Reviews 2023 dataset. It
supports product discovery; recommendation quality and business impact still
require evaluation before making performance claims.

Recommendations depend on prepared data and trained models. On-demand queries
reflect prepared table updates, while trained model updates still require the
existing publication/cache workflow. The service does not retrain automatically
from new feedback. Some shoppers may receive fewer suggestions or none.

Supabase stores the prepared information and model files. The backend serves
these to the frontend through an API and is configured for Docker deployment on
Render.

## On-demand serving

The API does not download complete User, Item, or Review tables at startup or
on the first request. Random shoppers are sampled in Supabase, popular products
are ordered and limited there, and product/profile/review pages use targeted
queries. Any valid shopper ID or direct product link works independently of the
initial sample. The frontend can independently call `/api/v1/randomusers?n=5`
and `/api/v1/popular_items?limit=5`, then request the selected shopper's
recommendations. Automatic selection remains the frontend's responsibility;
this change does not modify the frontend.

One background worker loads the trained engines, their complete saved matrices,
ID mappings, and content metadata. Browsing works while models load; recommendation
requests return 503 until both models are ready. Collaborative results use the
saved interaction matrix without querying raw reviews. Content fallback retrieves
only that shopper's complete paginated inference history. Candidate coverage,
seen-item filtering, weighting, and ranking rules remain intact.

Serving queries live in `utils/supabase_utils.py`, model validation and reusable
ID indexes live in `models_loader/models_loader.py`, and `app.py` owns the single
background initialization task. Lookup maps are prepared once per model set;
database metadata queries do not hold the numeric inference lock.

`/health` reports liveness; `/ready` reports model readiness and identifies serving
data as `on_demand`. No database connectivity is implied by either response.
Missing database dependencies return 503 rather than empty recommendations.
No persistent serving-data caches, retraining, packaged model releases, or automatic
version refresh are introduced. Cold instances still incur Python imports,
model-file downloads when missing, model deserialization, and Render provisioning.
No deployed startup speedup has been measured.

Before using random shoppers, manually review/apply
[`sql/001_on_demand_serving.sql`](sql/001_on_demand_serving.sql). It provides the
random-sampling function, restricted execute permissions, and query indexes.
No live database changes or deployment were performed by this implementation.

## Technical guide

See [DEPLOYMENT.md](DEPLOYMENT.md) for setup, API endpoints, deployment, testing,
and maintenance.
