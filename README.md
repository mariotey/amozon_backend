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

Recommendations depend on prepared data and trained models. Updates require a
managed data/model refresh; the service does not learn automatically from new
feedback in real time. Some shoppers may receive fewer suggestions or none.

Supabase stores the prepared information and model files. The backend serves
these to the frontend through an API and is configured for Docker deployment on
Render.

## Technical guide

See [DEPLOYMENT.md](DEPLOYMENT.md) for setup, API endpoints, deployment, testing,
and maintenance.
