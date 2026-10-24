# Amozon Recommendation System

Amozon helps people discover software that is relevant to their interests. It
uses previous customer feedback to personalise suggestions and provides general
recommendations when there is not enough information about a customer.

## Purpose

A software catalogue can contain more products than customers can comfortably
explore. This project supports product discovery by presenting a focused list
of suggestions through one consistent recommendation service.

## How recommendations are selected

The system follows a predictable order:

1. **Learn from similar customers.** For customers already supported by the
   trained recommendation model, suggest products based on shared preferences.
2. **Learn from the customer's interests.** When the first approach cannot
   provide suggestions, use the characteristics of products the customer rated
   positively.
3. **Offer general recommendations.** When personal history is insufficient,
   suggest highly rated products the customer has not already reviewed.

Each response identifies which approach was used and explains the choice.
The selection follows fixed rules and does not depend on a conversational AI
making a decision.

## What the project provides

- Personalised software recommendations where sufficient history is available.
- A general recommendation path for new customers and other customers without
  a usable personal history.
- A single entry point for applications requesting recommendations.
- An explanation of the recommendation approach used for each request.

## Current scope

The project uses the Software subset of the Amazon Reviews 2023 dataset. It
serves recommendations from prepared customer feedback, product information,
and trained models. New feedback affects results once the relevant data and
models are refreshed; recommendations do not update automatically in real time.

Results depend on the available data and products. The system can return fewer
suggestions when suitable candidates are limited. Recommendation quality and
business impact still need evaluation before making performance claims.

## Further information

[DEPLOYMENT.md](DEPLOYMENT.md) covers
local setup, hosting, API usage, testing, and troubleshooting.
