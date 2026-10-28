"""On-demand database bounds, model-only startup, and request-isolated inference."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from threading import Barrier, Event, Lock
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPBASE_SECRET_KEY", "test-key")

import app
from models_loader import validate_inference_models
from utils.supabase_utils import CARD_FIELDS, HISTORY_FIELDS, PROFILE_FIELDS, ServingData
from app import ModelResources


class Query:
    def __init__(self, db, table, params=None):
        self.db, self.table, self.params = db, table, params
        self.fields, self.count = None, None
        self.filters, self.orders = [], []
        self.requested_limit, self.interval = None, None

    def select(self, fields, count=None):
        self.fields, self.count = fields, count
        return self

    def eq(self, field, value):
        self.filters.append(("eq", field, value))
        return self

    def gt(self, field, value):
        self.filters.append(("gt", field, value))
        return self

    def in_(self, field, values):
        self.filters.append(("in", field, values))
        return self

    def order(self, field, desc=False, nullsfirst=None):
        self.orders.append((field, desc, nullsfirst))
        return self

    def limit(self, n):
        self.requested_limit = n
        return self

    def range(self, start, end):
        self.interval = (start, end)
        return self

    def execute(self):
        with self.db.lock:
            self.db.calls.append(self)
        if self.table in self.db.fail:
            raise RuntimeError("private credential must not appear in responses")
        if self.table == "rpc:random_shoppers":
            ids = list(dict.fromkeys(row["user_id"] for row in self.db.tables["Review"] if row["user_id"] is not None))
            # A deterministic fake, not an implementation of database randomness.
            return SimpleNamespace(data=[{"user_id": uid} for uid in ids[:self.params["sample_size"]]], count=None)
        rows = list(self.db.tables[self.table])
        for operation, field, value in self.filters:
            if operation == "eq":
                rows = [row for row in rows if row.get(field) == value]
            elif operation == "gt":
                rows = [row for row in rows if row.get(field) is not None and row[field] > value]
            else:
                rows = [row for row in rows if row.get(field) in value]
        count = len(rows)
        for field, desc, nullsfirst in reversed(self.orders):
            nulls = [row for row in rows if row.get(field) is None]
            nonnulls = sorted((row for row in rows if row.get(field) is not None),
                              key=lambda row: row[field], reverse=desc)
            rows = nulls + nonnulls if nullsfirst else nonnulls + nulls
        if self.interval is not None:
            rows = rows[self.interval[0]:self.interval[1] + 1]
        if self.requested_limit is not None:
            rows = rows[:self.requested_limit]
        rows = rows[:self.db.response_cap]
        if self.db.history_barrier is not None and self.fields and self.fields.startswith("review_id,user_id,parent_asin,review_rating"):
            if not any(op == "gt" for op, _, _ in self.filters):
                self.db.history_barrier.wait(timeout=5)
        if self.fields:
            rows = [{field: row.get(field) for field in self.fields.split(",")} for row in rows]
        return SimpleNamespace(data=rows, count=count if self.count == "exact" else None)


class Database:
    def __init__(self):
        self.tables = {
            "Item": [{"parent_asin": asin, "item_title": asin, "main_category": "Software",
                      "is_free": True, "price": 0., "avg_rating": 4., "num_reviews": count,
                      "quality_score": .8, "store": "store", "popularity_segment": "popular",
                      "description": "Description"}
                     for asin, count in (("seen", 100), ("b", 10), ("a", 10), ("c", 5))],
            "User": [{"user_id": "u", "num_reviews": 1, "avg_rating_given": 3.}],
            "Review": [self.review("1", "u", "seen", 3.),
                       self.review("2", "alpha", "seen", 5.),
                       self.review("3", "beta", "b", 5.),
                       self.review("4", "alpha", "a", 1.)],
        }
        self.calls, self.fail = [], set()
        self.lock = Lock()
        self.response_cap = 1000
        self.history_barrier = None

    @staticmethod
    def review(rid, user, asin, rating):
        return {"review_id": rid, "user_id": user, "parent_asin": asin, "review_rating": rating,
                "recency_weight": 1., "review_date": "2026-01-01", "review_title": "Review",
                "review_text": "Text", "verified_purchase": True, "helpful_vote": 1}

    def table(self, name):
        return Query(self, name)

    def rpc(self, name, params):
        return Query(self, "rpc:" + name, params)


def model_fixture(db):
    model = Mock()
    model.user_factors = np.array([[1., 0.]])
    model.item_factors = np.ones((4, 2))
    model.recommend.return_value = (np.array([1, 2]), np.array([.9, .8]))
    meta = pd.DataFrame(db.tables["Item"])[["parent_asin", "item_title", "main_category", "is_free"]]
    return SimpleNamespace(model_artefacts={
        "collaborative_filtering": {
            "model": model, "interactions": csr_matrix([[1., 0., 0., 0.]]),
            "user_ids": ["u"], "item_ids": ["seen", "b", "a", "c"],
        },
        "content_based_filtering": {
            "item_matrix": csr_matrix([[1., 0.], [.9, .1], [.8, .2], [.2, .8]]),
            "metadata": meta,
        },
    })


class OnDemandTests(unittest.TestCase):
    def setUp(self):
        self.db = Database()
        self.models = model_fixture(self.db)
        self.store = ServingData(self.db)
        self.gate = Event()
        self.entered = Event()
        self.failure = None
        self.load_count = 0

    def load(self, stop):
        self.load_count += 1
        self.entered.set()
        while not self.gate.wait(.01):
            if stop.is_set():
                raise InterruptedError()
        if self.failure:
            raise self.failure
        return validate_inference_models(self.models)

    @contextmanager
    def client(self, ready=True, **options):
        if ready:
            self.gate.set()
        self.resources = ModelResources(self.load)
        with patch.object(app, "create_model_resources", return_value=self.resources), \
             patch.object(app, "serving_data", self.store), \
             patch("data_loader.data_loader.load_data", side_effect=AssertionError("Bulk loader forbidden")), \
             patch("utils.supabase_utils.extract_table_from_supabase", side_effect=AssertionError("Bulk extraction forbidden")), \
             TestClient(app.app, **options) as client:
            if ready:
                self.wait_ready(client, 200)
            yield client

    def wait_ready(self, client, code):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = client.get("/ready")
            if response.status_code == code and (code != 503 or response.json()["status"] == "degraded"):
                return response
            time.sleep(.01)
        self.fail("Models did not reach expected readiness")

    def test_all_existing_endpoint_payloads_and_validation(self):
        with self.client() as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            self.assertEqual(client.get("/ready").json()["status"], "ready")
            result = client.get("/api/v1/users/u/recommendations?n=2").json()
            self.assertEqual(set(result), {"user_id", "engine_used", "reason", "items"})
            self.assertEqual(result["user_id"], "u")
            self.assertEqual(result["engine_used"], "collaborative")
            self.assertEqual([row["parent_asin"] for row in result["items"]], ["b", "a"])
            product = client.get("/api/v1/items/seen").json()
            self.assertEqual(product["parent_asin"], "seen")
            profile = client.get("/api/v1/users/u").json()
            self.assertEqual(profile["user_id"], "u")
            self.assertEqual(profile["num_reviews"], 1)
            page_fields = {"items", "page", "page_size", "total_count", "page_count"}
            reviews = client.get("/api/v1/items/seen/reviews?page=1&page_size=1").json()
            self.assertEqual(set(reviews), page_fields)
            self.assertEqual((len(reviews["items"]), reviews["total_count"], reviews["page_count"]), (1, 2, 2))
            history = client.get("/api/v1/users/alpha/history?page_size=1").json()
            self.assertEqual(set(history), page_fields | {"user_id", "verified_only"})
            self.assertTrue(history["verified_only"])
            self.assertIsInstance(history["items"][0]["item"], dict)
            shoppers = client.get("/api/v1/randomusers?n=2").json()
            self.assertEqual(len(shoppers), 2)
            self.assertTrue(all(set(row) == {"user_id"} for row in shoppers))
            for path in ("/api/v1/items/missing", "/api/v1/items/missing/reviews", "/api/v1/users/missing"):
                with self.subTest(path=path):
                    self.assertEqual(client.get(path).status_code, 404)
            for path in ("/api/v1/randomusers?n=0", "/api/v1/users/u/recommendations?n=101",
                         "/api/v1/items/seen/reviews?page=0", "/api/v1/users/u/history?page_size=101",
                         "/api/v1/users/%20/recommendations"):
                with self.subTest(path=path):
                    self.assertEqual(client.get(path).status_code, 422)

    def test_review_and_history_pages_fill_under_small_server_response_caps(self):
        self.db.tables["Review"].extend(Database.review(str(i), "alpha", "seen", 4.) for i in range(5, 10))
        self.db.response_cap = 2
        with self.client(ready=False) as client:
            for route, key in (("/api/v1/items/seen/reviews", "parent_asin"),
                               ("/api/v1/users/alpha/history", "user_id")):
                with self.subTest(route=route):
                    first = client.get(route + "?page=1&page_size=4").json()
                    second = client.get(route + "?page=2&page_size=4").json()
                    self.assertEqual((len(first["items"]), first["total_count"], first["page_count"]), (4, 7, 2))
                    self.assertEqual(len(second["items"]), 3)
                    first_ids = {row["review_id"] for row in first["items"]}
                    second_ids = {row["review_id"] for row in second["items"]}
                    self.assertFalse(first_ids & second_ids)
                    self.assertEqual(len(first_ids | second_ids), 7)

    def test_startup_and_browsing_do_not_extract_serving_tables_or_wait_for_models(self):
        with self.client(ready=False) as client:
            self.assertTrue(self.entered.wait(5))
            self.assertEqual(self.db.calls, [])
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            self.assertEqual(client.get("/ready").status_code, 503)
            self.assertEqual(client.get("/api/v1/randomusers").status_code, 200)
            self.assertEqual(client.get("/api/v1/items/c").status_code, 200)
            self.assertEqual(client.get("/api/v1/items/c/reviews").status_code, 200)
            self.assertEqual(client.get("/api/v1/users/u").status_code, 200)
            response = client.get("/api/v1/users/alpha/recommendations")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["Retry-After"], "5")
            self.assertEqual(self.load_count, 1)
        self.assertTrue(self.resources.task.done())
        self.assertIsNone(self.resources.value)

    def test_production_startup_factory_initializes_models_only_once(self):
        with patch.object(app, "ModelsLoader", return_value=self.models) as loader, \
             patch.object(app, "serving_data", self.store), \
             patch("data_loader.data_loader.load_data", side_effect=AssertionError("Bulk load")), \
             patch("utils.supabase_utils.extract_table_from_supabase", side_effect=AssertionError("Bulk extraction")), \
             TestClient(app.app) as client:
            self.wait_ready(client, 200)
            client.get("/health")
            client.get("/ready")
            loader.assert_called_once()
            self.assertIsInstance(loader.call_args.kwargs["stop_event"], Event)
            self.assertEqual(self.db.calls, [])


    def test_all_product_endpoints_return_decoded_images_without_changing_stored_data(self):
        expected = {}
        for item in self.db.tables["Item"]:
            images = {"variant": ["MAIN"], "large": [f"https://example.com/{item['parent_asin']}.jpg"],
                      "hi_res": [None], "thumb": [None]}
            expected[item["parent_asin"]] = images
            item["item_images"] = json.dumps(images)

        def check(items):
            self.assertTrue(items)
            for item in items:
                self.assertEqual(item["item_images"], expected[item["parent_asin"]])
                self.assertTrue(set(CARD_FIELDS) <= set(item))

        with self.client(ready=False) as client:
            check([client.get("/api/v1/items/a").json()])
            check([row["item"] for row in client.get("/api/v1/users/alpha/history").json()["items"]])
            self.gate.set()
            self.wait_ready(client, 200)
            for user, engine in (("u", "collaborative"), ("alpha", "content_based"), ("unknown", "popular")):
                response = client.get(f"/api/v1/users/{user}/recommendations").json()
                self.assertEqual(response["engine_used"], engine)
                check(response["items"])
                if engine == "content_based":
                    self.assertTrue(all("score" in item for item in response["items"]))
        self.assertTrue(all(isinstance(item["item_images"], str) for item in self.db.tables["Item"]))
        self.assertTrue(all("item_images" in query.fields.split(",")
                            for query in self.db.calls if query.table == "Item"))


    def test_history_is_filtered_and_fully_keyset_paginated_even_with_short_pages(self):
        self.db.tables["Review"].extend(Database.review(str(i), "alpha", "c", 2.) for i in range(5, 12))
        self.db.response_cap = 2
        self.store.history_page_size = 1000
        history = self.store.history("alpha")
        self.assertEqual(len(history), 9)
        self.assertEqual(history.columns.tolist(), HISTORY_FIELDS)
        self.assertEqual(set(history.user_id), {"alpha"})
        self.assertIn(1., history.review_rating.tolist())
        self.assertGreater(len(self.db.calls), 2)
        for query in self.db.calls:
            self.assertIn(("eq", "user_id", "alpha"), query.filters)
            self.assertEqual(query.orders, [("review_id", False, None)])
            self.assertEqual(query.requested_limit, 1000)
        self.assertTrue(self.store.history("no-such-shopper").empty)


    def test_collaborative_wins_without_fetching_history_and_preserves_rank(self):
        self.models.model_artefacts["collaborative_filtering"]["model"].recommend.return_value = (
            np.array([1, 2, 0, -1]), np.array([.9, .8, -np.finfo(np.float32).max, np.nan]),
        )
        with self.client() as client:
            response = client.get("/api/v1/users/%20u%20/recommendations?n=4")
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertEqual(set(result), {"user_id", "engine_used", "reason", "items"})
            self.assertEqual(result["user_id"], "u")
            self.assertEqual(result["engine_used"], "collaborative")
            self.assertEqual([item["parent_asin"] for item in result["items"]], ["b", "a"])
            self.assertTrue(set(CARD_FIELDS) <= set(result["items"][0]))
            self.assertEqual(len(self.db.calls), 1)
            self.assertEqual(self.db.calls[0].table, "Item")
            self.assertIn(("in", "parent_asin", ["b", "a"]), self.db.calls[0].filters)

    def test_content_fallback_keeps_full_candidates_seen_exclusion_and_order(self):
        with self.client() as client:
            # This shopper was never selected by randomusers and isn't in the ALS mapping.
            client.get("/api/v1/randomusers?n=1")
            response = client.get("/api/v1/users/alpha/recommendations?n=100")
            result = response.json()
            self.assertEqual(result["engine_used"], "content_based")
            self.assertEqual([item["parent_asin"] for item in result["items"]], ["b", "c"])
            self.assertTrue(all("score" in item for item in result["items"]))
            metadata = [query for query in self.db.calls if query.table == "Item"]
            self.assertEqual(len(metadata), 1)
            self.assertIn(("in", "parent_asin", ["b", "c"]), metadata[0].filters)


    def test_concurrent_shoppers_never_share_histories(self):
        self.db.history_barrier = Barrier(2)
        with self.client() as client, ThreadPoolExecutor(max_workers=2) as executor:
            futures = {uid: executor.submit(client.get, f"/api/v1/users/{uid}/recommendations?n=100")
                       for uid in ("alpha", "beta")}
            results = {uid: future.result(timeout=10).json() for uid, future in futures.items()}
            alpha = {item["parent_asin"] for item in results["alpha"]["items"]}
            beta = {item["parent_asin"] for item in results["beta"]["items"]}
            self.assertEqual(alpha, {"b", "c"})
            self.assertEqual(beta, {"seen", "a", "c"})

    def test_empty_history_returns_popular_but_history_or_metadata_failures_are_503(self):
        with self.client() as client:
            result = client.get("/api/v1/users/unknown/recommendations").json()
            self.assertEqual(result["engine_used"], "popular")
            self.assertEqual([item["parent_asin"] for item in result["items"]], ["seen", "a", "b", "c"])
            self.db.fail.add("Review")
            response = client.get("/api/v1/users/alpha/recommendations")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("credential", response.text)
            self.db.fail.clear()
            self.db.fail.add("Item")
            self.assertEqual(client.get("/api/v1/users/u/recommendations").status_code, 503)
            self.assertEqual(client.get("/api/v1/users/unknown/recommendations").status_code, 503)


    def test_popular_exclusion_fills_results_under_lower_database_response_cap(self):
        self.db.response_cap = 1
        items = self.store.popular_items(2, exclude_ids={"seen", "a"})
        self.assertEqual([item["parent_asin"] for item in items], ["b", "c"])
        self.assertEqual([query.interval[0] for query in self.db.calls], [0, 1, 2, 3])
        self.assertTrue(all(query.interval[1] - query.interval[0] + 1 <= 100 for query in self.db.calls))
        self.assertTrue(all(query.fields == ",".join(CARD_FIELDS) for query in self.db.calls))


    def test_inference_errors_do_not_trigger_fallback_or_look_like_empty_results(self):
        self.models.model_artefacts["collaborative_filtering"]["model"].recommend.side_effect = RuntimeError("Bad inference")
        with self.client(raise_server_exceptions=False) as client:
            self.assertEqual(client.get("/api/v1/users/u/recommendations").status_code, 500)
            self.assertEqual(self.db.calls, [])


    def test_model_failure_does_not_disable_browsing_or_fallback_to_empty_results(self):
        self.failure = ValueError("Incompatible model")
        with self.client(ready=False) as client:
            self.gate.set()
            report = self.wait_ready(client, 503)
            self.assertEqual(report.json()["resources"]["models"], "error")
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/api/v1/randomusers").status_code, 200)
            self.assertEqual(client.get("/api/v1/items/a").status_code, 200)
            self.assertEqual(client.get("/api/v1/users/u/recommendations").status_code, 503)


    def test_all_endpoint_urls_and_model_alignment_checks(self):
        validate_inference_models(self.models)
        broken = deepcopy(self.models)
        broken.model_artefacts["collaborative_filtering"]["model"].user_factors = np.ones((2, 2))
        with self.assertRaises(ValueError):
            validate_inference_models(broken)
        with self.client(ready=False) as client:
            self.assertEqual(set(client.get("/openapi.json").json()["paths"]), {
                "/health", "/ready", "/api/v1/randomusers",
                "/api/v1/users/{user_id}", "/api/v1/users/{user_id}/recommendations",
                "/api/v1/users/{user_id}/history",
                "/api/v1/items/{parent_asin}", "/api/v1/items/{parent_asin}/reviews",
            })
            self.assertEqual(client.get("/docs").status_code, 200)
            self.assertEqual(client.get("/redoc").status_code, 200)


if __name__ == "__main__":
    unittest.main()
