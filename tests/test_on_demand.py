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
        "collaborative_filtering": (model, csr_matrix([[1., 0., 0., 0.]]), {0: "u"},
                                    dict(enumerate(["seen", "b", "a", "c"]))),
        "content_based_filtering": (SimpleNamespace(vocabulary_={"x": 0, "y": 1}, idf_=np.ones(2)),
                                    csr_matrix([[1., 0.], [.9, .1], [.8, .2], [.2, .8]]), meta),
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
             patch.object(app.routing.collaborative, "get_data_loader", side_effect=AssertionError("Shared data forbidden")), \
             patch.object(app.routing.content, "get_data_loader", side_effect=AssertionError("Shared data forbidden")), \
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

    def test_startup_and_browsing_do_not_extract_serving_tables_or_wait_for_models(self):
        with self.client(ready=False) as client:
            self.assertTrue(self.entered.wait(5))
            self.assertEqual(self.db.calls, [])
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            self.assertEqual(client.get("/ready").status_code, 503)
            self.assertEqual(client.get("/api/v1/randomusers").status_code, 200)
            self.assertEqual(client.get("/api/v1/popular_items").status_code, 200)
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

    def test_random_and_popular_responses_are_bounded_and_ordered(self):
        self.db.tables["Review"].append(Database.review("5", "u", "a", 5.))
        self.db.tables["Review"].append(Database.review("6", None, "a", 5.))
        self.db.tables["Item"].extend([
            {**self.db.tables["Item"][0], "parent_asin": "null-count", "num_reviews": None},
            {**self.db.tables["Item"][0], "parent_asin": "null-rating", "avg_rating": None},
        ])
        with self.client(ready=False) as client:
            self.assertEqual(len(client.get("/api/v1/randomusers?n=2").json()), 2)
            rpc = self.db.calls[-1]
            self.assertEqual(rpc.table, "rpc:random_shoppers")
            self.assertEqual(rpc.params, {"sample_size": 2})
            response = client.get("/api/v1/popular_items?limit=5")
            self.assertEqual([row["parent_asin"] for row in response.json()], ["seen", "null-rating", "a", "b", "c"])
            query = self.db.calls[-1]
            self.assertEqual(query.requested_limit, 5)
            self.assertEqual(query.orders, [("num_reviews", True, False), ("avg_rating", True, False),
                                            ("parent_asin", False, False)])
            self.assertEqual(query.fields, ",".join(CARD_FIELDS))

    def test_empty_browsing_results_are_successful_and_profiles_keep_their_fields(self):
        with self.client(ready=False) as client:
            profile = client.get("/api/v1/users/u")
            self.assertEqual(profile.status_code, 200)
            self.assertEqual(profile.json()["user_id"], "u")
            self.assertEqual(profile.json()["num_reviews"], 1)
            self.assertEqual(set(profile.json()), set(PROFILE_FIELDS.split(",")))
            self.db.tables["Review"] = []
            self.db.tables["Item"] = []
            self.assertEqual(client.get("/api/v1/randomusers").json(), [])
            self.assertEqual(client.get("/api/v1/popular_items").json(), [])

    def test_image_serialization_preserves_dictionaries_and_handles_bad_values(self):
        images = {"variant": ["MAIN", "PT01"], "large": ["https://example.com/Main.JPG", None],
                  "hi_res": [None, None], "thumb": [None, "https://example.com/Other.jpg"]}
        for value, expected in (
            (json.dumps(images), images), (images, images), ({}, {}), ("{}", {}),
            (None, None), (np.nan, None), ("", None), ("{bad json", None),
            ("[]", None), ([], None), ("null", None), ("42", None), ('"url"', None),
        ):
            with self.subTest(value=value):
                rows = [{"parent_asin": "a", "score": .9, "item_images": value}]
                history = [{"review_id": "r", "item": {"parent_asin": "a", "item_images": value}},
                           {"review_id": "missing", "item": None}]
                result = app._serialize_items(rows)
                history_result = app._serialize_items(history)
                self.assertEqual(result[0]["item_images"], expected)
                self.assertEqual(history_result[0]["item"]["item_images"], expected)
                self.assertEqual(result[0]["score"], .9)
                self.assertIsNone(history_result[1]["item"])
                self.assertNotIn("item_images", history_result[0])
                self.assertIs(rows[0]["item_images"], value)  # No mutation of fetched data.

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
            check(client.get("/api/v1/popular_items").json())
            check([client.get("/api/v1/items/a").json()])
            check([row["item"] for row in client.get("/api/v1/users/alpha/history").json()["items"]])
            self.gate.set()
            self.wait_ready(client, 200)
            for user, engine in (("u", "collaborative"), ("alpha", "content_based")):
                response = client.get(f"/api/v1/users/{user}/recommendations").json()
                self.assertEqual(response["engine_used"], engine)
                check(response["items"])
                if engine == "content_based":
                    self.assertTrue(all("score" in item for item in response["items"]))
        self.assertTrue(all(isinstance(item["item_images"], str) for item in self.db.tables["Item"]))
        self.assertTrue(all("item_images" in query.fields.split(",")
                            for query in self.db.calls if query.table == "Item"))

    def test_bad_images_do_not_break_product_responses(self):
        for item, value in zip(self.db.tables["Item"], ("{broken", None, "[]", "")):
            item["item_images"] = value
        with self.client() as client:
            for path in ("/api/v1/popular_items", "/api/v1/items/a", "/api/v1/users/alpha/history",
                         "/api/v1/users/u/recommendations", "/api/v1/users/alpha/recommendations"):
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200)
                    payload = response.json()
                    items = payload if isinstance(payload, list) else payload.get("items", [payload])
                    for row in items:
                        product = row.get("item", row)
                        self.assertIsNone(product["item_images"])
                        self.assertIsNotNone(product["item_title"])

    def test_existing_image_objects_are_accepted_by_product_endpoints(self):
        images = {"variant": ["MAIN"], "large": ["https://example.com/Original.JPG"],
                  "hi_res": [None], "thumb": [None]}
        for item in self.db.tables["Item"]:
            item["item_images"] = deepcopy(images)
        with self.client(ready=False) as client:
            self.assertEqual(client.get("/api/v1/items/a").json()["item_images"], images)
            history = client.get("/api/v1/users/alpha/history").json()
            self.assertTrue(all(row["item"]["item_images"] == images for row in history["items"]))

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

    def test_user_history_defaults_to_verified_reviews_without_models_or_profile(self):
        self.db.tables["Review"].extend([
            {**Database.review("5", "alpha", "b", 5.), "verified_purchase": False,
             "review_date": "2026-02-01"},
            {**Database.review("6", "alpha", "a", 2.), "review_date": "2026-01-02"},
            {**Database.review("7", "alpha", "c", 4.), "review_date": None},
        ])
        with self.client(ready=False) as client:
            response = client.get("/api/v1/users/alpha/history?page_size=2")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["user_id"], "alpha")
            self.assertTrue(payload["verified_only"])
            self.assertEqual((payload["page"], payload["page_size"]), (1, 2))
            self.assertEqual((payload["total_count"], payload["page_count"]), (4, 2))
            self.assertEqual([row["review_id"] for row in payload["items"]], ["6", "2"])
            self.assertEqual(payload["items"][0]["item"]["parent_asin"], "a")
            self.assertEqual(set(payload["items"][0]["item"]), set(CARD_FIELDS))
            query = self.db.calls[0]
            self.assertEqual(query.filters, [("eq", "user_id", "alpha"),
                                             ("eq", "verified_purchase", True)])
            self.assertEqual(query.interval, (0, 1))
            self.assertEqual(query.orders, [("review_date", True, False), ("review_id", False, None)])
            self.assertEqual([query.table for query in self.db.calls], ["Review", "Item"])
            second = client.get("/api/v1/users/alpha/history?page=2&page_size=2").json()
            self.assertEqual([row["review_id"] for row in second["items"]], ["4", "7"])
            self.assertIsNone(second["items"][-1]["review_date"])

    def test_user_history_can_include_unverified_reviews_and_preserves_missing_products(self):
        self.db.tables["Review"].extend([
            {**Database.review("5", "alpha", "missing", 1.), "verified_purchase": False},
            {**Database.review("6", "alpha", "a", 2.), "verified_purchase": None},
        ])
        with self.client(ready=False) as client:
            payload = client.get("/api/v1/users/alpha/history?verified_only=false").json()
            self.assertFalse(payload["verified_only"])
            self.assertEqual((payload["total_count"], payload["page_size"]), (4, 20))
            self.assertEqual([row["review_id"] for row in payload["items"]], ["2", "4", "5", "6"])
            self.assertIsNone(payload["items"][2]["item"])
            self.assertIsNone(payload["items"][3]["verified_purchase"])
            self.assertEqual(self.db.calls[0].filters, [("eq", "user_id", "alpha")])
            verified = client.get("/api/v1/users/alpha/history").json()
            self.assertEqual(verified["total_count"], 2)

    def test_user_history_fills_a_bounded_page_with_lower_server_caps(self):
        self.db.response_cap = 1
        self.db.tables["Review"].append(Database.review("5", "alpha", "seen", 2.))
        with self.client(ready=False) as client:
            payload = client.get("/api/v1/users/alpha/history?page_size=2").json()
            self.assertEqual([row["review_id"] for row in payload["items"]], ["2", "4"])
            self.assertEqual([row["item"]["parent_asin"] for row in payload["items"]], ["seen", "a"])
            queries = [query for query in self.db.calls if query.table == "Review"]
            self.assertEqual([query.interval for query in queries], [(0, 1), (1, 1)])
            last = client.get("/api/v1/users/alpha/history?page=2&page_size=2").json()
            self.assertEqual(last["items"][0]["parent_asin"], "seen")
            self.assertEqual(len(last["items"]), 1)
            self.assertEqual(last["total_count"], 3)

    def test_user_history_empty_results_do_not_fetch_metadata(self):
        with self.client(ready=False) as client:
            for path, count in (("unknown/history", 0), ("alpha/history?page=100", 2)):
                self.db.calls.clear()
                response = client.get("/api/v1/users/" + path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["items"], [])
                self.assertEqual(response.json()["total_count"], count)
                self.assertEqual([query.table for query in self.db.calls], ["Review"])
            self.db.tables["Review"][0]["verified_purchase"] = False
            self.assertEqual(client.get("/api/v1/users/u/history").json()["items"], [])

    def test_user_history_validation_and_upstream_failures(self):
        with self.client(ready=False) as client:
            for path in ("%20/history", "alpha/history?page=0", "alpha/history?page_size=0",
                         "alpha/history?page_size=101", "alpha/history?verified_only=invalid"):
                self.assertEqual(client.get("/api/v1/users/" + path).status_code, 422)
            self.assertEqual(self.db.calls, [])
            for table in ("Review", "Item"):
                self.db.fail = {table}
                response = client.get("/api/v1/users/alpha/history")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.headers["Retry-After"], "5")
                self.assertEqual(response.json(), {"detail": "Supabase request failed"})
                self.assertNotIn("credential", response.text)

    def test_collaborative_wins_without_fetching_history_and_preserves_rank(self):
        self.models.model_artefacts["collaborative_filtering"][0].recommend.return_value = (
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

    def test_serving_reuses_model_lookup_maps_without_rebuilding_them(self):
        class NoIteration(dict):
            def items(self):
                raise AssertionError("User index must not be rebuilt per request")

            def values(self):
                raise AssertionError("Eligibility must use the prepared user index")

        with self.client() as client:
            cf = self.models.model_artefacts["collaborative_filtering"]
            self.models.model_artefacts["collaborative_filtering"] = (cf[0], cf[1], NoIteration(cf[2]), cf[3])
            user_index, content_indexes = self.models.user_index, self.models.content_indexes
            with patch("src.content_based_filtering.recommender.build_item_index_maps",
                       side_effect=AssertionError("Content index must not be rebuilt per request")):
                self.assertEqual(client.get("/api/v1/users/u/recommendations").json()["engine_used"], "collaborative")
                self.assertEqual(client.get("/api/v1/users/alpha/recommendations").json()["engine_used"], "content_based")
            self.assertIs(self.models.user_index, user_index)
            self.assertIs(self.models.content_indexes, content_indexes)

    def test_collaborative_inference_is_locked_but_metadata_query_is_not(self):
        model = self.models.model_artefacts["collaborative_filtering"][0]
        fetch_cards = self.store.cards

        def recommend(**kwargs):
            self.assertTrue(app._inference_lock.locked())
            return model.recommend.return_value

        def cards(ids):
            acquired = app._inference_lock.acquire(blocking=False)
            self.assertTrue(acquired, "Database fetching must not hold the inference lock")
            try:
                return fetch_cards(ids)
            finally:
                if acquired:
                    app._inference_lock.release()

        model.recommend.side_effect = recommend
        with self.client() as client, patch.object(self.store, "cards", side_effect=cards):
            self.assertEqual(client.get("/api/v1/users/u/recommendations").status_code, 200)

    def test_empty_collaborative_result_permits_content_fallback(self):
        self.models.model_artefacts["collaborative_filtering"][0].recommend.return_value = (np.array([], dtype=int), np.array([]))
        with self.client() as client:
            response = client.get("/api/v1/users/u/recommendations")
            self.assertEqual(response.json()["engine_used"], "content_based")
            self.assertIn("Collaborative filtering returned no items", response.json()["reason"])

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

    def test_empty_history_is_success_but_history_or_metadata_failures_are_503(self):
        with self.client() as client:
            result = client.get("/api/v1/users/unknown/recommendations").json()
            self.assertIsNone(result["items"])
            self.assertIsNone(result["engine_used"])
            self.db.fail.add("Review")
            response = client.get("/api/v1/users/alpha/recommendations")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("credential", response.text)
            self.db.fail.clear()
            self.db.fail.add("Item")
            self.assertEqual(client.get("/api/v1/users/u/recommendations").status_code, 503)

    def test_inference_errors_do_not_trigger_fallback_or_look_like_empty_results(self):
        self.models.model_artefacts["collaborative_filtering"][0].recommend.side_effect = RuntimeError("Bad inference")
        with self.client(raise_server_exceptions=False) as client:
            self.assertEqual(client.get("/api/v1/users/u/recommendations").status_code, 500)
            self.assertEqual(self.db.calls, [])

    def test_product_links_reviews_profiles_missing_ids_and_validation(self):
        with self.client(ready=False) as client:
            self.assertEqual(client.get("/api/v1/items/c").json()["description"], "Description")
            query = self.db.calls[-1]
            self.assertEqual(query.filters, [("eq", "parent_asin", "c")])
            self.assertEqual(query.requested_limit, 1)
            self.assertEqual(client.get("/api/v1/items/no-such-id").status_code, 404)
            self.assertEqual(client.get("/api/v1/users/no-such-id").status_code, 404)
            response = client.get("/api/v1/items/seen/reviews?page=2&page_size=1")
            self.assertEqual(response.status_code, 200)
            review_query = self.db.calls[-1]
            self.assertEqual(self.db.calls[-2].fields, "parent_asin")
            self.assertEqual(review_query.interval, (1, 1))
            self.assertEqual(review_query.orders, [("review_date", True, False), ("review_id", False, None)])
            self.assertEqual(response.json()["total_count"], 2)
            self.assertEqual(response.json()["page_count"], 2)
            self.assertEqual(response.json()["items"][0]["user_id"], "alpha")
            self.assertEqual(client.get("/api/v1/items/seen/reviews?page=100").json()["items"], [])
            calls = len(self.db.calls)
            self.assertEqual(client.get("/api/v1/items/unknown/reviews").status_code, 404)
            self.assertEqual(len(self.db.calls), calls + 1)  # Existence lookup only.
            for path in ("/api/v1/randomusers?n=0", "/api/v1/popular_items?limit=101",
                         "/api/v1/items/a/reviews?page=0", "/api/v1/items/a/reviews?page_size=101",
                         "/api/v1/users/u/recommendations?n=101", "/api/v1/users/%20/recommendations"):
                self.assertEqual(client.get(path).status_code, 422)
            self.db.fail.add("Item")
            self.assertEqual(client.get("/api/v1/items/a").status_code, 503)
            self.db.fail.add("rpc:random_shoppers")
            self.assertEqual(client.get("/api/v1/randomusers").status_code, 503)

    def test_model_failure_does_not_disable_browsing_or_fallback_to_empty_results(self):
        self.failure = ValueError("Incompatible model")
        with self.client(ready=False) as client:
            self.gate.set()
            report = self.wait_ready(client, 503)
            self.assertEqual(report.json()["resources"]["models"], "error")
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/api/v1/randomusers").status_code, 200)
            self.assertEqual(client.get("/api/v1/popular_items").status_code, 200)
            self.assertEqual(client.get("/api/v1/users/u/recommendations").status_code, 503)

    def test_metadata_batch_handles_lower_server_cap_and_preserves_enrichment_order(self):
        self.db.response_cap = 1
        with patch.object(app, "serving_data", self.store):
            result = app._card_items([{"parent_asin": "b", "score": .9}, {"parent_asin": "a", "score": .8}])
        self.assertEqual([row["parent_asin"] for row in result], ["b", "a"])
        self.assertEqual([row["score"] for row in result], [.9, .8])
        self.assertEqual(len(self.db.calls), 2)
        self.assertTrue(all(query.table == "Item" for query in self.db.calls))

    def test_all_endpoint_urls_and_model_alignment_checks(self):
        validate_inference_models(self.models)
        broken = deepcopy(self.models)
        broken.model_artefacts["collaborative_filtering"][0].user_factors = np.ones((2, 2))
        with self.assertRaises(ValueError):
            validate_inference_models(broken)
        with self.client(ready=False) as client:
            self.assertEqual(set(client.get("/openapi.json").json()["paths"]), {
                "/health", "/ready", "/api/v1/randomusers", "/api/v1/popular_items",
                "/api/v1/users/{user_id}", "/api/v1/users/{user_id}/recommendations",
                "/api/v1/users/{user_id}/history",
                "/api/v1/items/{parent_asin}", "/api/v1/items/{parent_asin}/reviews",
            })
            self.assertEqual(client.get("/docs").status_code, 200)
            self.assertEqual(client.get("/redoc").status_code, 200)

if __name__ == "__main__":
    unittest.main()
