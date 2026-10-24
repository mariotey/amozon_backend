"""Routing and HTTP contracts tested with small datasets and mocked inference."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import pandas as pd
from fastapi.testclient import TestClient

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPBASE_SECRET_KEY", "test-key")

import app
from src import routing


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = pd.DataFrame([
            {"parent_asin": asin, "item_title": asin, "main_category": "Software",
             "is_free": True, "quality_score": quality, "num_reviews": reviews,
             "price": 0.0, "avg_rating": 4.5, "store": "store",
             "description": "Product description", "popularity_segment": "popular"}
            for asin, quality, reviews in [("seen", 1.0, 100), ("b", .8, 10),
                                            ("a", .8, 10), ("c", .8, 5)]
        ])
        self.data = SimpleNamespace(
            item_df=self.catalogue,
            user_item_df=pd.DataFrame([
                {"user_id": "u", "parent_asin": "seen", "review_rating": 3},
            ]),
        )
        self.models = SimpleNamespace(model_artefacts={
            "collaborative_filtering": (None, None, {0: "u"}, {}),
            "content_based_filtering": (None, None, pd.DataFrame({"parent_asin": ["seen"]})),
        })
        for target, value in [
            ("get_data_loader", self.data), ("get_models_loader", self.models),
        ]:
            mocked = patch.object(routing.collaborative, target, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)
        cf = patch.object(routing.collaborative, "get_user_recommendations",
                          return_value=[{"parent_asin": "cf"}])
        cb = patch.object(routing.content, "get_user_recommendations",
                          return_value=[{"parent_asin": "cb"}])
        self.cf = cf.start()
        self.cb = cb.start()
        self.addCleanup(cf.stop)
        self.addCleanup(cb.stop)

    def test_cf_wins_when_both_engines_qualify(self):
        result = routing.get_user_recommendations("u", n=3)
        self.assertEqual(result["engine_used"], "collaborative")
        self.cf.assert_called_once_with("u", n=3)
        self.cb.assert_not_called()

    def test_cf_does_not_require_positive_reviews(self):
        self.data.user_item_df["review_rating"] = 1
        self.assertEqual(routing.get_user_recommendations("u")["engine_used"], "collaborative")

    def test_untrained_user_with_qualifying_history_uses_content(self):
        self.models.model_artefacts["collaborative_filtering"][2].clear()
        self.assertEqual(routing.get_user_recommendations("u")["engine_used"], "content_based")
        self.cf.assert_not_called()

    def test_empty_cf_falls_back_to_content(self):
        self.cf.return_value = []
        self.assertEqual(routing.get_user_recommendations("u")["engine_used"], "content_based")

    def test_empty_personalised_results_return_none(self):
        self.cf.return_value = self.cb.return_value = []
        result = routing.get_user_recommendations("u", n=3)
        self.assertEqual(result["engine_used"], None)
        self.assertIsNone(result["items"])
        self.assertIn("returned no items", result["reason"])

    def test_positive_review_outside_index_does_not_qualify(self):
        self.models.model_artefacts["collaborative_filtering"][2].clear()
        self.models.model_artefacts["content_based_filtering"][2]["parent_asin"] = ["other"]
        self.assertEqual(routing.get_user_recommendations("u")["engine_used"], None)
        self.cb.assert_not_called()

    def test_low_rating_does_not_qualify_for_content(self):
        self.models.model_artefacts["collaborative_filtering"][2].clear()
        self.data.user_item_df["review_rating"] = 2
        self.assertEqual(routing.get_user_recommendations("u")["engine_used"], None)
        self.cb.assert_not_called()

    def test_unknown_user_gets_no_recommendations(self):
        result = routing.get_user_recommendations("new", n=2)
        self.assertEqual(result["engine_used"], None)
        self.assertIsNone(result["items"])
        self.cf.assert_not_called()
        self.cb.assert_not_called()

    def test_no_candidates_returns_none(self):
        self.cf.return_value = self.cb.return_value = []
        self.data.item_df = self.catalogue[self.catalogue["parent_asin"] == "seen"]
        self.assertIsNone(routing.get_user_recommendations("u")["items"])

    def test_model_errors_are_not_silently_hidden(self):
        for error in (RuntimeError("broken model"), ValueError("bad matrix")):
            with self.subTest(error=type(error).__name__):
                self.cf.side_effect = error
                with self.assertRaises(type(error)):
                    routing.get_user_recommendations("u")
        self.cb.assert_not_called()

    def test_identifiers_are_trimmed_and_input_is_validated(self):
        self.assertEqual(routing.get_user_recommendations(" u ")["user_id"], "u")
        for user, n in [(" ", 10), ("u", 0), ("u", 101)]:
            with self.subTest(user=user, n=n), self.assertRaises(ValueError):
                routing.get_user_recommendations(user, n)

    def test_http_contract_and_public_routes(self):
        with patch.object(app, "DataLoader", return_value=self.data), \
             patch.object(app, "ModelsLoader", return_value=self.models), \
             TestClient(app.app) as client:
            url = "/api/v1/users/u/recommendations"
            response = client.get(url, params={"n": 3})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(set(response.json()), {"user_id", "engine_used", "reason", "items"})
            for n in (0, 101, "bad"):
                self.assertEqual(client.get(url, params={"n": n}).status_code, 422)
            self.assertEqual(client.get("/api/v1/users/%20/recommendations").status_code, 422)
            no_recommendations = client.get("/api/v1/users/new/recommendations")
            self.assertEqual(no_recommendations.status_code, 200)
            self.assertIsNone(no_recommendations.json()["engine_used"])
            self.assertIsNone(no_recommendations.json()["items"])
            for path in (
                "/api/v1/collaborative/users/u/recommendations",
                "/api/v1/content/users/u/recommendations",
                "/api/v1/content/items/item/similar",
            ):
                self.assertEqual(client.get(path).status_code, 404)
            self.assertEqual(
                set(client.get("/openapi.json").json()["paths"]),
                {"/health", "/api/v1/users/{user_id}/recommendations", "/api/v1/users/{user_id}",
                 "/api/v1/randomusers", "/api/v1/items/popular",
                 "/api/v1/items/{parent_asin}", "/api/v1/items/{parent_asin}/reviews"},
            )
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            self.assertEqual(client.get("/docs").status_code, 200)

    def test_user_details(self):
        from unittest.mock import MagicMock
        query = MagicMock()
        query.select.return_value = query
        query.eq.return_value = query
        query.limit.return_value = query
        profile = {"user_id": "u", "num_reviews": 3, "avg_rating_given": 4.5,
                   "user_segment": "occasional", "avg_price_purchased": None}
        query.execute.return_value = SimpleNamespace(data=[profile])
        with patch.object(app, "DataLoader", return_value=self.data), \
             patch.object(app, "ModelsLoader", return_value=self.models), \
             patch.object(app.SUPABASE_CLIENT, "table", return_value=query) as table, \
             TestClient(app.app) as client:
            response = client.get("/api/v1/users/%20u%20")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), profile)
            table.assert_called_once_with("User")
            query.eq.assert_called_once_with("user_id", "u")
            query.limit.assert_called_once_with(1)
            query.execute.return_value = SimpleNamespace(data=[])
            response = client.get("/api/v1/users/missing")
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json(), {"detail": "User not found"})
            query.execute.reset_mock()
            self.assertEqual(client.get("/api/v1/users/%20").status_code, 422)
            query.execute.assert_not_called()

    def test_catalogue_endpoints_and_enriched_recommendations(self):
        self.cf.return_value = [{"parent_asin": "a", "score": .9}]
        with patch.object(app, "DataLoader", return_value=self.data), \
             patch.object(app, "ModelsLoader", return_value=self.models), \
             TestClient(app.app) as client:
            self.assertEqual(client.get("/api/v1/randomusers?n=5").json(), [{"user_id": "u"}])
            popular = client.get("/api/v1/items/popular?limit=2").json()
            self.assertEqual([item["parent_asin"] for item in popular], ["seen", "a"])
            card = client.get("/api/v1/users/u/recommendations").json()["items"][0]
            self.assertTrue(set(app.CARD_FIELDS).issubset(card))
            self.assertEqual(card["avg_rating"], 4.5)
            self.assertEqual(card["score"], .9)
            self.assertEqual(client.get("/api/v1/items/a").json()["description"], "Product description")
            self.assertEqual(client.get("/api/v1/items/missing").status_code, 404)
            for url in ("/api/v1/randomusers?n=0", "/api/v1/items/popular?limit=101",
                        "/api/v1/items/a/reviews?page=0", "/api/v1/items/a/reviews?page_size=101"):
                self.assertEqual(client.get(url).status_code, 422)

    def test_review_pagination_and_missing_products(self):
        from unittest.mock import MagicMock
        query = MagicMock()
        for method in ("select", "eq", "order", "range"):
            getattr(query, method).return_value = query
        query.execute.return_value = SimpleNamespace(data=[{"review_id": "r"}], count=21)
        with patch.object(app, "DataLoader", return_value=self.data), \
             patch.object(app, "ModelsLoader", return_value=self.models), \
             patch.object(app.SUPABASE_CLIENT, "table", return_value=query) as table, \
             TestClient(app.app) as client:
            result = client.get("/api/v1/items/a/reviews?page=2&page_size=10").json()
            self.assertEqual(result, {"items": [{"review_id": "r"}], "page": 2,
                                     "page_size": 10, "total_count": 21, "page_count": 3})
            query.eq.assert_called_once_with("parent_asin", "a")
            query.range.assert_called_once_with(10, 19)
            self.assertEqual(query.order.call_args_list[0].args, ("review_date",))
            self.assertTrue(query.order.call_args_list[0].kwargs["desc"])
            table.reset_mock()
            self.assertEqual(client.get("/api/v1/items/missing/reviews").status_code, 404)
            table.assert_not_called()
            query.execute.return_value = SimpleNamespace(data=[], count=0)
            self.assertEqual(client.get("/api/v1/items/a/reviews").json()["page_count"], 0)


if __name__ == "__main__":
    unittest.main()
