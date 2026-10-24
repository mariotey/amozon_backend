"""Routing and HTTP contracts tested with small datasets and mocked inference."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import pandas as pd

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPBASE_SECRET_KEY", "test-key")

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



if __name__ == "__main__":
    unittest.main()
