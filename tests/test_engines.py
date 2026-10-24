"""Regression checks for exhausted candidate pools and model ranking."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPBASE_SECRET_KEY", "test-key")

from src.collaborative_filtering import main as collaborative
from src.collaborative_filtering.recommender import recommend_for_user as cf_recommend
from src.content_based_filtering.recommender import recommend_for_user as cb_recommend
from src.content_based_filtering.recommender import build_user_profile


class EngineTests(unittest.TestCase):
    def test_content_profile_preserves_rating_and_recency_weights(self):
        history = pd.DataFrame([
            {"user_id": "u", "parent_asin": "a", "review_rating": 5., "recency_weight": 1.},
            {"user_id": "u", "parent_asin": "b", "review_rating": 3., "recency_weight": .5},
            {"user_id": "u", "parent_asin": "a", "review_rating": 1., "recency_weight": 100.},
            {"user_id": "other", "parent_asin": "b", "review_rating": 5., "recency_weight": 100.},
        ])
        profile = build_user_profile("u", history, csr_matrix(np.eye(2)), {"a": 0, "b": 1})
        np.testing.assert_allclose(profile, [[1 / 1.3, .3 / 1.3]])

    def test_content_preserves_category_boost_free_preference_and_low_rating_seen_exclusion(self):
        meta = pd.DataFrame([
            {"parent_asin": asin, "item_title": asin, "main_category": category, "is_free": free}
            for asin, category, free in (("seen", "Liked", True), ("paid", "Liked", False),
                                         ("other", "Other", True), ("same", "Liked", True),
                                         ("seen-low", "Liked", True))
        ])
        history = pd.DataFrame([
            {"user_id": "u", "parent_asin": "seen", "review_rating": 5., "recency_weight": 1.},
            {"user_id": "u", "parent_asin": "seen-low", "review_rating": 1., "recency_weight": 1.},
        ])
        result = cb_recommend("u", history, (None, csr_matrix(np.ones((5, 2))), meta), n=100)
        self.assertEqual(result.parent_asin.tolist(), ["same", "other"])
        self.assertGreater(result.iloc[0].score, result.iloc[1].score)

    def test_collaborative_filters_exhausted_and_seen_results(self):
        model = Mock()
        model.recommend.return_value = (
            np.array([1, 0, 2, -1]),
            np.array([.8, -np.finfo(np.float32).max, -np.finfo(np.float32).max, np.nan]),
        )
        artifacts = (model, csr_matrix([[1, 0, 0]]), {0: "u"},
                     {0: "seen", 1: "unseen", 2: "padding"})
        self.assertEqual(cf_recommend("u", artifacts, n=4), ["unseen"])

    def test_collaborative_preserves_model_ranking(self):
        catalogue = pd.DataFrame([
            {"parent_asin": asin, "item_title": asin, "main_category": "Software", "is_free": True}
            for asin in ["a", "b"]
        ])
        with patch.object(collaborative, "get_data_loader", return_value=SimpleNamespace(item_df=catalogue)), \
             patch.object(collaborative, "get_models_loader", return_value=SimpleNamespace(
                 model_artefacts={collaborative.MODEL_NAME: ()})), \
             patch.object(collaborative, "recommend_for_user", return_value=["b", "a"]):
            result = collaborative.get_user_recommendations("u", n=2)
        self.assertEqual([item["parent_asin"] for item in result], ["b", "a"])

    def test_content_never_returns_seen_products_when_pool_is_exhausted(self):
        metadata = pd.DataFrame([
            {"parent_asin": asin, "item_title": asin, "main_category": "Software", "is_free": True}
            for asin in ["a", "b"]
        ])
        matrix = csr_matrix([[1., 0.], [1., 1.]])
        history = pd.DataFrame([
            {"user_id": "u", "parent_asin": "a", "review_rating": 5, "recency_weight": 1.}
        ])
        result = cb_recommend("u", history, (None, matrix, metadata), n=10)
        self.assertEqual(result["parent_asin"].tolist(), ["b"])
        history = pd.concat([history, history.assign(parent_asin="b")], ignore_index=True)
        self.assertTrue(cb_recommend("u", history, (None, matrix, metadata), n=10).empty)
