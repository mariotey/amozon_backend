"""Small real-model, bundle, metrics, and notebook regression checks; no network."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import DataLoader
from models_loader import save_model, load_model, validate_model, ModelsLoader
from src.collaborative_filtering import model as cf, recommender as cf_recommend
from src.content_based_filtering import model as cb, recommender as cb_recommend
from src.metrics import evaluate_predictions
from tests.test_supabase_data_workflow import save_dataset, items


def events():
    return pd.DataFrame([
        {"review_id": str(i), "user_id": f"u{i % 2}", "review_rating": 5.,
         "parent_asin": f"p{i % 6}" if i < 16 else ["p1", "p2", "p3", "p4"][i - 16],
         "review_date": f"2020-01-{i + 1:02d}"}
        for i in range(20)
    ])


class ModelTests(unittest.TestCase):
    def test_content_ranking_preserves_weighting_latest_reviews_and_metadata_rows(self):
        matrix = csr_matrix([[2., 0.], [0., 3.], [1., 1.], [2., 1.],
                             [1., 2.], [1., 1.], [0., 0.]], dtype=np.float32)
        metadata = pd.DataFrame({
            "parent_asin": [f"p{i}" for i in range(7)],
            "item_title": [f"Product {i}" for i in range(7)],
            "main_category": ["editor", "editor", "editor", "game", "game", "editor", "editor"],
            "is_free": False,
        }, index=range(10, 17))
        bundle = {"item_matrix": matrix, "metadata": metadata}
        history = pd.DataFrame({
            "user_id": "u", "parent_asin": ["p0", "p1", "p2", "p3", "unknown"],
            "review_rating": [5., 4., 1., 3., 5.], "recency_weight": [1., .5, 1., 1., 1.],
            "review_date": "2020-01-01",
        })
        before = history.copy(deep=True)
        expected_profile = np.array([[2., 1.2]]) / (1.4 + 1e-9)
        maps = cb_recommend.build_item_index_maps(metadata)
        np.testing.assert_allclose(cb_recommend.build_user_profile("u", history, matrix, maps[0]),
                                   expected_profile)
        result = cb_recommend.recommend_for_user("u", history, bundle, n=3)
        self.assertEqual(result.parent_asin.tolist(), ["p5", "p4", "p6"])
        expected_scores = cosine_similarity(expected_profile, matrix).ravel()[[5, 4, 6]] + [.1, 0., .1]
        np.testing.assert_allclose(result.score, expected_scores, rtol=1e-12, atol=1e-12)
        newer_dislike = history.iloc[[0]].assign(review_date="2020-01-02", review_rating=1.)
        updated = cb_recommend.recommend_for_user("u", pd.concat([history, newer_dislike]), bundle, n=3)
        self.assertEqual(updated.parent_asin.tolist(), ["p4", "p5", "p6"])
        pd.testing.assert_frame_equal(history, before)

    def test_content_free_filter_stays_after_candidate_pool_and_handles_exhausted_history(self):
        metadata = pd.DataFrame({"parent_asin": [f"p{i}" for i in range(13)],
                                 "item_title": "product", "main_category": "software",
                                 "is_free": [True] * 8 + [False] * 5})
        bundle = {"item_matrix": csr_matrix(np.ones((13, 2), dtype=np.float32)), "metadata": metadata}
        history = pd.DataFrame({"user_id": ["u"], "parent_asin": ["p0"],
                                "review_rating": [5.], "recency_weight": [1.]})
        # All similarities tie: the original descending argsort selects p12..p8.
        # They are paid, so filtering this pool returns nothing despite other free items.
        self.assertTrue(cb_recommend.recommend_for_user("u", history, bundle, n=1).empty)
        metadata.loc[12, "is_free"] = True
        self.assertEqual(cb_recommend.recommend_for_user("u", history, bundle, n=1).parent_asin.tolist(), ["p12"])
        all_seen = pd.DataFrame({"user_id": "u", "parent_asin": metadata.parent_asin,
                                 "review_rating": 5., "recency_weight": 1.})
        self.assertTrue(cb_recommend.recommend_for_user("u", all_seen, bundle).empty)
        self.assertTrue(cb_recommend.recommend_for_user("u", history.assign(review_rating=1.), bundle).empty)

    def test_content_evaluation_matches_individual_rankings_and_prepares_catalogue_once(self):
        from data_loader.data_loader import RecommendationData
        source = events().iloc[:6].assign(recency_weight=1.)
        bundle = cb.fit(items(source), max_features=100)
        targets = pd.DataFrame({"user_id": ["u0", "u1", "missing"], "parent_asin": ["p1", "p0", "p0"]})
        for dtype in (np.float32, np.float64):
            history = source.astype({"review_rating": dtype, "recency_weight": dtype})
            data = RecommendationData(items(source), history, targets, excluded_targets=7, excluded_users=3)
            predictions = {}
            for user in targets.user_id:
                rows = cb_recommend.recommend_for_user(user, history, bundle, n=3)
                predictions[user] = rows.parent_asin.tolist() if not rows.empty else []
            expected = evaluate_predictions(predictions, targets, k=3)
            with patch.object(cb_recommend, "_normalize_items", wraps=cb_recommend._normalize_items) as normalize_items:
                actual = cb_recommend.evaluate(bundle, data, k=3)
            normalize_items.assert_called_once()
            pd.testing.assert_frame_equal(actual["per_user"], expected["per_user"])
            self.assertEqual(actual["summary"], {**expected["summary"], "excluded_targets": 7, "excluded_users": 3})
            empty = cb_recommend.evaluate(bundle, RecommendationData(data.items, history, targets.iloc[:0]))
            self.assertEqual(empty["summary"]["users"], 0)
            self.assertIsNone(empty["summary"]["recall"])

    def test_signed_preferences_and_neutral_seen_items_are_preserved(self):
        source = events().iloc[:5].copy()
        source["user_id"] = "u"
        source["review_rating"] = [1., 2., 3., 4., 5.]
        matrix, users, products = cf.prepare_training_data(items(source), source)
        self.assertEqual(users, ["u"])
        np.testing.assert_array_equal(matrix.toarray(), [[-21., -11., 0., 11., 21.]])
        self.assertEqual(matrix.nnz, 5)
        self.assertEqual(products, ["p0", "p1", "p2", "p3", "p4"])
        newer = source.iloc[[0]].assign(review_id="new", review_date="2020-02-01", review_rating=5.)
        updated = cf.prepare_training_data(items(source), pd.concat([source, newer]))[0]
        self.assertEqual(updated[0, 0], 21.)

    def test_metrics_denominators_ranking_and_empty_populations(self):
        targets = pd.DataFrame({"user_id": ["u", "u", "missing"], "parent_asin": ["a", "b", "c"]})
        result = evaluate_predictions({"u": ["a", "x"]}, targets, k=2)
        self.assertEqual(result["summary"]["recall"], .25)
        self.assertEqual(result["summary"]["users"], 2)
        self.assertEqual(result["per_user"].set_index("user_id").loc["missing", "ndcg"], 0.)
        self.assertIsNone(evaluate_predictions({}, targets.iloc[:0])["summary"]["recall"])
        with self.assertRaises(ValueError):
            evaluate_predictions({"u": ["a", "a"]}, targets)
        with self.assertRaises(ValueError):
            evaluate_predictions({}, targets, k=0)

    def test_real_fitting_round_trip_and_alignment_validation(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_dataset(output, events())
            data = DataLoader(output_dir=output)
            for name, prepare, fit, recommend in (
                ("cf", data.for_collaborative, lambda p: cf.fit(p.items, p.history, factors=4, iterations=3), cf_recommend),
                ("cb", data.for_content_based, lambda p: cb.fit(p.items, max_features=100), cb_recommend),
            ):
                phase = prepare()
                bundle = fit(phase)
                path = output / name / "model.joblib"
                save_model(bundle, path)
                loaded = load_model(path)
                before = recommend.evaluate(bundle, phase, k=3)
                after = recommend.evaluate(loaded, phase, k=3)
                self.assertEqual(before["summary"], after["summary"])
                pd.testing.assert_frame_equal(before["per_user"], after["per_user"])
                self.assertEqual(len(list(path.parent.iterdir())), 1)
                if name == "cf":
                    invalid = {**loaded, "user_ids": []}
                else:
                    self.assertEqual(set(loaded), {"item_matrix", "metadata"})
                    self.assertEqual(loaded["item_matrix"].dtype, np.float32)
                    invalid = {**loaded, "metadata": loaded["metadata"].iloc[:-1]}
                    seed = loaded["metadata"].parent_asin.iloc[0]
                    self.assertNotIn(seed, cb_recommend.similar_items(seed, loaded, n=100).parent_asin.tolist())
                with self.assertRaises(ValueError):
                    validate_model(invalid)
                # Invalid replacement must not corrupt a previously saved bundle.
                with self.assertRaises(ValueError):
                    save_model(invalid, path)
                self.assertEqual(set(load_model(path)), set(loaded))

    def test_merged_notebooks_use_shared_exports_and_configuration(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_dataset(output, events())
            for short in ("cf", "cb"):
                notebook = json.loads(Path(f"notebook/{short}_recommend.ipynb").read_text(encoding="utf-8"))
                existing_models = set((output / "models").rglob("*.joblib"))
                scope = {}
                config_module = __import__(f"src.{'collaborative_filtering' if short == 'cf' else 'content_based_filtering'}.tool_config", fromlist=["tool_config"])
                with patch("IPython.display.display"), patch("builtins.print"), \
                        patch("config.ml_config.DATA_OUTPUT_DIR", output), \
                        patch.object(config_module, "DEFAULT_TOP_N", 3), \
                        patch("data_loader.data_loader.load_data", side_effect=AssertionError("Live data")), \
                        patch("utils.supabase_utils.ARTEFACTS_STORAGE") as storage, \
                        patch("utils.supabase_utils.SUPABASE_CLIENT") as registry:
                    for cell in notebook["cells"]:
                        if cell["cell_type"] != "code":
                            continue
                        source = "".join(cell["source"])
                        exec(source, scope)
                        if "CANDIDATES =" in source:
                            # Baseline fitting defaults come from the algorithm configuration.
                            expected = ({"factors": config_module.ALS_FACTORS, "regularization": config_module.ALS_REG,
                                         "iterations": config_module.ALS_ITERA, "random_state": config_module.ALS_RANDOM_STATE}
                                        if short == "cf" else
                                        {"max_features": config_module.TFIDF_MAX_FEATURES,
                                         "ngram_range": config_module.TFIDF_NGRAM_RANGE,
                                         "sublinear_tf": config_module.TFIDF_SUBLINEAR_TF})
                            self.assertEqual(scope["CANDIDATES"], [expected])
                            scope["CANDIDATES"] = ([{"factors": 4, "iterations": 2}] if short == "cf" else [{"max_features": 100}])
                            # Preserve notebook save controls while isolating all writes.
                            scope["MODEL_ARTEFACT_DIR"] = output / "models" / short
                    self.assertEqual(scope["selected_validation"]["summary"]["users"], 2)
                    self.assertEqual(scope["selected_validation"]["summary"]["k"], 3)
                    test_source = next("".join(c["source"]) for c in notebook["cells"]
                                       if c["cell_type"] == "code" and "final_test = None" in "".join(c["source"]))
                    if "RUN_FINAL_TEST = False" in test_source:
                        self.assertIsNone(scope["final_bundle"])
                        self.assertEqual(set((output / "models").rglob("*.joblib")), existing_models)
                        exec(test_source.replace("RUN_FINAL_TEST = False", "RUN_FINAL_TEST = True"), scope)
                    self.assertEqual(scope["final_test"]["summary"]["users"], 2)
                    self.assertLess(scope["test"].history.review_date.max(), scope["data"].test_df.review_date.min())
                    save_source = next("".join(c["source"]) for c in notebook["cells"]
                                       if c["cell_type"] == "code" and "save_model(final_bundle" in "".join(c["source"]))
                    exec(save_source.replace("SAVE_MODEL = False", "SAVE_MODEL = True"), scope)
                    load_model(scope["model_path"])

    def test_notebook_config_imports_work_from_root_and_notebook_directory(self):
        root = Path(__file__).resolve().parents[1]
        for short in ("cf", "cb"):
            notebook = json.loads((root / f"notebook/{short}_recommend.ipynb").read_text(encoding="utf-8"))
            source = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
            self.assertIn("from config.ml_config import", source)
            for directory in (root, root / "notebook"):
                with self.subTest(algorithm=short, directory=directory):
                    previous = Path.cwd()
                    try:
                        os.chdir(directory)
                        scope = {}
                        exec(source, scope)
                    finally:
                        os.chdir(previous)
                    self.assertEqual(scope["repo_root"], root)
                    self.assertEqual(scope["DATA_OUTPUT_DIR"], root / "data/output")
                    self.assertEqual(scope["MODEL_ARTEFACT_DIR"], root / "models")

    def test_inline_publication_uploads_then_registers_without_helper(self):
        from src.collaborative_filtering import tool_config as cf_config
        from src.content_based_filtering import tool_config as cb_config
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_dataset(root, events())
            data = DataLoader(output_dir=root)
            phase = data.for_collaborative()
            bundles = {"cf": cf.fit(phase.items, phase.history, factors=4, iterations=2),
                       "cb": cb.fit(data.for_content_based().items)}
            for short, settings in (("cf", cf_config), ("cb", cb_config)):
                notebook = json.loads(Path(f"notebook/{short}_recommend.ipynb").read_text(encoding="utf-8"))
                source = next("".join(c["source"]) for c in notebook["cells"]
                              if c["cell_type"] == "code" and "published_model_id =" in "".join(c["source"]))
                self.assertNotIn("publish_model(", source)
                path = root / settings.MODEL_NAME / settings.MODEL_FILENAME
                save_model(bundles[short], path)
                for failure in (False, True):
                    with self.subTest(algorithm=short, upload_failure=failure):
                        calls = []
                        storage, registry = Mock(), Mock()
                        def upload(**kwargs):
                            calls.append("upload")
                            self.assertEqual(kwargs["file"].read(), path.read_bytes())
                            if failure:
                                raise RuntimeError("Upload failed")
                        storage.upload.side_effect = upload
                        registry.table.return_value.insert.return_value.execute.side_effect = lambda: calls.append("register")
                        scope = {"MODEL_ARTEFACT_DIR": root, "tool_config": settings}
                        with patch("utils.supabase_utils.ARTEFACTS_STORAGE", storage), \
                                patch("utils.supabase_utils.SUPABASE_CLIENT", registry), \
                                patch("config.ml_config.MODELREGISTRY_TABLE_NAME", "ConfiguredRegistry"), \
                                patch("builtins.print"):
                            if failure:
                                with self.assertRaisesRegex(RuntimeError, "Upload failed"):
                                    exec(source, scope)
                                registry.table.assert_not_called()
                            else:
                                exec(source, scope)
                                self.assertEqual(calls, ["upload", "register"])
                                self.assertEqual(registry.table.call_args.args, ("ConfiguredRegistry",))
                                record = registry.table.return_value.insert.call_args.args[0]
                                self.assertEqual(record["model_id"], scope["published_model_id"])
                                self.assertEqual(record["tool"], settings.MODEL_NAME)
                                self.assertEqual(record["storage_path"], f'{settings.MODEL_NAME}/{record["model_id"]}')
                                self.assertEqual(storage.upload.call_args.kwargs["path"],
                                                 f'{record["storage_path"]}/{settings.MODEL_FILENAME}')

    def test_missing_bundles_use_api_config_published_ids(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_dataset(root, events())
            data = DataLoader(output_dir=root)
            phase = data.for_collaborative()
            bundles = {"collaborative_filtering": cf.fit(phase.items, phase.history, factors=4, iterations=2)}
            phase = data.for_content_based()
            bundles["content_based_filtering"] = cb.fit(phase.items)
            pinned = {"collaborative_filtering": "configured-cf", "content_based_filtering": "configured-cb"}

            def download(tool, model_id, filename, stop_event):
                self.assertEqual(model_id, pinned[tool])
                save_model(bundles[tool], root / tool / filename)

            with patch("models_loader.models_loader.MODEL_ARTEFACT_DIR", root), \
                    patch("models_loader.models_loader.MODEL_IDS", pinned), \
                    patch("utils.supabase_utils.download_model_from_supabase", side_effect=download) as fetch:
                loaded = ModelsLoader()
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(set(loaded.model_artefacts), set(pinned))

    def test_local_serving_loader_needs_no_supabase(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_dataset(root, events())
            data = DataLoader(output_dir=root)
            phase = data.for_collaborative()
            save_model(cf.fit(phase.items, phase.history, factors=4, iterations=2), root / "collaborative_filtering" / "model.joblib")
            phase = data.for_content_based()
            save_model(cb.fit(phase.items), root / "content_based_filtering" / "model.joblib")
            with patch("models_loader.models_loader.MODEL_ARTEFACT_DIR", root):
                models = ModelsLoader()
            self.assertEqual(set(models.model_artefacts), {"collaborative_filtering", "content_based_filtering"})


if __name__ == "__main__":
    unittest.main()
