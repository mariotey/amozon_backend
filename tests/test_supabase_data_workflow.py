"""Offline coverage for the manual Supabase-to-local-splits workflow."""
import importlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from config.ml_config import ITEM_TABLE_NAME, REVIEW_TABLE_NAME
from data_loader.data_loader import DataLoader, load_data

splitting = importlib.import_module("data_loader.train_test_split")


def reviews():
    return pd.DataFrame([
        {"review_id": str(day), "user_id": "u", "parent_asin": f"p{day}",
         "review_rating": [5., 1., 3., 4., 2.][day % 5],
         "review_date": f"2020-01-{day + 1:02d}"}
        for day in range(20)
    ])


def items(source):
    return pd.DataFrame({"parent_asin": source.parent_asin.unique()}).assign(
        item_title="software", description="useful editor", features="editing",
        main_category="software", is_free=True, num_reviews=999999, avg_rating=5.,
    )


def save_dataset(output, source):
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(("train", "validation", "test"), splitting.train_test_split(source)):
        frame.to_parquet(output / f"{name}.parquet", index=False)
    items(source).to_parquet(output / "items.parquet", index=False)


class ReplacementClient:
    """Small fake replacing complete tables, including stale/null-ID rows."""
    def __init__(self, failure=None):
        self.tables = {"Review": [{"review_id": "old"}, {"review_id": None}],
                       "User": [{"user_id": "old"}], "Item": [{"parent_asin": "old"}],
                       "ModelRegistry": [{"model_id": "unchanged"}]}
        self.calls = []
        self.failure = failure

    def table(self, name):
        return ReplacementQuery(self, name)


class ReplacementQuery:
    def __init__(self, client, name):
        self.client, self.name = client, name
        self.action, self.rows, self.predicate = None, None, None

    def delete(self, *, returning):
        assert returning == "minimal"
        self.action = "delete"
        return self

    def or_(self, predicate):
        self.predicate = predicate
        return self

    def insert(self, rows, *, returning):
        assert returning == "minimal"
        self.action, self.rows = "insert", rows
        return self

    def execute(self):
        self.client.calls.append((self.action, self.name, self.predicate))
        if self.client.failure == (self.action, self.name):
            raise RuntimeError("failed batch")
        if self.action == "delete":
            self.client.tables[self.name] = []
        else:
            self.client.tables[self.name].extend(self.rows)
        return SimpleNamespace(data=[])


class SupabaseSplitWorkflowTests(unittest.TestCase):
    def test_notebook_round_trip_uses_only_saved_data(self):
        source = reviews().assign(custom_feature="preserved")
        notebook = json.loads(Path("notebook/train_test_split_data.ipynb").read_text())
        fake = SimpleNamespace(extract_table_from_supabase=Mock(
            side_effect=lambda table, **kwargs: source.copy() if table == REVIEW_TABLE_NAME else items(source),
        ))
        with TemporaryDirectory() as temporary, patch.dict(sys.modules, {"utils.supabase_utils": fake}):
            namespace = {}
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    text = "".join(cell["source"])
                    # Redirect the notebook output without changing its orchestration.
                    text = text.replace('output_dir = DATA_OUTPUT_DIR',
                                        'output_dir = test_output')
                    namespace["test_output"] = Path(temporary)
                    exec(text, namespace)
            self.assertEqual(fake.extract_table_from_supabase.call_count, 2)
            with patch("data_loader.data_loader.load_data", side_effect=AssertionError("Live fetch")):
                data = DataLoader(output_dir=Path(temporary))
            self.assertEqual([len(data.train_df), len(data.validation_df), len(data.test_df)], [16, 2, 2])
            combined = pd.concat([data.train_df, data.validation_df, data.test_df])
            self.assertEqual(set(combined.review_id), set(source.review_id))
            self.assertTrue(combined.custom_feature.eq("preserved").all())
            self.assertEqual(set(combined.review_rating), {1., 2., 3., 4., 5.})
            self.assertEqual(len(list(Path(temporary).iterdir())), 4)

    def test_export_is_consumed_by_both_notebooks_from_root_and_notebook_directory(self):
        root = Path.cwd().resolve()
        export = json.loads((root / "notebook/train_test_split_data.ipynb").read_text(encoding="utf-8"))
        consumers = [json.loads((root / f"notebook/{short}_recommend.ipynb").read_text(encoding="utf-8"))
                     for short in ("cf", "cb")]
        source = reviews()
        fake = SimpleNamespace(extract_table_from_supabase=Mock(
            side_effect=lambda table, **kwargs: source.copy() if table == REVIEW_TABLE_NAME else items(source),
        ))
        try:
            with TemporaryDirectory() as temporary, patch.dict(sys.modules, {"utils.supabase_utils": fake}), \
                    patch("IPython.display.display"):
                for index, workdir in enumerate((root, root / "notebook")):
                    output = Path(temporary) / str(index)
                    os.chdir(workdir)
                    with patch("config.ml_config.DATA_OUTPUT_DIR", output):
                        namespace = {}
                        for cell in export["cells"]:
                            if cell["cell_type"] == "code":
                                exec("".join(cell["source"]), namespace)
                        for consumer in consumers:
                            namespace = {}
                            for cell in consumer["cells"]:
                                text = "".join(cell["source"])
                                if cell["cell_type"] == "code" and (
                                    "CANDIDATES =" in text or "data = DataLoader(" in text
                                ):
                                    exec(text, namespace)
                            self.assertEqual(len(namespace["data"].train_df), 16)
                            self.assertIn("test", namespace["prepared"])
                    self.assertEqual(len(list(output.glob("*.parquet"))), 4)
        finally:
            os.chdir(root)

    def test_publication_notebook_downloads_transforms_and_replaces_without_local_exports(self):
        downloader = importlib.import_module("data_loader.data_download")
        utility = importlib.import_module("utils.supabase_utils")
        from config.ml_config import DATASET_NAME, DATASET_CATEGORY, DATASET_SPLIT_PERCENTAGE

        raw_reviews = pd.DataFrame({
            "asin": ["a", "b", "a"], "parent_asin": ["a", "b", "a"],
            "user_id": ["u", "u", "v"], "title": ["Good", "Bad", "Useful"],
            "text": ["works", "does not work", "useful app"], "images": [[], [], []],
            "rating": [5., 1., 4.], "timestamp": [1577836800123, 1577923200456, 1578009600789],
            "verified_purchase": [True, True, False], "helpful_vote": [0, 1, 0],
        })
        raw_items = pd.DataFrame([
            {"parent_asin": product, "title": "Software", "main_category": "Software",
             "categories": ["Education"], "videos": {"url": []}, "rating_number": 1,
             "price": price, "description": ["Useful software"], "features": ["Editing"],
             "details": {}, "images": {"large": ["image"]}, "store": "store"}
            for product, price in (("a", 0.), ("b", None))
        ])
        sources = [SimpleNamespace(to_pandas=lambda: raw_reviews), SimpleNamespace(to_pandas=lambda: raw_items)]
        database = ReplacementClient()
        notebook = json.loads(Path("notebook/publish_dataset.ipynb").read_text(encoding="utf-8"))
        with patch.object(downloader, "load_dataset", side_effect=sources) as download, \
                patch.object(utility, "SUPABASE_CLIENT", database), \
                patch("pandas.read_parquet", side_effect=AssertionError("Local read")), \
                patch("pandas.DataFrame.to_parquet", side_effect=AssertionError("Local export")):
            namespace = {}
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec("".join(cell["source"]), namespace)
        self.assertEqual([call.args for call in download.call_args_list],
                         [(DATASET_NAME, f"raw_review_{DATASET_CATEGORY}"),
                          (DATASET_NAME, f"raw_meta_{DATASET_CATEGORY}")])
        self.assertEqual([call.kwargs["split"] for call in download.call_args_list],
                         [f"full[:{DATASET_SPLIT_PERCENTAGE}%]", "full"])
        self.assertEqual(namespace["published_rows"], {"User": 2, "Item": 2, "Review": 3})
        self.assertEqual([event[:2] for event in database.calls[:3]],
                         [("delete", "Review"), ("delete", "User"), ("delete", "Item")])
        self.assertEqual([event[:2] for event in database.calls[3:]],
                         [("insert", "User"), ("insert", "Item"), ("insert", "Review")])
        self.assertEqual(database.tables["Review"][0]["review_date"], "2020-01-01T00:00:00.123000Z")
        self.assertEqual(database.tables["ModelRegistry"], [{"model_id": "unchanged"}])
        json.dumps(database.tables, allow_nan=False)
        for table, key in (("User", "user_id"), ("Item", "parent_asin"), ("Review", "review_id")):
            self.assertTrue(all(row[key] not in (None, "old") for row in database.tables[table]))

    def test_replacement_sanitizes_values_and_reruns_replace_instead_of_appending(self):
        utility = importlib.import_module("utils.supabase_utils")
        database = ReplacementClient()
        review = pd.DataFrame({"review_id": ["1"], "user_id": ["u"], "parent_asin": ["p"],
                               "date": [pd.Timestamp("2020-01-01T00:00:00.123456Z")],
                               "weight": [np.inf]})
        user = pd.DataFrame({"user_id": ["u"], "segment": pd.Categorical(["power"]), "missing": [pd.NA]})
        item = pd.DataFrame({"parent_asin": ["p"], "rating_std": [np.nan]})
        with patch.object(utility, "SUPABASE_CLIENT", database):
            for _ in range(2):
                utility.replace_data_in_supabase(review, user, item, batch_size=1, show_progress=False)
        self.assertEqual([len(database.tables[name]) for name in ("Review", "User", "Item")], [1, 1, 1])
        self.assertIsNone(database.tables["Review"][0]["weight"])
        self.assertIsNone(database.tables["User"][0]["missing"])
        self.assertIsNone(database.tables["Item"][0]["rating_std"])
        self.assertEqual(database.tables["Review"][0]["date"], "2020-01-01T00:00:00.123456Z")
        self.assertEqual(database.calls[0][2], "review_id.is.null,review_id.not.is.null")
        json.dumps(database.tables, allow_nan=False)

    def test_replacement_rejects_bad_data_or_serialization_before_deleting(self):
        utility = importlib.import_module("utils.supabase_utils")
        database = ReplacementClient()
        review = pd.DataFrame({"review_id": ["1"], "user_id": ["u"], "parent_asin": ["p"]})
        user, item = pd.DataFrame({"user_id": ["u"]}), pd.DataFrame({"parent_asin": ["p"]})
        with patch.object(utility, "SUPABASE_CLIENT", database):
            for broken in (review.iloc[:0], pd.concat([review, review]), review.assign(parent_asin="unknown")):
                with self.assertRaises(ValueError):
                    utility.replace_data_in_supabase(broken, user, item, show_progress=False)
            with patch("pandas.DataFrame.to_json", side_effect=TypeError("bad serialization")):
                with self.assertRaisesRegex(TypeError, "serialization"):
                    utility.replace_data_in_supabase(review, user, item, show_progress=False)
        self.assertEqual(database.calls, [])
        self.assertEqual(database.tables["Review"][0]["review_id"], "old")

    def test_replacement_stops_on_failed_batch_without_claiming_success(self):
        utility = importlib.import_module("utils.supabase_utils")
        database = ReplacementClient(failure=("insert", "Item"))
        review = pd.DataFrame({"review_id": ["1"], "user_id": ["u"], "parent_asin": ["p"]})
        user, item = pd.DataFrame({"user_id": ["u"]}), pd.DataFrame({"parent_asin": ["p"]})
        with patch.object(utility, "SUPABASE_CLIENT", database):
            with self.assertRaisesRegex(RuntimeError, "failed batch"):
                utility.replace_data_in_supabase(review, user, item, show_progress=False)
        self.assertNotIn(("insert", "Review", None), database.calls)
        self.assertEqual(database.tables["Review"], [])

    def test_timestamp_ties_stay_together_and_shuffle_does_not_change_split(self):
        source = reviews()
        source.loc[15, "review_date"] = source.loc[16, "review_date"]
        original = source.copy(deep=True)
        first = splitting.train_test_split(source)
        second = splitting.train_test_split(source.sample(frac=1, random_state=7))
        self.assertEqual([len(frame) for frame in first], [15, 3, 2])
        for one, two in zip(first, second):
            pd.testing.assert_frame_equal(one, two)
        pd.testing.assert_frame_equal(source, original)
        with self.assertRaisesRegex(ValueError, "empty split"):
            splitting.train_test_split(source.assign(review_date="2020-01-01"))

    def test_repeated_reviews_resolve_in_past_only_and_statistics_do_not_leak(self):
        source = reviews()
        source.loc[16, ["parent_asin", "review_rating"]] = ["p0", 1.]
        source.loc[17, ["parent_asin", "review_rating"]] = ["p16", 5.]
        source.loc[18, ["parent_asin", "review_rating"]] = ["p17", 5.]
        source["recency_weight"] = 999.
        with TemporaryDirectory() as temporary:
            save_dataset(Path(temporary), source)
            data = DataLoader(output_dir=Path(temporary))
            validation, test = data.for_collaborative(), data.for_collaborative(phase="test")
        self.assertEqual(validation.history.loc[validation.history.parent_asin.eq("p0"), "review_rating"].item(), 5.)
        self.assertEqual(test.history.loc[test.history.parent_asin.eq("p0"), "review_rating"].item(), 1.)
        self.assertEqual(validation.items.loc[validation.items.parent_asin.eq("p0"), "num_reviews"].item(), 1)
        self.assertEqual(test.items.loc[test.items.parent_asin.eq("p0"), "num_reviews"].item(), 2)
        self.assertTrue(validation.history.recency_weight.between(0, 1).all())
        self.assertNotIn("p16", set(validation.items.parent_asin))
        self.assertIn("p16", set(test.items.parent_asin))
        # CF reports cold items as outside its supported evaluation scenario.
        self.assertNotIn("p16", set(validation.targets.parent_asin))
        self.assertNotIn("p17", set(test.targets.parent_asin))
        self.assertEqual(validation.excluded_targets, 1)
        self.assertEqual(test.excluded_targets, 1)

    def test_targets_use_first_new_product_review_not_later_update(self):
        source = reviews()
        source.loc[16, ["parent_asin", "review_rating"]] = ["new", 1.]
        source.loc[17, ["parent_asin", "review_rating"]] = ["new", 5.]
        with TemporaryDirectory() as temporary:
            save_dataset(Path(temporary), source)
            phase = DataLoader(output_dir=Path(temporary)).for_content_based()
        self.assertTrue(phase.targets.empty)

    def test_algorithms_prepare_their_own_catalogues_and_supported_targets(self):
        source = reviews().assign(review_rating=5.)
        source.loc[16, ["parent_asin", "review_rating"]] = ["p1", 5.]
        source.loc[17, ["parent_asin", "review_rating"]] = ["p17", 5.]
        source.loc[18, "user_id"] = "cold_user"
        # u has seen p1; add a second supported user whose heldout p1 is new.
        source.loc[0, "user_id"] = "supported"
        source.loc[16, "user_id"] = "supported"
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_dataset(output, source)
            metadata = pd.read_parquet(output / "items.parquet")
            metadata = pd.concat([metadata, metadata.iloc[[0]].assign(
                parent_asin="metadata_only", item_title="new product",
            )], ignore_index=True)
            metadata.to_parquet(output / "items.parquet", index=False)
            data = DataLoader(output_dir=output)
            cf, cb = data.for_collaborative(), data.for_content_based()
            self.assertEqual(set(cf.targets.parent_asin), {"p1"})
            self.assertEqual(set(cb.targets.parent_asin), {"p1", "p17"})
            self.assertEqual((cf.excluded_targets, cf.excluded_users), (1, 1))
            self.assertEqual((cb.excluded_targets, cb.excluded_users), (0, 0))
            self.assertNotIn("metadata_only", set(cf.items.parent_asin))
            self.assertIn("metadata_only", set(cb.items.parent_asin))
            self.assertNotIn("cold_user", set(data.for_content_based(phase="test").targets.user_id))
            for method in (data.for_collaborative, data.for_content_based):
                train = method(phase="train")
                self.assertTrue(train.targets.empty)
                self.assertEqual((train.excluded_targets, train.excluded_users), (0, 0))
                self.assertTrue(train.history.review_date.lt(data.validation_df.review_date.min()).all())
                with self.assertRaisesRegex(ValueError, "phase"):
                    method(phase="invalid")
            # Neither preparation method mutates the saved event frames.
            self.assertEqual(len(data.train_df), 16)

    def test_content_filters_empty_metadata_and_users_without_positive_profiles(self):
        source = reviews().assign(review_rating=1.)
        source.loc[16, "review_rating"] = 5.
        source.loc[17, "review_rating"] = 5.
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_dataset(output, source)
            metadata = pd.read_parquet(output / "items.parquet")
            metadata.loc[metadata.parent_asin.eq("p16"), ["item_title", "description", "features"]] = ""
            metadata.to_parquet(output / "items.parquet", index=False)
            data = DataLoader(output_dir=output)
            for method in (data.for_collaborative, data.for_content_based):
                phase = method()
                self.assertTrue(phase.targets.empty)
                self.assertEqual((phase.excluded_targets, phase.excluded_users), (2, 1))
                self.assertEqual(set(phase.history.review_rating), {1.})
            self.assertNotIn("p16", set(data.for_content_based().items.parent_asin))

    def test_invalid_events_and_fractions_fail_clearly(self):
        for column, value in (("review_id", "0"), ("review_rating", True),
                              ("review_rating", float("inf")), ("review_rating", None), ("review_date", None),
                              ("review_date", 1234567), ("user_id", " ")):
            source = reviews().astype({column: object})
            source.loc[1, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                splitting.train_test_split(source)
        nullable = reviews().astype({"review_rating": "Float64"})
        nullable.loc[0, "review_rating"] = pd.NA
        with self.assertRaisesRegex(ValueError, "review_rating"):
            splitting.train_test_split(nullable)
        with self.assertRaisesRegex(ValueError, "review_date"):
            splitting.train_test_split(reviews().drop(columns="review_date"))
        for kwargs in ({"train_fraction": 0}, {"train_fraction": True},
                       {"train_fraction": float("nan")}, {"train_fraction": .95}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                splitting.train_test_split(reviews(), **kwargs)

    def test_loader_rejects_overlapping_or_reordered_splits_and_missing_metadata(self):
        for broken in ("overlap", "chronology", "metadata"):
            with self.subTest(broken=broken), TemporaryDirectory() as temporary:
                output = Path(temporary)
                save_dataset(output, reviews())
                if broken == "overlap":
                    frame = pd.read_parquet(output / "validation.parquet")
                    frame.loc[0, "review_id"] = "0"
                    frame.to_parquet(output / "validation.parquet", index=False)
                elif broken == "chronology":
                    frame = pd.read_parquet(output / "validation.parquet")
                    frame.loc[0, "review_date"] = pd.Timestamp("2019-01-01", tz="UTC")
                    frame.to_parquet(output / "validation.parquet", index=False)
                else:
                    frame = pd.read_parquet(output / "items.parquet").iloc[1:]
                    frame.to_parquet(output / "items.parquet", index=False)
                with self.assertRaises(ValueError):
                    DataLoader(output_dir=output)

    def test_load_data_never_reads_or_writes_local_source_files(self):
        expected = reviews()
        extract = Mock(return_value=expected)
        fake = SimpleNamespace(extract_table_from_supabase=extract)
        with patch.dict(sys.modules, {"utils.supabase_utils": fake}), \
                patch("pandas.read_parquet") as read, patch("pandas.DataFrame.to_parquet") as write:
            self.assertIs(load_data(REVIEW_TABLE_NAME, ["review_id"]), expected)
        extract.assert_called_once_with(REVIEW_TABLE_NAME, columns=["review_id"], order_by="review_id")
        read.assert_not_called()
        write.assert_not_called()

    def test_supabase_errors_propagate_without_local_fallback(self):
        fake = SimpleNamespace(extract_table_from_supabase=Mock(side_effect=RuntimeError("offline")))
        with patch.dict(sys.modules, {"utils.supabase_utils": fake}), \
                patch("pandas.read_parquet") as read:
            with self.assertRaisesRegex(RuntimeError, "offline"):
                load_data(ITEM_TABLE_NAME)
        read.assert_not_called()

    def test_supabase_pagination_handles_server_capped_pages(self):
        # Load the utility with a fake client; no credentials or network are used.
        spec = importlib.util.spec_from_file_location(
            "_offline_supabase_utils", Path("utils/supabase_utils.py"),
        )
        utility = importlib.util.module_from_spec(spec)
        query = Mock()
        query.select.return_value = query
        query.order.return_value = query
        query.range.return_value = query
        query.execute.side_effect = [
            SimpleNamespace(data=[{"review_id": "a"}, {"review_id": "b"}]),
            SimpleNamespace(data=[{"review_id": "c"}]),
            SimpleNamespace(data=[]),
        ]
        client = Mock()
        client.table.return_value = query
        with patch("supabase.create_client", return_value=client):
            spec.loader.exec_module(utility)
        frame = utility.extract_table_from_supabase(
            REVIEW_TABLE_NAME, batch_size=1000, show_progress=False,
            columns=["review_id"], order_by="review_id",
        )
        self.assertEqual(frame.review_id.tolist(), ["a", "b", "c"])
        self.assertEqual([call.args for call in query.range.call_args_list],
                         [(0, 999), (2, 1001), (3, 1002)])
        self.assertEqual([call.args for call in query.order.call_args_list],
                         [("review_id",)] * 3)

    def test_missing_split_requires_explicit_preparation(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "train_test_split_data.ipynb"):
                DataLoader(output_dir=Path(temporary))


class FirstRunSetupTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "_first_run_setup", Path(__file__).resolve().parents[1] / "setup.py",
        )
        self.setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.setup)

    def test_skip_install_creates_folders_and_preserves_existing_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.multiple(self.setup, REPO_ROOT=root, DATA_OUTPUT_DIR=root / "data/output",
                                MODEL_ARTEFACT_DIR=root / "models"), \
                    patch.object(self.setup.subprocess, "run") as run, patch("builtins.print"):
                self.assertEqual(self.setup.main(["--skip-install"]), 0)
                self.assertTrue((root / "data/output").is_dir())
                for tool in self.setup.TOOLS:
                    self.assertTrue((root / "models" / tool).is_dir())
                self.assertIn("SUPBASE_SECRET_KEY=", (root / ".env").read_text())
                (root / ".env").write_text("existing credentials\n")
                model = root / "models" / self.setup.TOOLS[0] / "model.joblib"
                model.write_bytes(b"existing model")
                self.assertEqual(self.setup.main(["--skip-install"]), 0)
                self.assertEqual((root / ".env").read_text(), "existing credentials\n")
                self.assertEqual(model.read_bytes(), b"existing model")
                run.assert_not_called()

    def test_install_uses_locked_environment_and_environment_local_kernel(self):
        with patch.object(self.setup.shutil, "which", return_value="uv-executable"), \
                patch.object(self.setup.subprocess, "run") as run:
            self.setup.install_dependencies()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0],
                         ["uv-executable", "sync", "--frozen", "--no-install-project"])
        kernel = run.call_args_list[1].args[0]
        self.assertEqual(kernel[:7],
                         ["uv-executable", "run", "--no-sync", "python", "-m", "ipykernel", "install"])
        self.assertIn("--sys-prefix", kernel)
        self.assertNotIn("--user", kernel)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs, {"cwd": self.setup.REPO_ROOT, "check": True})

    def test_missing_uv_is_installed_and_failure_stops_setup(self):
        with patch.object(self.setup.shutil, "which", return_value=None), \
                patch.object(self.setup.subprocess, "run") as run:
            self.setup.install_dependencies()
            self.assertEqual(run.call_args_list[0].args[0],
                             [sys.executable, "-m", "pip", "install", "uv"])
            self.assertEqual(run.call_args_list[1].args[0][:3],
                             [sys.executable, "-m", "uv"])
            run.reset_mock()
            run.side_effect = self.setup.subprocess.CalledProcessError(1, "pip")
            with patch.object(self.setup, "prepare_local_files"), patch("builtins.print"):
                self.assertEqual(self.setup.main([]), 1)
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
