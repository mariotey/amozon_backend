"""Save/load one named bundle per algorithm and validate serving alignment."""

import importlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event

import joblib
import numpy as np
from scipy.sparse import isspmatrix_csr

from config.api_config import MODEL_ARTEFACT_DIR, MODEL_IDS, TOOLS


def _require(condition, message):
    """
    Raise ValueError with the supplied message when a condition is false.

    Args:
    - condition(bool): Condition that must be truthy for validation to succeed.
    - message(str): Error message raised when the condition is false.

    Returns:
    - None: Completes the operation without returning a value.
    """
    if not condition:
        raise ValueError(message)


def validate_model(bundle: dict, algorithm: str | None = None) -> dict:
    """
    Reject incomplete or misaligned bundles before saving or serving.

    Args:
    - bundle(dict): Named serving model bundle to validate or save.
    - algorithm(str | None): Expected algorithm; None infers the kind from bundle fields.

    Returns:
    - dict: Original bundle after schema, numeric, and row-alignment checks.
    """
    _require(isinstance(bundle, dict), "Model bundle must be a dictionary")
    if "model" in bundle:
        _require(algorithm in (None, "collaborative_filtering"), "Wrong algorithm bundle")
        _require(
            {"model", "interactions", "user_ids", "item_ids"} <= bundle.keys(),
            "Incomplete CF bundle",
        )
        model, matrix = bundle["model"], bundle["interactions"]
        _require(isspmatrix_csr(matrix) and min(matrix.shape) > 0, "Invalid CF interaction matrix")
        _require(np.isfinite(matrix.data).all(), "Invalid CF weights")
        for ids, count in (
            (bundle["user_ids"], matrix.shape[0]),
            (bundle["item_ids"], matrix.shape[1]),
        ):
            _require(isinstance(ids, list) and len(ids) == count, "CF IDs/matrix mismatch")
            _require(
                all(isinstance(value, str) and bool(value.strip()) for value in ids)
                and len(set(ids)) == count,
                "Invalid or duplicate CF IDs",
            )
        users, items = model.user_factors, model.item_factors
        _require(
            users.ndim == items.ndim == 2
            and users.shape[0] == matrix.shape[0]
            and items.shape[0] == matrix.shape[1]
            and users.shape[1] == items.shape[1] > 0,
            "ALS factors/matrix mismatch",
        )
        _require(np.isfinite(users).all() and np.isfinite(items).all(), "Invalid ALS factors")
        _require(callable(getattr(model, "recommend", None)), "Invalid ALS model")
    else:
        _require(algorithm in (None, "content_based_filtering"), "Wrong algorithm bundle")
        _require({"item_matrix", "metadata"} <= bundle.keys(), "Incomplete content bundle")
        matrix, meta = bundle["item_matrix"], bundle["metadata"]
        _require(
            {"parent_asin", "item_title", "main_category", "is_free"} <= set(meta.columns),
            "Invalid content metadata schema",
        )
        _require(
            not meta.empty
            and meta.parent_asin.is_unique
            and meta.parent_asin.map(
                lambda value: isinstance(value, str) and bool(value.strip())
            ).all(),
            "Invalid content product IDs",
        )
        _require(
            meta.is_free.map(lambda value: isinstance(value, (bool, np.bool_))).all(),
            "Invalid content free flags",
        )
        _require(
            isspmatrix_csr(matrix) and matrix.shape[0] == len(meta) and matrix.shape[1] > 0,
            "Content matrix/metadata mismatch",
        )
        _require(
            np.isfinite(matrix.data).all() and (matrix.data >= 0).all(), "Invalid content weights"
        )
    return bundle


def save_model(bundle: dict, path: str | Path) -> None:
    """
    Validate and atomically replace one local bundle; never upload it.

    Args:
    - bundle(dict): Named serving model bundle to validate or save.
    - path(str | Path): Local model bundle path.

    Returns:
    - None: Completes the operation without returning a value.
    """
    validate_model(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(path: str | Path) -> dict:
    """
    Load a trusted local bundle and validate its internal alignment.

    Args:
    - path(str | Path): Local model bundle path.

    Returns:
    - dict: Validated serving bundle loaded from the trusted local file.
    """
    return validate_model(joblib.load(Path(path)))


class ModelsLoader:
    """Load configured bundles locally, downloading missing bundles on demand."""

    def __init__(self, tools: list[str] | None = None, stop_event: Event | None = None) -> None:
        """
        Load selected bundles, downloading missing files and respecting cancellation.

        Args:
        - tools(list[str] | None): Algorithms to load; None loads all configured tools.
        - stop_event(Event | None): Optional cancellation signal checked during model loading.

        Returns:
        - None: Initializes the instance state.
        """
        self.model_artefacts = {}
        for tool in TOOLS if tools is None else tools:
            if tool not in TOOLS:
                raise ValueError(f"Unknown recommendation tool: {tool}")
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Model loading stopped")
            settings = importlib.import_module(f"src.{tool}.tool_config")
            path = MODEL_ARTEFACT_DIR / tool / settings.MODEL_FILENAME
            if not path.is_file():
                from utils.supabase_utils import download_model_from_supabase

                download_model_from_supabase(
                    tool, MODEL_IDS[tool], settings.MODEL_FILENAME, stop_event
                )
            bundle = load_model(path)
            self.model_artefacts[tool] = validate_model(bundle, tool)
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("Model loading stopped")


def validate_inference_models(loader):
    """
    Validate the two serving bundles and build lookup maps once per load.

    Args:
    - loader(ModelsLoader): Loaded bundles to validate and index.

    Returns:
    - ModelsLoader: Original loader with validated bundles and cached index maps.
    """
    cf = validate_model(
        loader.model_artefacts["collaborative_filtering"], "collaborative_filtering"
    )
    cb = validate_model(
        loader.model_artefacts["content_based_filtering"], "content_based_filtering"
    )
    loader.user_index = {value: i for i, value in enumerate(cf["user_ids"])}
    products = cb["metadata"].parent_asin.tolist()
    loader.content_indexes = (
        {value: i for i, value in enumerate(products)},
        dict(enumerate(products)),
    )
    return loader
