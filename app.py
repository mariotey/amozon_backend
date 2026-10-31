"""FastAPI storefront: named model bundles and on-demand Supabase queries.

Run uvicorn app:app. Model loading happens once in the background; browsing stays
available while it loads. Recommendation code receives explicit bundles and
request-local history and never loads notebook data or runs training.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from threading import Event, Lock
from time import perf_counter
from typing import Any, cast

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models_loader import ModelsLoader, validate_inference_models
from src import routing
from utils.supabase_utils import (
    CARD_FIELDS,
    DETAIL_FIELDS,
    DependencyUnavailable,
    ServingData,
    timed,
)

_inference_lock = Lock()
serving_data = ServingData()
logger = logging.getLogger(__name__)


class ModelResources:
    """Own one model-loading task and synchronize readiness with request threads."""

    def __init__(self, loader):
        """
        Initialize the model loader, cancellation event, and synchronized state.

        Args:
        - loader(Callable[[Event], ModelsLoader]): Blocking loading callback receiving the shutdown
          event.

        Returns:
        - None: Initializes the instance state.
        """
        self.loader = loader
        self.stop = Event()
        self.lock = Lock()
        self.state = "queued"
        self.value = None
        self.task: asyncio.Task[None] | None = None

    def start(self):
        """
        Schedule model loading once on the running event loop.

        Args:
        - None: No explicit arguments.

        Returns:
        - None: Completes the operation without returning a value.
        """
        if self.task is not None:
            raise RuntimeError("Models already scheduled")
        self.task = asyncio.create_task(self._load(), name="load-inference-models")

    def _blocking_load(self):
        """
        Load models in a worker thread unless shutdown has started.

        Args:
        - None: No explicit arguments.

        Returns:
        - ModelsLoader: Loaded and validated inference models.
        """
        if self.stop.is_set():
            raise InterruptedError("Model initialization stopped")
        with self.lock:
            self.state = "loading"
        return self.loader(self.stop)

    async def _load(self):
        """
        Load models asynchronously and record readiness, cancellation, or failure.

        Args:
        - None: No explicit arguments.

        Returns:
        - None: Completes the operation without returning a value.
        """
        started = perf_counter()
        try:
            result = await asyncio.to_thread(self._blocking_load)
            with self.lock:
                if not self.stop.is_set():
                    self.value, self.state = result, "ready"
                else:
                    self.state = "stopped"
            logger.info("Model readiness %s after %.3fs", self.state, perf_counter() - started)
        except Exception as exc:
            with self.lock:
                self.state = "stopped" if self.stop.is_set() else "error"
            logger.error(
                "Model initialization %s after %.3fs (%s)",
                self.state,
                perf_counter() - started,
                type(exc).__name__,
            )

    def require(self):
        """
        Return loaded models or raise HTTP 503 when resources are unavailable.

        Args:
        - None: No explicit arguments.

        Returns:
        - ModelsLoader: Ready models; unavailable resources raise HTTP 503.
        """
        with self.lock:
            if self.stop.is_set():
                raise HTTPException(503, detail="Service is stopping")
            if self.state != "ready":
                failed = self.state == "error"
                raise HTTPException(
                    503,
                    detail={
                        "message": (
                            "Required resources failed to load"
                            if failed
                            else "Required resources are still loading"
                        ),
                        "resources": {"models": self.state},
                    },
                    headers=None if failed else {"Retry-After": "5"},
                )
            return self.value

    def report(self):
        """
        Read synchronized model readiness and build the public status report.

        Args:
        - None: No explicit arguments.

        Returns:
        - tuple[bool, dict]: Readiness flag and public resource status payload.
        """
        with self.lock:
            state = "stopped" if self.stop.is_set() else self.state
            status = "ready" if state == "ready" else "degraded" if state == "error" else "loading"
            if self.stop.is_set():
                status = "stopping"
            return state == "ready", {
                "status": status,
                "resources": {
                    "models": state,
                    "serving_data": "on_demand",
                },
            }

    async def close(self):
        """
        Signal shutdown, wait for model loading, and clear loaded resources.

        Args:
        - None: No explicit arguments.

        Returns:
        - None: Completes the operation without returning a value.
        """
        self.stop.set()
        if self.task is not None:
            # Do not abandon workers that might still be writing model cache files.
            await asyncio.gather(self.task, return_exceptions=True)
        with self.lock:
            self.value, self.state = None, "stopped"


def create_model_resources():
    """
    Create the background model loader and its readiness state.

    Args:
    - None: No explicit arguments.

    Returns:
    - ModelResources: New resource owner with a deferred model-loading callback.
    """
    return ModelResources(lambda stop: validate_inference_models(ModelsLoader(stop_event=stop)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Schedule models without extracting User, Item, or Review tables.

    Args:
    - app(FastAPI): Application whose model lifecycle is managed.

    Returns:
    - AsyncIterator[None]: Startup/shutdown context that waits for the model worker during cleanup.
    """
    resources = create_model_resources()
    app.state.models = resources
    resources.start()
    try:
        yield
    finally:
        try:
            await resources.close()
        finally:
            app.state.models = None


app = FastAPI(title="Amozon Recommendation API", lifespan=lifespan)


# Existing public, read-only storefront endpoints; browsers send no credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Accept"],
    expose_headers=["Retry-After"],
)


@app.exception_handler(DependencyUnavailable)
async def upstream_unavailable(request, exc):
    """
    Translate an upstream dependency failure into a retryable HTTP 503 response.

    Args:
    - request(Request): Request associated with the upstream dependency failure.
    - exc(DependencyUnavailable): Upstream failure translated into HTTP 503.

    Returns:
    - JSONResponse: HTTP 503 response with a five-second Retry-After header.
    """
    return JSONResponse(
        {"detail": "Supabase request failed"}, status_code=503, headers={"Retry-After": "5"}
    )


def _model_resources():
    """
    Return application model resources or raise HTTP 503 before startup.

    Args:
    - None: No explicit arguments.

    Returns:
    - ModelResources: Resources attached to the running application.
    """
    resources = getattr(app.state, "models", None)
    if resources is None:
        raise HTTPException(503, detail="Service is not running")
    return resources


def _validate_identifier(identifier: str) -> str:
    """
    Strip surrounding whitespace or raise HTTP 422 for a blank identifier.

    Args:
    - identifier(str): Identifier to trim; a blank value raises HTTP 422.

    Returns:
    - str: Nonempty identifier with surrounding whitespace removed.
    """
    if not identifier.strip():
        raise HTTPException(status_code=422, detail="ID must not be blank")
    return identifier.strip()


def _decode_item_images(value: Any) -> dict[str, Any] | None:
    """
    Accept a stored JSON string or object, preserving the source image dictionary.

    Args:
    - value(Any): Scalar or collection to normalize or inspect.

    Returns:
    - dict[str, Any] | None: Image dictionary, or None for unsupported or invalid JSON values.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _serialize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Serialize values and decode images on products and nested history products.

    Args:
    - items(list[dict[str, Any]]): Product or review records to serialize.

    Returns:
    - list[dict[str, Any]]: JSON-compatible records with product and nested product images decoded.
    """
    records = json.loads(cast(str, pd.DataFrame(items).to_json(orient="records")))
    for row in records:
        for product in (row, row.get("item")):
            if isinstance(product, dict) and "item_images" in product:
                product["item_images"] = _decode_item_images(product["item_images"])
    return records


def _product(parent_asin: str) -> dict:
    """
    Look up one product, including direct links to unsampled products.

    Args:
    - parent_asin(str): Product identifier to look up or use as a similarity seed.

    Returns:
    - dict: Product metadata; an unknown product raises HTTP 404.
    """
    product = serving_data.product(parent_asin)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


def _card_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Enrich ranked results with card metadata while preserving engine fields.

    Args:
    - items(list[dict[str, Any]]): Ranked recommendation records to enrich with card metadata.

    Returns:
    - list[dict[str, Any]]: Serialized cards in engine order, retaining recommendation fields.
    """
    catalogue = {
        row["parent_asin"]: row
        for row in serving_data.cards([item["parent_asin"] for item in items])
    }
    enriched = []
    for item in items:
        metadata = catalogue.get(item["parent_asin"], {})
        merged = {**item, **metadata}
        # Preserve engine-specific fields such as recommendation scores.
        enriched.append({**merged, **{key: merged.get(key) for key in CARD_FIELDS}})
    return _serialize_items(enriched)


@app.get("/health")
def health() -> dict[str, str]:
    """
    Process liveness; does not wait for models or probe the database.

    Args:
    - None: No explicit arguments.

    Returns:
    - dict[str, str]: Process liveness status after confirming application resources exist.
    """
    _model_resources()
    return {"status": "ok"}


@app.get("/ready")
def readiness():
    """
    Model readiness; serving data is fetched on demand and is not bulk-loaded.

    Args:
    - None: No explicit arguments.

    Returns:
    - JSONResponse: Model status report with HTTP 200 when ready and HTTP 503 otherwise.
    """
    ready, report = _model_resources().report()
    return JSONResponse(report, status_code=200 if ready else 503)


@app.get("/api/v1/users/{user_id}/recommendations")
def user_recommendations(
    user_id: str,
    n: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    """
    Try collaborative, content-based, then unseen popular products.

    Args:
    - user_id(str): Shopper identifier used to select history or recommendations.
    - n(int): Maximum number of results to return.

    Returns:
    - dict[str, Any]: User ID, selected engine, fallback reason, and ranked unseen product cards.
    """
    user_id = _validate_identifier(user_id)
    models = _model_resources().require()
    with timed("Recommendation generation"):
        result = routing.get_user_recommendations(
            user_id,
            n=n,
            model_artefacts=models.model_artefacts,
            history_loader=serving_data.history,
            item_loader=serving_data.cards,
            popular_loader=serving_data.popular_items,
            inference_lock=_inference_lock,
            user_index=models.user_index,
            content_indexes=models.content_indexes,
        )
        if result["items"] is not None:
            # CF already queried its ranked IDs in a batch, including card fields.
            result["items"] = (
                _card_items(result["items"])
                if result["engine_used"] == "content_based"
                else _serialize_items(result["items"])
            )
        return result


@app.get("/api/v1/items/{parent_asin}")
def item_details(parent_asin: str) -> dict[str, Any]:
    """
    Query one product's card/detail metadata, or return 404 for an unknown ID.

    Args:
    - parent_asin(str): Product identifier to look up or use as a similarity seed.

    Returns:
    - dict[str, Any]: Product card and detail fields with decoded images.
    """
    product = _product(_validate_identifier(parent_asin))
    return _serialize_items([{key: product.get(key) for key in DETAIL_FIELDS}])[0]


@app.get("/api/v1/items/{parent_asin}/reviews")
def item_reviews(
    parent_asin: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    """
    Fetch one review page from Supabase, newest first with stable review-ID ties.

    Args:
    - parent_asin(str): Product identifier to look up or use as a similarity seed.
    - page(int): One-based page number.
    - page_size(int): Maximum number of reviews included in one page.

    Returns:
    - dict[str, Any]: Serialized review page with total and page counts.
    """
    parent_asin = _validate_identifier(parent_asin)
    if not serving_data.product_exists(parent_asin):
        raise HTTPException(status_code=404, detail="Product not found")
    response = serving_data.reviews(parent_asin, page, page_size)
    return {**response, "items": _serialize_items(response["items"])}


@app.get("/api/v1/users/{user_id}/history")
def user_history(
    user_id: str,
    verified_only: bool = Query(True, description="Only reviews marked as verified purchases."),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """
    Reviewed products, newest first, with product card metadata under `item`.

    Defaults to verified purchase reviews. Dates are review dates, not purchase dates, and
    unreviewed purchases are unavailable. Unknown users and pages beyond the end return empty items.
    Missing products have `item: null`. This endpoint works independently of recommendation model
    readiness.

    Args:
    - user_id(str): Shopper identifier used to select history or recommendations.
    - verified_only(bool): Whether to include only verified purchase reviews.
    - page(int): One-based page number.
    - page_size(int): Maximum number of reviews included in one page.

    Returns:
    - dict[str, Any]: Review page with nested cards, verification setting, and pagination counts.
    """
    response = serving_data.user_history(
        _validate_identifier(user_id),
        page,
        page_size,
        verified_only=verified_only,
    )
    return {**response, "items": _serialize_items(response["items"])}


@app.get("/api/v1/users/{user_id}")
def user_details(user_id: str) -> dict[str, Any]:
    """
    Fetch a shopper's prepared profile from Supabase without loading all users.

    Args:
    - user_id(str): Shopper identifier used to select history or recommendations.

    Returns:
    - dict[str, Any]: Shopper profile; an unknown shopper raises HTTP 404.
    """
    user_id = _validate_identifier(user_id)
    response = serving_data.profile(user_id)
    if response is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _serialize_items([response])[0]
