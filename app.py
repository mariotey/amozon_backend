"""
FastAPI entry points for the Amozon recommendation tools.

The default user endpoint delegates deterministic CF-first selection to src.routing.
The API exposes user recommendations and a deployment health check. Both reuse
startup resources, and inference is serialized to limit temporary memory usage.

Run locally:
    uv run uvicorn app:app --reload
"""
from contextlib import asynccontextmanager
import json
from threading import Lock
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from data_loader import DataLoader
from models_loader import ModelsLoader
from src import routing
from src.collaborative_filtering import main as collaborative
from src.content_based_filtering import main as content

_inference_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load shared inference resources before the server accepts requests.

    Both tools receive the same datasets and model loader. Resources remain fixed
    for the process lifetime; restarting reloads them using the existing cache policy.
    Startup fails if required Supabase resources or local artefacts cannot be loaded.
    """
    data = DataLoader(inference_only=True)
    models = ModelsLoader()
    for tool in (collaborative, content):
        tool._data_loader_obj = data
        tool._models_loader_obj = models
    try:
        yield
    finally:
        for tool in (collaborative, content):
            tool._data_loader_obj = None
            tool._models_loader_obj = None


app = FastAPI(title="Amozon Recommendation API", lifespan=lifespan)


def _validate_identifier(identifier: str) -> str:
    """Strip surrounding whitespace or raise HTTP 422 for a blank identifier."""
    if not identifier.strip():
        raise HTTPException(status_code=422, detail="ID must not be blank")
    return identifier.strip()


def _serialize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert pandas/NumPy scalars and missing metadata into JSON-compatible values."""
    return json.loads(pd.DataFrame(items).to_json(orient="records"))



@app.get("/health")
def health() -> dict[str, str]:
    """Report process health after startup resource loading succeeds."""
    return {"status": "ok"}


@app.get("/api/v1/users/{user_id}/recommendations")
def user_recommendations(
    user_id: str,
    n: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    """
    Recommend for any user using collaborative, content-based, then popular fallback.

    Args:
    - user_id (str): User identifier, including users unknown to the trained models.
    - n (int): Maximum number of recommendations, from 1 to 100 (default 10).

    Returns:
    - dict[str, Any]: user_id, engine_used, reason, and items. Empty catalogue
                      results remain HTTP 200. Unexpected inference errors remain
                      server errors and do not silently trigger fallback.
    """
    user_id = _validate_identifier(user_id)
    with _inference_lock:
        result = routing.get_user_recommendations(user_id, n=n)
        result["items"] = _serialize_items(result["items"])
        return result
