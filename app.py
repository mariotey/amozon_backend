"""Serve the existing recommendation tools through FastAPI."""
from contextlib import asynccontextmanager
from threading import Lock
import json

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from data_loader import DataLoader
from models_loader import ModelsLoader
from src.collaborative_filtering import main as collaborative
from src.content_based_filtering import main as content


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Share small inference datasets and models between both tools.
    data = DataLoader(inference_only=True)
    models = ModelsLoader()
    for tool in (collaborative, content):
        tool._data_loader_obj = data
        tool._models_loader_obj = models
    yield


app = FastAPI(title="Amozon Recommendation API", lifespan=lifespan)
inference_lock = Lock()


def recommend(handler, identifier: str, n: int):
    if not identifier.strip():
        raise HTTPException(status_code=422, detail="ID must not be blank")
    # One calculation at a time keeps memory use down on Render Free.
    with inference_lock:
        try:
            records = handler(identifier, n=n)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        # Convert missing metadata to JSON null.
        return json.loads(pd.DataFrame(records).to_json(orient="records"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/collaborative/users/{user_id}/recommendations")
def collaborative_recommendations(user_id: str, n: int = Query(10, ge=1, le=100)):
    return recommend(collaborative.get_user_recommendations, user_id, n)


@app.get("/api/v1/content/users/{user_id}/recommendations")
def content_recommendations(user_id: str, n: int = Query(10, ge=1, le=100)):
    return recommend(content.get_user_recommendations, user_id, n)


@app.get("/api/v1/content/items/{parent_asin}/similar")
def similar_items(parent_asin: str, n: int = Query(10, ge=1, le=100)):
    return recommend(content.get_similar_items, parent_asin, n)
