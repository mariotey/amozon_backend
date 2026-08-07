"""
Shared, lazily-cached access to the datasets and artefacts the orchestration tools need.

Rather than build its own loaders, this module reuses the lazy-cached singletons already exposed
by the recommender tools' handler modules, so a warm process holds one copy of the interaction
table and one copy of the ALS artefacts regardless of how many orchestration tools run.
"""
import logging
from typing import Any
import pandas as pd
from data_loader import DataLoader
from src.collaborative_filtering import tool_config as cf_config
from src.collaborative_filtering.main import (
    get_data_loader as _cf_get_data_loader,
    get_models_loader as _cf_get_models_loader,
)

logger = logging.getLogger(__name__)

# Position of the index-to-user-id mapping within the collaborative-filtering artefact tuple.
# Artefacts are returned in `MODEL_ARTEFACTS` insertion order (`models_loader.read_local_artefacts`),
# so derive the offset from the config rather than hard-coding it.
_UID_MAPPING_POSITION: int = list(cf_config.MODEL_ARTEFACTS).index("uid_mapping_filename")

def get_data_loader() -> DataLoader:
    """
    Return the shared data loader holding the user, item and interaction tables.

    Returns:
    - DataLoader: Cached data loader instance.
    """
    return _cf_get_data_loader()

def get_user_item_df() -> pd.DataFrame:
    """
    Return the user-item interaction table.

    Returns:
    - pd.DataFrame: One row per review, including "user_id", "parent_asin" and "review_rating".
    """
    return get_data_loader().user_item_df

def get_item_df() -> pd.DataFrame:
    """
    Return the item metadata table.

    Returns:
    - pd.DataFrame: One row per product, including "main_category", "is_free", "quality_score",
                    "num_reviews" and "popularity_segment".
    """
    return get_data_loader().item_df

def get_known_cf_user_ids() -> set[str]:
    """
    Return the set of user identifiers present in the trained ALS matrix.

    Collaborative filtering raises `ValueError` for users outside this set, so membership is the
    precondition the analysis step checks before offering the collaborative engine.

    Returns:
    - set[str]: User identifiers the ALS model was trained on.

    Raises:
    - FileNotFoundError: If the collaborative-filtering artefacts have not been built.
    """
    artefacts: tuple[Any, ...] = _cf_get_models_loader().model_artefacts[cf_config.MODEL_NAME]
    idx_to_userid: dict[int, str] = artefacts[_UID_MAPPING_POSITION]

    return set(idx_to_userid.values())
