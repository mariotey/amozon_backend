"""
Utility helper functions for interfacing with Supabase functionalities.

Import creates the shared Supabase client/storage handle. Table helpers support
manual data preparation and targeted API queries. Bundle downloads validate
before replacing local files. Database exceptions propagate.
"""

from contextlib import contextmanager
from typing import Any, cast
import logging
import json
import os
from time import perf_counter
import pandas as pd
from threading import Event
from postgrest.types import CountMethod, ReturnMethod
from supabase import create_client
from dotenv import load_dotenv
from tqdm.auto import tqdm
from config.api_config import (
    ITEM_TABLE_NAME,
    REVIEW_TABLE_NAME,
    USER_TABLE_NAME,
    MODELREGISTRY_TABLE_NAME,
    MODEL_ARTEFACT_BUCKET,
    MODEL_ARTEFACT_DIR,
)

# Load environment variables
load_dotenv()

SUPABASE_CLIENT = create_client(
    cast(str, os.getenv("SUPABASE_URL")), cast(str, os.getenv("SUPBASE_SECRET_KEY"))
)

ARTEFACTS_STORAGE = SUPABASE_CLIENT.storage.from_(MODEL_ARTEFACT_BUCKET)


def replace_data_in_supabase(
    reviews: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    batch_size: int = 800,
    show_progress: bool = True,
) -> dict[str, int]:
    """
    Replace all rows in Review/User/Item while preserving their schemas.

    Validate and JSON-serialize all incoming batches before deleting anything. Delete child reviews
    first; insert parent users/items before new reviews. Requests are not one transaction: failures
    propagate and a complete rerun is required after correcting them. ModelRegistry and Storage are
    untouched.

    Args:
    - reviews(pd.DataFrame): Review events to validate, split, or publish.
    - users(pd.DataFrame): Prepared shopper profiles referenced by the published reviews.
    - items(pd.DataFrame): Product metadata for the supplied workflow.
    - batch_size(int): Maximum rows sent or fetched per Supabase request.
    - show_progress(bool): Whether to display a progress bar.

    Returns:
    - dict[str, int]: Inserted counts keyed by configured User, Item, and Review table names.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    tables = (
        (USER_TABLE_NAME, users, "user_id"),
        (ITEM_TABLE_NAME, items, "parent_asin"),
        (REVIEW_TABLE_NAME, reviews, "review_id"),
    )
    for name, frame, key in tables:
        if frame.empty or not frame.columns.is_unique or key not in frame:
            raise ValueError(f"{name} must be nonempty with unique columns and {key}")
        if not frame[key].is_unique or not cast(
            bool,
            frame[key]
            .map(
                lambda value: isinstance(value, str)
                and bool(value.strip())
                and value == value.strip()
            )
            .all(),
        ):
            raise ValueError(f"{name}.{key} must contain unique nonempty trimmed string IDs")
    if not {"user_id", "parent_asin"} <= set(reviews.columns):
        raise ValueError("Review requires user_id and parent_asin")
    if (
        not reviews.user_id.isin(users.user_id).all()
        or not reviews.parent_asin.isin(items.parent_asin).all()
    ):
        raise ValueError("Every review must have a matching User and Item")

    def records(frame):
        """
        Convert a DataFrame batch into JSON records with nulls and UTC date strings.

        Args:
        - frame(pd.DataFrame): Input table to normalize, validate, or serialize.

        Returns:
        - list[dict]: JSON-compatible records with missing and nonfinite values as null.
        """
        # Retain microsecond precision when serializing nullable and NumPy values.
        return json.loads(frame.to_json(orient="records", date_format="iso", date_unit="us"))

    for _, frame, _ in tables:
        for start in range(0, len(frame), batch_size):
            records(frame.iloc[start : start + batch_size])
    for name, _, key in (tables[2], tables[0], tables[1]):
        # The explicit predicate covers even legacy rows whose IDs are null.
        SUPABASE_CLIENT.table(name).delete(returning=cast(ReturnMethod, "minimal")).or_(
            f"{key}.is.null,{key}.not.is.null"
        ).execute()
    for name, frame, _ in tables:
        batches = range(0, len(frame), batch_size)
        if show_progress:
            batches = tqdm(batches, desc=f"Replacing `{name}`", unit="batch")
        for start in batches:
            SUPABASE_CLIENT.table(name).insert(
                records(frame.iloc[start : start + batch_size]),
                returning=cast(ReturnMethod, "minimal"),
            ).execute()
    return {name: len(frame) for name, frame, _ in tables}


def extract_table_from_supabase(
    table_name: str,
    batch_size: int = 1000,
    show_progress: bool = True,
    columns: list[str] | None = None,
    order_by: str | None = None,
) -> pd.DataFrame:
    """
    Extract all records from a Supabase table in batches and concatenated into a pandas DataFrame.

    Args:
    - table_name(str): Supabase table to extract.
    - batch_size(int): Maximum rows sent or fetched per Supabase request.
    - show_progress(bool): Whether to display a progress bar.
    - columns(list[str] | None): Columns to select; None selects every source column.
    - order_by(str | None): Unique column for stable pagination; avoid source writes during
      extraction.

    Returns:
    - pd.DataFrame: Fetched rows with selected columns preserved even for empty tables.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    offset = 0
    all_rows = []

    pbar = tqdm(
        unit="rows",
        desc=f"Fetching from `{table_name}` ",
        disable=not show_progress,
    )

    while True:
        query = SUPABASE_CLIENT.table(table_name).select(",".join(columns) if columns else "*")
        if order_by is not None:
            query = query.order(order_by)
        res = query.range(offset, offset + batch_size - 1).execute()

        data = res.data
        if not data:
            break

        all_rows.extend(data)

        batch_len = len(data)
        offset += batch_len

        pbar.update(batch_len)

    pbar.close()

    return (
        pd.DataFrame(all_rows, columns=columns) if columns is not None else pd.DataFrame(all_rows)
    )


def download_model_from_supabase(
    tool_name: str, model_id: str, filename: str, stop_event: Event | None = None
) -> None:
    """
    Download exactly the configured bundle. Existing remote models are not rewritten.

    Args:
    - tool_name(str): Configured recommendation algorithm name.
    - model_id(str): Explicit published model identifier in ModelRegistry.
    - filename(str): Bundle filename within the model storage directory.
    - stop_event(Event | None): Optional cancellation signal checked during model loading.

    Returns:
    - None: Completes the operation without returning a value.
    """
    from pathlib import Path
    from tempfile import NamedTemporaryFile
    from models_loader import load_model, validate_model

    if stop_event is not None and stop_event.is_set():
        raise InterruptedError("Model loading stopped")
    rows = cast(
        list[dict[str, str]],
        (
            SUPABASE_CLIENT.table(MODELREGISTRY_TABLE_NAME)
            .select("storage_path")
            .eq("tool", tool_name)
            .eq("model_id", model_id)
            .limit(1)
            .execute()
        ).data,
    )
    if not rows:
        raise FileNotFoundError(f"No configured model bundle for {tool_name}: {model_id}")
    content = ARTEFACTS_STORAGE.download(f"{rows[0]['storage_path']}/{filename}")
    destination = MODEL_ARTEFACT_DIR / tool_name / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=".joblib", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        validate_model(load_model(temporary), tool_name)
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Model loading stopped")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


# Targeted API queries; bulk extraction above remains available to offline workflows.
logger = logging.getLogger(__name__)
CARD_FIELDS = [
    "parent_asin",
    "item_title",
    "main_category",
    "price",
    "is_free",
    "avg_rating",
    "num_reviews",
    "item_images",
]
DETAIL_FIELDS = CARD_FIELDS + ["store", "popularity_segment", "quality_score", "description"]
HISTORY_FIELDS = [
    "user_id",
    "parent_asin",
    "review_rating",
    "recency_weight",
    "review_date",
    "review_id",
]
PROFILE_FIELDS = (
    "user_id,num_reviews,avg_rating_given,min_rating_given,max_rating_given,"
    "first_review_date,last_review_date,days_active,user_segment,"
    "num_verified_purchases,verified_purchase_ratio,avg_price_purchased,"
    "free_app_ratio,total_helpful_votes_received"
)
REVIEW_FIELDS = (
    "review_id,user_id,parent_asin,review_title,review_text,review_rating,"
    "review_date,verified_purchase,helpful_vote"
)


class DependencyUnavailable(RuntimeError):
    """An upstream failure, distinct from a successful query returning no rows."""


@contextmanager
def timed(operation):
    """
    Measure an operation and log elapsed time even if its body raises an exception.

    Args:
    - operation(str): Operation name included in the elapsed-time log.

    Returns:
    - Iterator[None]: Context logging elapsed time on success or failure.
    """
    started = perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.3fs", operation, perf_counter() - started)


class ServingData:
    """Fetch request-scoped storefront data through targeted, paginated Supabase queries."""

    def __init__(self, client=None):
        """
        Select a Supabase client and validate the history pagination limit.

        Args:
        - client(Client | None): Supabase client; None uses the shared configured client.

        Returns:
        - None: Initializes the instance state.
        """
        self.client = SUPABASE_CLIENT if client is None else client
        self.history_page_size = int(os.getenv("HISTORY_PAGE_SIZE", "1000"))
        if not 1 <= self.history_page_size <= 1000:
            raise ValueError("HISTORY_PAGE_SIZE must be between 1 and 1000")

    def _execute(self, query: Any) -> Any:
        """
        Execute a serving query and translate upstream failures into DependencyUnavailable.

        Args:
        - query(Any): Supabase query builder exposing execute().

        Returns:
        - Any: Supabase response or compatible test double with list-valued data.
        """
        try:
            response = query.execute()
            if not isinstance(response.data, list):
                raise ValueError("Expected a list of database rows")
            return response
        except Exception as exc:
            logger.warning("Supabase serving query failed (%s)", type(exc).__name__)
            raise DependencyUnavailable("Supabase request failed") from None

    def popular_items(self, limit, *, exclude_ids=()) -> list[dict[str, Any]]:
        """
        Rank popular cards, paging bounded batches when excluding seen products.

        Recommendation fallback supplies one shopper's seen IDs and scans only as far as needed to
        fill the result or exhaust the catalog. A shopper with no history needs no exclusions. No
        bulk table cache is built.

        Args:
        - limit(int): Maximum product card count, from 1 to 100.
        - exclude_ids(Iterable[str]): Seen product IDs to exclude from popular results.

        Returns:
        - list[dict]: Unseen cards ranked by review count, average rating, and product ID.
        """
        if not 1 <= limit <= 100:
            raise ValueError("Popular-item limit must be between 1 and 100")
        excluded = set(exclude_ids)

        def ranked_query():
            """
            Build a product query ordered by review count, average rating, and product ID.

            Args:
            - None: No explicit arguments.

            Returns:
            - SyncSelectRequestBuilder: Product query with deterministic popularity ordering.
            """
            return (
                self.client.table(ITEM_TABLE_NAME)
                .select(",".join(CARD_FIELDS))
                .order("num_reviews", desc=True, nullsfirst=False)
                .order("avg_rating", desc=True, nullsfirst=False)
                .order("parent_asin", nullsfirst=False)
            )

        with timed("Popular products"):
            if not excluded:
                return self._execute(ranked_query().limit(limit)).data
            rows, offset = [], 0
            while len(rows) < limit:
                batch = self._execute(ranked_query().range(offset, offset + 99)).data
                if not batch:
                    break
                rows.extend(row for row in batch if row["parent_asin"] not in excluded)
                # A short page may be a server response cap, not the catalog end.
                offset += len(batch)
            return rows[:limit]

    def product(self, parent_asin, *, fields=DETAIL_FIELDS) -> dict[str, Any] | None:
        """
        Fetch product metadata, or return None when the product is absent.

        Args:
        - parent_asin(str): Product identifier to look up or use as a similarity seed.
        - fields(list[str]): Product metadata columns to select.

        Returns:
        - dict | None: Product metadata, or None when the product does not exist.
        """
        rows = self._execute(
            self.client.table(ITEM_TABLE_NAME)
            .select(",".join(fields))
            .eq("parent_asin", parent_asin)
            .limit(1)
        ).data
        return rows[0] if rows else None

    def product_exists(self, parent_asin):
        """
        Check whether a product exists using only its identifier field.

        Args:
        - parent_asin(str): Product identifier to look up or use as a similarity seed.

        Returns:
        - bool: Whether a matching product record exists.
        """
        return self.product(parent_asin, fields=["parent_asin"]) is not None

    def profile(self, user_id) -> dict[str, Any] | None:
        """
        Fetch a prepared shopper profile, or return None when the shopper is unknown.

        Args:
        - user_id(str): Shopper identifier used to select history or recommendations.

        Returns:
        - dict | None: Prepared profile, or None when the shopper is unknown.
        """
        rows = self._execute(
            self.client.table(USER_TABLE_NAME)
            .select(PROFILE_FIELDS)
            .eq("user_id", user_id)
            .limit(1)
        ).data
        return rows[0] if rows else None

    def reviews(self, parent_asin, page, page_size):
        """
        Fetch a stable page of product reviews and return pagination counts.

        Args:
        - parent_asin(str): Product identifier to look up or use as a similarity seed.
        - page(int): One-based page number.
        - page_size(int): Maximum number of reviews included in one page.

        Returns:
        - dict: Reviews with page, page_size, total_count, and page_count fields.
        """
        start = (page - 1) * page_size
        rows, total = [], 0
        # Fill this page even when the server caps each response below page_size.
        while len(rows) < page_size:
            response = self._execute(
                self.client.table(REVIEW_TABLE_NAME)
                .select(REVIEW_FIELDS, count=cast(CountMethod, "exact"))
                .eq("parent_asin", parent_asin)
                .order("review_date", desc=True, nullsfirst=False)
                .order("review_id")
                .range(start + len(rows), start + page_size - 1)
            )
            total = response.count or 0
            rows.extend(response.data)
            if not response.data or start + len(rows) >= total:
                break
        return {
            "items": rows,
            "page": page,
            "page_size": page_size,
            "total_count": total,
            "page_count": (total + page_size - 1) // page_size,
        }

    def user_history(self, user_id, page, page_size, *, verified_only=True):
        """
        One page of reviewed products; verification is not a complete order history.

        Args:
        - user_id(str): Shopper identifier used to select history or recommendations.
        - page(int): One-based page number.
        - page_size(int): Maximum number of reviews included in one page.
        - verified_only(bool): Whether to include only verified purchase reviews.

        Returns:
        - dict: Review page with nested cards; missing products have item=None.
        """
        start = (page - 1) * page_size
        rows, total = [], 0
        # Fill only the requested page even when Supabase caps individual responses.
        while len(rows) < page_size:
            query = (
                self.client.table(REVIEW_TABLE_NAME)
                .select(REVIEW_FIELDS, count=cast(CountMethod, "exact"))
                .eq("user_id", user_id)
            )
            if verified_only:
                query = query.eq("verified_purchase", True)
            response = self._execute(
                query.order("review_date", desc=True, nullsfirst=False)
                .order("review_id")
                .range(start + len(rows), start + page_size - 1)
            )
            total = response.count or 0
            rows.extend(response.data)
            if not response.data or start + len(rows) >= total:
                break

        catalogue = {
            item["parent_asin"]: item
            for item in self.cards(
                [row["parent_asin"] for row in rows if row.get("parent_asin") is not None]
            )
        }
        return {
            "user_id": user_id,
            "verified_only": verified_only,
            "items": [{**row, "item": catalogue.get(row.get("parent_asin"))} for row in rows],
            "page": page,
            "page_size": page_size,
            "total_count": total,
            "page_count": (total + page_size - 1) // page_size,
        }

    def cards(self, ids) -> list[dict[str, Any]]:
        """
        Fetch card metadata for at most 100 distinct product IDs with bounded pagination.

        Args:
        - ids(Iterable[str]): Product IDs; duplicates are removed before querying.

        Returns:
        - list[dict]: Product cards; callers restore recommendation ranking separately.
        """
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        if len(ids) > 100:
            raise ValueError("At most 100 recommendation card IDs are supported")
        rows = []
        # Count is restricted to the requested IDs; pagination handles a lower server row cap.
        while len(rows) < len(ids):
            response = self._execute(
                self.client.table(ITEM_TABLE_NAME)
                .select(",".join(CARD_FIELDS), count=cast(CountMethod, "exact"))
                .in_("parent_asin", ids)
                .order("parent_asin")
                .range(len(rows), len(ids) - 1)
            )
            if not response.data:
                break
            rows.extend(response.data)
            if response.count is not None and len(rows) >= response.count:
                break
        return rows

    def history(self, user_id):
        """
        All of one shopper's inference history, including low ratings and seen products.

        Args:
        - user_id(str): Shopper identifier used to select history or recommendations.

        Returns:
        - pd.DataFrame: Complete HISTORY_FIELDS records, including low ratings and seen products.
        """
        rows, cursor, pages = [], None, 0
        with timed("Shopper history retrieval"):
            while True:
                query = (
                    self.client.table(REVIEW_TABLE_NAME)
                    .select(
                        "review_id,"
                        + ",".join(field for field in HISTORY_FIELDS if field != "review_id")
                    )
                    .eq("user_id", user_id)
                    .order("review_id")
                    .limit(self.history_page_size)
                )
                if cursor is not None:
                    query = query.gt("review_id", cursor)
                batch = self._execute(query).data
                pages += 1
                if not batch:
                    break
                if any(
                    row.get("user_id") != user_id or row.get("review_id") is None for row in batch
                ):
                    raise DependencyUnavailable("Invalid shopper history response")
                next_cursor = batch[-1]["review_id"]
                if next_cursor == cursor:
                    raise DependencyUnavailable("Shopper history pagination did not advance")
                rows.extend(batch)
                cursor = next_cursor
                # Never stop on a short page: the server may impose a lower response cap.
        logger.info("Shopper history retrieved %s rows across %s requests", len(rows), pages)
        return pd.DataFrame(rows, columns=HISTORY_FIELDS)
