# Change Log

## 2026-08-02

- Created `.env` with dummy Supabase credentials (`SUPABASE_URL`, `SUPBASE_SECRET_KEY`) so import-time client construction in `utils/supabase_utils.py:20` succeeds without real Supabase access. File is gitignored.
- `data_loader/data_download.py`: fixed import of `ITEM_META_FILENAME` → `ITEM_METADATA_FILENAME` (matching `config.py:51`) in both the import block (`:17`) and the export statement (`:237`).

## 2026-08-08

- Verified live Supabase state (tables `User`/`Item`/`Review`/`ModelRegistry` populated, `ModelArtefacts` storage bucket populated with both tools' pinned `MODEL_ID` artefacts) and recorded it in `docs/SUPABASE_STATE.md` for future sessions.
- Added a `force_remote: bool = False` flag to make inference prefer Supabase over the local cache, while leaving build/setup behavior local-cache-first as before:
  - `data_loader/data_loader.py`: `load_data()` and `DataLoader.__init__` accept `force_remote`; when `True`, skips the local Parquet read and always fetches from Supabase (still refreshes the local cache file with the result).
  - `models_loader/models_loader.py`: `load_artefacts()` and `ModelsLoader.__init__` accept `force_remote`; when `True`, bypasses the "all cached locally" check and always re-downloads artefacts from Supabase Storage.
  - `src/collaborative_filtering/main.py` and `src/content_based_filtering/main.py`: `get_data_loader()`/`get_models_loader()` accept `force_remote`, applied only on the call that creates the cached singleton. `get_user_recommendations` (and `get_similar_items` in CBF) now pass `force_remote=True`; `build_model()` keeps the default `False`.
  - Caveat: since loaders are cached module-level singletons per warm process, `force_remote` only takes effect on first instantiation (or after a cache reset in `build_model`), not on every individual inference call.
- Documented the `force_remote` flag and the full inference-time cache-loading flow across `docs/DATA_LOADER.md`, `docs/MODELS_LOADER.md`, `docs/CB_CF_FILTERING.md`, and a new `docs/ARCHITECTURE.md#10-inference-time-loading-into-cache` section covering: the three caching layers (local disk / Supabase / in-process singleton), what happens on cold vs. warm serverless invocations, per-instance scale-out behavior, and the approximate in-memory footprint of cached data/artefacts.
