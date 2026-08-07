# AGENTS.md

Guidance for AI agents (and humans) operating on this repository.

## Project

`amozon_backend` is a Python 3.13 recommender backend with two tools — **Collaborative Filtering (ALS)** and **Content-Based Filtering (TF-IDF)** — over the Amazon Reviews 2023 Software subset, persisting data and model artefacts to Supabase and serving recommendations via importable handlers for serverless runtimes. See `docs/` for full documentation.

## Environment setup
- Python ≥ 3.13, managed with `uv` (NOT Anaconda/Miniconda).
- `.env` in repo root (gitignored) with:
  ```
  SUPABASE_URL=...
  SUPABASE_PUBLIC_KEY=...      # documented only; not read by source
  SUPBASE_SECRET_KEY=...       # NOTE: deliberate typo "SUPBASE" — must match exactly
  ```
- Bootstrap: `pip install uv && uv sync && python -m setup` (creates dirs, pulls tables + artefacts from Supabase).

## Common commands

| Task | Command |
|---|---|
| Build CF model | `python -m src.collaborative_filtering.main --mode build` |
| Build CBF model | `python -m src.content_based_filtering.main --mode build` |
| CF recommendations | `python -m src.collaborative_filtering.main --mode user --user_id <id> --n 10` |
| CBF user recs | `python -m src.content_based_filtering.main --mode user --user_id <id> --n 10` |
| CBF similar items | `python -m src.content_based_filtering.main --mode item --asin <parent_asin> --n 10` |
| Natural-language recommendations (needs `API_KEY`) | `python -m src.orchestration.main --mode query --query "..."` |
| Routing verdict for a user (offline) | `python -m src.orchestration.main --mode analyze --user_id <id>` |
| Popularity fallback (offline) | `python -m src.orchestration.main --mode popular --n 10` |
| Re-download raw data | `python -m data_loader.data_download` (runs module-level code) |
| Re-run feature engineering | `python -m data_loader.data_transform` |
| Push processed tables to Supabase | `python -m data_loader.push_into_supabase` |
| Push built model artefacts to Supabase | `python -m models_loader.push_into_supabase` |

There is **no test suite / lint / typecheck config** in this repo. If the user asks you to verify, run a `--mode build` followed by a `--mode user`/`--mode item` invocation against a known `user_id`/`asin` to sanity-check.

## Lint / typecheck
None configured. `pyproject.toml` declares dependencies only — no `[tool.ruff]`, `[tool.mypy]`, `pytest`, etc. Do not invent lint commands; if asked, ask the user for the command and suggest adding it here.

## Codebase conventions

### Tool-config contract
Every recommender tool lives under `src/<tool_name>/` and MUST ship a `tool_config.py` exposing:
- `MODEL_NAME` — derived from `Path(__file__).parent.name`.
- `MODEL_ID` — UUID pinned for Supabase downloads.
- `MODEL_ARTEFACTS` — ordered dict of logical name → filename (order = positional tuple order for save/load).
Plus a `model.py` with `build_and_save(...)`, a `recommender.py` with pure inference logic, and a `main.py` with lazy-cached `get_data_loader()` / `get_models_loader()` + public handlers + a CLI dispatcher.

Register new tools in `config.TOOLS`.

**Exception — `src/orchestration/`.** The agentic orchestration layer lives under `src/` but is
NOT a recommender tool: it owns no ML artefacts, its config module is named
`orchestration_config.py` (never `tool_config.py`), and it must never be added to `config.TOOLS`.
It wraps the other tools' handlers as agent tools. See `docs/ORCHESTRATION_IMPLEMENTATION.md`.

### Artefact I/O
`models_loader/models_loader.py` dispatches by file suffix: `.joblib` (joblib), `.npz` (scipy.sparse), `.json` (int/str key coercion), `.parquet` (pandas). Any new artefact type must extend `read_local_artefacts` + `save_local_artefacts`.

### Supabase
- Tables: `User`, `Item`, `Review`, `ModelRegistry`.
- Storage bucket: `ModelArtefacts` with layout `<tool_name>/<model_id>/<filename>`.
- `reconcile_registry(tool_name)` runs at the start of every push — keeps table and bucket in sync.
- After publishing new artefacts, manually update the pinned `MODEL_ID` in the tool's `tool_config.py` (push generates a fresh UUID).

### Module-level code caution
`data_loader/data_download.py`, `data_loader/data_transform.py`, `data_loader/push_into_supabase.py`, and `models_loader/push_into_supabase.py` execute on import (no `__main__` guard). Do not import them casually — run as scripts.

### Lazy-cached singletons
Both `main.py` files keep `_data_loader_obj` / `_models_loader_obj` at module scope for warm serverless reuse. `build_model()` resets them to `None`. Preserve this pattern when editing `main.py`.

## Known bugs / caveats (do not "fix" without asking)
1. `data_loader/push_into_supabase.py:64` passes `USER_FILENAME` (`"user.parquet"`) instead of `USER_TABLE_NAME` (`"User"`).
2. `wilson_lower_bound` (`data_transform.py:116`) ignores its `confidence` arg; z is hardcoded to `1.96`.
3. Rename `num_review_images_sum→total_review_images` in `data_transform.py:252` is a no-op (actual column is `num_review_img_sum`).
4. `SUPBASE_SECRET_KEY` env var typo is intentional in code — the `.env` must match.
5. CF `recommender.py` docstring claims category boost + free filtering; implementation does pure ALS only.
6. `joblib` is not in `pyproject.toml` explicitly (transitive via scikit-learn).
7. `fastapi` and `uvicorn` are declared dependencies but not yet imported anywhere — reserved for a future HTTP layer.

## Documentation
- `docs/PROJECT_DESCRIPTION.md` — high-level overview.
- `docs/ARCHITECTURE.md` — complete architecture, layers, flows, Supabase schema.
- `docs/DATA_LOADER.md` — data ETL pipeline in full.
- `docs/MODELS_LOADER.md` — model artefact lifecycle in full.
- `docs/CB_CF_FILTERING.md` — both recommendation systems in detail.

## Directory layout (quick reference)
```
config.py, setup.py, pyproject.toml, uv.lock, .python-version, .gitignore, README.md
utils/supabase_utils.py
data_loader/{__init__,data_download,data_transform,data_loader,push_into_supabase}.py
models_loader/{__init__,models_loader,push_into_supabase}.py
src/collaborative_filtering/{__init__,tool_config,model,recommender,main}.py
src/content_based_filtering/{__init__,tool_config,model,recommender,main}.py
notebook/{cb_recommend,cf_recommend,test}.ipynb
data/{input,output}/   (gitignored)
models/<tool>/         (gitignored)
```

## Style notes
- No comments in code unless explicitly requested (matches repo convention — code is mostly self-documenting with docstrings).
- Follow existing import grouping (stdlib → third-party → internal).
- Use `logging` (module-level `logger = logging.getLogger(__name__)`) as the existing modules do; do not introduce `print` in library code (bootstrap scripts already use `print`).
