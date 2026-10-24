FROM python:3.13-slim-bookworm

WORKDIR /app

# Required to build the implicit recommendation library.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY . .
RUN uv sync --frozen --no-install-project --no-cache

ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

CMD ["sh", "-c", "exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
