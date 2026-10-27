# syntax=docker/dockerfile:1
FROM python:3.10.20-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

# Wheels supply SentencePiece and Transformers' native extensions. WeasyPrint
# additionally needs Pango/Harfbuzz and fonts at runtime (not a C compiler).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Editable installation preserves the existing project-relative YAML/data paths.
# Source is baked into the image; no source bind mount is required.
COPY src ./src
COPY config.yaml ./
COPY data ./data
COPY tests ./tests
COPY pytest.ini ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen \
    && useradd --create-home --uid 1000 workshop \
    && mkdir /app/results \
    && chown workshop:workshop /app/results

USER workshop
EXPOSE 8000
CMD ["uvicorn", "tokenizer_workshop.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
