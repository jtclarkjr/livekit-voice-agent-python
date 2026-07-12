ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.11.28

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm AS base

COPY --from=uv /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

WORKDIR /app

FROM base AS build

COPY pyproject.toml uv.lock uv.toml .python-version ./

# uv.toml enforces locked, wheel-only dependency installation. The module-level
# command discovers installed LiveKit plugins without importing application code.
RUN uv sync --locked --no-dev --no-install-project \
    && uv run --no-dev --no-sync --module livekit.agents download-files

COPY src ./src
RUN python -m compileall -q src

FROM base AS production

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

COPY --from=build --chown=appuser:appuser /app /app

ENV HOME=/app \
    PATH="/app/.venv/bin:${PATH}"

USER appuser

CMD ["python", "src/agent.py", "start"]
