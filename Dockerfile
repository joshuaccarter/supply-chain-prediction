# Multi-stage production build for Supply Chain Shipping Prediction FastAPI Service
# Stage 1: Build dependency wheelhouse using Astral uv for fast, reliable layer caching
FROM ghcr.io/astral-sh/uv:0.6-python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install production dependencies according to uv.lock
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev


# Stage 2: Final minimal runtime image
FROM python:3.12-slim-bookworm AS runtime

# Install system libraries (libgomp1 required for LightGBM/XGBoost OpenMP, curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user and group for security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -d /app appuser

WORKDIR /app

# Copy pre-compiled virtual environment from builder stage
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Configure environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    WORKERS=2

# Copy configuration, trained production model artifacts, and source code
COPY --chown=appuser:appgroup config/ /app/config/
COPY --chown=appuser:appgroup models/ /app/models/
COPY --chown=appuser:appgroup src/ /app/src/

# Switch to non-root user
USER appuser

# Expose FastAPI listening port
EXPOSE 8000

# Container healthcheck querying the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production Uvicorn ASGI server
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-2}"]
