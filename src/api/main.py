"""FastAPI application entrypoint for the Supply Chain Shipping Duration & SLA API.

This module initializes the production FastAPI application, configures CORS
middleware, registers custom exception handlers for structured error formatting,
manages startup/shutdown lifecycle events (including model pre-warming), and mounts
inference and diagnostics API routers.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import router
from src.models.predict import load_trained_model

logger = logging.getLogger("supply_chain_api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle manager for the FastAPI application.

    During startup, this hook pre-warms the primary LightGBM dual-quantile
    (P50 and P90) model artifact into memory, verifying that trained weights
    and feature transformation pipelines are deserialized and ready for
    sub-millisecond inference. During shutdown, it coordinates resource cleanup.

    Args:
        app: The FastAPI application instance being started or shut down.

    Yields:
        None: Passes execution control back to the FastAPI ASGI web server.
    """
    logger.info("Initializing Supply Chain Shipping API service...")
    try:
        model = load_trained_model()
        logger.info(
            "Trained model successfully cached in memory (%d encoded features).",
            len(model.feature_names),
        )
    except FileNotFoundError:
        logger.warning(
            "Model artifact not found during startup warmup. Endpoints will return 503 until artifact is created."
        )
    except Exception as err:  # noqa: BLE001
        logger.error("Error loading model artifact during startup: %s", err)
    yield
    logger.info("Shutting down Supply Chain Shipping API service.")


app = FastAPI(
    title="Supply Chain Shipping Duration & SLA Prediction API",
    description=(
        "Production REST API for estimating actual order shipping transit days (`days_for_shipping_real`). "
        "Delivers dual quantiles: **P50** (median expected duration) and **P90** (conservative SLA commitment buffer). "
        "Accepts raw checkout characteristics known at submission time and computes all derived features internally."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle 422 schema validation errors with structured JSON output.

    Transforms raw FastAPI and Pydantic validation error lists into a clean,
    standardized dictionary detailing the parameter location, error message,
    and error type for intuitive API client troubleshooting.

    Args:
        request: The incoming HTTP request that encountered validation failure.
        exc: The validation exception raised by FastAPI / Pydantic.

    Returns:
        JSONResponse: A 422 Unprocessable Entity response containing error details.
    """
    error_details = []
    for error in exc.errors():
        loc = " -> ".join(str(p) for p in error.get("loc", []))
        msg = error.get("msg", "Validation error")
        error_details.append(
            {"location": loc, "message": msg, "type": error.get("type")}
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Unprocessable Entity",
            "detail": error_details,
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle HTTP exceptions with a standardized error response schema.

    Ensures that all client errors (4xx) and application-level HTTP errors
    conform to a predictable, uniform schema containing top-level error,
    detail, and status_code fields.

    Args:
        request: The incoming HTTP request that resulted in an HTTP error.
        exc: The Starlette / FastAPI HTTPException instance.

    Returns:
        JSONResponse: A structured JSON response matching the exception status code.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail if isinstance(exc.detail, str) else "HTTP Error",
            "detail": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled internal errors and prevent unformatted trace leakage.

    Logs full traceback details internally while returning a sanitized 500
    Internal Server Error response to prevent exposing system internals or
    stack traces to external clients.

    Args:
        request: The incoming HTTP request triggering an unhandled exception.
        exc: The uncaught Python Exception instance.

    Returns:
        JSONResponse: A 500 Internal Server Error JSON response.
    """
    logger.exception(
        "Unhandled server exception on %s %s: %s", request.method, request.url.path, exc
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred processing your request.",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )


# Mount routes
app.include_router(router)
