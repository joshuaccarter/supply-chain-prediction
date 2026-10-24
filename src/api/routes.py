"""FastAPI router providing inference, batch prediction, health check, and model info endpoints.

This module exposes the production REST endpoints for querying quantile delivery
predictions (P50 median transit days and P90 SLA commitment days), evaluating service
readiness, and inspecting model architecture metadata and benchmark metrics.
"""

import json
from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    OrderSubmission,
    PredictionResult,
)
from src.models.predict import PROJECT_ROOT, get_default_model_path, load_trained_model

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service and model health check",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Verify service availability and model artifact readiness.

    Performs a heartbeat check confirming the REST API is responsive and that the
    trained dual-quantile LightGBM model artifact (`.joblib`) exists and can be
    successfully accessed in memory.

    Returns:
        HealthResponse: A payload confirming the healthy service status, target
            parameter name, configured quantile levels, and artifact storage path.

    Raises:
        HTTPException: 503 Service Unavailable if the model artifact file is missing.
        HTTPException: 500 Internal Server Error if an unexpected error occurs while
            loading the model artifact.
    """
    try:
        load_trained_model()
        model_path = get_default_model_path()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            target_parameter="days_for_shipping_real",
            quantiles={"p50": 0.50, "p90": 0.90},
            artifact_path=str(
                model_path.relative_to(PROJECT_ROOT)
                if model_path.is_relative_to(PROJECT_ROOT)
                else model_path
            ),
        )
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact not found: {err}",
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model health check failed: {err}",
        ) from err


@router.post(
    "/predict",
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
    summary="Predict P50 and P90 shipping days for a single order",
    tags=["Predictions"],
)
def predict_single_order(submission: OrderSubmission) -> PredictionResult:
    """Compute median duration (P50) and SLA commitment (P90) from checkout characteristics.

    Accepts raw checkout characteristics known at submission time (e.g. shipping tier,
    payment type, product unit price, discount rate, destination region and country,
    customer segment, checkout timestamp). Automatically constructs cyclical calendar
    features, price ratios, basket aggregations, and route classifications internally
    without requiring any post-checkout leakage fields.

    Args:
        submission: Validated OrderSubmission payload containing raw order attributes.

    Returns:
        PredictionResult: Dual-quantile predictions including discrete P50 shipping days,
            discrete P90 shipping days, safety buffer cushion (P90 - P50), unrounded continuous
            scores, and calendar delivery date timestamps if checkout date was provided.

    Raises:
        HTTPException: 503 Service Unavailable if the trained model artifact is missing.
        HTTPException: 400 Bad Request if feature transformation fails due to invalid values.
        HTTPException: 500 Internal Server Error if inference execution encounters an error.
    """
    try:
        model = load_trained_model()
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Trained model artifact is unavailable: {err}",
        ) from err

    payload = submission.model_dump(exclude_none=True)
    try:
        result = model.predict_p50_p90(payload)
        return PredictionResult(**result)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature transformation or inference error: {err}",
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {err}",
        ) from err


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch predict P50 and P90 shipping days (up to 1,000 orders)",
    tags=["Predictions"],
)
def predict_batch_orders(
    batch_request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Run batch P50 and P90 quantile inference across multiple order submissions.

    Processes an array of 1 to 1,000 order records in a vectorized manner, executing
    feature engineering, design-matrix encoding, and quantile regressor passes
    simultaneously for high throughput.

    Args:
        batch_request: Validated BatchPredictionRequest containing a list of OrderSubmission items.

    Returns:
        BatchPredictionResponse: Summary object containing batch count, list of PredictionResult
            records, target parameter name, and model family description.

    Raises:
        HTTPException: 503 Service Unavailable if the trained model artifact is missing.
        HTTPException: 400 Bad Request if batch feature transformation fails.
        HTTPException: 500 Internal Server Error if batch inference fails.
    """
    try:
        model = load_trained_model()
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Trained model artifact is unavailable: {err}",
        ) from err

    records = [order.model_dump(exclude_none=True) for order in batch_request.orders]
    try:
        raw_results = model.predict_p50_p90(records)
        predictions = [PredictionResult(**r) for r in raw_results]
        return BatchPredictionResponse(
            count=len(predictions),
            predictions=predictions,
            target_parameter="days_for_shipping_real",
            model_family="LightGBM Quantile Regressor",
        )
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch transformation or inference error: {err}",
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference execution failed: {err}",
        ) from err


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve metadata, feature schema, and performance benchmarks",
    tags=["System"],
)
def get_model_info() -> ModelInfoResponse:
    """Return model architecture details, encoded feature names, and benchmark test metrics.

    Provides introspection into the production deployment, detailing the target parameter,
    quantile levels (0.50 and 0.90), discrete output bounds [0, 6], split methodology,
    the complete ordered sequence of one-hot and scaled design-matrix features, and held-out
    test evaluation metrics.

    Returns:
        ModelInfoResponse: Structured metadata, feature schema, and benchmark summary.

    Raises:
        HTTPException: 503 Service Unavailable if the model artifact cannot be located.
    """
    try:
        model = load_trained_model()
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Trained model artifact is unavailable: {err}",
        ) from err

    metrics_payload: dict[str, Any] | None = None
    metrics_path = PROJECT_ROOT / "models" / "lgbm_metrics.json"
    if metrics_path.exists():
        try:
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            metrics_payload = None

    return ModelInfoResponse(
        target_parameter="days_for_shipping_real",
        quantiles={"p50": 0.50, "p90": 0.90},
        target_bounds={"min": model.min_delay, "max": model.max_delay},
        split_type=model.metadata.get("split_type", "chronological"),
        n_encoded_features=len(model.feature_names),
        feature_names=model.feature_names,
        test_metrics=metrics_payload.get("chronological", {}).get("test")
        if metrics_payload
        else model.metadata.get("test_summary"),
    )
