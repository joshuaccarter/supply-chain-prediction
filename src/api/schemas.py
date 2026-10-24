"""Pydantic schemas for FastAPI request validation, response serialization, and health checks.

This module defines data validation schemas for order submissions, single and batch
inference responses, system health statuses, model architecture introspection,
and standardized HTTP error payloads.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OrderSubmission(BaseModel):
    """Raw characteristics known at checkout/order submission time.

    Derived features (e.g., cyclical temporal encodings, basket metrics, financial totals,
    and cross-border indicators) are automatically computed by the feature pipeline
    without requiring any downstream delivery or leakage information.

    Attributes:
        shipping_mode: Fulfillment shipping tier ('Standard Class', 'First Class',
            'Second Class', or 'Same Day').
        type: Payment transaction method ('DEBIT', 'TRANSFER', 'PAYMENT', or 'CASH').
        order_region: Destination geographic sales region.
        market: Global market territory ('LATAM', 'Europe', 'Pacific Asia', 'USCA', or 'Africa').
        customer_country: Customer residence country (e.g., 'EE. UU.', 'Puerto Rico', 'USA').
        customer_segment: Customer market segment ('Consumer', 'Corporate', or 'Home Office').
        department_id: Internal department identifier (1-12). Inferred from department_name if omitted.
        department_name: Optional department name automatically mapped to department_id.
        order_country: Destination delivery country.
        order_item_quantity: Quantity of units ordered for the item line (>= 1).
        product_price: Unit catalog price of the product in USD.
        order_item_discount_rate: Item promotion discount rate between 0.0 (0%) and 1.0 (100%).
        order_item_profit_ratio: Expected profit margin ratio on the line item.
        order_date: Checkout timestamp (ISO-8601 or standard string format).
        latitude: Store/origin GPS latitude coordinate.
        longitude: Store/origin GPS longitude coordinate.
        order_id: Optional enterprise order identifier for tracking and multi-item grouping.
        days_for_shipment_scheduled: Optional scheduled fulfillment quote in days (0, 1, 2, or 4).
    """

    model_config = ConfigDict(extra="ignore")

    shipping_mode: str = Field(
        default="Standard Class",
        description="Fulfillment shipping tier: 'Standard Class', 'First Class', 'Second Class', or 'Same Day'.",
        examples=["Standard Class"],
    )
    type: str = Field(
        default="DEBIT",
        description="Payment transaction method: 'DEBIT', 'TRANSFER', 'PAYMENT', or 'CASH'.",
        examples=["DEBIT"],
    )
    order_region: str = Field(
        default="Central America",
        description="Destination geographic sales region.",
        examples=["Central America", "Western Europe", "West of USA"],
    )
    market: str = Field(
        default="LATAM",
        description="Global market territory: 'LATAM', 'Europe', 'Pacific Asia', 'USCA', or 'Africa'.",
        examples=["LATAM"],
    )
    customer_country: str = Field(
        default="EE. UU.",
        description="Customer residence country (e.g. 'EE. UU.', 'Puerto Rico', 'USA').",
        examples=["EE. UU."],
    )
    customer_segment: str = Field(
        default="Consumer",
        description="Customer account market segment: 'Consumer', 'Corporate', or 'Home Office'.",
        examples=["Consumer"],
    )
    department_id: str | None = Field(
        default="7",
        description="Internal department identifier (1-12). If omitted, inferred from department_name.",
        examples=["7"],
    )
    department_name: str | None = Field(
        default=None,
        description="Department name (e.g. 'Fan Shop', 'Apparel', 'Outdoors'). Automatically mapped to department_id.",
        examples=["Fan Shop"],
    )
    order_country: str | None = Field(
        default="Estados Unidos",
        description="Destination delivery country.",
        examples=["Estados Unidos", "Francia", "México"],
    )
    order_item_quantity: int = Field(
        default=1,
        ge=1,
        le=1000,
        description="Quantity of units ordered for the item line (>= 1).",
        examples=[1],
    )
    product_price: float = Field(
        default=100.0,
        ge=0.0,
        description="Unit catalog price of the product.",
        examples=[100.0],
    )
    order_item_discount_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Item promotion discount rate between 0.0 (0%) and 1.0 (100%).",
        examples=[0.05],
    )
    order_item_profit_ratio: float = Field(
        default=0.15,
        ge=-1.0,
        le=1.0,
        description="Expected profit margin ratio on the line item.",
        examples=[0.15],
    )
    order_date: datetime | str | None = Field(
        default="2018-01-15 12:00:00",
        description="Checkout timestamp (ISO-8601 or standard string format). Used to compute delivery dates.",
        examples=["2018-01-15 12:00:00"],
    )
    latitude: float | None = Field(
        default=18.2208,
        description="Store/origin GPS latitude coordinate.",
        examples=[18.2208],
    )
    longitude: float | None = Field(
        default=-66.5901,
        description="Store/origin GPS longitude coordinate.",
        examples=[-66.5901],
    )
    order_id: str | None = Field(
        default=None,
        description="Optional enterprise order identifier for tracking and multi-item grouping.",
        examples=["10001"],
    )
    # Support backward-compatible scheduled days parameter if supplied
    days_for_shipment_scheduled: int | None = Field(
        default=None,
        description="Optional scheduled fulfillment quote in days (0, 1, 2, or 4). Inferred from shipping_mode if omitted.",
        examples=[4],
    )


class PredictionResult(BaseModel):
    """P50 expected shipping duration and P90 SLA commitment response.

    Attributes:
        order_id: Enterprise order identifier passed in the submission if available.
        shipping_mode: Canonical shipping tier used for fulfillment.
        p50_shipping_days: Median predicted shipping duration in calendar days (P50).
        p90_shipping_days: Conservative SLA commitment shipping duration in calendar days (P90).
        safety_buffer_days: Operational buffer cushion computed as P90 - P50 days.
        p50_raw: Unrounded continuous P50 prediction score from the quantile regressor.
        p90_raw: Unrounded continuous P90 prediction score from the quantile regressor.
        p50_estimated_delivery_date: Expected arrival timestamp based on P50 shipping duration.
        p90_sla_delivery_date: Guaranteed SLA arrival deadline based on P90 shipping duration.
        p50: Direct alias for p50_shipping_days.
        p90: Direct alias for p90_shipping_days.
        days_for_shipment_scheduled: Scheduled shipping tier quote days (0, 1, 2, or 4).
    """

    order_id: str | None = Field(
        default=None,
        description="Enterprise order identifier passed in the submission.",
    )
    shipping_mode: str = Field(
        description="Canonical shipping tier used for fulfillment.",
        examples=["Standard Class"],
    )
    p50_shipping_days: int = Field(
        description="Median predicted shipping duration in calendar days (P50).",
        examples=[4],
    )
    p90_shipping_days: int = Field(
        description="Conservative SLA commitment shipping duration in calendar days (P90).",
        examples=[5],
    )
    safety_buffer_days: int = Field(
        description="Operational buffer cushion: P90 - P50 days.",
        examples=[1],
    )
    p50_raw: float = Field(
        description="Unrounded continuous P50 prediction score.",
        examples=[3.9214],
    )
    p90_raw: float = Field(
        description="Unrounded continuous P90 prediction score.",
        examples=[4.8912],
    )
    p50_estimated_delivery_date: str | None = Field(
        default=None,
        description="Expected arrival timestamp based on P50 shipping duration (ISO-8601).",
        examples=["2018-01-19 12:00:00"],
    )
    p90_sla_delivery_date: str | None = Field(
        default=None,
        description="Guaranteed SLA arrival deadline based on P90 shipping duration (ISO-8601).",
        examples=["2018-01-20 12:00:00"],
    )
    p50: int = Field(
        description="Direct alias for p50_shipping_days.",
        examples=[4],
    )
    p90: int = Field(
        description="Direct alias for p90_shipping_days.",
        examples=[5],
    )
    days_for_shipment_scheduled: int | None = Field(
        default=None,
        description="Scheduled shipping tier quote days (0, 1, 2, or 4).",
    )


class BatchPredictionRequest(BaseModel):
    """Batch order inference request (1 to 1,000 orders).

    Attributes:
        orders: List of OrderSubmission payloads to evaluate concurrently.
    """

    orders: list[OrderSubmission] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of order submissions to evaluate.",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction results containing individual order predictions and metadata.

    Attributes:
        count: Number of predictions returned in the response.
        predictions: List of PredictionResult objects matching each submitted order.
        target_parameter: Physical supply chain quantity predicted by the model.
        model_family: High-level model architecture description.
    """

    count: int = Field(
        description="Number of predictions returned.",
        examples=[1],
    )
    predictions: list[PredictionResult] = Field(
        description="List of P50 and P90 prediction results.",
    )
    target_parameter: str = Field(
        default="days_for_shipping_real",
        description="Physical quantity predicted by the model.",
    )
    model_family: str = Field(
        default="LightGBM Quantile Regressor",
        description="Model family.",
    )


class HealthResponse(BaseModel):
    """API health status and model readiness.

    Attributes:
        status: Overall service health status string (e.g. 'healthy').
        model_loaded: Boolean flag indicating whether the trained model artifact is active in memory.
        target_parameter: Physical target quantity predicted by the deployed pipeline.
        quantiles: Dictionary of active quantile thresholds (P50 and P90).
        artifact_path: Relative or absolute path to the serialized model file on disk.
    """

    status: str = Field(
        default="healthy",
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        default=True,
        examples=[True],
    )
    target_parameter: str = Field(
        default="days_for_shipping_real",
        examples=["days_for_shipping_real"],
    )
    quantiles: dict[str, float] = Field(
        default_factory=lambda: {"p50": 0.50, "p90": 0.90},
        examples=[{"p50": 0.50, "p90": 0.90}],
    )
    artifact_path: str | None = Field(
        default=None,
        examples=["models/lgbm_shipping_delay_model.joblib"],
    )


class ModelInfoResponse(BaseModel):
    """Metadata, feature schema, and validation/test performance of the loaded model.

    Attributes:
        target_parameter: Target column predicted by the model.
        quantiles: Dictionary mapping quantile names to their numerical alpha values.
        target_bounds: Minimum and maximum discrete bounds applied to predictions.
        split_type: Splitting strategy used during model training.
        n_encoded_features: Total number of features in the one-hot and scaled design matrix.
        feature_names: Ordered list of feature names consumed by the LightGBM regressors.
        test_metrics: Dictionary of test set evaluation metrics and benchmarks.
    """

    target_parameter: str = "days_for_shipping_real"
    quantiles: dict[str, float] = Field(
        default_factory=lambda: {"p50": 0.50, "p90": 0.90}
    )
    target_bounds: dict[str, int] = Field(default_factory=lambda: {"min": 0, "max": 6})
    split_type: str = "chronological"
    n_encoded_features: int = 0
    feature_names: list[str] = Field(default_factory=list)
    test_metrics: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Standardized API error response format.

    Attributes:
        error: High-level error classification string.
        detail: Detailed error explanation or nested validation issues.
        status_code: HTTP status code corresponding to the error.
    """

    error: str = Field(description="High-level error classification.")
    detail: Any = Field(description="Detailed error message or validation breakdowns.")
    status_code: int = Field(description="HTTP status code corresponding to the error.")
