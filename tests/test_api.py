"""Integration and unit tests for the FastAPI Supply Chain Prediction API."""

from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest

from src.api.main import app

client = TestClient(app)


def test_health_check() -> None:
    """Verify GET /health returns 200 with model readiness and metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["target_parameter"] == "days_for_shipping_real"
    assert data["quantiles"] == {"p50": 0.50, "p90": 0.90}
    assert "artifact_path" in data


def test_model_info_endpoint() -> None:
    """Verify GET /model-info returns 200 with schema, target bounds, and test metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["target_parameter"] == "days_for_shipping_real"
    assert data["target_bounds"] == {"min": 0, "max": 6}
    assert data["quantiles"] == {"p50": 0.50, "p90": 0.90}
    assert data["n_encoded_features"] > 0
    assert isinstance(data["feature_names"], list)
    assert len(data["feature_names"]) == data["n_encoded_features"]
    assert data["test_metrics"] is not None


def test_predict_single_order_full_payload() -> None:
    """Verify POST /predict returns valid P50, P90, safety buffer, and delivery dates."""
    payload = {
        "shipping_mode": "Standard Class",
        "type": "DEBIT",
        "order_region": "Central America",
        "market": "LATAM",
        "customer_country": "EE. UU.",
        "customer_segment": "Consumer",
        "department_id": "7",
        "order_country": "Estados Unidos",
        "order_item_quantity": 2,
        "product_price": 120.0,
        "order_item_discount_rate": 0.05,
        "order_item_profit_ratio": 0.20,
        "order_date": "2018-01-15 12:00:00",
        "latitude": 18.2208,
        "longitude": -66.5901,
        "order_id": "ORD-12345",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["order_id"] == "ORD-12345"
    assert data["shipping_mode"] == "Standard Class"
    assert 0 <= data["p50_shipping_days"] <= data["p90_shipping_days"] <= 6
    assert data["p50"] == data["p50_shipping_days"]
    assert data["p90"] == data["p90_shipping_days"]
    assert data["safety_buffer_days"] == data["p90_shipping_days"] - data["p50_shipping_days"]
    assert isinstance(data["p50_raw"], float)
    assert isinstance(data["p90_raw"], float)
    assert data["p50_estimated_delivery_date"] == "2018-01-19 12:00:00"
    assert data["p90_sla_delivery_date"] == "2018-01-21 12:00:00"


def test_predict_single_order_minimal_defaults() -> None:
    """Verify POST /predict succeeds with empty payload, using all checkout defaults."""
    response = client.post("/predict", json={})
    assert response.status_code == 200
    data = response.json()

    assert data["shipping_mode"] == "Standard Class"
    assert 0 <= data["p50_shipping_days"] <= data["p90_shipping_days"] <= 6
    assert data["safety_buffer_days"] == data["p90_shipping_days"] - data["p50_shipping_days"]
    assert data["p50_estimated_delivery_date"] is not None
    assert data["p90_sla_delivery_date"] is not None


def test_predict_with_aliases() -> None:
    """Verify POST /predict resolves convenience aliases (department_name, First Class shipping)."""
    payload = {
        "shipping_mode": "first class",
        "department_name": "Fan Shop",
        "customer_country": "USA",
        "order_date": "2018-01-20 09:00:00",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["shipping_mode"] == "First Class"
    # First Class in DataCo dataset is 100% physically 2 days
    assert data["p50_shipping_days"] == 2
    assert data["p90_shipping_days"] == 6
    assert data["p50_estimated_delivery_date"] == "2018-01-22 09:00:00"


def test_predict_batch_orders() -> None:
    """Verify POST /predict/batch handles multiple order submissions in a single request."""
    batch_payload = {
        "orders": [
            {
                "shipping_mode": "Same Day",
                "order_date": "2018-01-15 10:00:00",
                "order_id": "BATCH-01",
            },
            {
                "shipping_mode": "First Class",
                "order_date": "2018-01-15 14:00:00",
                "order_id": "BATCH-02",
            },
            {
                "shipping_mode": "Second Class",
                "order_date": "2018-01-15 18:00:00",
                "order_id": "BATCH-03",
            },
            {
                "shipping_mode": "Standard Class",
                "order_date": "2018-01-15 22:00:00",
                "order_id": "BATCH-04",
            },
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 4
    assert len(data["predictions"]) == 4
    assert data["target_parameter"] == "days_for_shipping_real"

    for idx, expected_id in enumerate(["BATCH-01", "BATCH-02", "BATCH-03", "BATCH-04"]):
        pred = data["predictions"][idx]
        assert pred["order_id"] == expected_id
        assert 0 <= pred["p50_shipping_days"] <= pred["p90_shipping_days"] <= 6


def test_validation_errors_return_422() -> None:
    """Verify Pydantic schema constraints trigger HTTP 422 Unprocessable Entity with error details."""
    # 1. Invalid order_item_quantity (< 1)
    res_qty = client.post("/predict", json={"order_item_quantity": 0})
    assert res_qty.status_code == 422
    assert res_qty.json()["error"] == "Unprocessable Entity"
    assert len(res_qty.json()["detail"]) > 0

    # 2. Invalid discount rate (> 1.0)
    res_disc = client.post("/predict", json={"order_item_discount_rate": 1.5})
    assert res_disc.status_code == 422

    # 3. Invalid negative price (< 0.0)
    res_price = client.post("/predict", json={"product_price": -25.0})
    assert res_price.status_code == 422

    # 4. Batch request with empty orders list (min_length=1)
    res_empty_batch = client.post("/predict/batch", json={"orders": []})
    assert res_empty_batch.status_code == 422


def test_missing_model_artifact_returns_503() -> None:
    """Verify HTTP 503 Service Unavailable is raised when model artifact is missing."""
    with patch("src.api.routes.load_trained_model", side_effect=FileNotFoundError("Artifact missing")):
        res_health = client.get("/health")
        assert res_health.status_code == 503
        assert "not found" in res_health.json()["detail"]

        res_pred = client.post("/predict", json={})
        assert res_pred.status_code == 503
        assert "unavailable" in res_pred.json()["detail"]

        res_batch = client.post("/predict/batch", json={"orders": [{}]})
        assert res_batch.status_code == 503
        assert "unavailable" in res_batch.json()["detail"]
