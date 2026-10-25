"""Unit and integration tests for the LightGBM P50/P90 model, evaluation metrics, and API inference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    evaluate_p50_p90,
    evaluate_predictions,
)
from src.models.predict import (
    LightGBMP50P90Model,
    get_default_model_path,
    load_trained_model,
    predict_p50_p90,
    prepare_inference_dataframe,
)
from src.models.train import train_lightgbm_p50_p90
from src.utils.config_loader import load_config


@pytest.fixture(scope="module")
def synthetic_training_df() -> pd.DataFrame:
    """Create a synthetic 60-row dataset spanning multiple orders and shipping tiers for fast training tests."""
    rng = np.random.default_rng(42)
    n_rows = 60
    modes = np.tile(
        ["Same Day", "First Class", "Second Class", "Standard Class"], n_rows // 4
    )
    actual_days = np.tile([0, 2, 4, 4], n_rows // 4)
    order_ids = [f"ORD_{i // 2:03d}" for i in range(n_rows)]
    base_dates = pd.date_range("2017-01-01 09:00:00", periods=n_rows, freq="12h")

    return pd.DataFrame(
        {
            "days_for_shipping_real": actual_days,
            "shipping_mode": pd.Series(modes, dtype="category"),
            "order_item_quantity": rng.integers(1, 5, size=n_rows),
            "product_price": rng.uniform(25.0, 250.0, size=n_rows).round(2),
            "order_item_discount_rate": rng.choice(
                [0.0, 0.05, 0.10, 0.15], size=n_rows
            ),
            "order_item_profit_ratio": rng.uniform(-0.10, 0.35, size=n_rows).round(2),
            "latitude": rng.uniform(18.0, 42.0, size=n_rows).round(4),
            "longitude": rng.uniform(-120.0, -66.0, size=n_rows).round(4),
            "type": pd.Series(
                rng.choice(["DEBIT", "TRANSFER", "CASH", "PAYMENT"], size=n_rows),
                dtype="category",
            ),
            "customer_country": pd.Series(
                rng.choice(["EE. UU.", "Puerto Rico"], size=n_rows),
                dtype="category",
            ),
            "customer_segment": pd.Series(
                rng.choice(["Consumer", "Corporate", "Home Office"], size=n_rows),
                dtype="category",
            ),
            "market": pd.Series(
                rng.choice(
                    ["LATAM", "Europe", "USCA", "Pacific Asia", "Africa"], size=n_rows
                ),
                dtype="category",
            ),
            "department_id": pd.Series(
                rng.choice(["2", "4", "7", "10"], size=n_rows),
                dtype="category",
            ),
            "order_region": pd.Series(
                rng.choice(
                    ["Central America", "Western Europe", "West of USA"], size=n_rows
                ),
                dtype="category",
            ),
            "order_country": pd.Series(
                rng.choice(
                    ["Estados Unidos", "Francia", "México", "Alemania"], size=n_rows
                ),
                dtype="string",
            ),
            "order_id": pd.Series(order_ids, dtype="string"),
            "order_item_id": pd.Series([str(i) for i in range(n_rows)], dtype="string"),
            "customer_id": pd.Series(
                [f"CUST_{i % 10}" for i in range(n_rows)], dtype="string"
            ),
            "order_date_dateorders": base_dates,
        }
    )


def test_prepare_inference_dataframe_aliases_and_defaults() -> None:
    """Verify API convenience aliases (shipping_mode, department_name, order_date, country) are resolved."""
    payload = {
        "shipping_mode": "Second Class",
        "department_name": "Technology",
        "customer_country": "USA",
        "order_date": "2018-01-15 15:30:00",
        "product_price": 150.0,
        "order_item_quantity": 2,
        "order_line_item_count": 3,
        "order_total_units": 6,
        "order_total_sales": 750.0,
    }
    std_df, explicit_basket = prepare_inference_dataframe(payload)

    assert len(std_df) == 1
    assert std_df.loc[0, "days_for_shipment_scheduled"] == 2
    assert std_df.loc[0, "department_id"] == "10"
    assert std_df.loc[0, "customer_country"] == "EE. UU."
    assert std_df.loc[0, "order_date_dateorders"] == "2018-01-15 15:30:00"
    assert explicit_basket["order_line_item_count"].iloc[0] == 3
    assert explicit_basket["order_total_units"].iloc[0] == 6
    assert np.isclose(explicit_basket["order_total_sales"].iloc[0], 750.0)


def test_evaluate_predictions_and_p50_p90_metrics() -> None:
    """Verify standard regression, classification, ordinal, and SLA buffer metrics."""
    y_true = np.array([0, 1, 2, 3, 4, 6])
    p50_pred = np.array([0, 1, 2, 3, 4, 5])
    p90_pred = np.array([1, 2, 3, 4, 6, 6])
    sched = np.array([0, 1, 2, 4, 4, 4])

    m_p50 = evaluate_predictions(y_true, p50_pred, model_name="TestP50")
    assert np.isclose(m_p50["exact_accuracy"], 5 / 6)
    assert np.isclose(m_p50["within_1_day_accuracy"], 1.0)
    assert np.isclose(m_p50["mae"], 1 / 6)
    assert np.isclose(m_p50["sla_compliance_pct"], (5 / 6) * 100.0)
    assert np.isclose(m_p50["late_breach_pct"], (1 / 6) * 100.0)

    joint_rep = evaluate_p50_p90(
        y_true=y_true,
        p50_pred=p50_pred,
        p90_pred=p90_pred,
        scheduled_days=sched,
    )
    assert joint_rep["p90"]["sla_compliance_pct"] == 100.0
    assert joint_rep["p90"]["late_breach_pct"] == 0.0
    assert joint_rep["joint"]["quantile_non_crossing_rate"] == 1.0
    assert len(joint_rep["tier_breakdown"]) == 4


def test_train_and_serialize_lightgbm_on_synthetic_data(
    synthetic_training_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verify LightGBMP50P90Model trains, saves, reloads, and predicts on synthetic data."""
    artifact_file = tmp_path / "test_lgbm.joblib"
    metrics_file = tmp_path / "test_metrics.json"

    model, report = train_lightgbm_p50_p90(
        df=synthetic_training_df,
        split_type="chronological",
        save_artifact=True,
        artifact_path=artifact_file,
        metrics_path=metrics_file,
    )

    assert artifact_file.exists()
    assert metrics_file.exists()
    assert "validation" in report
    assert "test" in report

    reloaded = LightGBMP50P90Model.load(artifact_file)
    pred = reloaded.predict_p50_p90(
        {
            "shipping_mode": "First Class",
            "type": "DEBIT",
            "customer_country": "EE. UU.",
            "customer_segment": "Consumer",
            "department_id": "7",
            "market": "USCA",
            "order_region": "West of USA",
            "order_country": "Estados Unidos",
            "product_price": 99.0,
            "order_item_quantity": 1,
            "order_date_dateorders": "2017-01-10 10:00:00",
        }
    )
    assert isinstance(pred, dict)
    assert 0 <= pred["p50"] <= pred["p90"] <= 6
    assert pred["p50_shipping_days"] <= pred["p90_shipping_days"]


def test_production_model_artifact_and_api_predictions() -> None:
    """Verify the trained production LightGBM model artifact meets standard metric benchmarks and serves API inputs."""
    cfg = load_config()
    model_path = get_default_model_path(config=cfg)
    assert model_path.exists(), f"Expected trained model artifact at {model_path}"

    model = load_trained_model(model_path=model_path)

    # 1. Single characteristic dict prediction
    single_char = {
        "shipping_mode": "Second Class",
        "type": "TRANSFER",
        "customer_country": "Puerto Rico",
        "customer_segment": "Corporate",
        "department_name": "Fan Shop",
        "market": "Europe",
        "order_region": "Western Europe",
        "order_country": "Francia",
        "latitude": 18.2208,
        "longitude": -66.5901,
        "product_price": 120.0,
        "order_item_quantity": 2,
        "order_item_discount_rate": 0.05,
        "order_item_profit_ratio": 0.20,
        "order_date": "2018-01-15 14:30:00",
    }
    single_out = predict_p50_p90(single_char, model=model)
    assert isinstance(single_out, dict)
    for key in (
        "p50",
        "p90",
        "p50_raw",
        "p90_raw",
        "p50_shipping_days",
        "p90_shipping_days",
        "safety_buffer_days",
        "shipping_mode",
    ):
        assert key in single_out
    assert single_out["p50"] == single_out["p50_shipping_days"]
    assert single_out["p90"] == single_out["p90_shipping_days"]
    assert 0 <= single_out["p50"] <= single_out["p90"] <= 6
    assert single_out["safety_buffer_days"] == single_out["p90"] - single_out["p50"]
    assert "p50_estimated_delivery_date" in single_out
    assert "p90_sla_delivery_date" in single_out

    # 2. Verify Same Day cutoff behavior (0 to 1 days)
    same_day_morning = {
        **single_char,
        "shipping_mode": "Same Day",
        "order_date": "2018-01-15 10:00:00",
    }
    same_day_night = {
        **single_char,
        "shipping_mode": "Same Day",
        "order_date": "2018-01-15 21:00:00",
    }
    batch_out = predict_p50_p90([same_day_morning, same_day_night], model=model)
    assert isinstance(batch_out, list) and len(batch_out) == 2
    assert 0 <= batch_out[0]["p50"] <= 1
    assert 0 <= batch_out[1]["p50"] <= 1

    # 3. Verify saved benchmark metrics meet expected accuracy, MAE, QWK, and P90 SLA standards
    metrics_path = model_path.parent / cfg["models"]["metrics_name"]
    assert metrics_path.exists()
    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))

    for split_key in ("chronological", "group_random"):
        test_p50 = metrics_data[split_key]["test"]["p50"]
        test_p90 = metrics_data[split_key]["test"]["p90"]

        assert test_p50["exact_accuracy"] >= 0.35
        assert test_p50["within_1_day_accuracy"] >= 0.65
        assert test_p50["mae"] <= 1.05
        assert test_p50["r2"] >= 0.35
        assert test_p50["qwk"] >= 0.50
        assert test_p50["pinball_loss_p50"] <= 0.50

        assert test_p90["sla_compliance_pct"] >= 99.5
        assert test_p90["late_breach_pct"] <= 0.5
        assert test_p90["pinball_loss_p90"] <= 0.30
