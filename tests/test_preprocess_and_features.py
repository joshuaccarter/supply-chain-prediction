"""Unit and integration tests for data preprocessing, splitting, and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import load_raw_data
from src.data.preprocess import (
    clean_and_preprocess,
    drop_and_prune_columns,
    get_columns_to_drop,
    split_dataset,
)
from src.features.build_features import (
    SupplyChainFeatureBuilder,
    add_basket_features,
    add_financial_ratio_features,
    add_temporal_features,
    build_features,
)
from src.utils.config_loader import load_config


@pytest.fixture(scope="module")
def sample_preprocessed_df() -> pd.DataFrame:
    """Create a synthetic DataFrame matching the raw schema for fast unit testing."""
    return pd.DataFrame(
        {
            # Target & legitimate core numerics
            "days_for_shipping_real": [3, 3, 3, 1, 6, 2],
            "days_for_shipment_scheduled": [4, 2, 2, 1, 4, 4],
            "order_item_quantity": [2, 1, 3, 1, 4, 2],
            "product_price": [100.0, 50.0, 20.0, 200.0, 30.0, 80.0],
            "order_item_discount_rate": [0.10, 0.0, 0.05, 0.20, 0.10, 0.0],
            "order_item_profit_ratio": [0.25, -0.10, 0.15, 0.30, -0.50, 0.10],
            "latitude": [18.2, 34.0, 34.0, 40.7, 18.2, 41.8],
            "longitude": [-66.5, -118.2, -118.2, -74.0, -66.5, -87.6],
            # Core categoricals & geography
            "type": pd.Series(["DEBIT", "TRANSFER", "TRANSFER", "CASH", "DEBIT", "PAYMENT"], dtype="category"),
            "customer_country": pd.Series(
                ["Puerto Rico", "EE. UU.", "EE. UU.", "EE. UU.", "Puerto Rico", "EE. UU."],
                dtype="category",
            ),
            "order_country": pd.Series(
                ["Estados Unidos", "Francia", "Francia", "Puerto Rico", "Alemania", "Estados Unidos"],
                dtype="string",
            ),
            # Entity & time keys
            "order_id": pd.Series(["101", "102", "102", "103", "104", "105"], dtype="string"),
            "order_item_id": pd.Series(["1", "2", "3", "4", "5", "6"], dtype="string"),
            "customer_id": pd.Series(["C1", "C2", "C2", "C1", "C3", "C4"], dtype="string"),
            "order_date_dateorders": pd.to_datetime(
                [
                    "2018-01-05 14:30:00",  # Friday 14:30
                    "2018-01-06 21:15:00",  # Saturday 21:15 (weekend + after hours)
                    "2018-01-06 21:15:00",
                    "2018-01-07 06:00:00",  # Sunday 06:00 (weekend + after hours)
                    "2018-01-08 10:00:00",  # Monday 10:00
                    "2018-01-09 16:45:00",  # Tuesday 16:45
                ]
            ),
            # Leakage columns (should be dropped by preprocess)
            "shipping_delay_days": [-1, 1, 1, 0, 2, -2],
            "shipping_date_dateorders": pd.to_datetime(["2018-01-08"] * 6),
            "delivery_status": pd.Series(["Advance shipping"] * 6, dtype="category"),
            "late_delivery_risk": pd.Series([0, 1, 1, 0, 1, 0], dtype="category"),
            "order_status": pd.Series(["COMPLETE"] * 6, dtype="category"),
            # Missing / Constant / PII / Duplicates / Collinear (should be dropped by preprocess)
            "product_description": [None] * 6,
            "order_zipcode": [None] * 6,
            "product_status": pd.Series([0] * 6, dtype="category"),
            "customer_email": ["XXXXXXXXX"] * 6,
            "order_profit_per_order": [45.0, -5.0, 8.55, 48.0, -54.0, 16.0],
            "sales_per_customer": [180.0, 50.0, 57.0, 160.0, 108.0, 160.0],
            "order_item_product_price": [100.0, 50.0, 20.0, 200.0, 30.0, 80.0],
            "sales": [200.0, 50.0, 60.0, 200.0, 120.0, 160.0],
            "order_item_discount": [20.0, 0.0, 3.0, 40.0, 12.0, 0.0],
            "order_item_total": [180.0, 50.0, 57.0, 160.0, 108.0, 160.0],
            "benefit_per_order": [45.0, -5.0, 8.55, 48.0, -54.0, 16.0],
            "shipping_mode": pd.Series(["Standard Class"] * 6, dtype="category"),
        }
    )


def test_get_columns_to_drop_respects_flags() -> None:
    """Verify get_columns_to_drop toggles collinear and entity_and_time_keys properly."""
    cfg = load_config()
    base_drop = get_columns_to_drop(config=cfg, include_collinear=False, include_entity_keys=False)
    collinear_drop = get_columns_to_drop(config=cfg, include_collinear=True, include_entity_keys=False)
    full_drop = get_columns_to_drop(config=cfg, include_collinear=True, include_entity_keys=True)

    # Leakage and duplicates are always in base_drop
    assert "shipping_delay_days" in base_drop
    assert "days_for_shipping_real" not in base_drop
    assert "delivery_status" in base_drop
    assert "late_delivery_risk" in base_drop
    assert "order_profit_per_order" in base_drop

    # Collinear only when include_collinear=True
    assert "sales" not in base_drop
    assert "sales" in collinear_drop

    # Entity keys only when include_entity_keys=True
    assert "order_id" not in collinear_drop
    assert "order_date_dateorders" not in collinear_drop
    assert "order_id" in full_drop
    assert "order_date_dateorders" in full_drop


def test_clean_and_preprocess_removes_leakage_and_retains_keys(
    sample_preprocessed_df: pd.DataFrame,
) -> None:
    """Verify clean_and_preprocess drops leakage/collinear columns while keeping entity keys and target."""
    cleaned = clean_and_preprocess(
        sample_preprocessed_df, include_collinear=True, include_entity_keys=False
    )

    # Target and entity keys must still be present
    assert "days_for_shipping_real" in cleaned.columns
    assert "order_id" in cleaned.columns
    assert "customer_id" in cleaned.columns
    assert "order_date_dateorders" in cleaned.columns

    # Leakage, PII, duplicates, and collinear columns must be gone
    for dropped_col in [
        "shipping_delay_days",
        "shipping_date_dateorders",
        "delivery_status",
        "late_delivery_risk",
        "order_status",
        "product_description",
        "customer_email",
        "order_profit_per_order",
        "sales",
        "order_item_total",
        "benefit_per_order",
        "days_for_shipment_scheduled",
    ]:
        assert dropped_col not in cleaned.columns


def test_split_dataset_prevents_order_id_leakage(
    sample_preprocessed_df: pd.DataFrame,
) -> None:
    """Verify group-aware splitting keeps all rows with the same order_id in the same split."""
    cleaned = clean_and_preprocess(sample_preprocessed_df)
    train_df, val_df, test_df = split_dataset(
        cleaned, test_size=0.25, val_size=0.25, random_state=42, group_col="order_id"
    )

    train_orders = set(train_df["order_id"])
    val_orders = set(val_df["order_id"])
    test_orders = set(test_df["order_id"])

    # Disjoint order_id sets across train, val, and test
    assert train_orders.isdisjoint(val_orders)
    assert train_orders.isdisjoint(test_orders)
    assert val_orders.isdisjoint(test_orders)
    assert len(train_df) + len(val_df) + len(test_df) == len(cleaned)


def test_split_dataset_chronological_preserves_temporal_order(
    sample_preprocessed_df: pd.DataFrame,
) -> None:
    """Verify chronological splitting orders splits strictly in time with zero order_id overlap."""
    cleaned = clean_and_preprocess(sample_preprocessed_df)
    train_df, val_df, test_df = split_dataset(
        cleaned,
        test_size=0.20,
        val_size=0.20,
        group_col="order_id",
        split_type="chronological",
    )

    train_orders = set(train_df["order_id"])
    val_orders = set(val_df["order_id"])
    test_orders = set(test_df["order_id"])

    assert train_orders.isdisjoint(val_orders)
    assert train_orders.isdisjoint(test_orders)
    assert val_orders.isdisjoint(test_orders)
    assert len(train_df) + len(val_df) + len(test_df) == len(cleaned)
    assert train_df["order_date_dateorders"].max() <= val_df["order_date_dateorders"].min()
    assert val_df["order_date_dateorders"].max() <= test_df["order_date_dateorders"].min()


def test_feature_engineering_calculations_and_leakage_free_transform(
    sample_preprocessed_df: pd.DataFrame,
) -> None:
    """Verify SupplyChainFeatureBuilder computes expected features and drops entity keys."""
    cleaned = clean_and_preprocess(sample_preprocessed_df, include_collinear=True)
    builder = SupplyChainFeatureBuilder(drop_entity_keys=True)
    feat_df = builder.fit_transform(cleaned)

    # Entity and time keys should be dropped after feature building
    for key_col in ["order_id", "order_item_id", "customer_id", "order_date_dateorders"]:
        assert key_col not in feat_df.columns

    # Check temporal features on row 0 (2018-01-05 14:30:00 -> Friday, dow=4, hour=14, doy=5, q=1)
    assert feat_df.loc[0, "order_year"] == 2018
    assert feat_df.loc[0, "order_quarter"] == 1
    assert feat_df.loc[0, "order_month"] == 1
    assert feat_df.loc[0, "order_day"] == 5
    assert feat_df.loc[0, "order_dayofyear"] == 5
    assert feat_df.loc[0, "order_dayofweek"] == 4
    assert feat_df.loc[0, "order_hour"] == 14
    assert feat_df.loc[0, "is_weekend_order"] == 0
    assert feat_df.loc[0, "is_after_hours_order"] == 0
    assert feat_df.loc[0, "is_q4_peak_season"] == 0
    assert np.isclose(feat_df.loc[0, "order_month_sin"], np.sin(2.0 * np.pi * 1.0 / 12.0))
    assert np.isclose(feat_df.loc[0, "order_month_cos"], np.cos(2.0 * np.pi * 1.0 / 12.0))
    assert np.isclose(feat_df.loc[0, "order_doy_sin"], np.sin(2.0 * np.pi * 5.0 / 365.25))
    assert np.isclose(feat_df.loc[0, "order_doy_cos"], np.cos(2.0 * np.pi * 5.0 / 365.25))

    # Row 1 (Saturday 21:15 -> weekend=1, after_hours=1)
    assert feat_df.loc[1, "is_weekend_order"] == 1
    assert feat_df.loc[1, "is_after_hours_order"] == 1

    # Check financial ratio features
    # Row 0: price=100, discount_rate=0.10 -> effective_unit_price = 90.0, profit_ratio=0.25 -> not loss
    assert np.isclose(feat_df.loc[0, "effective_unit_price"], 90.0)
    assert feat_df.loc[0, "is_loss_making_item"] == 0
    # Row 1: profit_ratio = -0.10 -> loss making item = 1
    assert feat_df.loc[1, "is_loss_making_item"] == 1

    # Check basket features on order "102" (rows 1 and 2 have quantities 1 and 3 -> total_units=4, count=2)
    assert feat_df.loc[1, "order_line_item_count"] == 2
    assert feat_df.loc[2, "order_line_item_count"] == 2
    assert feat_df.loc[1, "order_total_units"] == 4
    assert np.isclose(feat_df.loc[1, "item_share_of_order_units"], 0.25)
    assert np.isclose(feat_df.loc[2, "item_share_of_order_units"], 0.75)

    # Check geographic features
    # Row 0: customer_country="Puerto Rico", order_country="Estados Unidos"
    assert feat_df.loc[0, "origin_is_puerto_rico"] == 1
    assert feat_df.loc[0, "is_cross_border_dest"] == 0
    # Row 1: customer_country="EE. UU.", order_country="Francia"
    assert feat_df.loc[1, "origin_is_puerto_rico"] == 0
    assert feat_df.loc[1, "is_cross_border_dest"] == 1

    # Verify historical lookup features and high-cardinality raw columns are excluded
    assert "customer_distinct_orders" not in feat_df.columns
    assert "dest_country_log_volume" not in feat_df.columns
    assert "order_country" not in feat_df.columns


def test_full_dataset_preprocess_and_feature_pipeline() -> None:
    """Integration test verifying the full 180,519-row dataset runs through preprocess and build_features."""
    raw_df = load_raw_data()
    clean_df = clean_and_preprocess(raw_df)

    # 54 raw columns - 29 dropped/pruned columns = 25 columns retained in clean_df
    assert clean_df.shape == (180519, 25)
    assert "days_for_shipping_real" in clean_df.columns
    assert "shipping_delay_days" not in clean_df.columns
    assert "late_delivery_risk" not in clean_df.columns
    assert "days_for_shipment_scheduled" not in clean_df.columns
    assert "shipping_mode" in clean_df.columns

    train_df, val_df, test_df = split_dataset(clean_df)
    assert set(train_df["order_id"]).isdisjoint(set(test_df["order_id"]))
    assert set(train_df["order_id"]).isdisjoint(set(val_df["order_id"]))

    builder = SupplyChainFeatureBuilder(drop_entity_keys=True)
    X_train = builder.fit_transform(train_df)
    X_test = builder.transform(test_df)

    assert X_train.shape[0] == train_df.shape[0]
    assert X_test.shape[0] == test_df.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert "order_id" not in X_train.columns
    assert "order_hour_sin" in X_train.columns
    assert "order_total_units" in X_train.columns
    assert "effective_unit_price" in X_train.columns
