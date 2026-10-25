"""Unit tests for data loading and schema configuration."""

import re
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import load_raw_data, to_snake_case
from src.utils.config_loader import load_config


@pytest.mark.parametrize(
    "raw_name,expected",
    [
        ("Days for shipping (real)", "days_for_shipping_real"),
        ("order date (DateOrders)", "order_date_dateorders"),
        ("Late_delivery_risk", "late_delivery_risk"),
        ("Order Item Product Price", "order_item_product_price"),
        ("Benefit per order!", "benefit_per_order"),
        ("  Multiple   Spaces & Symbols@# ", "multiple_spaces_symbols"),
    ],
)
def test_to_snake_case(raw_name: str, expected: str) -> None:
    """Verify column names are converted to snake_case without punctuation."""
    assert to_snake_case(raw_name) == expected


def test_load_raw_data_with_mock_csv(tmp_path: Path) -> None:
    """Verify load_raw_data handles latin1 accented characters, dtypes, dates, and target calculation."""
    csv_file = tmp_path / "sample.csv"
    # Include accented characters (e.g., Rajastán, Bogotá) encoded in latin1
    csv_content = (
        "Type,Days for shipping (real),Days for shipment (scheduled),Benefit per order,Order State,order date (DateOrders)\n"
        "DEBIT,3,4,91.25,Rajastán,1/31/2018 22:56\n"
        "TRANSFER,5,4,-249.09,Bogotá,1/13/2018 12:27\n"
        "CASH,2,2,50.00,Caguas,1/15/2018 08:00\n"
    )
    csv_file.write_bytes(csv_content.encode("latin1"))

    mock_config = {
        "data": {
            "raw_path": str(csv_file),
            "encoding": "latin1",
            "target": {
                "name": "shipping_delay_days",
                "dtype": "int64",
                "actual_days_col": "days_for_shipping_real",
                "scheduled_days_col": "days_for_shipment_scheduled",
            },
            "date_columns": ["order_date_dateorders"],
            "columns": {
                "Type": {"name": "type", "dtype": "category"},
                "Days for shipping (real)": {
                    "name": "days_for_shipping_real",
                    "dtype": "int64",
                },
                "Days for shipment (scheduled)": {
                    "name": "days_for_shipment_scheduled",
                    "dtype": "int64",
                },
                "Benefit per order": {"name": "benefit_per_order", "dtype": "float64"},
                "Order State": {"name": "order_state", "dtype": "string"},
                "order date (DateOrders)": {
                    "name": "order_date_dateorders",
                    "dtype": "datetime64[ns]",
                },
            },
        }
    }

    df = load_raw_data(config=mock_config)

    assert df.shape == (3, 7)
    assert list(df.columns) == [
        "type",
        "days_for_shipping_real",
        "days_for_shipment_scheduled",
        "benefit_per_order",
        "order_state",
        "order_date_dateorders",
        "shipping_delay_days",
    ]
    assert isinstance(df["type"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_integer_dtype(df["days_for_shipping_real"])
    assert pd.api.types.is_integer_dtype(df["days_for_shipment_scheduled"])
    assert pd.api.types.is_float_dtype(df["benefit_per_order"])
    assert pd.api.types.is_string_dtype(df["order_state"])
    assert pd.api.types.is_datetime64_any_dtype(df["order_date_dateorders"])
    assert pd.api.types.is_integer_dtype(df["shipping_delay_days"])
    assert df.loc[0, "order_state"] == "Rajastán"
    assert df.loc[1, "order_state"] == "Bogotá"

    # Verify target calculation: real - scheduled (early < 0, delayed > 0, on-time == 0)
    assert df["shipping_delay_days"].tolist() == [-1, 1, 0]


def test_load_raw_data_missing_file() -> None:
    """Verify FileNotFoundError is raised when the raw CSV path does not exist."""
    bad_config = {"data": {"raw_path": "data/raw/non_existent_file.csv"}}
    with pytest.raises(FileNotFoundError):
        load_raw_data(config=bad_config)


def test_load_actual_raw_dataset() -> None:
    """Integration test verifying the full raw dataset loads according to config.yaml."""
    config = load_config()
    df = load_raw_data(config=config)

    assert not df.empty
    assert df.shape == (180519, 54)

    # Ensure all column names are strictly lowercase snake_case without special characters
    snake_case_pattern = re.compile(r"^[a-z0-9_]+$")
    for col in df.columns:
        assert snake_case_pattern.match(col), f"Column '{col}' is not valid snake_case"

    # Verify configured date columns are parsed as datetime
    for date_col in config["data"]["date_columns"]:
        assert pd.api.types.is_datetime64_any_dtype(df[date_col])

    # Verify target column exists and contains valid actual shipping days in [0, 6]
    target_col = config["data"]["target"]["name"]
    assert target_col == "days_for_shipping_real"
    assert target_col in df.columns
    assert df[target_col].between(0, 6).all()
    assert pd.api.types.is_integer_dtype(df[target_col])
