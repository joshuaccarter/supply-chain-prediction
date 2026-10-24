"""Functions for loading, parsing, and cleaning the raw DataCo Supply Chain dataset.

This module reads the source CSV dataset according to configurations specified in
`config/config.yaml`, standardizes column headers to snake_case format, parses datetime
stamps, casts designated data types, and computes the delayed shipping duration target.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import DEFAULT_CONFIG_PATH, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def to_snake_case(column_name: str) -> str:
    """Convert an arbitrary column name string to standardized snake_case format.

    Strips non-alphanumeric punctuation and special characters, trims whitespace,
    and replaces remaining whitespace sequences with a single underscore.

    Args:
        column_name: Original raw column name string.

    Returns:
        Standardized lowercase snake_case column name string.

    Examples:
        >>> to_snake_case("Days for shipping (real)")
        'days_for_shipping_real'
        >>> to_snake_case("Order Region")
        'order_region'
    """
    cleaned = re.sub(r"[^\w\s]", "", column_name)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned.lower()


def compute_target_metric(df: pd.DataFrame, target_cfg: dict[str, Any]) -> pd.DataFrame:
    """Compute and format the shipping transit duration and delay target columns.

    Calculates the actual shipping transit delay (`shipping_delay_days` =
    `days_for_shipping_real` - `days_for_shipment_scheduled`), where early arrival
    produces a negative integer, on-time delivery produces 0, and late arrival
    produces a positive integer. Also ensures the primary target metric
    (`days_for_shipping_real`) is cast to its configured integer data type.

    Args:
        df: Pandas DataFrame containing snake_case normalized columns.
        target_cfg: Dictionary containing target configuration extracted from config.yaml.

    Returns:
        pd.DataFrame: The DataFrame with `shipping_delay_days` and target types applied.
    """
    if (
        "days_for_shipping_real" in df.columns
        and "days_for_shipment_scheduled" in df.columns
    ):
        df["shipping_delay_days"] = (
            df["days_for_shipping_real"] - df["days_for_shipment_scheduled"]
        ).astype("int64")

    if not target_cfg:
        return df

    target_name = target_cfg.get("name", "days_for_shipping_real")
    target_dtype = target_cfg.get("dtype", "int64")

    if target_name in df.columns:
        df[target_name] = df[target_name].astype(target_dtype)

    return df


def load_raw_data(
    config: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Load the raw DataCo Supply Chain dataset using YAML configuration settings.

    Reads the raw dataset from disk using the configured encoding (`latin1`),
    normalizes all raw column names to canonical snake_case identifiers, enforces
    configured column dtypes, parses configured date and timestamp columns into
    `pandas.Timestamp` objects, and computes the target metric (`days_for_shipping_real`
    and `shipping_delay_days`).

    Args:
        config: Optional pre-loaded configuration dictionary. If None, `load_config`
            is called with `config_path`.
        config_path: Path to the YAML configuration file if `config` is not supplied.

    Returns:
        pd.DataFrame: Loaded, parsed, and normalized pandas DataFrame.

    Raises:
        FileNotFoundError: If the raw CSV data file cannot be located at the configured path.
    """
    if config is None:
        config = load_config(config_path)

    data_cfg = config.get("data", {})
    raw_rel_path = data_cfg.get("raw_path", "data/raw/DataCoSupplyChainDataset.csv")
    raw_path = Path(raw_rel_path)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    encoding = data_cfg.get("encoding", "latin1")
    columns_cfg = data_cfg.get("columns", {})

    # Build mapping for read_csv dtypes (excluding datetime columns handled after/during load)
    raw_dtypes: dict[str, str] = {}
    rename_map: dict[str, str] = {}
    datetime_cols: list[str] = []

    for raw_col, col_meta in columns_cfg.items():
        if isinstance(col_meta, dict):
            snake_name = col_meta.get("name", to_snake_case(raw_col))
            dtype = col_meta.get("dtype", "string")
        else:
            snake_name = to_snake_case(raw_col)
            dtype = str(col_meta)

        rename_map[raw_col] = snake_name

        if dtype.startswith("datetime"):
            datetime_cols.append(snake_name)
        else:
            raw_dtypes[raw_col] = dtype

    # Read CSV with specified encoding and column dtypes
    df = pd.read_csv(
        raw_path,
        encoding=encoding,
        dtype=raw_dtypes if raw_dtypes else None,
        low_memory=False,
    )

    # Rename mapped columns and normalize any unmapped columns to snake_case
    df = df.rename(columns=rename_map)
    df.columns = [
        col if col in rename_map.values() else to_snake_case(col) for col in df.columns
    ]

    # Parse configured datetime columns
    configured_dates = set(datetime_cols) | set(data_cfg.get("date_columns", []))
    for date_col in configured_dates:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Automatically compute target metric if configured
    target_cfg = data_cfg.get("target", {})
    df = compute_target_metric(df, target_cfg)

    return df


if __name__ == "__main__":
    dataset = load_raw_data()
    print(f"Loaded dataset with shape: {dataset.shape}")
    print(
        dataset[
            [
                "days_for_shipping_real",
                "days_for_shipment_scheduled",
                "shipping_delay_days",
            ]
        ].head()
    )
