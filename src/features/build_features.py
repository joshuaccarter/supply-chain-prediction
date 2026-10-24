"""Functions and scikit-learn transformer for building and selecting model features.

This module extracts self-contained features directly from order attributes without
relying on historical lookups or future leakage information. Features include cyclical
calendar encodings, orthogonalized financial profit and discount ratios, intra-order
basket aggregations, and origin fulfillment / cross-border destination route indicators.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.config_loader import DEFAULT_CONFIG_PATH, load_config

ORIGIN_COUNTRIES: tuple[str, ...] = ("Estados Unidos", "Puerto Rico", "EE. UU.")
HIGH_CARD_COLS_DEFAULT: tuple[str, ...] = (
    "customer_state",
    "category_id",
    "product_card_id",
    "order_country",
    "customer_city",
    "order_state",
    "order_city",
)


def add_temporal_features(
    df: pd.DataFrame, date_col: str = "order_date_dateorders"
) -> pd.DataFrame:
    """Extract calendar and cyclical time features from the order timestamp.

    Computes calendar components (year, quarter, month, day, day of year, day of week, hour),
    binary peak indicators (weekend order, after-hours checkout, Q4 peak holiday season),
    and smooth sine/cosine cyclical harmonic encodings for hour of day (24h period),
    day of week (7d period), month of year (12m period), and day of year (365.25d period).

    Args:
        df: Input pandas DataFrame containing the order timestamp column.
        date_col: Name of the datetime column to extract temporal components from.

    Returns:
        pd.DataFrame: DataFrame copy with added calendar and cyclical feature columns.
    """
    out = df.copy()
    if date_col not in out.columns:
        return out

    dt = pd.to_datetime(out[date_col], errors="coerce")
    out["order_year"] = dt.dt.year.astype("int64")
    out["order_quarter"] = dt.dt.quarter.astype("int64")
    out["order_month"] = dt.dt.month.astype("int64")
    out["order_day"] = dt.dt.day.astype("int64")
    out["order_dayofyear"] = dt.dt.day_of_year.astype("int64")
    out["order_dayofweek"] = dt.dt.day_of_week.astype("int64")
    out["order_hour"] = dt.dt.hour.astype("int64")
    out["is_weekend_order"] = (out["order_dayofweek"] >= 5).astype("int64")
    out["is_after_hours_order"] = (
        (out["order_hour"] < 8) | (out["order_hour"] >= 18)
    ).astype("int64")
    out["is_q4_peak_season"] = (out["order_month"] >= 10).astype("int64")

    # Cyclical encodings for hour of day, day of week, month of year, and day of year
    out["order_hour_sin"] = np.sin(2.0 * np.pi * out["order_hour"] / 24.0)
    out["order_hour_cos"] = np.cos(2.0 * np.pi * out["order_hour"] / 24.0)
    out["order_dow_sin"] = np.sin(2.0 * np.pi * out["order_dayofweek"] / 7.0)
    out["order_dow_cos"] = np.cos(2.0 * np.pi * out["order_dayofweek"] / 7.0)
    out["order_month_sin"] = np.sin(2.0 * np.pi * out["order_month"] / 12.0)
    out["order_month_cos"] = np.cos(2.0 * np.pi * out["order_month"] / 12.0)
    out["order_doy_sin"] = np.sin(2.0 * np.pi * out["order_dayofyear"] / 365.25)
    out["order_doy_cos"] = np.cos(2.0 * np.pi * out["order_dayofyear"] / 365.25)

    return out


def add_financial_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute orthogonalized unit pricing and expected profitability features.

    Works robustly even after collinear columns (`sales`, `order_item_total`,
    `benefit_per_order`) have been pruned, deriving net effective unit price
    (`product_price * (1 - discount_rate)`) and binary margin indicators
    (`is_loss_making_item`).

    Args:
        df: Input pandas DataFrame containing checkout pricing and profit parameters.

    Returns:
        pd.DataFrame: DataFrame copy with `effective_unit_price` and `is_loss_making_item`.
    """
    out = df.copy()
    has_price = "product_price" in out.columns
    has_disc = "order_item_discount_rate" in out.columns
    has_qty = "order_item_quantity" in out.columns
    has_profit_ratio = "order_item_profit_ratio" in out.columns

    if has_price and has_disc:
        out["effective_unit_price"] = out["product_price"] * (
            1.0 - out["order_item_discount_rate"]
        )
    elif "order_item_total" in out.columns and has_qty:
        out["effective_unit_price"] = out["order_item_total"] / out[
            "order_item_quantity"
        ].replace(0, np.nan)

    if has_profit_ratio:
        out["is_loss_making_item"] = (out["order_item_profit_ratio"] < 0.0).astype(
            "int64"
        )
    elif "benefit_per_order" in out.columns:
        out["is_loss_making_item"] = (out["benefit_per_order"] < 0.0).astype("int64")

    return out


def add_basket_features(df: pd.DataFrame, order_col: str = "order_id") -> pd.DataFrame:
    """Compute order-level basket aggregations across line items sharing an `order_id`.

    Aggregates multi-line checkout baskets to measure total items in the shipment
    (`order_line_item_count`), cumulative units across all lines (`order_total_units`),
    total net revenue of the basket (`order_total_sales`), and current line unit share
    (`item_share_of_order_units`). Falls back cleanly to 1-item basket defaults for
    standalone items.

    Args:
        df: Input pandas DataFrame containing line items.
        order_col: Column name identifying common order groupings (default: "order_id").

    Returns:
        pd.DataFrame: DataFrame copy with order basket aggregation metrics added.
    """
    out = df.copy()

    # Determine net item sales from available columns
    if "order_item_total" in out.columns:
        net_item_sales = out["order_item_total"].astype(float)
    elif (
        "product_price" in out.columns
        and "order_item_quantity" in out.columns
        and "order_item_discount_rate" in out.columns
    ):
        net_item_sales = (
            out["product_price"].astype(float)
            * out["order_item_quantity"].astype(float)
            * (1.0 - out["order_item_discount_rate"].astype(float))
        )
    elif "product_price" in out.columns and "order_item_quantity" in out.columns:
        net_item_sales = out["product_price"].astype(float) * out[
            "order_item_quantity"
        ].astype(float)
    else:
        net_item_sales = pd.Series(0.0, index=out.index)

    out["_net_item_sales"] = net_item_sales

    if order_col in out.columns and "order_item_quantity" in out.columns:
        grp = out.groupby(order_col, observed=True)
        out["order_line_item_count"] = (
            grp["order_item_quantity"].transform("count").astype("int64")
        )
        out["order_total_units"] = (
            grp["order_item_quantity"].transform("sum").astype("int64")
        )
        out["order_total_sales"] = (
            grp["_net_item_sales"].transform("sum").astype("float64")
        )
        out["item_share_of_order_units"] = (
            out["order_item_quantity"] / out["order_total_units"].replace(0, 1)
        ).astype("float64")
    elif "order_item_quantity" in out.columns:
        out["order_line_item_count"] = 1
        out["order_total_units"] = out["order_item_quantity"].astype("int64")
        out["order_total_sales"] = out["_net_item_sales"].astype("float64")
        out["item_share_of_order_units"] = 1.0

    return out.drop(columns=["_net_item_sales"])


def add_geographic_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add origin fulfillment and cross-border destination route indicators.

    Identifies whether fulfillment originates from Puerto Rico (`origin_is_puerto_rico`)
    and whether the destination delivery country requires international cross-border
    customs transit (`is_cross_border_dest`).

    Args:
        df: Input pandas DataFrame containing geographic location fields.

    Returns:
        pd.DataFrame: DataFrame copy with `origin_is_puerto_rico` and `is_cross_border_dest`.
    """
    out = df.copy()

    if "customer_country" in out.columns:
        out["origin_is_puerto_rico"] = (
            out["customer_country"].astype(str) == "Puerto Rico"
        ).astype("int64")

    if "order_country" in out.columns:
        order_country_str = out["order_country"].astype(str)
        out["is_cross_border_dest"] = (
            ~order_country_str.isin(ORIGIN_COUNTRIES)
        ).astype("int64")

    return out


def drop_entity_and_time_keys(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Drop raw entity IDs and timestamp columns after feature extraction.

    Removes high-cardinality entity keys (`order_id`, `order_item_id`, `customer_id`)
    and raw datetime stamps (`order_date_dateorders`) that are no longer needed
    after temporal and basket feature computation.

    Args:
        df: Input DataFrame with engineered features.
        config: Optional pre-loaded configuration dictionary.
        config_path: Path to the YAML configuration file if `config` is not provided.

    Returns:
        pd.DataFrame: DataFrame copy with configured entity and timestamp keys dropped.
    """
    if config is None:
        config = load_config(config_path)

    entity_keys: list[str] = config.get("preprocessing", {}).get(
        "entity_and_time_keys",
        ["order_id", "order_item_id", "customer_id", "order_date_dateorders"],
    )
    existing = [c for c in entity_keys if c in df.columns]
    return df.drop(columns=existing).copy()


class SupplyChainFeatureBuilder(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer that builds self-contained supply chain features.

    Computes temporal/seasonal cycles, financial ratios, order-level basket aggregations,
    and origin/cross-border route indicators directly from order fields without relying on
    historical lookup tables (`customer_distinct_orders`, `dest_country_log_volume`, or
    high-cardinality frequency maps) that would be unavailable at API inference time.

    Attributes:
        drop_entity_keys: Whether to drop identifier and timestamp columns during transform.
        entity_keys: Specific list of entity/time column names to remove if drop_entity_keys is True.
        drop_high_card_cols: Whether to drop high-cardinality geographic columns.
        high_card_cols: Specific list of high-cardinality column names to remove.
        entity_keys_: Cached list of entity keys resolved during `fit`.
        high_card_cols_: Cached list of high-cardinality column names resolved during `fit`.
    """

    def __init__(
        self,
        drop_entity_keys: bool = True,
        entity_keys: list[str] | None = None,
        drop_high_card_cols: bool = True,
        high_card_cols: list[str] | None = None,
    ) -> None:
        """Initialize the SupplyChainFeatureBuilder transformer.

        Args:
            drop_entity_keys: Whether to drop entity and timestamp keys after extraction.
            entity_keys: Explicit list of entity key names. Defaults to config if None.
            drop_high_card_cols: Whether to drop high-cardinality string/location columns.
            high_card_cols: Explicit list of high-cardinality column names. Defaults to
                HIGH_CARD_COLS_DEFAULT if None.
        """
        self.drop_entity_keys = drop_entity_keys
        self.entity_keys = entity_keys
        self.drop_high_card_cols = drop_high_card_cols
        self.high_card_cols = high_card_cols

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "SupplyChainFeatureBuilder":
        """Configure entity and high-cardinality columns to drop after feature extraction.

        Args:
            X: Training DataFrame to inspect.
            y: Ignored (scikit-learn transformer protocol compatibility).

        Returns:
            SupplyChainFeatureBuilder: Fitted transformer instance.
        """
        if self.entity_keys is not None:
            self.entity_keys_: list[str] = list(self.entity_keys)
        else:
            cfg = load_config()
            self.entity_keys_ = cfg.get("preprocessing", {}).get(
                "entity_and_time_keys",
                ["order_id", "order_item_id", "customer_id", "order_date_dateorders"],
            )

        if self.high_card_cols is not None:
            self.high_card_cols_: list[str] = list(self.high_card_cols)
        else:
            self.high_card_cols_ = list(HIGH_CARD_COLS_DEFAULT)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all self-contained feature engineering transformations to a split DataFrame.

        Args:
            X: Input DataFrame containing order characteristics.

        Returns:
            pd.DataFrame: DataFrame containing engineered temporal, ratio, basket, and
                geographic features, with entity and high-cardinality columns pruned as configured.
        """
        out = X.copy()
        out = add_temporal_features(out)
        out = add_financial_ratio_features(out)
        out = add_basket_features(out)
        out = add_geographic_features(out)

        if self.drop_entity_keys:
            keys_to_drop = [
                c
                for c in getattr(
                    self,
                    "entity_keys_",
                    [
                        "order_id",
                        "order_item_id",
                        "customer_id",
                        "order_date_dateorders",
                    ],
                )
                if c in out.columns
            ]
            out = out.drop(columns=keys_to_drop)

        if self.drop_high_card_cols:
            hc_to_drop = [
                c
                for c in getattr(self, "high_card_cols_", list(HIGH_CARD_COLS_DEFAULT))
                if c in out.columns
            ]
            out = out.drop(columns=hc_to_drop)

        return out


def build_features(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    drop_entity_keys: bool = True,
) -> pd.DataFrame:
    """Convenience function to fit and apply `SupplyChainFeatureBuilder` on a DataFrame.

    Args:
        df: Preprocessed DataFrame.
        config: Optional pre-loaded configuration dictionary.
        config_path: Path to the YAML configuration file if `config` is not provided.
        drop_entity_keys: Whether to drop raw entity and timestamp keys after building features.

    Returns:
        pd.DataFrame: DataFrame with engineered features ready for modeling.
    """
    if config is None:
        config = load_config(config_path)

    entity_keys = config.get("preprocessing", {}).get(
        "entity_and_time_keys",
        ["order_id", "order_item_id", "customer_id", "order_date_dateorders"],
    )
    builder = SupplyChainFeatureBuilder(
        drop_entity_keys=drop_entity_keys,
        entity_keys=entity_keys,
    )
    return builder.fit_transform(df)
