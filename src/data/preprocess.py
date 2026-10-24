"""Functions for data preprocessing, column pruning, and group-aware dataset splitting.

This module provides data cleaning routines that prune leakage columns, invariant/constant
attributes, high-cardinality metadata, and collinear features. It also implements both
group-aware random partitioning and out-of-time chronological splitting that preserve
order-level boundaries across train, validation, and test sets.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.utils.config_loader import DEFAULT_CONFIG_PATH, load_config

DROP_CATEGORIES_DEFAULT: tuple[str, ...] = (
    "drop_leakage",
    "drop_missing_or_constant",
    "drop_pii_or_metadata",
    "drop_exact_duplicates",
)


def get_columns_to_drop(
    config: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    include_collinear: bool = True,
    include_entity_keys: bool = False,
) -> list[str]:
    """Retrieve the deduplicated list of columns to drop/prune from configuration.

    Reads categories defined under `preprocessing` in `config/config.yaml`
    (`drop_leakage`, `drop_missing_or_constant`, `drop_pii_or_metadata`,
    `drop_exact_duplicates`, and optionally `prune_collinear` and `entity_and_time_keys`).

    Args:
        config: Optional pre-loaded configuration dictionary.
        config_path: Path to the YAML configuration file if `config` is not provided.
        include_collinear: Whether to include algebraically/deterministically collinear
            columns (`prune_collinear`) in the drop list.
        include_entity_keys: Whether to also include `entity_and_time_keys` (`order_id`,
            `order_item_id`, `customer_id`, `order_date_dateorders`). Kept False during
            initial preprocessing so keys remain available for group splitting and
            feature engineering.

    Returns:
        list[str]: Ordered list of unique column names configured for removal.
    """
    if config is None:
        config = load_config(config_path)

    prep_cfg = config.get("preprocessing", {})
    categories = list(DROP_CATEGORIES_DEFAULT)
    if include_collinear:
        categories.append("prune_collinear")
    if include_entity_keys:
        categories.append("entity_and_time_keys")

    cols_to_drop: list[str] = []
    seen: set[str] = set()
    for category in categories:
        for col in prep_cfg.get(category, []):
            if col not in seen:
                seen.add(col)
                cols_to_drop.append(col)

    return cols_to_drop


def drop_and_prune_columns(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    include_collinear: bool = True,
    include_entity_keys: bool = False,
) -> pd.DataFrame:
    """Drop target leakage, missing/constant, PII, duplicate, and collinear columns.

    Args:
        df: Input DataFrame (typically from `load_raw_data`).
        config: Optional pre-loaded configuration dictionary.
        config_path: Path to the YAML configuration file if `config` is not provided.
        include_collinear: Whether to prune collinear columns (`prune_collinear`).
        include_entity_keys: Whether to drop entity and timestamp keys (`entity_and_time_keys`).

    Returns:
        pd.DataFrame: New DataFrame with the specified columns removed.
    """
    cols_to_drop = get_columns_to_drop(
        config=config,
        config_path=config_path,
        include_collinear=include_collinear,
        include_entity_keys=include_entity_keys,
    )
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    return df.drop(columns=existing_cols).copy()


def clean_and_preprocess(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    include_collinear: bool = True,
    include_entity_keys: bool = False,
) -> pd.DataFrame:
    """Clean and prune the raw dataset for downstream splitting and feature engineering.

    Drops leakage, missing/constant, PII, exact duplicates, and (optionally) collinear
    columns, and strips unobserved category levels from categorical columns.

    Args:
        df: Raw DataFrame loaded by `load_raw_data`.
        config: Optional pre-loaded configuration dictionary.
        config_path: Path to the YAML configuration file if `config` is not provided.
        include_collinear: Whether to prune collinear columns (`prune_collinear`).
        include_entity_keys: Whether to drop entity and timestamp keys (`entity_and_time_keys`).

    Returns:
        pd.DataFrame: Cleaned and pruned pandas DataFrame ready for splitting.
    """
    cleaned = drop_and_prune_columns(
        df=df,
        config=config,
        config_path=config_path,
        include_collinear=include_collinear,
        include_entity_keys=include_entity_keys,
    )

    # Remove unused categories on categorical columns if any exist
    for col in cleaned.select_dtypes(include=["category"]).columns:
        cleaned[col] = cleaned[col].cat.remove_unused_categories()

    return cleaned


def split_dataset(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    test_size: float | None = None,
    val_size: float | None = None,
    random_state: int | None = None,
    group_col: str | None = None,
    split_type: str | None = None,
    date_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test DataFrames using group-aware or chronological splitting.

    Because multiple order line items share the same `order_id` and identical shipping
    outcomes, splitting by `order_id` prevents cross-split order leakage. When
    `split_type="chronological"` (or `"time_series"`), orders are sorted chronologically
    by `date_col` (`order_date_dateorders`) before partitioning into past (train),
    subsequent (val), and future (test) horizons while keeping `order_id` groups intact.

    Args:
        df: Preprocessed DataFrame containing the group and/or date column.
        config: Optional pre-loaded configuration dictionary.
        config_path: Path to the YAML configuration file if `config` is not provided.
        test_size: Fraction of total data to allocate to the test split (default from YAML: 0.20).
        val_size: Fraction of total data to allocate to the validation split (default from YAML: 0.10).
        random_state: Random seed for reproducible splitting (default from YAML: 42).
        group_col: Column name to group on (default from YAML: "order_id").
        split_type: Splitting strategy: `"group_random"` (default) or `"chronological"` / `"time_series"`.
        date_col: Timestamp column name used when `split_type` is chronological
            (default: `"order_date_dateorders"`).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple of `(train_df, val_df, test_df)` with reset indices.

    Raises:
        ValueError: If split fractions are non-positive or exceed 1.0, or if the specified date column
            is missing when chronological splitting is requested.
    """
    if config is None:
        config = load_config(config_path)

    split_cfg = config.get("preprocessing", {}).get("split", {})
    test_frac = float(
        test_size if test_size is not None else split_cfg.get("test_size", 0.20)
    )
    val_frac = float(
        val_size if val_size is not None else split_cfg.get("val_size", 0.10)
    )
    seed = int(
        random_state if random_state is not None else split_cfg.get("random_state", 42)
    )
    grp_col = (
        group_col if group_col is not None else split_cfg.get("group_col", "order_id")
    )
    mode = (
        split_type
        if split_type is not None
        else split_cfg.get("split_type", "group_random")
    ).lower()
    dt_col = (
        date_col
        if date_col is not None
        else split_cfg.get("date_col", "order_date_dateorders")
    )

    if test_frac <= 0 or val_frac < 0 or (test_frac + val_frac) >= 1.0:
        raise ValueError(
            f"Invalid split fractions: test_size={test_frac}, val_size={val_frac}. "
            "Must satisfy test_size > 0, val_size >= 0, and test_size + val_size < 1.0."
        )

    if mode in ("chronological", "time_series"):
        if dt_col not in df.columns:
            raise ValueError(
                f"Date column '{dt_col}' is required in DataFrame for split_type='{mode}'."
            )
        sort_cols = (
            [dt_col, grp_col] if (grp_col and grp_col in df.columns) else [dt_col]
        )
        sorted_df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        if grp_col and grp_col in sorted_df.columns:
            ordered_groups = sorted_df[grp_col].drop_duplicates().to_numpy()
            n_groups = len(ordered_groups)
            train_end = round(n_groups * (1.0 - test_frac - val_frac))
            val_end = round(n_groups * (1.0 - test_frac))

            train_groups = set(ordered_groups[:train_end])
            val_groups = set(ordered_groups[train_end:val_end])
            test_groups = set(ordered_groups[val_end:])

            train_df = sorted_df[sorted_df[grp_col].isin(train_groups)].reset_index(
                drop=True
            )
            val_df = (
                sorted_df[sorted_df[grp_col].isin(val_groups)].reset_index(drop=True)
                if val_frac > 0
                else sorted_df.iloc[0:0].copy()
            )
            test_df = sorted_df[sorted_df[grp_col].isin(test_groups)].reset_index(
                drop=True
            )
        else:
            n_rows = len(sorted_df)
            train_end = round(n_rows * (1.0 - test_frac - val_frac))
            val_end = round(n_rows * (1.0 - test_frac))
            train_df = sorted_df.iloc[:train_end].reset_index(drop=True)
            val_df = (
                sorted_df.iloc[train_end:val_end].reset_index(drop=True)
                if val_frac > 0
                else sorted_df.iloc[0:0].copy()
            )
            test_df = sorted_df.iloc[val_end:].reset_index(drop=True)

        return train_df, val_df, test_df

    if grp_col and grp_col in df.columns:
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(df, groups=df[grp_col]))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        if val_frac > 0:
            relative_val_frac = val_frac / (1.0 - test_frac)
            gss_val = GroupShuffleSplit(
                n_splits=1, test_size=relative_val_frac, random_state=seed + 1
            )
            train_idx, val_idx = next(
                gss_val.split(train_val_df, groups=train_val_df[grp_col])
            )
            train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
            val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        else:
            train_df = train_val_df
            val_df = df.iloc[0:0].copy()
    else:
        train_val_df, test_df = train_test_split(
            df, test_size=test_frac, random_state=seed
        )
        if val_frac > 0:
            relative_val_frac = val_frac / (1.0 - test_frac)
            train_df, val_df = train_test_split(
                train_val_df, test_size=relative_val_frac, random_state=seed + 1
            )
        else:
            train_df = train_val_df
            val_df = df.iloc[0:0].copy()
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    from src.data.load_data import load_raw_data

    raw_df = load_raw_data()
    clean_df = clean_and_preprocess(raw_df)
    print(f"Raw shape: {raw_df.shape} -> Preprocessed shape: {clean_df.shape}")
    print(f"Remaining columns ({len(clean_df.columns)}): {list(clean_df.columns)}")
