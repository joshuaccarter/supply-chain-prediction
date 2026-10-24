"""Prediction engine and API-ready inference logic for the LightGBM P50/P90 model.

This module provides the runtime prediction engine that consumes raw checkout characteristics
from client requests, executes feature engineering and column transformations, and runs joint
quantile inference using trained LightGBM estimators for P50 (median) and P90 (SLA buffer).
"""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer

from src.features.build_features import SupplyChainFeatureBuilder
from src.utils.config_loader import DEFAULT_CONFIG_PATH, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOW_CARD_CATS_DEFAULT: tuple[str, ...] = (
    "type",
    "shipping_mode",
    "customer_country",
    "customer_segment",
    "market",
    "department_id",
    "order_region",
)

SHIPPING_MODE_TO_SCHEDULED_DAYS: dict[str, int] = {
    "same day": 0,
    "first class": 1,
    "second class": 2,
    "standard class": 4,
}

SCHEDULED_DAYS_TO_SHIPPING_MODE: dict[int, str] = {
    0: "Same Day",
    1: "First Class",
    2: "Second Class",
    4: "Standard Class",
}

CANONICAL_SHIPPING_MODES: dict[str, str] = {
    "same day": "Same Day",
    "first class": "First Class",
    "second class": "Second Class",
    "standard class": "Standard Class",
}

DEPARTMENT_NAME_TO_ID: dict[str, str] = {
    "fitness": "2",
    "footwear": "3",
    "apparel": "4",
    "golf": "5",
    "outdoors": "6",
    "fan shop": "7",
    "book shop": "8",
    "discs shop": "9",
    "technology": "10",
    "pet shop": "11",
    "health and beauty": "12",
}

CUSTOMER_COUNTRY_ALIASES: dict[str, str] = {
    "ee. uu.": "EE. UU.",
    "ee.uu.": "EE. UU.",
    "estados unidos": "EE. UU.",
    "united states": "EE. UU.",
    "usa": "EE. UU.",
    "us": "EE. UU.",
    "puerto rico": "Puerto Rico",
    "pr": "Puerto Rico",
}

API_CHARACTERISTIC_DEFAULTS: dict[str, Any] = {
    "type": "DEBIT",
    "shipping_mode": "Standard Class",
    "customer_country": "EE. UU.",
    "customer_segment": "Consumer",
    "department_id": "7",
    "latitude": 18.2208,
    "longitude": -66.5901,
    "market": "LATAM",
    "order_country": "Estados Unidos",
    "order_date_dateorders": "2018-01-15 12:00:00",
    "order_item_discount_rate": 0.0,
    "order_item_profit_ratio": 0.15,
    "order_item_quantity": 1,
    "order_region": "Central America",
    "product_price": 100.0,
}

_CACHED_MODEL: "LightGBMP50P90Model | None" = None
_CACHED_MODEL_PATH: Path | None = None


def normalize_category_text(series: pd.Series) -> pd.Series:
    """Normalize string categorical series by trimming whitespace and collapsing internal spaces.

    Args:
        series: Pandas Series containing text categorical values.

    Returns:
        pd.Series: Cleaned string series with uniform single spaces.
    """
    return series.astype(str).str.strip().apply(lambda s: re.sub(r"\s+", " ", s))


def prepare_inference_dataframe(
    characteristics: Mapping[str, Any] | Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Standardize raw API characteristic dictionary/records or DataFrame for feature building.

    Supports convenient API aliases (`shipping_mode` -> `days_for_shipment_scheduled`,
    `department_name` -> `department_id`, `order_date` -> `order_date_dateorders`,
    country aliases) and preserves optional pre-aggregated basket characteristics
    (`order_line_item_count`, `order_total_units`, `order_total_sales`) when provided
    for single-item API calls.

    Args:
        characteristics: Single dictionary of order characteristics, sequence of dictionaries,
            or pandas DataFrame.

    Returns:
        tuple[pd.DataFrame, dict[str, pd.Series]]: Tuple of `(standardized_df, explicit_basket_overrides)`.

    Raises:
        TypeError: If characteristics is not a dict, sequence of dicts, or pandas DataFrame.
        ValueError: If input characteristics is empty.
    """
    if isinstance(characteristics, pd.DataFrame):
        df = characteristics.copy().reset_index(drop=True)
    elif isinstance(characteristics, Mapping):
        df = pd.DataFrame([dict(characteristics)])
    elif isinstance(characteristics, Sequence):
        df = pd.DataFrame([dict(item) for item in characteristics])
    else:
        raise TypeError(
            "characteristics must be a dict, sequence of dicts, or pandas DataFrame."
        )

    if df.empty:
        raise ValueError("Input characteristics cannot be empty.")

    # Resolve shipping_mode and days_for_shipment_scheduled
    if "shipping_mode" in df.columns and df["shipping_mode"].notna().any():
        df["shipping_mode"] = (
            df["shipping_mode"]
            .astype(str)
            .str.strip()
            .apply(lambda v: CANONICAL_SHIPPING_MODES.get(v.lower(), v))
        )
        mapped_days = (
            df["shipping_mode"].str.lower().map(SHIPPING_MODE_TO_SCHEDULED_DAYS)
        )
        if "days_for_shipment_scheduled" not in df.columns:
            df["days_for_shipment_scheduled"] = mapped_days
        else:
            df["days_for_shipment_scheduled"] = df[
                "days_for_shipment_scheduled"
            ].fillna(mapped_days)
    elif "days_for_shipment_scheduled" in df.columns:
        df["shipping_mode"] = (
            pd.to_numeric(df["days_for_shipment_scheduled"], errors="coerce")
            .map(SCHEDULED_DAYS_TO_SHIPPING_MODE)
            .fillna("Standard Class")
        )

    # Resolve department_name -> department_id if department_id missing
    if (
        "department_id" not in df.columns or df["department_id"].isna().any()
    ) and "department_name" in df.columns:
        mapped_dept = (
            df["department_name"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(DEPARTMENT_NAME_TO_ID)
        )
        if "department_id" not in df.columns:
            df["department_id"] = mapped_dept
        else:
            df["department_id"] = df["department_id"].fillna(mapped_dept)

    # Resolve order_date alias -> order_date_dateorders
    if "order_date_dateorders" not in df.columns and "order_date" in df.columns:
        df["order_date_dateorders"] = df["order_date"]

    # Fill any missing default characteristics for partial API payloads
    for col, default_val in API_CHARACTERISTIC_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_val
        elif df[col].isna().any():
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype(str).replace("nan", str(default_val))
            else:
                df[col] = df[col].fillna(default_val)

    # Normalize customer_country aliases (e.g. "USA" -> "EE. UU.")
    df["customer_country"] = (
        df["customer_country"]
        .astype(str)
        .str.strip()
        .apply(lambda v: CUSTOMER_COUNTRY_ALIASES.get(v.lower(), v))
    )

    # Strip leakage or target columns if present in inference input
    for leak_col in (
        "days_for_shipping_real",
        "shipping_delay_days",
        "delivery_status",
        "late_delivery_risk",
    ):
        if leak_col in df.columns:
            df = df.drop(columns=[leak_col])

    # Ensure numeric types on numeric characteristics
    df["days_for_shipment_scheduled"] = (
        pd.to_numeric(df["days_for_shipment_scheduled"], errors="coerce")
        .fillna(4)
        .astype("int64")
    )
    df["order_item_quantity"] = (
        pd.to_numeric(df["order_item_quantity"], errors="coerce")
        .fillna(1)
        .astype("int64")
    )
    for float_col, default_f in [
        ("product_price", 100.0),
        ("order_item_discount_rate", 0.0),
        ("order_item_profit_ratio", 0.15),
        ("latitude", 18.2208),
        ("longitude", -66.5901),
    ]:
        df[float_col] = (
            pd.to_numeric(df[float_col], errors="coerce")
            .fillna(default_f)
            .astype("float64")
        )

    # Capture explicit basket overrides when order_id is not grouping multiple rows
    has_multi_row_order = (
        "order_id" in df.columns and len(df) > 1 and df["order_id"].nunique() < len(df)
    )
    explicit_basket: dict[str, pd.Series] = {}
    if not has_multi_row_order:
        for b_col in (
            "order_line_item_count",
            "order_total_units",
            "order_total_sales",
            "item_share_of_order_units",
        ):
            if b_col in df.columns and df[b_col].notna().all():
                explicit_basket[b_col] = pd.to_numeric(df[b_col], errors="coerce")

    return df, explicit_basket


class LightGBMP50P90Model:
    """Self-contained LightGBM Quantile Regression bundle returning P50 and P90 shipping days.

    Wraps the feature builder (`SupplyChainFeatureBuilder`), categorical/numeric design-matrix
    preprocessor (`ColumnTransformer`), and trained `P50` (alpha=0.50) and `P90` (alpha=0.90)
    LightGBM quantile regressors into a single serializable artifact for API inference.

    Attributes:
        feature_builder: Configured SupplyChainFeatureBuilder transformer.
        preprocessor: Fitted ColumnTransformer for scaling and one-hot encoding.
        model_p50: Fitted LightGBM regressor predicting the median transit days (alpha=0.50).
        model_p90: Fitted LightGBM regressor predicting the SLA transit days buffer (alpha=0.90).
        numeric_cols: Ordered list of continuous numerical columns scaled by preprocessor.
        low_card_cats: Ordered list of categorical columns encoded by preprocessor.
        feature_names: Full ordered list of encoded column names in design matrix.
        drop_cols: List of non-stationary columns dropped prior to inference (e.g. ['order_year']).
        min_delay: Lower bound for discrete shipping day predictions (default: 0).
        max_delay: Upper bound for discrete shipping day predictions (default: 6).
        metadata: Dictionary containing training parameters, split types, and evaluation metrics.
    """

    def __init__(
        self,
        feature_builder: SupplyChainFeatureBuilder,
        preprocessor: ColumnTransformer,
        model_p50: LGBMRegressor,
        model_p90: LGBMRegressor,
        numeric_cols: list[str],
        low_card_cats: list[str],
        feature_names: list[str],
        drop_cols: list[str] | None = None,
        min_delay: int = 0,
        max_delay: int = 6,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the LightGBMP50P90Model bundle with estimators and preprocessors.

        Args:
            feature_builder: SupplyChainFeatureBuilder transformer.
            preprocessor: Fitted ColumnTransformer.
            model_p50: Trained P50 LightGBM regressor.
            model_p90: Trained P90 LightGBM regressor.
            numeric_cols: List of numeric feature names.
            low_card_cats: List of categorical feature names.
            feature_names: Full list of encoded feature names.
            drop_cols: Optional list of columns to drop from feature matrix.
            min_delay: Minimum discrete bound for transit predictions.
            max_delay: Maximum discrete bound for transit predictions.
            metadata: Optional dictionary with training metadata.
        """
        self.feature_builder = feature_builder
        self.preprocessor = preprocessor
        self.model_p50 = model_p50
        self.model_p90 = model_p90
        self.numeric_cols = list(numeric_cols)
        self.low_card_cats = list(low_card_cats)
        self.feature_names = list(feature_names)
        self.drop_cols = list(drop_cols) if drop_cols is not None else ["order_year"]
        self.min_delay = int(min_delay)
        self.max_delay = int(max_delay)
        self.metadata = metadata or {}

    def transform_features(
        self,
        characteristics: Mapping[str, Any] | Sequence[Mapping[str, Any]] | pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Transform raw characteristics or a DataFrame into the encoded design matrix.

        Args:
            characteristics: Single dict, sequence of dicts, or DataFrame of order characteristics.

        Returns:
            tuple[np.ndarray, pd.DataFrame]: Tuple of `(X_encoded_matrix, standardized_input_df)`.
        """
        std_df, explicit_basket = prepare_inference_dataframe(characteristics)
        feat_df = self.feature_builder.transform(std_df)

        for leak_col in ("shipping_delay_days", "days_for_shipping_real"):
            if leak_col in feat_df.columns:
                feat_df = feat_df.drop(columns=[leak_col])

        # Apply explicit single-item basket overrides if supplied by API caller
        if explicit_basket:
            for b_col, series_val in explicit_basket.items():
                if b_col in feat_df.columns:
                    feat_df[b_col] = series_val.to_numpy()
            if (
                "order_total_units" in explicit_basket
                and "item_share_of_order_units" not in explicit_basket
            ):
                denom = np.where(
                    feat_df["order_total_units"].to_numpy() == 0,
                    1.0,
                    feat_df["order_total_units"].to_numpy(),
                )
                feat_df["item_share_of_order_units"] = (
                    feat_df["order_item_quantity"].to_numpy() / denom
                ).astype("float64")

        if self.drop_cols:
            existing_drop = [c for c in self.drop_cols if c in feat_df.columns]
            if existing_drop:
                feat_df = feat_df.drop(columns=existing_drop)

        for cat_col in self.low_card_cats:
            if cat_col in feat_df.columns:
                feat_df[cat_col] = normalize_category_text(feat_df[cat_col])

        X_mat = self.preprocessor.transform(feat_df)
        return X_mat, std_df

    def predict_from_matrix(self, X_mat: np.ndarray) -> dict[str, np.ndarray]:
        """Run P50 and P90 LightGBM quantile inference directly on an encoded design matrix.

        Enforces quantile monotonicity (`p90 >= p50`) and clips discrete predictions to
        `[self.min_delay, self.max_delay]`.

        Args:
            X_mat: 2D numpy array containing the scaled and one-hot encoded design matrix.

        Returns:
            dict[str, np.ndarray]: Dictionary with keys 'p50', 'p90', 'p50_raw', and 'p90_raw'.
        """
        p50_raw = np.asarray(self.model_p50.predict(X_mat), dtype=float)
        p90_raw = np.asarray(self.model_p90.predict(X_mat), dtype=float)
        p90_raw = np.maximum(p50_raw, p90_raw)

        p50_int = np.clip(np.round(p50_raw), self.min_delay, self.max_delay).astype(int)
        p90_int = np.clip(np.round(p90_raw), self.min_delay, self.max_delay).astype(int)
        p90_int = np.maximum(p50_int, p90_int)

        return {
            "p50": p50_int,
            "p90": p90_int,
            "p50_raw": p50_raw,
            "p90_raw": p90_raw,
        }

    def predict_p50_p90(
        self,
        characteristics: Mapping[str, Any] | Sequence[Mapping[str, Any]] | pd.DataFrame,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Return P50 and P90 shipping days and SLA buffer predictions from API characteristics.

        Args:
            characteristics: A single dictionary of order/item characteristics, a list of
                dictionaries, or a pandas DataFrame.

        Returns:
            dict[str, Any] | list[dict[str, Any]]: A single prediction dictionary (when input
            is a `Mapping`) or a list of prediction dictionaries (when input is a sequence
            of dicts or `DataFrame`).
        """
        single_input = isinstance(characteristics, Mapping)
        X_mat, std_df = self.transform_features(characteristics)
        preds = self.predict_from_matrix(X_mat)

        results: list[dict[str, Any]] = []
        for idx in range(len(std_df)):
            p50_d = int(preds["p50"][idx])
            p90_d = int(preds["p90"][idx])
            p50_r = float(preds["p50_raw"][idx])
            p90_r = float(preds["p90_raw"][idx])
            safety_buffer = max(0, p90_d - p50_d)
            shipping_mode_val = (
                str(std_df.loc[idx, "shipping_mode"])
                if "shipping_mode" in std_df.columns
                else "Standard Class"
            )

            res: dict[str, Any] = {
                "shipping_mode": shipping_mode_val,
                "p50_shipping_days": p50_d,
                "p90_shipping_days": p90_d,
                "safety_buffer_days": safety_buffer,
                "p50_raw": round(p50_r, 4),
                "p90_raw": round(p90_r, 4),
                "p50": p50_d,
                "p90": p90_d,
                "p50_delay_days": p50_d,
                "p90_delay_days": p90_d,
            }

            if "order_id" in std_df.columns and pd.notna(std_df.loc[idx, "order_id"]):
                res["order_id"] = str(std_df.loc[idx, "order_id"])

            if "days_for_shipment_scheduled" in std_df.columns and pd.notna(
                std_df.loc[idx, "days_for_shipment_scheduled"]
            ):
                res["days_for_shipment_scheduled"] = int(
                    std_df.loc[idx, "days_for_shipment_scheduled"]
                )

            # Compute delivery dates if order date is present
            date_val = None
            if "order_date_dateorders" in std_df.columns and pd.notna(
                std_df.loc[idx, "order_date_dateorders"]
            ):
                date_val = std_df.loc[idx, "order_date_dateorders"]
            elif "order_date" in std_df.columns and pd.notna(
                std_df.loc[idx, "order_date"]
            ):
                date_val = std_df.loc[idx, "order_date"]

            if date_val is not None:
                try:
                    dt = pd.to_datetime(date_val)
                    res["p50_estimated_delivery_date"] = (
                        dt + pd.Timedelta(days=p50_d)
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    res["p90_sla_delivery_date"] = (
                        dt + pd.Timedelta(days=p90_d)
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError, OverflowError):
                    res["p50_estimated_delivery_date"] = None
                    res["p90_sla_delivery_date"] = None

            results.append(res)

        return results[0] if single_input else results

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return a DataFrame of top feature importances across the P50 and P90 LightGBM models.

        Args:
            top_n: Number of highest ranking features to return (default: 20).

        Returns:
            pd.DataFrame: DataFrame containing feature name, P50 splits, P90 splits, and total splits.
        """
        imp_p50 = np.asarray(self.model_p50.feature_importances_, dtype=float)
        imp_p90 = np.asarray(self.model_p90.feature_importances_, dtype=float)
        df_imp = pd.DataFrame(
            {
                "feature": self.feature_names,
                "p50_splits": imp_p50,
                "p90_splits": imp_p90,
                "total_splits": imp_p50 + imp_p90,
            }
        ).sort_values("total_splits", ascending=False)
        return df_imp.head(top_n).reset_index(drop=True)

    def save(self, path: str | Path) -> Path:
        """Serialize the LightGBMP50P90Model bundle to disk via joblib.

        Args:
            path: Target file path for the serialized `.joblib` model bundle.

        Returns:
            Path: Resolved Path instance pointing to the persisted file.
        """
        out_path = Path(path)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMP50P90Model":
        """Load a serialized LightGBMP50P90Model bundle from disk.

        Args:
            path: Path to the `.joblib` model artifact file.

        Returns:
            LightGBMP50P90Model: Deserialized model bundle ready for inference.

        Raises:
            FileNotFoundError: If the artifact does not exist on disk.
            TypeError: If the deserialized object is not an instance of LightGBMP50P90Model.
        """
        in_path = Path(path)
        if not in_path.is_absolute():
            in_path = PROJECT_ROOT / in_path
        if not in_path.exists():
            raise FileNotFoundError(f"Trained model artifact not found at: {in_path}")
        loaded = joblib.load(in_path)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Expected LightGBMP50P90Model artifact, got {type(loaded)!r}"
            )
        return loaded


def get_default_model_path(
    config: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """Resolve the configured model artifact path from `config/config.yaml`.

    Args:
        config: Optional pre-loaded configuration dictionary.
        config_path: Path to configuration file if `config` is None.

    Returns:
        Path: Absolute or project-root resolved path to the `.joblib` model artifact.
    """
    if config is None:
        config = load_config(config_path)
    models_cfg = config.get("models", {})
    model_dir = Path(models_cfg.get("dir", "models"))
    artifact_name = models_cfg.get("artifact_name", "lgbm_shipping_delay_model.joblib")
    rel_path = model_dir / artifact_name
    return rel_path if rel_path.is_absolute() else PROJECT_ROOT / rel_path


def load_trained_model(
    model_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    force_reload: bool = False,
) -> LightGBMP50P90Model:
    """Load and cache the trained LightGBMP50P90Model artifact for fast API inference.

    Caches the deserialized artifact in module-level global memory so that subsequent
    calls return instantly without file I/O overhead.

    Args:
        model_path: Optional explicit path to the model file. If None, resolves from config.
        config: Optional pre-loaded configuration dictionary.
        force_reload: If True, forces reloading from disk even if already cached.

    Returns:
        LightGBMP50P90Model: Deserialized and cached model instance.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH

    resolved_path = (
        Path(model_path)
        if model_path is not None
        else get_default_model_path(config=config)
    )
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    if (
        not force_reload
        and _CACHED_MODEL is not None
        and resolved_path == _CACHED_MODEL_PATH
    ):
        return _CACHED_MODEL

    model = LightGBMP50P90Model.load(resolved_path)
    _CACHED_MODEL = model
    _CACHED_MODEL_PATH = resolved_path
    return model


def predict_p50_p90(
    characteristics: Mapping[str, Any] | Sequence[Mapping[str, Any]] | pd.DataFrame,
    model: LightGBMP50P90Model | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict P50 and P90 shipping delay values based on order characteristics provided to the API.

    Convenience wrapper that automatically loads the active model bundle if not provided,
    runs feature extraction and design matrix encoding, and returns prediction dictionaries.

    Args:
        characteristics: Single dictionary of order/item characteristics, list of dictionaries,
            or pandas DataFrame.
        model: Optional pre-loaded `LightGBMP50P90Model` instance.
        model_path: Optional path to a saved `.joblib` model artifact if `model` is not provided.

    Returns:
        dict[str, Any] | list[dict[str, Any]]: Dictionary (for single-item input) or list of
            dictionaries containing `p50`, `p90`, `p50_delay_days`, `p90_delay_days`, `p50_raw`,
            `p90_raw`, `days_for_shipment_scheduled`, `p50_shipping_days`, and `p90_shipping_days`.
    """
    active_model = (
        model if model is not None else load_trained_model(model_path=model_path)
    )
    return active_model.predict_p50_p90(characteristics)
