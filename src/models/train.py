"""Script and utilities for training and evaluating the LightGBM P50/P90 prediction model.

This module orchestrates data preparation, feature building, design matrix encoding
via standard scaling and one-hot encoding, quantile regression fitting for P50 and P90
objectives, out-of-time evaluation, and serialization of the production model artifact.
"""

import json
import time
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_raw_data
from src.data.preprocess import clean_and_preprocess, split_dataset
from src.features.build_features import SupplyChainFeatureBuilder
from src.models.evaluate import evaluate_p50_p90
from src.models.predict import (
    LOW_CARD_CATS_DEFAULT,
    PROJECT_ROOT,
    SHIPPING_MODE_TO_SCHEDULED_DAYS,
    LightGBMP50P90Model,
    get_default_model_path,
    normalize_category_text,
)
from src.utils.config_loader import DEFAULT_CONFIG_PATH, load_config


def encode_and_scale_splits(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    low_card_cats: list[str] | None = None,
    drop_cols: list[str] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ColumnTransformer,
    list[str],
    list[str],
    list[str],
]:
    """Fit `StandardScaler` + `OneHotEncoder` on `X_train_df` and transform all three splits.

    Separates continuous numerical features and discrete categorical attributes, drops any
    designated non-stationary columns, cleans category string formatting, and constructs
    a unified scikit-learn `ColumnTransformer` with `StandardScaler` for numeric columns
    and `OneHotEncoder(drop='first', handle_unknown='ignore')` for categorical features.

    Args:
        X_train_df: Training feature DataFrame (excluding target column).
        X_val_df: Validation feature DataFrame (excluding target column).
        X_test_df: Test feature DataFrame (excluding target column).
        low_card_cats: List of low-cardinality categorical column names to one-hot encode.
            Defaults to LOW_CARD_CATS_DEFAULT if None.
        drop_cols: Optional list of non-stationary columns to drop (e.g. `['order_year']`).

    Returns:
        tuple: A 7-element tuple containing:
            - X_tr_mat (np.ndarray): Scaled and encoded training design matrix.
            - X_va_mat (np.ndarray): Scaled and encoded validation design matrix.
            - X_te_mat (np.ndarray): Scaled and encoded test design matrix.
            - preprocessor (ColumnTransformer): Fitted ColumnTransformer instance.
            - feature_names (list[str]): Full list of encoded column names.
            - num_cols (list[str]): List of continuous numeric column names.
            - active_cats (list[str]): List of active categorical column names.
    """
    cats = (
        list(low_card_cats)
        if low_card_cats is not None
        else list(LOW_CARD_CATS_DEFAULT)
    )
    X_tr = X_train_df.copy()
    X_va = X_val_df.copy()
    X_te = X_test_df.copy()

    if drop_cols:
        existing_drop = [c for c in drop_cols if c in X_tr.columns]
        if existing_drop:
            X_tr = X_tr.drop(columns=existing_drop)
            X_va = X_va.drop(columns=[c for c in existing_drop if c in X_va.columns])
            X_te = X_te.drop(columns=[c for c in existing_drop if c in X_te.columns])

    active_cats = [c for c in cats if c in X_tr.columns]
    num_cols = [c for c in X_tr.columns if c not in active_cats]

    for col in active_cats:
        X_tr[col] = normalize_category_text(X_tr[col])
        if len(X_va) > 0:
            X_va[col] = normalize_category_text(X_va[col])
        if len(X_te) > 0:
            X_te[col] = normalize_category_text(X_te[col])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                active_cats,
            ),
        ],
        remainder="drop",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        X_tr_mat = preprocessor.fit_transform(X_tr)
        X_va_mat = (
            preprocessor.transform(X_va)
            if len(X_va) > 0
            else np.empty((0, X_tr_mat.shape[1]))
        )
        X_te_mat = (
            preprocessor.transform(X_te)
            if len(X_te) > 0
            else np.empty((0, X_tr_mat.shape[1]))
        )

    ohe_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(active_cats)
    )
    feature_names = num_cols + ohe_names
    return (
        X_tr_mat,
        X_va_mat,
        X_te_mat,
        preprocessor,
        feature_names,
        num_cols,
        active_cats,
    )


def fit_lgbm_quantile_estimator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    lgbm_params: dict[str, Any] | None = None,
) -> tuple[LGBMRegressor, float]:
    """Fit a single LightGBM quantile regressor for quantile level `alpha`.

    Initializes a LightGBM regressor with quantile objective at pinball alpha level,
    applies hyperparameters from configuration, adjusts min_child_samples for small
    training fixtures, and trains with early stopping against the validation set.

    Args:
        X_train: Encoded training design matrix.
        y_train: Training target array (`days_for_shipping_real`).
        alpha: Quantile level in `(0, 1)`, e.g. `0.50` for P50 or `0.90` for P90.
        X_val: Optional encoded validation design matrix for early stopping.
        y_val: Optional validation target array for early stopping.
        lgbm_params: Optional hyperparameter dictionary from `config.yaml`.

    Returns:
        tuple[LGBMRegressor, float]: Tuple of `(fitted_estimator, train_time_sec)`.
    """
    params: dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 7,
        "num_leaves": 63,
        "min_child_samples": 50,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if lgbm_params:
        params.update(lgbm_params)

    early_stopping_rounds = int(params.pop("early_stopping_rounds", 25))
    # Adapt min_child_samples gracefully when training on tiny synthetic test fixtures
    if len(y_train) < params.get("min_child_samples", 50) * 2:
        params["min_child_samples"] = max(1, len(y_train) // 4)

    estimator = LGBMRegressor(
        objective="quantile",
        alpha=float(alpha),
        **params,
    )

    t0 = time.perf_counter()
    if (
        X_val is not None
        and y_val is not None
        and len(y_val) > 0
        and early_stopping_rounds > 0
    ):
        estimator.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            ],
        )
    else:
        estimator.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    return estimator, train_time


def train_lightgbm_p50_p90(
    df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    split_type: str | None = None,
    save_artifact: bool = True,
    artifact_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> tuple[LightGBMP50P90Model, dict[str, Any]]:
    """Train the LightGBM P50/P90 shipping duration model and evaluate it on validation and test splits.

    Orchestrates the entire model lifecycle:
    1. Loads and preprocesses raw data if not passed in.
    2. Performs group-aware chronological or random dataset partitioning.
    3. Builds cyclical calendar, price ratio, basket, and geographic route features.
    4. Encodes and standardizes features into numeric and one-hot matrix representations.
    5. Fits P50 (median) and P90 (conservative SLA) LightGBM quantile regressors.
    6. Bundles transformers and estimators into a unified LightGBMP50P90Model object.
    7. Evaluates predictions on validation and held-out test splits across regression,
       classification, ordinal, and SLA operational metrics.
    8. Persists `.joblib` model artifact and `.json` metrics summary to disk.

    Args:
        df: Optional raw or preprocessed DataFrame. If None, loads dataset via `load_raw_data`.
        config: Optional configuration dictionary. If None, loads from `config_path`.
        config_path: Path to YAML config file if `config` is not provided.
        split_type: Dataset split strategy (`"chronological"` or `"group_random"`). Defaults to config.
        save_artifact: Whether to persist the trained model `.joblib` and metrics `.json` to disk.
        artifact_path: Optional custom path for the saved `.joblib` artifact.
        metrics_path: Optional custom path for the saved `.json` evaluation metrics.

    Returns:
        tuple[LightGBMP50P90Model, dict[str, Any]]: Tuple containing the trained model
            bundle and comprehensive evaluation report dictionary.
    """
    if config is None:
        config = load_config(config_path)

    models_cfg = config.get("models", {})
    active_split_type = (
        split_type
        if split_type is not None
        else models_cfg.get("split_type", "chronological")
    )
    drop_cols: list[str] = list(
        models_cfg.get("drop_non_stationary_cols", ["order_year"])
    )
    low_card_cats: list[str] = list(
        models_cfg.get("low_card_cats", LOW_CARD_CATS_DEFAULT)
    )
    quantiles_cfg = models_cfg.get("quantiles", {"p50": 0.50, "p90": 0.90})
    tau_p50 = float(quantiles_cfg.get("p50", 0.50))
    tau_p90 = float(quantiles_cfg.get("p90", 0.90))
    bounds_cfg = models_cfg.get(
        "target_bounds", models_cfg.get("delay_bounds", {"min": 0, "max": 6})
    )
    min_delay = int(bounds_cfg.get("min", 0))
    max_delay = int(bounds_cfg.get("max", 6))
    lgbm_params = dict(models_cfg.get("lgbm_params", {}))

    raw_df = df if df is not None else load_raw_data(config=config)
    clean_df = clean_and_preprocess(
        raw_df,
        config=config,
        include_collinear=True,
        include_entity_keys=False,
    )

    train_df, val_df, test_df = split_dataset(
        clean_df,
        config=config,
        split_type=active_split_type,
    )

    feature_builder = SupplyChainFeatureBuilder(drop_entity_keys=True)
    X_train_df = feature_builder.fit_transform(train_df)
    X_val_df = feature_builder.transform(val_df)
    X_test_df = feature_builder.transform(test_df)

    target_col = (
        config.get("data", {}).get("target", {}).get("name", "days_for_shipping_real")
    )
    y_train = X_train_df.pop(target_col).to_numpy(dtype=int)
    y_val = X_val_df.pop(target_col).to_numpy(dtype=int)
    y_test = X_test_df.pop(target_col).to_numpy(dtype=int)

    if "days_for_shipment_scheduled" in X_val_df.columns:
        sched_val = X_val_df["days_for_shipment_scheduled"].to_numpy(dtype=int)
        sched_test = X_test_df["days_for_shipment_scheduled"].to_numpy(dtype=int)
    elif "shipping_mode" in X_val_df.columns:
        sched_val = (
            X_val_df["shipping_mode"]
            .astype(str)
            .str.lower()
            .map(SHIPPING_MODE_TO_SCHEDULED_DAYS)
            .fillna(4)
            .to_numpy(dtype=int)
        )
        sched_test = (
            X_test_df["shipping_mode"]
            .astype(str)
            .str.lower()
            .map(SHIPPING_MODE_TO_SCHEDULED_DAYS)
            .fillna(4)
            .to_numpy(dtype=int)
        )
    else:
        sched_val = None
        sched_test = None

    (
        X_train_mat,
        X_val_mat,
        X_test_mat,
        preprocessor,
        feature_names,
        numeric_cols,
        active_cats,
    ) = encode_and_scale_splits(
        X_train_df=X_train_df,
        X_val_df=X_val_df,
        X_test_df=X_test_df,
        low_card_cats=low_card_cats,
        drop_cols=drop_cols,
    )

    model_p50, train_time_p50 = fit_lgbm_quantile_estimator(
        X_train=X_train_mat,
        y_train=y_train,
        alpha=tau_p50,
        X_val=X_val_mat,
        y_val=y_val,
        lgbm_params=lgbm_params,
    )
    model_p90, train_time_p90 = fit_lgbm_quantile_estimator(
        X_train=X_train_mat,
        y_train=y_train,
        alpha=tau_p90,
        X_val=X_val_mat,
        y_val=y_val,
        lgbm_params=lgbm_params,
    )

    metadata: dict[str, Any] = {
        "model_family": "LightGBM Quantile Regressor (P50 & P90)",
        "split_type": active_split_type,
        "quantiles": {"p50": tau_p50, "p90": tau_p90},
        "n_train_rows": len(y_train),
        "n_val_rows": len(y_val),
        "n_test_rows": len(y_test),
        "n_engineered_features": int(
            X_train_df.shape[1] - len([c for c in drop_cols if c in X_train_df.columns])
        ),
        "n_encoded_features": len(feature_names),
        "best_iteration_p50": int(
            getattr(model_p50, "best_iteration_", model_p50.n_estimators)
            or model_p50.n_estimators
        ),
        "best_iteration_p90": int(
            getattr(model_p90, "best_iteration_", model_p90.n_estimators)
            or model_p90.n_estimators
        ),
        "train_time_p50_sec": round(train_time_p50, 4),
        "train_time_p90_sec": round(train_time_p90, 4),
    }

    bundle = LightGBMP50P90Model(
        feature_builder=feature_builder,
        preprocessor=preprocessor,
        model_p50=model_p50,
        model_p90=model_p90,
        numeric_cols=numeric_cols,
        low_card_cats=active_cats,
        feature_names=feature_names,
        drop_cols=drop_cols,
        min_delay=min_delay,
        max_delay=max_delay,
        metadata=metadata,
    )

    # Evaluate on Validation and Test sets
    t_val0 = time.perf_counter()
    val_preds = bundle.predict_from_matrix(X_val_mat)
    val_pred_time = time.perf_counter() - t_val0

    t_test0 = time.perf_counter()
    test_preds = bundle.predict_from_matrix(X_test_mat)
    test_pred_time = time.perf_counter() - t_test0

    val_report = evaluate_p50_p90(
        y_true=y_val,
        p50_pred=val_preds["p50"],
        p90_pred=val_preds["p90"],
        scheduled_days=sched_val,
        model_name="LightGBM Quantile",
        train_time_p50=train_time_p50,
        train_time_p90=train_time_p90,
        pred_time_sec=val_pred_time,
        p50_raw=val_preds["p50_raw"],
        p90_raw=val_preds["p90_raw"],
    )
    test_report = evaluate_p50_p90(
        y_true=y_test,
        p50_pred=test_preds["p50"],
        p90_pred=test_preds["p90"],
        scheduled_days=sched_test,
        model_name="LightGBM Quantile",
        train_time_p50=train_time_p50,
        train_time_p90=train_time_p90,
        pred_time_sec=test_pred_time,
        p50_raw=test_preds["p50_raw"],
        p90_raw=test_preds["p90_raw"],
    )

    full_report: dict[str, Any] = {
        "metadata": metadata,
        "validation": val_report,
        "test": test_report,
        "top_features": bundle.get_feature_importance(top_n=15).to_dict(
            orient="records"
        ),
    }
    bundle.metadata["test_summary"] = {
        "p50_exact_accuracy": test_report["p50"]["exact_accuracy"],
        "p50_within_1_day_accuracy": test_report["p50"]["within_1_day_accuracy"],
        "p50_mae": test_report["p50"]["mae"],
        "p50_rmse": test_report["p50"]["rmse"],
        "p50_r2": test_report["p50"]["r2"],
        "p50_qwk": test_report["p50"]["qwk"],
        "p50_pinball_loss": test_report["p50"]["pinball_loss_p50"],
        "p90_sla_compliance_pct": test_report["p90"]["sla_compliance_pct"],
        "p90_late_breach_pct": test_report["p90"]["late_breach_pct"],
        "p90_mean_buffer_cushion_days": test_report["p90"]["mean_buffer_cushion_days"],
        "p90_pinball_loss": test_report["p90"]["pinball_loss_p90"],
    }

    if save_artifact:
        out_model_path = (
            Path(artifact_path)
            if artifact_path is not None
            else get_default_model_path(config=config)
        )
        saved_model_path = bundle.save(out_model_path)
        full_report["metadata"]["artifact_path"] = str(saved_model_path)

        if metrics_path is None:
            model_dir = Path(models_cfg.get("dir", "models"))
            metrics_name = models_cfg.get("metrics_name", "lgbm_metrics.json")
            out_metrics_path = model_dir / metrics_name
        else:
            out_metrics_path = Path(metrics_path)
        if not out_metrics_path.is_absolute():
            out_metrics_path = PROJECT_ROOT / out_metrics_path
        out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        out_metrics_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
        full_report["metadata"]["metrics_path"] = str(out_metrics_path)

    return bundle, full_report


def _print_evaluation_summary(split_label: str, report: dict[str, Any]) -> None:
    """Print a formatted summary of validation and test metrics for a trained model.

    Formats training metadata, quantile performance tables across validation and
    test horizons, and per-shipping-tier diagnostic breakdowns into terminal logs.

    Args:
        split_label: String label for the split strategy (e.g. 'chronological', 'group_random').
        report: Dictionary containing model metadata and evaluated metrics.
    """
    meta = report["metadata"]
    print(f"\n{'=' * 88}")
    print(
        f"LIGHTGBM QUANTILE (P50 & P90) TRAINING & EVALUATION — SPLIT: {split_label.upper()}"
    )
    print(f"{'=' * 88}")
    print(
        f"Rows -> Train: {meta['n_train_rows']:,} | Val: {meta['n_val_rows']:,} | Test: {meta['n_test_rows']:,} | "
        f"Encoded Features: {meta['n_encoded_features']}"
    )
    print(
        f"Best Iteration -> P50: {meta['best_iteration_p50']} ({meta['train_time_p50_sec']:.2f}s) | "
        f"P90: {meta['best_iteration_p90']} ({meta['train_time_p90_sec']:.2f}s)"
    )

    rows = []
    for part_label, part_key in [("Validation", "validation"), ("Test", "test")]:
        for q_label, q_key in [("P50 (Central)", "p50"), ("P90 (SLA Buffer)", "p90")]:
            m = report[part_key][q_key]
            rows.append(
                {
                    "Split": part_label,
                    "Quantile": q_label,
                    "Exact Acc (%)": round(m["exact_accuracy"] * 100.0, 2),
                    "Within-1d Acc (%)": round(m["within_1_day_accuracy"] * 100.0, 2),
                    "MAE (d)": round(m["mae"], 4),
                    "RMSE (d)": round(m["rmse"], 4),
                    "R2": round(m["r2"], 4),
                    "QWK": round(m["qwk"], 4),
                    "Pinball(p50)": round(m["pinball_loss_p50"], 4),
                    "Pinball(p90)": round(m["pinball_loss_p90"], 4),
                    "SLA Comp (%)": round(m["sla_compliance_pct"], 2),
                    "Late Breach (%)": round(m["late_breach_pct"], 2),
                    "Cushion (d)": round(m["mean_buffer_cushion_days"], 4),
                }
            )
    summary_df = pd.DataFrame(rows)
    print("\n--- Standard & SLA Metrics (Validation vs. Held-Out Test) ---")
    print(summary_df.to_string(index=False))

    tier_df = pd.DataFrame(report["test"]["tier_breakdown"])
    print("\n--- Held-Out Test Set Breakdown by Scheduled Shipping Tier ---")
    print(tier_df.to_string(index=False))


if __name__ == "__main__":
    cfg = load_config()
    raw_dataset = load_raw_data(config=cfg)

    # 1. Evaluate on Group-Random Split for benchmark comparison
    _, random_report = train_lightgbm_p50_p90(
        df=raw_dataset,
        config=cfg,
        split_type="group_random",
        save_artifact=False,
    )
    _print_evaluation_summary("group_random", random_report)

    # 2. Train & persist primary production model on Out-of-Time Chronological Split
    prod_model, chrono_report = train_lightgbm_p50_p90(
        df=raw_dataset,
        config=cfg,
        split_type="chronological",
        save_artifact=True,
    )
    _print_evaluation_summary("chronological (saved production model)", chrono_report)

    # Save combined benchmark + production metrics JSON
    combined_metrics_path = Path(cfg["models"]["dir"]) / cfg["models"]["metrics_name"]
    if not combined_metrics_path.is_absolute():
        combined_metrics_path = PROJECT_ROOT / combined_metrics_path
    combined_payload = {
        "production_split": "chronological",
        "chronological": chrono_report,
        "group_random": random_report,
    }
    combined_metrics_path.write_text(
        json.dumps(combined_payload, indent=2), encoding="utf-8"
    )

    print("\n--- Top 10 Feature Importances (Production Model) ---")
    print(prod_model.get_feature_importance(top_n=10).to_string(index=False))

    # Demonstrate API characteristic inference across the 4 shipping modes
    sample_api_payloads = [
        {
            "shipping_mode": "Same Day",
            "type": "DEBIT",
            "customer_country": "EE. UU.",
            "customer_segment": "Consumer",
            "department_name": "Fan Shop",
            "market": "USCA",
            "order_region": "West of USA",
            "order_country": "Estados Unidos",
            "latitude": 34.0522,
            "longitude": -118.2437,
            "product_price": 120.0,
            "order_item_quantity": 2,
            "order_item_discount_rate": 0.05,
            "order_item_profit_ratio": 0.25,
            "order_date": "2018-01-15 10:30:00",
        },
        {
            "shipping_mode": "First Class",
            "type": "TRANSFER",
            "customer_country": "Puerto Rico",
            "customer_segment": "Corporate",
            "department_name": "Apparel",
            "market": "Europe",
            "order_region": "Western Europe",
            "order_country": "Francia",
            "latitude": 18.2208,
            "longitude": -66.5901,
            "product_price": 85.0,
            "order_item_quantity": 3,
            "order_item_discount_rate": 0.10,
            "order_item_profit_ratio": 0.18,
            "order_date": "2018-01-16 20:15:00",
        },
        {
            "shipping_mode": "Second Class",
            "type": "PAYMENT",
            "customer_country": "EE. UU.",
            "customer_segment": "Home Office",
            "department_name": "Technology",
            "market": "LATAM",
            "order_region": "Central America",
            "order_country": "México",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "product_price": 250.0,
            "order_item_quantity": 1,
            "order_item_discount_rate": 0.0,
            "order_item_profit_ratio": 0.30,
            "order_date": "2018-01-17 14:00:00",
        },
        {
            "shipping_mode": "Standard Class",
            "type": "CASH",
            "customer_country": "EE. UU.",
            "customer_segment": "Consumer",
            "department_name": "Outdoors",
            "market": "Pacific Asia",
            "order_region": "Oceania",
            "order_country": "Australia",
            "latitude": 41.8781,
            "longitude": -87.6298,
            "product_price": 60.0,
            "order_item_quantity": 4,
            "order_item_discount_rate": 0.15,
            "order_item_profit_ratio": -0.05,
            "order_date": "2018-01-20 22:00:00",
        },
    ]
    sample_preds = prod_model.predict_p50_p90(sample_api_payloads)
    print("\n--- Sample API Predictions (P50 Central vs. P90 SLA Buffer) ---")
    print(pd.DataFrame(sample_preds).to_string(index=False))
    print(f"\nSaved model artifact to: {chrono_report['metadata']['artifact_path']}")
    print(f"Saved metrics report to: {combined_metrics_path}")
