"""Utilities for evaluating LightGBM P50 and P90 model performance against standard metrics.

This module calculates standard regression error benchmarks (MAE, RMSE, R-squared),
quantile pinball losses for median and SLA horizons (P50 and P90), discrete and ordinal
classification metrics (exact accuracy, within-1-day accuracy, macro/weighted F1, QWK),
and supply chain operational metrics (SLA compliance rate, late breach percentage, mean buffer cushion).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_pinball_loss,
    r2_score,
    root_mean_squared_error,
)

SHIPPING_DAYS_CLASSES: np.ndarray = np.arange(7, dtype=int)
DELAY_CLASSES: np.ndarray = SHIPPING_DAYS_CLASSES
SCHEDULED_TIER_NAMES: dict[int, str] = {
    0: "Same Day (0d)",
    1: "First Class (1d)",
    2: "Second Class (2d)",
    4: "Standard Class (4d)",
}


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int | float] | np.ndarray | pd.Series,
    model_name: str = "LightGBM",
    train_time_sec: float = 0.0,
    pred_time_sec: float = 0.0,
    min_delay: int = 0,
    max_delay: int = 6,
) -> dict[str, Any]:
    """Compute standard regression, discrete classification, ordinal, and SLA buffer metrics.

    Evaluates predictions against ground truth transit days, calculating:
    - Regression metrics: MAE, RMSE, R2, pinball losses at alpha=0.50 and alpha=0.90
    - Discrete classification metrics: Exact accuracy, within-1-day accuracy, balanced accuracy,
      macro F1, weighted F1, and Quadratic Weighted Kappa (QWK)
    - Supply chain operational SLA metrics: SLA compliance rate (y_true <= y_pred), late breach
      rate (y_true > y_pred), mean buffer cushion, and signed bias

    Args:
        y_true: Ground-truth integer shipping transit days in `[min_delay, max_delay]`.
        y_pred: Predicted shipping transit days (rounded and clipped to `[min_delay, max_delay]`).
        model_name: Identifier for the evaluated model/quantile operating point.
        train_time_sec: Model training wall-clock time in seconds.
        pred_time_sec: Model inference wall-clock time in seconds.
        min_delay: Minimum discrete shipping transit bound (default: 0).
        max_delay: Maximum discrete shipping transit bound (default: 6).

    Returns:
        dict[str, Any]: Dictionary containing regression, quantile, classification, ordinal,
            and SLA metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_int = np.clip(
        np.round(np.asarray(y_pred, dtype=float)), min_delay, max_delay
    ).astype(int)
    diff = (
        y_pred_int - y_true_arr
    )  # >= 0 means shipment arrived on/before predicted bound

    sla_compliance_rate = float(np.mean(y_true_arr <= y_pred_int))
    late_breach_rate = float(np.mean(y_true_arr > y_pred_int))

    return {
        "model": model_name,
        "n_samples": len(y_true_arr),
        # Standard Regression & Quantile Error Metrics
        "mae": float(mean_absolute_error(y_true_arr, y_pred_int)),
        "rmse": float(root_mean_squared_error(y_true_arr, y_pred_int)),
        "r2": float(r2_score(y_true_arr, y_pred_int)),
        "pinball_loss_p50": float(
            mean_pinball_loss(y_true_arr, y_pred_int, alpha=0.50)
        ),
        "pinball_loss_p90": float(
            mean_pinball_loss(y_true_arr, y_pred_int, alpha=0.90)
        ),
        # Standard Discrete / Ordinal Classification Metrics
        "exact_accuracy": float(accuracy_score(y_true_arr, y_pred_int)),
        "within_1_day_accuracy": float(np.mean(np.abs(y_true_arr - y_pred_int) <= 1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_int)),
        "macro_f1": float(
            f1_score(y_true_arr, y_pred_int, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_arr, y_pred_int, average="weighted", zero_division=0)
        ),
        "qwk": float(cohen_kappa_score(y_true_arr, y_pred_int, weights="quadratic")),
        # Supply Chain Operational SLA & Buffer Metrics
        "sla_compliance_rate": sla_compliance_rate,
        "sla_compliance_pct": sla_compliance_rate * 100.0,
        "late_breach_rate": late_breach_rate,
        "late_breach_pct": late_breach_rate * 100.0,
        "mean_buffer_cushion_days": float(np.mean(np.maximum(0, diff))),
        "mean_signed_bias_days": float(np.mean(diff)),
        # Runtime Metrics
        "train_time_sec": float(train_time_sec),
        "pred_time_sec": float(pred_time_sec),
    }


def evaluate_by_scheduled_tier(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    p50_pred: Sequence[int | float] | np.ndarray | pd.Series,
    p90_pred: Sequence[int | float] | np.ndarray | pd.Series,
    scheduled_days: Sequence[int] | np.ndarray | pd.Series,
) -> list[dict[str, Any]]:
    """Compute P50 and P90 performance broken down by scheduled shipping tier (0, 1, 2, 4 days).

    Partitions predictions and ground truth across the 4 commercial shipping modes to verify
    calibration and compliance independently for same day, first class, second class, and standard class.

    Args:
        y_true: Ground-truth shipping transit or delay days.
        p50_pred: P50 central predictions.
        p90_pred: P90 SLA buffer predictions.
        scheduled_days: Scheduled shipment days (`0`, `1`, `2`, `4`).

    Returns:
        list[dict[str, Any]]: List of per-tier metric dictionaries sorted by `scheduled_days`.
    """
    y_arr = np.asarray(y_true, dtype=int)
    p50_arr = np.clip(np.round(np.asarray(p50_pred, dtype=float)), 0, 6).astype(int)
    p90_arr = np.clip(np.round(np.asarray(p90_pred, dtype=float)), 0, 6).astype(int)
    sched_arr = np.asarray(scheduled_days, dtype=int)

    tier_rows: list[dict[str, Any]] = []
    for tier in sorted(np.unique(sched_arr)):
        mask = sched_arr == tier
        y_t = y_arr[mask]
        p50_t = p50_arr[mask]
        p90_t = p90_arr[mask]

        tier_rows.append(
            {
                "scheduled_days": int(tier),
                "shipping_tier": SCHEDULED_TIER_NAMES.get(
                    int(tier), f"Tier {int(tier)}d"
                ),
                "n_samples": int(np.sum(mask)),
                "actual_mean_delay": float(np.mean(y_t)),
                "actual_max_delay": int(np.max(y_t)),
                "p50_mean_pred": float(np.mean(p50_t)),
                "p50_exact_accuracy": float(accuracy_score(y_t, p50_t)),
                "p50_within_1_day_accuracy": float(np.mean(np.abs(y_t - p50_t) <= 1)),
                "p50_mae": float(mean_absolute_error(y_t, p50_t)),
                "p50_late_breach_pct": float(np.mean(y_t > p50_t) * 100.0),
                "p90_mean_buffer": float(np.mean(p90_t)),
                "p90_sla_compliance_pct": float(np.mean(y_t <= p90_t) * 100.0),
                "p90_late_breach_pct": float(np.mean(y_t > p90_t) * 100.0),
                "p90_pinball_loss": float(mean_pinball_loss(y_t, p90_t, alpha=0.90)),
                "p90_mean_cushion_days": float(np.mean(np.maximum(0, p90_t - y_t))),
            }
        )
    return tier_rows


def evaluate_p50_p90(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    p50_pred: Sequence[int | float] | np.ndarray | pd.Series,
    p90_pred: Sequence[int | float] | np.ndarray | pd.Series,
    scheduled_days: Sequence[int] | np.ndarray | pd.Series | None = None,
    model_name: str = "LightGBM Quantile",
    train_time_p50: float = 0.0,
    train_time_p90: float = 0.0,
    pred_time_sec: float = 0.0,
    p50_raw: np.ndarray | None = None,
    p90_raw: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate a joint (P50, P90) prediction pair on all standard and operational metrics.

    Calculates separate evaluation reports for P50 and P90, verifies quantile non-crossing
    integrity (p90 >= p50), evaluates interval coverage between P50 and P90, generates
    confusion matrices, and optionally includes tier-level performance breakdowns.

    Args:
        y_true: Ground-truth integer shipping transit or delay days.
        p50_pred: Discrete P50 predictions.
        p90_pred: Discrete P90 predictions.
        scheduled_days: Optional array of scheduled shipping days for tier-level diagnostics.
        model_name: Name of the model family being evaluated.
        train_time_p50: Training time in seconds for the P50 estimator.
        train_time_p90: Training time in seconds for the P90 estimator.
        pred_time_sec: Joint inference time in seconds.
        p50_raw: Optional unrounded continuous P50 predictions for continuous pinball loss check.
        p90_raw: Optional unrounded continuous P90 predictions for continuous pinball loss check.

    Returns:
        dict[str, Any]: Structured dictionary with `p50`, `p90`, `joint`, and optional `tier_breakdown` reports.
    """
    y_arr = np.asarray(y_true, dtype=int)
    p50_arr = np.clip(np.round(np.asarray(p50_pred, dtype=float)), 0, 6).astype(int)
    p90_arr = np.clip(np.round(np.asarray(p90_pred, dtype=float)), 0, 6).astype(int)

    p50_metrics = evaluate_predictions(
        y_true=y_arr,
        y_pred=p50_arr,
        model_name=f"{model_name} (P50)",
        train_time_sec=train_time_p50,
        pred_time_sec=pred_time_sec,
    )
    p90_metrics = evaluate_predictions(
        y_true=y_arr,
        y_pred=p90_arr,
        model_name=f"{model_name} (P90)",
        train_time_sec=train_time_p90,
        pred_time_sec=pred_time_sec,
    )

    if p50_raw is not None:
        p50_metrics["continuous_pinball_loss_p50"] = float(
            mean_pinball_loss(y_arr, np.asarray(p50_raw, dtype=float), alpha=0.50)
        )
        p50_metrics["continuous_mae"] = float(
            mean_absolute_error(y_arr, np.asarray(p50_raw, dtype=float))
        )
    if p90_raw is not None:
        p90_metrics["continuous_pinball_loss_p90"] = float(
            mean_pinball_loss(y_arr, np.asarray(p90_raw, dtype=float), alpha=0.90)
        )

    joint_summary: dict[str, Any] = {
        "n_samples": len(y_arr),
        "quantile_non_crossing_rate": float(np.mean(p90_arr >= p50_arr)),
        "mean_interval_width_days": float(np.mean(p90_arr - p50_arr)),
        "interval_coverage_p50_to_p90_pct": float(
            np.mean((y_arr >= p50_arr) & (y_arr <= p90_arr)) * 100.0
        ),
        "late_breach_reduction_pct_points": float(
            p50_metrics["late_breach_pct"] - p90_metrics["late_breach_pct"]
        ),
        "p50_confusion_matrix": confusion_matrix(
            y_arr, p50_arr, labels=DELAY_CLASSES
        ).tolist(),
        "p90_confusion_matrix": confusion_matrix(
            y_arr, p90_arr, labels=DELAY_CLASSES
        ).tolist(),
    }

    report: dict[str, Any] = {
        "model_name": model_name,
        "p50": p50_metrics,
        "p90": p90_metrics,
        "joint": joint_summary,
    }

    if scheduled_days is not None:
        report["tier_breakdown"] = evaluate_by_scheduled_tier(
            y_true=y_arr,
            p50_pred=p50_arr,
            p90_pred=p90_arr,
            scheduled_days=scheduled_days,
        )

    return report
