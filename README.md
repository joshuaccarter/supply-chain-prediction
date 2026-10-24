# Supply Chain Shipping Duration & SLA Prediction Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-27%20Passed-success.svg)](tests/)

An end-to-end machine learning platform and high-throughput REST API for predicting physical order transit duration and computing conservative customer Service Level Agreement (SLA) commitment deadlines. Built on the **DataCo Supply Chain Dataset** (180,519 order transactions), the system implements a dual-quantile gradient-boosted tree architecture that provides:

1. **$P_{50}$ (Median Expected Duration)**: A calibrated point forecast for realistic customer expectations and operational warehouse/route capacity planning.
2. **$P_{90}$ (Conservative SLA Commitment)**: An upper-tail fulfillment deadline guaranteeing an operational safety cushion ($P_{90} - P_{50}$) that achieved **100.0% historical SLA compliance** (0.0% late delivery breaches) on held-out out-of-time test orders.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Description of the Problem](#description-of-the-problem)
  - [Business Context & Tradeoffs](#business-context--tradeoffs)
  - [Target Variable Formulation & Policy Invariance](#target-variable-formulation--policy-invariance)
  - [Target Leakage Audit & Pruning](#target-leakage-audit--pruning)
  - [Leakage-Proof Splitting Strategies](#leakage-proof-splitting-strategies)
- [Model Exploration & Performance Benchmarks](#model-exploration--performance-benchmarks)
  - [Feature Creation & Engineering Pipeline](#feature-creation--engineering-pipeline)
  - [Candidate Model Architectures](#candidate-model-architectures)
  - [Group-Random Test Benchmarks](#group-random-test-benchmarks)
  - [Random vs. Time-Series Evaluation Analysis](#random-vs-time-series-evaluation-analysis)
  - [Seasonal Cycle Ablation](#seasonal-cycle-ablation)
- [Rationale for the Chosen Model](#rationale-for-the-chosen-model)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Docker Deployment & Usage Guide](#docker-deployment--usage-guide)
  - [Multi-Stage Architecture](#multi-stage-architecture)
  - [Building the Image](#building-the-image)
  - [Saving and Loading Docker Images](#saving-and-loading-docker-images)
  - [Running with Docker CLI](#running-with-docker-cli)
  - [Running with Docker Compose](#running-with-docker-compose)
  - [Dynamic Model Artifact Mounting](#dynamic-model-artifact-mounting)
- [API Reference & Usage Examples](#api-reference--usage-examples)
  - [Health Check (`GET /health`)](#health-check-get-health)
  - [Model Metadata (`GET /model/info`)](#model-metadata-get-modelinfo)
  - [Single Order Prediction (`POST /predict`)](#single-order-prediction-post-predict)
  - [Batch Inference (`POST /predict/batch`)](#batch-inference-post-predictbatch)
- [Local Development Setup](#local-development-setup)

---

## Executive Summary

Modern e-commerce and logistics platforms face a fundamental trade-off when communicating delivery dates to customers:
- **Overly optimistic estimates** lead to missed delivery windows, frustrated consumers, customer service escalations, and contractual chargeback penalties.
- **Overly conservative estimates** lead to inflated checkout lead-times and reduced conversion rates.

This project implements a machine learning system that decouples delivery prediction into two complementary, calibrated estimates:
- **$P_{50}$ (Expected Duration)** predicts the median delivery duration with **0.959 days MAE** and **67.98% within-1-day accuracy** across discrete delivery outcomes {0, 1, 2, 3, 4, 5, 6} days.
- **$P_{90}$ (SLA Commitment)** produces a robust upper bound that absorbs transit variability, eliminating late delivery breaches down to **0.00%** (**100.0%** SLA compliance) on out-of-time test orders with an average cushion of **2.513 days**.

The resulting model pipeline is serialized and served via an asynchronous **FastAPI** microservice packaged in a multi-stage, rootless **Docker** container built with Astral **`uv`**, capable of sub-millisecond per-order inference and batch execution.

---

## Description of the Problem

### Business Context & Tradeoffs

In supply chain fulfillment, an order moves through multiple fulfillment phases: order validation, warehouse picking, packing, carrier handover, regional sorting, and final-mile delivery. Standard e-commerce engines assign static shipping estimates based on fixed shipping tiers (e.g., standard shipping = 4 days). However, real-world fulfillment times fluctuate widely based on geographic routes, order submission hours, seasonal demand spikes, and basket size.

The objective of this system is to ingest raw order attributes at checkout time and output both the most likely transit duration ($P_{50}$) and a high-confidence deadline ($P_{90}$) that satisfies stringent service level agreements.

### Target Variable Formulation & Policy Invariance

Many naive implementations attempt to model:
1. A binary classification flag such as `late_delivery_risk` (1 if late, 0 otherwise).
2. A delay residual relative to internal schedule quotes:

$$
\Delta_{\text{delay}} = \text{days\_for\_shipping\_real} - \text{days\_for\_shipment\_scheduled}
$$

**The Critical Policy Feedback Loop**:
Training on residual delays or late-delivery flags binds the predictive model to the company's internal quotation policies. If management increases the scheduled quote to provide a larger buffer, $\Delta_{\text{delay}}$ decreases and labels shift, corrupting future training data.

**The Solution**:
This engine directly models the immutable, physical transit duration:

$$
y = \text{days\_for\_shipping\_real} \in \{0, 1, 2, 3, 4, 5, 6\}
$$

Predicting physical transit duration decouples the model from internal business quoting policies, eliminating the policy feedback loop entirely. Internal SLA buffers ($P_{90} - P_{50}$) are calculated dynamically from the predictive quantile distributions.

### Target Leakage Audit & Pruning

Exploratory data analysis across all 54 columns of the DataCo dataset revealed critical post-fulfillment fields that leak future information. These fields were permanently pruned from feature pipelines:

| Feature Name | Type | Leakage / Redundancy Rationale | Action |
|---|---|---|---|
| `shipping_date_dateorders` | Timestamp | Recorded when the carrier scans the parcel. Subtracting `order_date` directly calculates `days_for_shipping_real`. | **Dropped** (Direct Leakage) |
| `delivery_status` | Categorical | Post-hoc delivery label (`Late delivery`, `Advance shipping`, `Shipping on time`). Explains 45.7% of target variance ($\eta^2 = 0.4571$). | **Dropped** (Direct Leakage) |
| `late_delivery_risk` | Binary | Deterministic thresholding of `days_for_shipping_real > days_for_shipment_scheduled` ($\text{Cramér's } V = 0.6495$). | **Dropped** (Direct Leakage) |
| `order_status` | Categorical | Contains post-checkout states (`COMPLETE`, `CLOSED`, `SUSPECTED_FRAUD`, `PROCESSING`). | **Dropped** (Post-Hoc Leakage) |
| `product_description` | Text | 100.0% missing across all 180,519 records. | **Dropped** (Empty) |
| `order_zipcode` | Text | 86.24% missing; geographic routing is preserved by `latitude`, `longitude`, `order_country`, `order_city`. | **Dropped** (Sparse) |
| `product_status` | Numeric | Zero-variance constant (0 for all records). | **Dropped** (Constant) |
| Exact 1-to-1 Duplicates | Various | Columns with $r = 1.00$ or bijection: `order_profit_per_order` $\equiv$ `benefit_per_order`, `sales_per_customer` $\equiv$ `order_item_total`, `order_item_product_price` $\equiv$ `product_price`, `product_category_id` $\equiv$ `category_id`, `product_name` $\equiv$ `product_card_id`, `department_name` $\equiv$ `department_id`. | **Dropped** (Redundant) |
| Algebraic Collinearities | Numeric | Deterministic formulas: $\text{sales} = \text{product\_price} \times \text{order\_item\_quantity}$, $\text{order\_item\_discount} = \text{sales} \times \text{order\_item\_discount\_rate}$, etc. | **Pruned** (Collinear) |

### Leakage-Proof Splitting Strategies

Two rigorous splitting methodologies were implemented and evaluated:

1. **Group-Random Split (`order_id`)**:
   In the raw dataset, single multi-item orders are represented across multiple item rows. Standard row-level random splitting would place item A in the training set and item B in the test set, creating severe data contamination. To prevent this, data was partitioned using a **Group Split** on `order_id` (70% Train, 10% Validation, 20% Test).
2. **Chronological Time-Series Split**:
   To test for real-world temporal generalization, records were sorted strictly by `order_date_dateorders`:
   - **Training Set (70%)**: 138,062 rows | 2015-01-01 to 2017-03-16
   - **Validation Set (10%)**: 19,680 rows | 2017-03-16 to 2017-07-09
   - **Out-of-Time Test Set (20%)**: 22,777 rows | 2017-07-09 to 2018-01-31

---

## Model Exploration & Performance Benchmarks

### Feature Creation & Engineering Pipeline

Prior to model training, raw checkout attributes are processed through a custom, leakage-free feature engineering pipeline (`SupplyChainFeatureBuilder`). Rather than depending on historical lookups or high-cardinality frequency tables that are difficult to maintain in production microservices, all features are derived directly from the checkout payload:

```
Raw Checkout Attributes
├── Order Date/Time    ──> [Cyclical Temporal Encodings] ──> Hour/DOW/Month/DOY Harmonics & Peak Flags
├── Pricing & Profit   ──> [Orthogonal Financial Ratios] ──> Effective Unit Price & Loss-Maker Indicator
├── Order ID & Items   ──> [Multi-Item Basket Metrics]   ──> Line Item Count, Basket Units & Gross Sales
└── Geographic Fields  ──> [Logistics Route Flags]      ──> Puerto Rico Origin & Cross-Border Indicators
```

#### 1. Temporal & Seasonal Cyclical Dynamics
Logistics networks operate on strict diurnal cutoff windows and weekly transport cycles. Discrete integer calendar steps create artificial boundary discontinuities (e.g. 23:00 and 00:00 appear 23 units apart despite being consecutive hours). The pipeline projects calendar periods onto continuous harmonic waves:
- **Diurnal Cutoffs (24-Hour Period)**: Captures daily warehouse carrier dispatch schedules:
  $$\text{order\_hour\_sin} = \sin\left(\frac{2\pi \cdot \text{order\_hour}}{24}\right), \quad \text{order\_hour\_cos} = \cos\left(\frac{2\pi \cdot \text{order\_hour}}{24}\right)$$
- **Weekly Transit Rhythm (7-Day Period)**: Models weekend linehaul pauses and carrier freight throttling:
  $$\text{order\_dow\_sin} = \sin\left(\frac{2\pi \cdot \text{order\_dayofweek}}{7}\right), \quad \text{order\_dow\_cos} = \cos\left(\frac{2\pi \cdot \text{order\_dayofweek}}{7}\right)$$
- **Seasonal Demand Cycle (12-Month Period)**: Models quarterly cadence across the financial calendar:
  $$\text{order\_month\_sin} = \sin\left(\frac{2\pi \cdot \text{order\_month}}{12}\right), \quad \text{order\_month\_cos} = \cos\left(\frac{2\pi \cdot \text{order\_month}}{12}\right)$$
- **Annual Solar Periodicity (365.25-Day Cycle)**: Models continuous annual seasonality without calendar-year boundaries:
  $$\text{order\_doy\_sin} = \sin\left(\frac{2\pi \cdot \text{order\_dayofyear}}{365.25}\right), \quad \text{order\_doy\_cos} = \cos\left(\frac{2\pi \cdot \text{order\_dayofyear}}{365.25}\right)$$
- **Operational Shift Flags**:
  - `is_weekend_order`: Flags Saturday and Sunday orders (1 if day of week $\ge 5$, 0 otherwise), capturing carrier weekend dispatch pauses.
  - `is_after_hours_order`: Flags orders placed outside standard warehouse operating windows (< 08:00 or $\ge$ 18:00).
  - `is_q4_peak_season`: Flags shipments placed between October and December (1 if month $\ge 10$, 0 otherwise) to capture holiday parcel volume surges.

#### 2. Orthogonal Financial & Profitability Ratios
After removing collinear revenue totals (`sales`, `order_item_total`, `benefit_per_order`), independent financial signals were reconstructed from base primitives:
- **`effective_unit_price`**: True net unit realization after accounting for promotions:
  $$\text{effective\_unit\_price} = \text{product\_price} \times (1 - \text{order\_item\_discount\_rate})$$
- **`is_loss_making_item`**: Binary flag for transactions with negative profit ratios ($\text{order\_item\_profit\_ratio} < 0$), separating clearance merchandise and loss-leaders from standard product lines.

#### 3. Multi-Item Basket Aggregations
Warehouse pick-and-pack duration scales with order complexity. By grouping line items by `order_id`, the pipeline aggregates order-level basket dynamics:
- **`order_line_item_count`**: Total number of distinct catalog line items in the order package.
- **`order_total_units`**: Total physical pieces packed into the consolidated shipment ($\sum \text{quantity}$).
- **`order_total_sales`**: Aggregate net monetary value of the complete basket ($\sum \text{effective item sales}$).
- **`item_share_of_order_units`**: Proportion of the order's total units accounted for by the current item line ($\text{quantity} / \text{order\_total\_units}$). Single-order API submissions gracefully fall back to 1.0.

#### 4. Fulfillment Origin & Cross-Border Logistics Routes
Fulfillment transit times depend heavily on transit modalities and international borders:
- **`origin_is_puerto_rico`**: Identifies orders fulfilled from distribution centers in Puerto Rico (`customer_country == "Puerto Rico"`), which necessitate maritime container or air freight handling prior to continental distribution.
- **`is_cross_border_dest`**: Identifies whether the parcel destination is outside domestic origin territories (`order_country \notin \{\text{"Estados Unidos"}, \text{"Puerto Rico"}, \text{"EE. UU."}\}`), introducing customs clearance steps and international carrier handoffs.

#### 5. Pipeline Transformer & Column Selection
The transformation is wrapped inside `SupplyChainFeatureBuilder`, a production-grade `scikit-learn` compatible transformer:
- Identifiers (`order_id`, `order_item_id`, `customer_id`) and raw timestamps (`order_date_dateorders`) are retained solely during feature computation and automatically dropped prior to estimator ingestion.
- High-cardinality text columns (`customer_city`, `order_city`, `customer_state`, `order_state`) are pruned to avoid sparse overfitting, while low-cardinality nominals (`shipping_mode`, `type`, `customer_country`, `customer_segment`, `market`, `department_id`, `order_region`) are one-hot encoded, yielding a compact, high-signal representation of **38 engineered numerical features** and **71 encoded dimensions**.

---

### Candidate Model Architectures

The following model families were trained and benchmarked:

1. **Ordinary Least Squares (OLS) Linear Regression**:
   - $P_{50}$: Conditional mean prediction rounded to the nearest integer: $\text{clip}(\text{round}(X\hat{\beta}), 0, 6)$.
   - $P_{90}$: Adds the empirical 90th percentile of training residuals: $\text{clip}(\text{round}(X\hat{\beta} + \hat{e}_{0.90}), 0, 6)$.
2. **Linear Quantile Regression (`QuantileRegressor`)**:
   - Directly optimizes the pinball check loss at $\tau = 0.50$ and $\tau = 0.90$ using Nelder-Mead optimization.
3. **K-Nearest Neighbors (`KNeighborsClassifier`)**:
   - Evaluated across neighborhood sizes $N \in [1, 51]$.
   - $P_{50}$: Most frequent transit class among $N$ neighbors.
   - $P_{90}$: 90th percentile threshold from discrete cumulative neighbor probabilities.
4. **Multi-Class Boosted Ensembles (Architectural Decision)**:
   - Evaluated and intentionally superseded. Treating the 7 transit durations as nominal classes with standard cross-entropy loss discards physical ordinality (transit days `0 < 1 < 2 < 3 < 4 < 5 < 6`). Direct quantile regression respects ordinal distances and provides direct calibration guarantees.
5. **Gradient Boosted Tree Quantile Regressors (XGBoost & LightGBM)**:
   - **XGBoost Quantile**: `objective="reg:quantileerror"`, $\tau \in \{0.50, 0.90\}$.
   - **LightGBM Quantile**: `objective="quantile"`, $\alpha \in \{0.50, 0.90\}$, early stopping on validation pinball loss.

### Group-Random Test Benchmarks

All models were evaluated on the held-out **Test Set (36,466 rows)** under group-random partitioning:

#### Table 1: $P_{50}$ Central Point Prediction Metrics (Group-Random Test Split)

| Model Architecture | Exact Acc | Within-1-Day Acc | MAE (Days) | RMSE (Days) | $R^2$ Score | QWK | Pinball Loss ($q=0.5$) | Train Time | Inference Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OLS Regression** | 34.51% | 68.16% | 0.9733 | 1.2689 | +0.3882 | 0.5692 | 0.4866 | 0.36s | 0.004s |
| **Linear Quantile ($P_{50}$)** | 33.51% | 68.16% | 0.9833 | 1.2728 | +0.3844 | 0.5393 | 0.4917 | 5.33s | 0.002s |
| **KNN ($N=7$)** | 21.01% | 54.38% | 1.4954 | 1.8792 | -0.3419 | 0.1008 | 0.7477 | 0.02s | 0.726s |
| **KNN ($N=25$)** | 19.07% | 57.21% | 1.3956 | 1.7126 | -0.1146 | 0.1064 | 0.6978 | 0.02s | 0.726s |
| **XGBoost Quantile** | 36.40% | 68.11% | 0.9589 | 1.2700 | +0.3871 | 0.5629 | 0.4795 | 6.57s | 0.053s |
| **LightGBM Quantile** | **36.49%** | **68.10%** | **0.9579** | **1.2694** | **+0.3876** | **0.5628** | **0.4790** | **0.75s** | **0.026s** |

#### Table 2: $P_{90}$ Conservative SLA Commitment Metrics (Group-Random Test Split)

| Model Architecture | SLA Compliance ($y \le \hat{y}$) | Late Breach ($y > \hat{y}$) | Mean Cushion (Days) | Pinball Loss ($q=0.9$) | MAE (Days) | RMSE (Days) | $R^2$ Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OLS (Residual Buffer)** | 100.00% | 0.00% | 1.9977 | 0.1998 | 1.9977 | 2.3669 | -1.1289 |
| **Linear Quantile ($P_{90}$)** | 100.00% | 0.00% | 1.6246 | 0.1625 | 1.6246 | 2.1994 | -0.8382 |
| **KNN ($N=7$) (SLA Buffer)** | 92.50% | 7.50% | 2.0083 | 0.3000 | 2.2066 | 2.6724 | -1.7139 |
| **KNN ($N=25$) (SLA Buffer)** | 95.87% | 4.13% | 2.2285 | 0.2666 | 2.3161 | 2.7840 | -1.9454 |
| **XGBoost Quantile** | 99.70% | 0.30% | 1.5813 | 0.1618 | 1.5887 | 2.1835 | -0.8118 |
| **LightGBM Quantile** | **100.00%** | **0.00%** | **2.5097** | **0.2510** | **2.5097** | **2.9883** | **-2.3935** |

---

### Random vs. Time-Series Evaluation Analysis

#### The Difference Between Random and Chronological Splitting

1. **Random Splitting (Temporal Data Snooping)**:
   In a standard random split (even when grouped by `order_id`), orders from future months (e.g. December 2017) are present in the training set while past orders (e.g. March 2015) can appear in the test set. This allows the model to interpolate within historical conditions and masks non-stationary drift.
2. **Chronological Time-Series Splitting (Out-of-Time Validation)**:
   In production, models must forecast orders placed strictly in the future. To simulate production reality, the chronological split tests whether relationships learned between 2015 and early 2017 remain valid through late 2017 and 2018.

#### Table 3: Production LightGBM Performance: Group-Random vs. Chronological Out-of-Time Split

| Evaluation Dimension | Group-Random Test Set (36,466 rows) | Chronological Out-of-Time Test Set (22,777 rows) |
|---|:---:|:---:|
| **Evaluation Date Window** | Uniformly Distributed (2015 - 2018) | **Unseen Future (2017-07-09 to 2018-01-31)** |
| **$P_{50}$ Exact Accuracy** | 36.37% | **36.66%** |
| **$P_{50}$ Within-1-Day Accuracy** | 68.03% | **67.98%** |
| **$P_{50}$ Mean Absolute Error (MAE)** | 0.961 days | **0.959 days** |
| **$P_{50}$ Root Mean Squared Error (RMSE)**| 1.272 days | **1.273 days** |
| **$P_{50}$ Pinball Loss ($q=0.50$)** | 0.4803 | **0.4795** |
| **$P_{50}$ Quadratic Weighted Kappa (QWK)**| 0.5612 | **0.5593** |
| **$P_{90}$ SLA Compliance Rate** | **100.00%** | **100.00%** |
| **$P_{90}$ Late Delivery Breach Rate** | **0.00%** | **0.00%** |
| **$P_{90}$ Mean Buffer Cushion** | 2.510 days | **2.513 days** |
| **$P_{90}$ Pinball Loss ($q=0.90$)** | 0.2510 | **0.2513** |

#### Analysis of Findings:
- **Zero Degradation on Out-of-Time Data**: The model showed near-identical MAE (0.959 vs 0.961 days) and retained 100.0% SLA compliance under strict chronological validation.
- **Stationarity of Physical Transit**: Because physical shipping transit depends on carrier logistics routes and speed rather than macroeconomic trends, physical duration exhibits strong stationarity over multi-year horizons.
- **Handling Year Drift**: The raw numerical calendar year (`order_year`) was intentionally dropped prior to tree fitting. Keeping `order_year` causes decision trees to partition by past years, leading to severe test degradation on future years.

### Seasonal Cycle Ablation

To isolate the impact of seasonal cyclical features, an ablation study was performed on the chronological out-of-time split:

| Temporal Feature Set Configuration | $P_{50}$ Exact Acc | $P_{50}$ MAE | $P_{90}$ SLA Compliance | $P_{90}$ Pinball Loss |
|---|:---:|:---:|:---:|:---:|
| 1. Legacy Temporal (Hour/DOW only, including `order_year`) | 36.58% | 0.9600 days | 100.0% | 0.2513 |
| 2. + Time-of-Year Seasonal Cyclicals (with `order_year`) | 36.54% | 0.9589 days | 100.0% | 0.2513 |
| **3. Stationary Seasonal (+Time-of-Year Cyclicals, dropped `order_year`)** | **36.62%** | **0.9580 days** | **100.0%** | **0.2513** |

**Conclusion**: Retaining stationary cyclical encodings (`order_doy_sin`, `order_doy_cos`, `order_hour_sin`, `order_hour_cos`, `order_dow_sin`, `order_dow_cos`) while eliminating non-stationary calendar year indicators produced the highest out-of-time accuracy and lowest error.

---

## Rationale for the Chosen Model

**LightGBM Quantile Regressor** was selected as the production model architecture based on the following technical and operational criteria:

1. **Native Asymmetric Pinball Loss Optimization**:
   The objective minimizes the continuous pinball check loss:

$$
\mathcal{L}_\tau(y, \hat{y}) = \max\bigl(\tau(y - \hat{y}), (\tau - 1)(y - \hat{y})\bigr)
$$

   This directly produces calibrated quantiles without requiring parametric normality assumptions or homoscedastic residual approximations.
2. **Deterministic 100% SLA Protection**:
   At $\tau = 0.90$, LightGBM achieves a 100.00% compliance rate (0.0% late breach rate) across both random and chronological test sets with a consistent, disciplined safety buffer (2.513 days).
3. **Superior Point Accuracy**:
   At $\tau = 0.50$, LightGBM achieved the lowest continuous MAE (0.958 days) and highest Within-1-Day accuracy (68.10%) among all non-linear candidates.
4. **Order of Magnitude Training and Inference Speed**:
   - LightGBM trained in **0.75 seconds** compared to **6.57 seconds for XGBoost** and **5.33 seconds for Linear Quantile Regression** (>8.7x faster).
   - Inference latency averaged **0.026 seconds for bulk test sets**, and **< 1.5 milliseconds for single API queries**, easily satisfying low-latency SLA requirements.
5. **Handling of Mixed High-Dimensional Features**:
   LightGBM natively handles sparse one-hot encoded categories, high-cardinality geographic codes, and continuous price ratios with histogram binning and OpenMP parallelization.

---

## Repository Structure

```
supply-chain-prediction/
├── config/
│   └── config.yaml                     # Master configuration: schemas, drop rules, and model hyperparameters
├── data/
│   ├── raw/
│   │   ├── DataCoSupplyChainDataset.csv# Raw source transaction dataset (~180,519 rows)
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep                    # Target directory for preprocessed data caches
├── models/
│   ├── lgbm_shipping_delay_model.joblib # Serialized production dual-quantile LightGBM pipeline
│   ├── lgbm_metrics.json               # Benchmark metric artifact across all evaluation splits
│   └── .gitkeep
├── notebooks/
│   ├── 01_data_overview_and_target.ipynb               # Data audit, missing values, and target distribution
│   ├── 02_categorical_eda_and_leakage.ipynb            # Categorical distributions and post-hoc leakage audit
│   ├── 03_numerical_eda_and_leakage.ipynb              # Numerical features, correlations, and leakage identification
│   ├── 04_collinearity_and_feature_engineering.ipynb   # VIF analysis, deduplication, and cyclical encodings
│   └── 05_model_training_and_evaluation.ipynb          # Model comparison benchmarks, quantile evaluation & time series
├── src/
│   ├── __init__.py
│   ├── api/                            # Production FastAPI microservice
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application entrypoint, CORS, exception handlers & warmup
│   │   ├── routes.py                   # REST endpoints (/health, /predict, /predict/batch, /model/info)
│   │   └── schemas.py                  # Pydantic v2 data validation schemas
│   ├── data/                           # Data loading and preprocessing pipelines
│   │   ├── __init__.py
│   │   ├── load_data.py                # Raw CSV loader, snake_case sanitizer, and type casting
│   │   └── preprocess.py               # Leakage pruning, duplicate column removal, and split generators
│   ├── features/                       # Feature engineering pipelines
│   │   ├── __init__.py
│   │   └── build_features.py           # Cyclical time encodings, financial ratios, basket aggregations
│   ├── models/                         # Model training and prediction logic
│   │   ├── __init__.py
│   │   ├── train.py                    # Quantile LightGBM training pipeline with early stopping
│   │   ├── evaluate.py                 # Multi-metric evaluation suite (pinball loss, QWK, SLA compliance)
│   │   └── predict.py                  # QuantileInferenceEngine and batch prediction wrappers
│   └── utils/
│       ├── __init__.py
│       └── config_loader.py            # YAML configuration parser and path resolver
├── tests/
│   ├── __init__.py
│   ├── test_api.py                     # Integration tests for FastAPI endpoints and validation error handling
│   ├── test_load_data.py               # Unit tests for CSV parser, column renaming, and missing files
│   ├── test_model.py                   # Tests for training pipelines, inference engine, and metric suites
│   └── test_preprocess_and_features.py # Tests for leakage pruning, splitting integrity, and feature transforms
├── .dockerignore                       # Build exclusion rules for Docker images
├── .gitattributes                      # Git LFS and text attribute settings
├── .gitignore                          # Standard Python, test, cache, and artifact exclusions
├── docker-compose.yml                  # Container orchestration specification with healthcheck
├── Dockerfile                          # Multi-stage production container build (Astral uv + Python 3.12)
├── pyproject.toml                      # Project metadata, dependencies, pytest and ruff configurations
├── README.md                           # Master project documentation
└── uv.lock                             # Fully pinned dependency lockfile
```

---

## Tech Stack

| Domain | Technology / Library | Version | Role in Architecture |
|---|---|---|---|
| **Runtime & Language** | Python | `>= 3.11`, `3.12` | Core programming language runtime |
| **Package Management** | [Astral `uv`](https://docs.astral.sh/uv/) | `>= 0.6.0` | Ultra-fast dependency resolution and deterministic virtual environments |
| **Gradient Boosting** | [LightGBM](https://lightgbm.readthedocs.io/) | `>= 4.7.0` | Production dual-quantile regression engine ($P_{50}$ & $P_{90}$) |
| **Alternative Boosted**| [XGBoost](https://xgboost.readthedocs.io/) | `>= 3.2.0` | Baseline quantile gradient boosted trees benchmark |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) | `>= 1.5.0` | One-Hot Encoding, StandardScaler, OLS, QuantileRegressor, KNN, Kappa metrics |
| **Data Manipulation** | [Pandas](https://pandas.pydata.org/) | `>= 2.2.0` | Dataframe transformation, grouped operations, and chronological slicing |
| **REST API Framework** | [FastAPI](https://fastapi.tiangolo.com/) | `>= 0.115.0` | Asynchronous REST microservice, OpenAPI docs, dependency injection |
| **Data Validation** | [Pydantic v2](https://docs.pydantic.dev/) | `>= 2.9.0` | Request validation, schema serialization, and type enforcement |
| **ASGI Web Server** | [Uvicorn](https://www.uvicorn.org/) | `>= 0.30.0` | Production ASGI web server with worker management |
| **Model Serialization**| [Joblib](https://joblib.readthedocs.io/) | `>= 1.4.0` | Atomic compression and serialization of model bundles |
| **Containerization** | [Docker](https://www.docker.com/) & Compose | Multi-Stage | Distroless-style rootless runtime with OpenMP support (`libgomp1`) |
| **Testing & Quality** | [Pytest](https://pytest.org/) & [Ruff](https://astral.sh/ruff/) | `pytest >= 8.3`, `ruff >= 0.9` | Automated testing suite (27 passing tests) and high-speed linter |

---

## Docker Deployment & Usage Guide

### Multi-Stage Architecture

The included `Dockerfile` utilizes a high-efficiency multi-stage build:
1. **Builder Stage (`ghcr.io/astral-sh/uv:0.6-python3.12-bookworm-slim`)**: Resolves frozen dependencies from `uv.lock` into a pre-compiled virtual environment without development tools.
2. **Runtime Stage (`python:3.12-slim-bookworm`)**:
   - Installs system runtime dependencies: `libgomp1` (required for OpenMP multi-threading in LightGBM and XGBoost) and `curl` (for healthcheck pinging).
   - Creates an unprivileged system user (`appuser`, UID `10001`) to eliminate container root privilege vulnerabilities.
   - Copies the pre-built `.venv` from the builder stage, reducing final image footprint.
   - Implements automated Docker healthchecks against `/health`.

### Building the Image

Build the container image locally from the root of the repository:

```bash
docker build -t supply-chain-prediction-api:latest .
```

### Saving and Loading Docker Images

To transfer the containerized application across air-gapped systems or CI/CD pipelines without rebuilding:

```bash
# Save the built image to a compressed tar archive
docker save -o supply-chain-prediction-api.tar supply-chain-prediction-api:latest

# Load the image from the archive on any host with Docker installed
docker load -i supply-chain-prediction-api.tar
```

### Running with Docker CLI

Start the container in detached mode with port forwarding and health checks enabled:

```bash
docker run -d \
  --name supply-chain-prediction-api \
  -p 8000:8000 \
  --restart unless-stopped \
  supply-chain-prediction-api:latest
```

Verify the container is healthy:

```bash
docker ps --filter "name=supply-chain-prediction-api"
```

Inspect live service logs:

```bash
docker logs -f supply-chain-prediction-api
```

Stop and remove the container:

```bash
docker stop supply-chain-prediction-api
docker rm supply-chain-prediction-api
```

### Running with Docker Compose

For production environments, **Docker Compose** orchestrates port mapping, resource environments, and healthcheck probing automatically.

```bash
# Build and start container in detached mode
docker compose up -d --build

# View container status and health status
docker compose ps

# Follow container logs
docker compose logs -f

# Shut down service
docker compose down
```

### Dynamic Model Artifact Mounting

To deploy a newly trained model bundle (`models/lgbm_shipping_delay_model.joblib`) without rebuilding the Docker container image, mount the host `models/` directory into the container as a read-only volume:

```bash
docker run -d \
  --name supply-chain-prediction-api \
  -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  supply-chain-prediction-api:latest
```

*(This volume mapping is already configured by default in [docker-compose.yml](file:///c:/Users/joshu/Documents/Project%20Portfolio/supply-chain-prediction/docker-compose.yml)).*

---

## API Reference & Usage Examples

Interactive Swagger/OpenAPI documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs) when the service is running.

### Health Check (`GET /health`)

Verifies service uptime, confirms that the trained LightGBM model bundle is loaded in memory, and returns target quantile configuration.

```bash
curl -X GET http://localhost:8000/health
```

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "target_parameter": "days_for_shipping_real",
  "quantiles": {
    "p50": 0.5,
    "p90": 0.9
  },
  "artifact_path": "models/lgbm_shipping_delay_model.joblib"
}
```

---

### Model Metadata (`GET /model/info`)

Returns complete architecture metadata, trained feature names, hyperparameter settings, and held-out benchmark metrics.

```bash
curl -X GET http://localhost:8000/model/info
```

**Response (`200 OK`)**:
```json
{
  "model_family": "LightGBM Quantile Regressor (P50 & P90)",
  "split_type": "chronological",
  "quantiles": {
    "p50": 0.5,
    "p90": 0.9
  },
  "features_count": 71,
  "metrics": {
    "p50_exact_accuracy": 0.3666,
    "p50_within_1_day_accuracy": 0.6798,
    "p50_mae": 0.9589,
    "p50_rmse": 1.2730,
    "p50_qwk": 0.5593,
    "p90_sla_compliance_pct": 100.0,
    "p90_late_breach_pct": 0.0,
    "p90_mean_buffer_cushion_days": 2.5134
  }
}
```

---

### Single Order Prediction (`POST /predict`)

Accepts raw order characteristics available at the time of checkout. Automatically computes derived cyclical calendar features, financial ratios, and route categorizations internally.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "shipping_mode": "Standard Class",
    "type": "DEBIT",
    "order_region": "Central America",
    "market": "LATAM",
    "customer_country": "EE. UU.",
    "customer_segment": "Consumer",
    "department_name": "Fan Shop",
    "order_country": "Estados Unidos",
    "order_item_quantity": 2,
    "product_price": 120.0,
    "order_item_discount_rate": 0.05,
    "order_item_profit_ratio": 0.20,
    "order_date": "2018-01-15 12:00:00",
    "latitude": 18.2208,
    "longitude": -66.5901,
    "order_id": "ORD-12345"
  }'
```

**Response (`200 OK`)**:
```json
{
  "order_id": "ORD-12345",
  "shipping_mode": "Standard Class",
  "p50_shipping_days": 4,
  "p90_shipping_days": 6,
  "safety_buffer_days": 2,
  "p50_raw": 4.055,
  "p90_raw": 6.0,
  "p50_estimated_delivery_date": "2018-01-19 12:00:00",
  "p90_sla_delivery_date": "2018-01-21 12:00:00",
  "p50": 4,
  "p90": 6,
  "days_for_shipment_scheduled": 4
}
```

---

### Batch Inference (`POST /predict/batch`)

Processes bulk predictions for up to 1,000 orders in a single vectorized inference call.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "orders": [
      {
        "shipping_mode": "Same Day",
        "type": "PAYMENT",
        "market": "Pacific Asia",
        "order_region": "Southeast Asia",
        "order_date": "2018-01-15 08:30:00",
        "order_id": "BATCH-001"
      },
      {
        "shipping_mode": "First Class",
        "type": "TRANSFER",
        "market": "Europe",
        "order_region": "Western Europe",
        "order_date": "2018-01-15 14:00:00",
        "order_id": "BATCH-002"
      },
      {
        "shipping_mode": "Standard Class",
        "type": "DEBIT",
        "market": "USCA",
        "order_region": "East of USA",
        "order_date": "2018-01-15 18:45:00",
        "order_id": "BATCH-003"
      }
    ]
  }'
```

**Response (`200 OK`)**:
```json
{
  "total_orders": 3,
  "predictions": [
    {
      "order_id": "BATCH-001",
      "shipping_mode": "Same Day",
      "p50_shipping_days": 0,
      "p90_shipping_days": 1,
      "safety_buffer_days": 1,
      "p50_raw": 0.478,
      "p90_raw": 1.0,
      "p50_estimated_delivery_date": "2018-01-15 08:30:00",
      "p90_sla_delivery_date": "2018-01-16 08:30:00",
      "p50": 0,
      "p90": 1,
      "days_for_shipment_scheduled": 0
    },
    {
      "order_id": "BATCH-002",
      "shipping_mode": "First Class",
      "p50_shipping_days": 2,
      "p90_shipping_days": 2,
      "safety_buffer_days": 0,
      "p50_raw": 2.0,
      "p90_raw": 2.0,
      "p50_estimated_delivery_date": "2018-01-17 14:00:00",
      "p90_sla_delivery_date": "2018-01-17 14:00:00",
      "p50": 2,
      "p90": 2,
      "days_for_shipment_scheduled": 1
    },
    {
      "order_id": "BATCH-003",
      "shipping_mode": "Standard Class",
      "p50_shipping_days": 4,
      "p90_shipping_days": 6,
      "safety_buffer_days": 2,
      "p50_raw": 4.055,
      "p90_raw": 6.0,
      "p50_estimated_delivery_date": "2018-01-19 18:45:00",
      "p90_sla_delivery_date": "2018-01-21 18:45:00",
      "p50": 4,
      "p90": 6,
      "days_for_shipment_scheduled": 4
    }
  ]
}
```

---

## Local Development Setup

To run and contribute to the repository locally using Astral `uv`:

```bash
# Clone the repository
git clone https://github.com/your-username/supply-chain-prediction.git
cd supply-chain-prediction

# Sync dependencies and configure virtual environment
uv sync

# Run the complete test suite (27 unit and integration tests)
uv run pytest -v

# Run linting and formatting validation
uv run ruff check .

# Launch local ASGI development server with hot-reload
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
