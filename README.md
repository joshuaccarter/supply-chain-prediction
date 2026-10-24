# Supply Chain Shipping Duration & SLA Prediction Engine

Production machine learning engine and REST API for predicting physical order transit time (`days_for_shipping_real`, $\{0, 1, 2, 3, 4, 5, 6\}$ days) using dual LightGBM quantile regression:
- **$P_{50}$ (Median Expected Duration)**: Accurate expected arrival estimates for operational planning and customer delivery estimates.
- **$P_{90}$ (Conservative SLA Commitment)**: High-confidence fulfillment deadline that provides an operational safety cushion ($P_{90} - P_{50}$) and protects customer SLAs ($100.0\%$ historical SLA compliance).

---

## Architecture Overview

```
supply-chain-prediction/
├── config/
│   └── config.yaml             # Single source of truth for targets, pruning, and model hyperparameters
├── data/
│   ├── raw/                    # Original DataCo Supply Chain Dataset (CSV)
│   └── processed/              # Preprocessed data caches
├── models/
│   ├── lgbm_shipping_delay_model.joblib  # Serialized production LightGBM quantile bundle
│   └── lgbm_metrics.json       # Benchmarks across chronological & group-random evaluations
├── notebooks/                  # Step-by-step EDA and modeling notebooks (01 to 05)
├── src/
│   ├── data/                   # Data loading, cleaning, leakage pruning, and splitting
│   ├── features/               # Cyclical temporal, financial, route, and basket feature builders
│   ├── models/                 # Model training, evaluation metrics, and inference engine
│   └── api/                    # FastAPI application, Pydantic schemas, and route handlers
├── tests/                      # Full unit and integration test suite (27 tests)
├── Dockerfile                  # Multi-stage production container build (Astral uv + Python 3.12)
├── docker-compose.yml          # Production container orchestration
└── pyproject.toml              # Project dependencies and packaging metadata
```

---

## Performance Summary (Held-Out Test Set)

| Metric | $P_{50}$ (Expected Duration) | $P_{90}$ (SLA Commitment) |
|---|---|---|
| **Exact Accuracy** | 36.66% | 15.64% |
| **Within-1-Day Accuracy** | 67.98% | 31.33% |
| **Mean Absolute Error (MAE)** | 0.959 days | 2.513 days |
| **Root Mean Squared Error (RMSE)** | 1.273 days | 2.991 days |
| **$R^2$ Score** | +0.383 | -2.405 |
| **Quadratic Weighted Kappa (QWK)** | 0.559 | 0.000 |
| **SLA Compliance Rate** | 68.34% | **100.00%** |
| **Late Breach Rate** | 31.66% | **0.00%** |
| **Mean Cushion** | 0.476 days | **2.513 days** |

---

## Quickstart

### 1. Local Setup with `uv`

Ensure [uv](https://docs.astral.sh/uv/) is installed:

```bash
# Clone and enter the repository
cd supply-chain-prediction

# Install dependencies and sync virtualenv
uv sync

# Run the test suite
uv run pytest -v

# Start the FastAPI server locally
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

### 2. Docker Deployment

The application includes a multi-stage Dockerfile built on `ghcr.io/astral-sh/uv` and `python:3.12-slim-bookworm`, running under an unprivileged user with OpenMP runtime support (`libgomp1`) and container healthchecks.

#### Using Docker Compose (Recommended)

```bash
# Build and start the container in detached mode
docker compose up -d --build

# Inspect service logs
docker compose logs -f

# Check container health status
docker compose ps

# Stop the container
docker compose down
```

#### Using Docker CLI Directly

```bash
# Build the Docker image
docker build -t supply-chain-prediction-api:latest .

# Run the container
docker run -d \
  --name supply-chain-prediction-api \
  -p 8000:8000 \
  --restart unless-stopped \
  supply-chain-prediction-api:latest

# Check health endpoint
curl http://localhost:8000/health
```

---

## API Endpoints & Examples

### Health Check

`GET /health`

```bash
curl -X GET http://localhost:8000/health
```

Response:
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

### Single Order Prediction

`POST /predict`

Inputs require only raw checkout characteristics known at order submission. All derived temporal (cyclical sine/cosine), financial, route, and basket metrics are computed internally.

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

Response:
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

### Batch Predictions

`POST /predict/batch`

Supports bulk inference up to 1,000 orders in a single request.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "orders": [
      {
        "shipping_mode": "Same Day",
        "order_date": "2018-01-15 10:00:00",
        "order_id": "BATCH-01"
      },
      {
        "shipping_mode": "First Class",
        "order_date": "2018-01-15 14:00:00",
        "order_id": "BATCH-02"
      }
    ]
  }'
```
