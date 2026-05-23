# SmartFeed

A production-style CTR prediction and item ranking system that learns from user behaviour, detects model drift, and retrains itself automatically.

---

## The Problem

Showing every user the same list of items is a missed opportunity. SmartFeed solves personalisation at scale: given a user and a set of candidate items, it predicts the probability each item gets clicked and returns them ranked highest-to-lowest. As user behaviour shifts over time, the system detects the drift and retrains the model without human intervention.

---

## Architecture

```
generate_data.py  →  smartfeed.db  →  train.py  →  model.pkl
                                                        ↓
                          predictions ←  api.py (FastAPI + uvicorn)
                               ↓
                        monitoring.py  →  PSI drift check (every 6h)
                               ↓ drift detected
                          retrain.py  →  new model.pkl (if AUC improves)
                               ↓
                         dashboard.py  →  Streamlit observability UI
```

| Component | What it does | Why it exists |
|-----------|-------------|---------------|
| **Data layer** (`generate_data.py`, `smartfeed.db`) | Generates 100k synthetic user-item interactions with realistic click signal and stores them in SQLite | Provides a reproducible training dataset without requiring a live production feed |
| **Model** (`train.py`, `model.pkl`) | Trains a LightGBM binary classifier on engineered features (user CTR, category CTR, price rank, recency) with a time-based train/test split | Learns which items each user is likely to click based on historical behaviour |
| **API** (`api.py`) | FastAPI service exposing `POST /rank` — scores candidate items, returns ranked list, logs every prediction | The serving layer external clients call; prediction logging feeds the drift monitor |
| **Caching** (in-memory `_state` dict) | Model and feature list loaded once at startup, not per request | Keeps p99 latency low — pickle deserialisation is expensive at request time |
| **Drift monitoring** (`monitoring.py`) | Computes PSI between training score distribution and live prediction scores every 6 hours | Detects when the live world has shifted away from what the model was trained on |
| **Auto-retraining** (`retrain.py`) | Retrains LightGBM on all accumulated interactions, promotes new model only if AUC improves by > 0.02 | Closes the loop — the system heals itself when drift is detected, no human needed |
| **Dashboard** (`dashboard.py`) | Streamlit UI showing model AUC, feature importances, recent predictions, daily PSI trend, and retrain history | Makes the system's internal state visible without querying the database manually |

---

## Tech Stack

- **LightGBM** — gradient boosted trees for CTR prediction
- **FastAPI** — async REST API framework
- **Uvicorn** — ASGI server
- **SQLite** — lightweight data store for interactions and predictions
- **APScheduler** — in-process job scheduler for drift checks and safety retrains
- **Streamlit** — internal observability dashboard
- **Altair** — declarative charting for PSI trend and feature importances
- **scikit-learn** — AUC evaluation
- **Docker** — containerised deployment
- **Render** — cloud hosting

---

## Live Demo

| | URL |
|-|-----|
| **API** | https://smartfeed-ml-d543.onrender.com |
| **Health check** | https://smartfeed-ml-d543.onrender.com/health |
| **Interactive docs** | https://smartfeed-ml-d543.onrender.com/docs |

> Note: hosted on Render's free tier — first request after inactivity may take ~30 seconds to wake up.

---

## How to Run Locally

**1. Clone and set up environment**
```bash
git clone https://github.com/bhasvana/smartfeed-ml.git
cd smartfeed-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Generate data and train model**
```bash
python generate_data.py   # creates smartfeed.db with 100k interactions
python train.py           # trains LightGBM, saves model.pkl and features.pkl
```

**3. Start the API**
```bash
uvicorn api:app --reload
# API running at http://localhost:8000
```

**4. Make a ranking request**
```bash
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "item_ids": [101, 202, 303, 404, 505]}'
```

**5. Start the dashboard** (separate terminal)
```bash
streamlit run dashboard.py
# Dashboard at http://localhost:8501
```

**6. Run drift monitoring** (separate terminal)
```bash
python monitoring.py
# Checks PSI every 6h, retrains if PSI > 0.20
```

**Or run with Docker**
```bash
docker build -t smartfeed .
docker run -p 8000:8000 smartfeed
```

---

## Key Technical Decisions

### 1. LightGBM over neural networks

A neural network would require significantly more data, longer training time, and careful hyperparameter tuning to outperform gradient boosting on tabular CTR data. LightGBM's leaf-wise tree growth naturally captures feature interactions (e.g. user × category affinity) that logistic regression misses, while training in seconds on 100k rows. At the data volumes typical of early-stage personalisation systems, gradient boosting consistently matches or beats deep learning with far less operational complexity.

### 2. PSI for drift detection

PSI (Population Stability Index) measures the shift between two score distributions using the formula `Σ (actual% − expected%) × ln(actual%/expected%)` across fixed buckets. Unlike feature-level monitoring (which requires tracking 7 separate distributions), PSI compresses the model's entire view of the world into a single number. A score above 0.20 is an industry-standard signal that the model is seeing a meaningfully different world than it was trained on — not just noise.

### 3. Automated retraining with an AUC gate

Retraining on more data almost always produces a marginally higher AUC due to additional signal, not genuine improvement. The 0.02 minimum improvement threshold prevents promoting a model that is statistically similar to the current one, which would waste compute and risk introducing noise. Tying retraining to drift detection (rather than a fixed schedule) means the system reacts to real behavioural shifts rather than burning resources on a calendar.

---

## What I Would Do Differently at Production Scale

| Current | Production replacement | Why |
|---------|----------------------|-----|
| SQLite | **PostgreSQL** | SQLite is a single file on disk — multiple service instances can't share it, and write throughput caps out quickly. Postgres handles concurrent reads/writes across services |
| APScheduler (in-process) | **Apache Airflow** | APScheduler lives inside the monitoring process — if it crashes, scheduling stops. Airflow gives persistent DAGs, retries, alerting, and a UI for job history |
| Feature engineering at query time | **Feature store (Feast / Tecton)** | Computing user CTR and category CTR inside the API on every request adds latency and couples feature logic to serving. A feature store pre-computes and caches features, keeping the API fast and feature definitions consistent between training and serving |
| Single model file | **Model registry (MLflow)** | `model.pkl` has no versioning, lineage, or rollback. A model registry tracks every experiment, links models to the data they were trained on, and enables one-command rollback if a new model regresses |
| Manual drift threshold | **Statistical process control** | A fixed PSI threshold of 0.20 doesn't account for sample size — PSI from 50 predictions is noise, from 50k it's signal. SPC methods like CUSUM adjust sensitivity based on data volume |
