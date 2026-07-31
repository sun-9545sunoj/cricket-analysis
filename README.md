# IPL Edge Analytics & Predictive Match Platform

![Version 1.0](https://img.shields.io/badge/version-1.0.0--baseline-brightgreen) ![Status](https://img.shields.io/badge/status-verified--passed-success) ![IEEE Compliance](https://img.shields.io/badge/IEEE-Std--830--1998-blue)

> **Version 1.0 Baseline Release**: Human-Centered Match Center • Real-Time Win Probabilities • 40-Over Timeline Curves • 10,000 Monte Carlo Simulator • 22-Player H2H Mismatch Engine • FastAPI REST Microservices

Compliant with **IEEE Std 830-1998 Software Requirements Specification (SRS)** format.

---

## 📌 Architecture Overview

```
+---------------------+     +--------------------+     +-----------------------+
|  Cricsheet Data     | --> | Python ETL Engine  | --> | Embedded DB (DuckDB)  |
|  (Raw Match JSONs)  |     | (pandas/polars)    |     | (Matches, Deliveries) |
+---------------------+     +--------------------+     +-----------------------+
                                                                   |
                                                                   v
+---------------------+     +--------------------+     +-----------------------+
| Streamlit Web App   | <-- | ML Inference Core  | <-- | Feature Pipeline      |
| (UI & Dashboard)    |     | (XGBoost / SHAP)   |     | (MPI, Phase Stats)    |
+---------------------+     +--------------------+     +-----------------------+
```

---

## ⚡ Key Modules (v1.0 Baseline)

- **Module 1: Data Pipeline & Advanced ETL (`src/etl.py`, `src/features.py`)**
  - High-speed DuckDB relational database for match metadata and ball-by-ball deliveries.
  - Innings phase breakdown: Powerplay (Overs 1–6), Middle (Overs 7–15), Death (Overs 16–20).
  - Dynamic metrics: Match Pressure Index (MPI) with human-friendly status levels (🟢 Comfortable, 🟡 Moderate, 🟠 High, 🔴 Panic Mode) and True Strike Rate ($SR_{true}$).

- **Module 2: Machine Learning Engines (`src/models.py`)**
  - **XGBoost Match Winner Predictor**: Real-time win probability outputs (Log-loss 0.4210, Accuracy 84.5%).
  - **SHAP Model Explainability**: Quantifies top 5 driving decision factors for local prediction transparency.
  - **Auction Valuation Model**: Phase capability regression to predict player bid prices & suggest replacement targets.

- **Module 3: Monte Carlo Simulator & What-If Engine (`src/simulator.py`)**
  - 10,000-iteration ball-by-ball match simulator for score distributions and 90% confidence bounds.
  - Tactical "What-If" scenario sandbox (Death Specialist Bowler vs Power Hitter Batter).

- **Module 4: Human-Centered Match Center (`app.py`, `src/utils.py`)**
  - **Plain English Verdict Banner**: Instant natural language match summary.
  - **40-Over Timeline Curve**: Dynamic ball-by-ball momentum visualization.
  - **22-Player H2H Heatmap**: $11 \times 11$ strike rate matrix, tactical alert cards, and Phase Capability radar charts.

- **Module 5: FastAPI Microservices & Containerization (`src/api.py`, `Dockerfile`)**
  - Production REST API endpoints (`/health`, `/api/v1/predict-win-prob`, `/api/v1/simulate-match`, `/api/v1/h2h-matrix`).
  - Interactive Swagger API playground at `http://localhost:8000/docs`.

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
cd /Users/nagasunojr/Documents/naga/projects/ipl
pip install -r requirements.txt
```

### 2. Launch Interactive Dashboard (v1.0 UI)

```bash
streamlit run app.py
```

### 3. Launch FastAPI REST API Server

```bash
uvicorn src.api:app --reload --port 8000
```

### 4. Run Verification Test Suite

```bash
python -m pytest -v
```

---

## 🧪 Verification Matrix Summary (v1.0)

| Requirement ID | Description | Testing Strategy | Pass Criteria | Status |
| :--- | :--- | :--- | :--- | :--- |
| **FR-1.1** | Cricsheet JSON Parsing | `test_etl.py` | Parses 1,000+ match files without error | **PASSED** |
| **FR-2.1** | XGBoost Probability Model | `test_models.py` | Log-Loss $\le 0.55$, Accuracy $\ge 70\%$ | **PASSED** |
| **FR-3.3** | 22-Player Matchup Matrix | `test_models.py` | Matrix populates $11 \times 11$ under 1.0s | **PASSED** |
| **NFR-1** | System Latency | End-to-End Benchmark | Total rerun time $< 1.5$s | **PASSED** |
| **API-1** | FastAPI Endpoints | `test_simulator_and_api.py` | 100% route health check and JSON responses | **PASSED** |

---

## 📄 License
MIT License.
