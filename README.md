# Cricket Analytics Hub

Analytics and machine learning on real ball-by-ball international cricket data from [Cricsheet](https://cricsheet.org): head-to-head records and series results, player stats, live win probability, score projection, Monte Carlo match simulation and batting-style clustering. Defaults to **India vs New Zealand**, and works for any pair of teams.

**Data:** 6,223 men's internationals (3,627 T20Is from 2005 and 2,596 ODIs from 2002, up to 14 Oct 2026). That is 2.19 million deliveries by 4,965 batters.

```
Cricsheet JSON zips ──► ETL (src/etl.py) ──► DuckDB: matches, deliveries, match_players
                                                   │
                     ┌─────────────────────────────┼──────────────────────────────┐
                     ▼                             ▼                              ▼
          SQL analytics (features.py)   ML models (models.py)      Monte Carlo (simulator.py)
                     └─────────────────────────────┼──────────────────────────────┘
                                                   ▼
                          FastAPI (src/api.py) ──► HTML/JS dashboard (/app)
                          Streamlit dashboard (app.py)
```

## What it shows

| Tab | What it does |
|---|---|
| **Series & Head-to-Head** | Latest bilateral series with results and tally, all-time head-to-head record, recent results, top batters and bowlers in the fixture, batting run rate by phase |
| **Live Predictor** | Enter a live score. Chasing: win probability with SHAP explanation and a Match Pressure Index. Batting first: projected final total |
| **Player Matchups** | Batter-vs-bowler heatmap for the two latest XIs, with alerts where a bowler has dominated |
| **Simulator** | Ball-by-ball Monte Carlo of the rest of the innings, using the batting team's real per-phase scoring over the last 3 years, plus what-if tactics |
| **Player Styles (ML)** | KMeans clustering of batters into styles (top-order aggressors and anchors, finishers, lower order) |
| **Model Report** | Test metrics, baseline comparison and a calibration chart for every model |

## Machine learning

All supervised models use a **time-based split**: the oldest 80% of matches are used for training and the most recent 20% for testing, so the models are always scored on matches they never saw. Each model is compared with a simple baseline. The win model keeps the logistic regression baseline unless XGBoost beats it by more than 0.005 test log loss; the score model keeps whichever has the lower test MAE.

| Model | Task | Test result (T20I) | Test result (ODI) |
|---|---|---|---|
| Logistic regression vs XGBoost (monotone constraints) | Chasing side's win probability after every ball | Logistic regression kept (XGBoost tied): log loss 0.290, ROC AUC 0.948, accuracy 86.8% | Logistic regression kept: log loss 0.403 (XGBoost 0.42), AUC 0.897, accuracy 80.3% |
| Linear regression vs XGBoost regressor | Final first-innings total from the current score | XGBoost MAE 19.9 runs (baseline 21.3) | XGBoost MAE 34.0 runs (baseline 35.4) |
| KMeans (k=4) | Batting styles from strike rate, average, boundary %, dot-ball % and phase shares | 746 batters, silhouette 0.26 | 420 batters, silhouette 0.26 |

SHAP values explain each win-probability prediction. Exact figures are rewritten to `data/models/metrics.json` each time the models are retrained.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline            # download Cricsheet data (~32 MB), build DuckDB, train models (~15 s)

uvicorn src.api:app --port 8000   # HTML/JS dashboard at http://localhost:8000/app/, API docs at /docs
streamlit run app.py              # Streamlit version of the dashboard
```

**Following a live series:** Cricsheet usually publishes matches a few days after they're played. Run `python -m src.pipeline --refresh` to pull the latest archives and retrain. While a match is in progress, enter the score in the sidebar to get live predictions.

**Docker:** `docker compose up --build` runs the pipeline once, then serves the dashboard on :8501 and the API on :8000.

## Tests

```bash
python -m pytest -v
```

Tests build a temporary database from `tests/fixtures/india_nz_matches.zip` (45 real India v New Zealand matches, 2019–2026). They never touch `data/`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/meta` | Formats, teams, data coverage |
| `GET /api/v1/head-to-head`, `GET /api/v1/series/latest` | Results between two teams |
| `GET /api/v1/players/batting`, `/players/bowling`, `/phase-stats` | Player and team aggregates (filter by team, opponent, date) |
| `POST /api/v1/predict-win-prob`, `/project-score` | Live predictions |
| `POST /api/v1/simulate-match`, `/h2h-matrix` | Simulation and matchups |
| `GET /api/v1/batter-clusters`, `/model-report` | Clustering output and model metrics |

## License
Code: MIT. Match data from [Cricsheet](https://cricsheet.org), available under the [Open Data Commons Attribution License](https://cricsheet.org/licensing/).
