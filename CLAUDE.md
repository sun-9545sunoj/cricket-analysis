# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m src.pipeline              # download Cricsheet zips if missing, rebuild DuckDB, train models
.venv/bin/python -m src.pipeline --refresh    # re-download latest archives first (new matches)
.venv/bin/python -m src.pipeline --skip-train

.venv/bin/uvicorn src.api:app --reload --port 8000   # HTML/JS dashboard at /app/, Swagger at /docs
.venv/bin/streamlit run app.py                       # Streamlit dashboard

.venv/bin/python -m pytest -v                                         # full suite (~2s)
.venv/bin/python -m pytest tests/test_models.py::test_scoreboard_state -v  # single test

docker compose up --build    # one-off pipeline service, then dashboard (:8501) + api (:8000)
```

No linter is configured. Run everything from the repo root, because modules import each other as `src.<module>`. The system `python3` (3.15) lacks the dependencies, so use `.venv`.

## Architecture

Data flow: Cricsheet JSON zips (`data/raw/`) → `src/etl.py` → DuckDB (`data/cricket.duckdb`) → `src/features.py` (SQL analytics), `src/models.py` (ML), `src/simulator.py` (Monte Carlo) → served by `src/api.py` (FastAPI, also mounts the static `web/` front end at `/app`) and by `app.py` (Streamlit, calling the Python functions directly). Everything under `data/` is generated and gitignored.

- **`src/config.py`** is the single source for formats (`T20`, `ODI`), overs per format and phase boundaries. Over numbers are 0-indexed. `phase_case_sql()` generates the SQL `CASE` used by every phase query, and `phase_of()` is the Python equivalent. Paths come from `data_dir()`, which reads the `CRICKET_DATA_DIR` environment variable on every call. This is how tests redirect all reads and writes.
- **ETL** is a full reload: `init_db()` drops and recreates the tables. Only men's T20Is and ODIs are kept (no Tests). Super-over innings are skipped. A tie's `winner` is the super-over eliminator. `target_runs` and `target_overs` come from the JSON's revised target when present. Key delivery flags: `is_legal` (not a wide or no-ball), `legal_ball_no` (cumulative legal balls in the innings) and `is_bowler_wicket` (excludes run outs and retirements). `match_players` holds each match's XI.
- **Stat conventions** (`features.py`): a batter's balls exclude wides. Bowler runs conceded = batter runs + wide/no-ball extras. Bowler balls are legal balls only. Queries use `?` parameters. Keep it that way, because team and player names are user input.
- **Models** (`models.py`), all trained by `train_all()`, which also writes `data/models/metrics.json`:
  - Win probability: second-innings ball states from completed, full-length, non-DLS chases.
  - Score projection: first-innings states → final total.
  - KMeans batting styles: computed on demand, cached with `lru_cache`, not persisted.
  - Supervised models use `_time_split` (most recent 20% of matches by date = test). Each fits a baseline (logistic or linear regression) and an XGBoost model. The win model keeps the baseline unless XGBoost's test log loss is lower by more than `MIN_LOG_LOSS_GAIN`, which keeps selection stable across retrains. The score model keeps whichever has the lower test MAE. Either type can be in the pickle, so `_explainer()` and `_shap_values()` handle both a `Pipeline(StandardScaler, LogisticRegression)` and an `XGBClassifier`.
  - Win-model features (`WIN_FEATURES`) are scoreboard-only. `scoreboard_state()` builds them, and every caller must use it. Cluster names come from centroids in `_name_clusters()`, not from fixed IDs.
- **Simulator** draws per-ball outcomes from empirical per-phase distributions (`phase_outcome_probabilities`, cached). With `batting_team`, it uses only that team's last 3 years. It runs vectorised across simulations with a seeded `default_rng`, so results are deterministic.
- **Front ends** duplicate presentation logic: `web/app.js` (Plotly.js from CDN, no build step, `#tab` deep links) and `app.py` + `src/utils.py` (Plotly in Python). Keep tab behaviour in sync when changing one of them. `xi_matchup()` (latest XIs, dropping XI members who never bowled) is shared by both.
- **Caches:** `load_model`, `_explainer`, `cluster_batters` and `phase_outcome_probabilities` are `lru_cache`d per process. Restart the API or Streamlit after re-running the pipeline.

## Tests

`tests/conftest.py` builds a session-scoped temporary database and models from `tests/fixtures/india_nz_matches.zip` (45 real India v NZ matches since 2019). It sets `CRICKET_DATA_DIR` and lowers `CLUSTER_MIN_BALLS` for the small dataset. Assertions on counts (e.g. 23 T20Is, 22 ODIs, the Jan 2026 tour) depend on that fixture.
