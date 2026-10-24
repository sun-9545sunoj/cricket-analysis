import os
import json
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error,
    roc_auc_score, silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FORMATS, model_path, phase_case_sql, total_balls
from src.etl import get_connection

WIN_FEATURES = ["runs_needed", "balls_left", "wickets_left", "req_rr", "curr_rr", "target_runs"]
SCORE_FEATURES = ["balls_bowled", "runs", "wickets", "run_rate"]
CLUSTER_FEATURES = ["strike_rate", "average", "boundary_pct", "dot_pct", "powerplay_share", "death_share"]

# Share of matches (by date) held out as the test set
TEST_FRACTION = 0.2
# How much lower XGBoost's test log loss must be before it replaces the logistic regression baseline
MIN_LOG_LOSS_GAIN = 0.005


def _query(sql, params, path=None):
    con = get_connection(path, read_only=True)
    try:
        return con.execute(sql, params).df()
    finally:
        con.close()


def _time_split(df):
    """Train on older matches, test on the most recent ones (no match appears in both)."""
    match_dates = df.groupby("match_id")["start_date"].first().sort_values()
    cutoff_idx = int(len(match_dates) * (1 - TEST_FRACTION))
    test_ids = set(match_dates.index[cutoff_idx:])
    is_test = df["match_id"].isin(test_ids)
    return df[~is_test], df[is_test], str(match_dates.iloc[cutoff_idx].date())


# ---------------- Win probability (second innings) ----------------

def build_chase_dataset(fmt, path=None):
    """One row per legal ball of completed, full-length chases: the scoreboard state and whether the chase won."""
    df = _query("""
        WITH states AS (
            SELECT d.match_id, m.start_date, m.target_runs, d.is_legal, d.legal_ball_no,
                   SUM(d.total_runs) OVER w AS runs,
                   SUM(d.is_wicket) OVER w AS wickets,
                   CASE WHEN m.winner = d.batting_team THEN 1 ELSE 0 END AS label
            FROM deliveries d JOIN matches m USING (match_id)
            WHERE m.format = ? AND d.innings = 2 AND m.result = 'normal' AND m.method = ''
              AND m.target_runs > 0 AND m.scheduled_overs = ?
              AND (m.target_overs IS NULL OR m.target_overs = m.scheduled_overs)
            WINDOW w AS (PARTITION BY d.match_id ORDER BY d.over, d.ball ROWS UNBOUNDED PRECEDING)
        )
        SELECT * FROM states WHERE is_legal
    """, [fmt, FORMATS[fmt]["overs"]], path)

    n_balls = total_balls(fmt)
    df["runs_needed"] = df["target_runs"] - df["runs"]
    df["balls_left"] = n_balls - df["legal_ball_no"]
    df["wickets_left"] = 10 - df["wickets"]
    df = df[(df["runs_needed"] > 0) & (df["balls_left"] > 0) & (df["wickets_left"] > 0)].copy()
    df["req_rr"] = df["runs_needed"] / df["balls_left"] * 6
    df["curr_rr"] = df["runs"] / df["legal_ball_no"] * 6
    return df


def train_win_model(fmt, path=None):
    """Trains logistic regression (baseline) and XGBoost win-probability models; keeps XGBoost."""
    df = build_chase_dataset(fmt, path)
    train, test, cutoff = _time_split(df)
    X_train, y_train = train[WIN_FEATURES], train["label"]
    X_test, y_test = test[WIN_FEATURES], test["label"]

    candidates = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        # Monotone constraints (same order as WIN_FEATURES): more runs needed or a higher required
        # rate never helps the chase; more balls, wickets or current run rate never hurts
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, min_child_weight=50,
            reg_lambda=5, monotone_constraints="(-1,1,1,-1,1,0)", eval_metric="logloss", random_state=42,
        ),
    }
    metrics = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "log_loss": round(float(log_loss(y_test, prob, labels=[0, 1])), 4),
            "brier": round(float(brier_score_loss(y_test, prob)), 4),
            "accuracy": round(float(accuracy_score(y_test, prob >= 0.5)), 4),
            # AUC is undefined when the test set has a single class (tiny datasets)
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4) if y_test.nunique() == 2 else None,
        }

    # Prefer the simpler baseline unless XGBoost is clearly better on the most recent matches;
    # near-ties otherwise flip between runs because XGBoost training is not bit-for-bit deterministic
    xgb_gain = metrics["Logistic Regression"]["log_loss"] - metrics["XGBoost"]["log_loss"]
    selected = "XGBoost" if xgb_gain > MIN_LOG_LOSS_GAIN else "Logistic Regression"
    best = candidates[selected]
    prob = best.predict_proba(X_test)[:, 1]
    bins = pd.cut(prob, np.linspace(0, 1, 11), include_lowest=True)
    calib = pd.DataFrame({"pred": prob, "actual": y_test.values, "bin": bins}).groupby("bin", observed=True).agg(
        predicted=("pred", "mean"), observed=("actual", "mean"), n=("pred", "size")).reset_index(drop=True)

    info = {
        "task": "Chase win probability (classification)",
        "train_matches": int(train["match_id"].nunique()),
        "test_matches": int(test["match_id"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_from": cutoff,
        "models": metrics,
        "selected": selected,
        "calibration": calib.round(4).to_dict(orient="records"),
    }
    background = X_train.sample(min(100, len(X_train)), random_state=42)
    _save(f"win_{fmt}.pkl", {"model": best, "features": WIN_FEATURES, "info": info, "background": background})
    return best, info


def predict_win_probability(fmt, state):
    """Chasing side's win probability for a scoreboard state, with SHAP feature contributions."""
    bundle = load_model(f"win_{fmt}.pkl")
    model = bundle["model"]
    X = pd.DataFrame([state])[WIN_FEATURES]
    chase_prob = float(model.predict_proba(X)[0][1])

    shap_values = _shap_values(fmt, X)
    impacts = sorted(
        ({"Feature": f, "Impact": round(float(v), 4)} for f, v in zip(WIN_FEATURES, shap_values)),
        key=lambda d: abs(d["Impact"]), reverse=True,
    )
    return {
        "chasing_win_pct": round(chase_prob * 100, 1),
        "defending_win_pct": round((1 - chase_prob) * 100, 1),
        "shap_impacts": impacts[:5],
    }


def scoreboard_state(fmt, target, current_score, balls_bowled, wickets_lost):
    """Builds the win-model feature dict from a live chase scoreboard."""
    balls_left = max(total_balls(fmt) - balls_bowled, 1)
    runs_needed = max(target - current_score, 0)
    return {
        "runs_needed": runs_needed,
        "balls_left": balls_left,
        "wickets_left": max(10 - wickets_lost, 1),
        "req_rr": round(runs_needed / balls_left * 6, 2),
        "curr_rr": round(current_score / max(balls_bowled, 1) * 6, 2),
        "target_runs": target,
    }


# ---------------- First-innings score projection ----------------

def build_first_innings_dataset(fmt, path=None):
    """One row per legal ball of completed full-length first innings, labelled with the final total."""
    df = _query("""
        WITH states AS (
            SELECT d.match_id, m.start_date, d.is_legal, d.legal_ball_no,
                   SUM(d.total_runs) OVER w AS runs,
                   SUM(d.is_wicket) OVER w AS wickets,
                   SUM(d.total_runs) OVER (PARTITION BY d.match_id) AS final_total
            FROM deliveries d JOIN matches m USING (match_id)
            WHERE m.format = ? AND d.innings = 1 AND m.result IN ('normal', 'tie') AND m.method = ''
              AND m.scheduled_overs = ?
            WINDOW w AS (PARTITION BY d.match_id ORDER BY d.over, d.ball ROWS UNBOUNDED PRECEDING)
        )
        SELECT * FROM states WHERE is_legal
    """, [fmt, FORMATS[fmt]["overs"]], path)
    df["balls_bowled"] = df["legal_ball_no"]
    df["run_rate"] = df["runs"] / df["balls_bowled"] * 6
    return df[df["balls_bowled"] < total_balls(fmt)].copy()


def train_score_model(fmt, path=None):
    """Trains linear regression (baseline) and XGBoost regressors for the final first-innings total."""
    df = build_first_innings_dataset(fmt, path)
    train, test, cutoff = _time_split(df)
    X_train, y_train = train[SCORE_FEATURES], train["final_total"]
    X_test, y_test = test[SCORE_FEATURES], test["final_total"]

    candidates = {
        "Linear Regression": LinearRegression(),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42,
        ),
    }
    halfway = test["balls_bowled"] == total_balls(fmt) // 2
    metrics = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics[name] = {
            "mae": round(float(mean_absolute_error(y_test, pred)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
            "mae_at_halfway": round(float(mean_absolute_error(y_test[halfway], pred[halfway.values])), 2),
        }

    selected = min(metrics, key=lambda name: metrics[name]["mae"])
    info = {
        "task": "First-innings final score (regression)",
        "train_matches": int(train["match_id"].nunique()),
        "test_matches": int(test["match_id"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_from": cutoff,
        "models": metrics,
        "selected": selected,
    }
    _save(f"score_{fmt}.pkl", {"model": candidates[selected], "features": SCORE_FEATURES, "info": info})
    return candidates[selected], info


def project_first_innings_score(fmt, current_score, balls_bowled, wickets_lost):
    model = load_model(f"score_{fmt}.pkl")["model"]
    X = pd.DataFrame([{
        "balls_bowled": balls_bowled,
        "runs": current_score,
        "wickets": wickets_lost,
        "run_rate": current_score / max(balls_bowled, 1) * 6,
    }])[SCORE_FEATURES]
    return max(round(float(model.predict(X)[0])), current_score)


# ---------------- Batting-style clustering ----------------

CLUSTER_MIN_BALLS = {"T20": 300, "ODI": 750}


def batter_profiles(fmt, min_balls=None, path=None):
    """Career batting profile per batter, including how much of their batting happens in each phase."""
    df = _query(f"""
        SELECT d.batter AS player,
               MODE(d.batting_team) AS team,
               SUM(d.runs_batter) AS runs,
               COUNT(*) FILTER (WHERE NOT d.is_wide) AS balls,
               SUM(CASE WHEN d.is_four OR d.is_six THEN 1 ELSE 0 END) AS boundaries,
               COUNT(*) FILTER (WHERE NOT d.is_wide AND d.runs_batter = 0) AS dots,
               COUNT(*) FILTER (WHERE NOT d.is_wide AND {phase_case_sql()} = 'Powerplay') AS pp_balls,
               COUNT(*) FILTER (WHERE NOT d.is_wide AND {phase_case_sql()} = 'Death') AS death_balls
        FROM deliveries d JOIN matches m USING (match_id)
        WHERE m.format = ?
        GROUP BY d.batter
        HAVING COUNT(*) FILTER (WHERE NOT d.is_wide) >= ?
    """, [fmt, min_balls or CLUSTER_MIN_BALLS[fmt]], path)
    outs = _query("""
        SELECT d.player_dismissed AS player, COUNT(*) AS outs
        FROM deliveries d JOIN matches m USING (match_id)
        WHERE m.format = ? AND d.player_dismissed IS NOT NULL
          AND d.wicket_type NOT IN ('retired hurt', 'retired not out')
        GROUP BY 1
    """, [fmt], path)
    df = df.merge(outs, on="player", how="left").fillna({"outs": 0})
    df["strike_rate"] = df["runs"] / df["balls"] * 100
    # Not-out-heavy careers have inflated averages; cap so they don't dominate the clustering
    df["average"] = (df["runs"] / df["outs"].clip(lower=1)).clip(upper=80)
    df["boundary_pct"] = df["boundaries"] / df["balls"] * 100
    df["dot_pct"] = df["dots"] / df["balls"] * 100
    df["powerplay_share"] = df["pp_balls"] / df["balls"]
    df["death_share"] = df["death_balls"] / df["balls"]
    return df


def _name_clusters(centroids):
    """Names each KMeans cluster from its standardised centroid: where in the innings they bat and how fast."""
    finisher = centroids["death_share"].idxmax()
    names = {}
    for idx, c in centroids.iterrows():
        if idx == finisher:
            name = "Finishers"
        elif c["strike_rate"] < 0 and c["average"] < -0.5:
            name = "Lower-order batters"
        else:
            position = "Top-order" if c["powerplay_share"] > 0 else "Middle-order"
            name = f"{position} {'aggressors' if c['strike_rate'] > 0 else 'anchors'}"
        names[idx] = name if name not in names.values() else f"{name} (group {idx + 1})"
    return names


@lru_cache(maxsize=4)
def cluster_batters(fmt, k=4, min_balls=None):
    """KMeans clustering of batters into playing styles. Returns (profiles with cluster labels, info)."""
    df = batter_profiles(fmt, min_balls)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[CLUSTER_FEATURES])
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
    df["cluster_id"] = km.labels_

    centroids_z = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_FEATURES)
    names = _name_clusters(centroids_z)
    df["style"] = df["cluster_id"].map(names)

    summary = df.groupby("style")[CLUSTER_FEATURES].mean().round(2)
    summary["players"] = df.groupby("style").size()
    info = {
        "task": "Batting-style segmentation (KMeans clustering)",
        "k": k,
        "players": int(len(df)),
        "min_balls": min_balls or CLUSTER_MIN_BALLS[fmt],
        "silhouette": round(float(silhouette_score(X, km.labels_)), 3),
        "clusters": summary.reset_index().to_dict(orient="records"),
    }
    return df, info


# ---------------- Persistence ----------------

def _save(name, bundle):
    path = model_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    load_model.cache_clear()
    _explainer.cache_clear()


@lru_cache(maxsize=8)
def load_model(name):
    path = model_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found; run `python -m src.pipeline` to train the models.")
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=4)
def _explainer(fmt):
    bundle = load_model(f"win_{fmt}.pkl")
    model = bundle["model"]
    if isinstance(model, xgb.XGBClassifier):
        return shap.TreeExplainer(model)
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    return shap.LinearExplainer(lr, scaler.transform(bundle["background"]))


def _shap_values(fmt, X):
    """SHAP contributions (log-odds) of each feature for one row, for either model type."""
    model = load_model(f"win_{fmt}.pkl")["model"]
    if not isinstance(model, xgb.XGBClassifier):
        X = model.named_steps["standardscaler"].transform(X)
    values = _explainer(fmt).shap_values(X)
    return np.asarray(values).reshape(-1)[:len(WIN_FEATURES)]


def model_report():
    """Evaluation metrics for every trained model, as written by train_all()."""
    path = model_path("metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def train_all(formats=None, path=None, cluster_min_balls=None):
    report = {}
    for fmt in formats or FORMATS:
        _, win_info = train_win_model(fmt, path)
        _, score_info = train_score_model(fmt, path)
        cluster_batters.cache_clear()
        _, cluster_info = cluster_batters(fmt, min_balls=cluster_min_balls)
        report[fmt] = {"win_probability": win_info, "score_projection": score_info, "batter_clusters": cluster_info}
        print(f"[{fmt}] win model: {win_info['models']}")
        print(f"[{fmt}] score model: {score_info['models']}")
        print(f"[{fmt}] clusters: silhouette {cluster_info['silhouette']} over {cluster_info['players']} batters")
    with open(model_path("metrics.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report
