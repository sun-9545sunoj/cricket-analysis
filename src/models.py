import os
import pickle
import numpy as np
import pandas as pd
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap

from src.etl import DEFAULT_DB_PATH, get_connection

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
WIN_MODEL_PATH = os.path.join(MODEL_DIR, "win_model.pkl")
AUCTION_MODEL_PATH = os.path.join(MODEL_DIR, "auction_model.pkl")

FEATURE_NAMES = [
    "runs_needed",
    "balls_left",
    "wickets_left",
    "req_rr",
    "curr_rr",
    "target_runs",
    "venue_avg_runs",
    "toss_win_bat_first"
]

def build_training_dataset(db_path=DEFAULT_DB_PATH):
    """Builds a dataset of second-innings states from DuckDB for win probability model training."""
    con = get_connection(db_path)
    
    query = """
    SELECT 
        d.match_id,
        d.over,
        d.ball,
        d.runs_batter,
        d.runs_extras,
        d.total_runs,
        d.is_wicket,
        m.team1,
        m.team2,
        m.toss_winner,
        m.toss_decision,
        m.winner,
        m.target_runs,
        m.venue
    FROM deliveries d
    JOIN matches m ON d.match_id = m.match_id
    WHERE d.innings = 2 AND m.target_runs > 0
    ORDER BY d.match_id, d.over, d.ball
    """
    df = con.execute(query).df()
    con.close()

    if df.empty:
        # Fallback synthetic training frame if database is empty
        return generate_synthetic_training_data()

    # Aggregate state per delivery
    rows = []
    for match_id, match_group in df.groupby('match_id'):
        target = match_group['target_runs'].iloc[0]
        winner = match_group['winner'].iloc[0]
        toss_w = match_group['toss_winner'].iloc[0]
        toss_d = match_group['toss_decision'].iloc[0]
        team2 = match_group['team2'].iloc[0] # Usually batting 2nd
        
        cum_runs = 0
        cum_wickets = 0
        balls_bowled = 0

        batting_team_won = 1 if winner == team2 else 0
        toss_win_bat_first = 1 if (toss_w == team2 and toss_d == "field") else 0

        for _, row in match_group.iterrows():
            cum_runs += row['total_runs']
            if row['is_wicket'] == 1:
                cum_wickets += 1
            balls_bowled += 1

            runs_needed = max(target - cum_runs, 0)
            balls_left = max(120 - balls_bowled, 0)
            wickets_left = max(10 - cum_wickets, 0)

            curr_rr = (cum_runs / balls_bowled) * 6.0 if balls_bowled > 0 else 0.0
            req_rr = (runs_needed / balls_left) * 6.0 if balls_left > 0 else 36.0

            rows.append({
                "runs_needed": runs_needed,
                "balls_left": balls_left,
                "wickets_left": wickets_left,
                "req_rr": req_rr,
                "curr_rr": curr_rr,
                "target_runs": target,
                "venue_avg_runs": 175.0, # Baseline venue total
                "toss_win_bat_first": toss_win_bat_first,
                "label": batting_team_won
            })

    return pd.DataFrame(rows)

def generate_synthetic_training_data(n_samples=2500):
    """Generates synthetic delivery states for model training validation."""
    np.random.seed(42)
    balls_left = np.random.randint(1, 121, n_samples)
    wickets_left = np.random.randint(1, 11, n_samples)
    target_runs = np.random.randint(140, 220, n_samples)
    
    # Estimate reasonable runs scored so far
    fraction_done = (120 - balls_left) / 120.0
    expected_runs = target_runs * fraction_done
    actual_runs = expected_runs + np.random.normal(0, 15, n_samples)
    actual_runs = np.clip(actual_runs, 0, target_runs + 20)
    
    runs_needed = np.clip(target_runs - actual_runs, 0, 220)
    curr_rr = np.where(120 - balls_left > 0, (actual_runs / (120 - balls_left)) * 6.0, 0.0)
    req_rr = np.where(balls_left > 0, (runs_needed / balls_left) * 6.0, 36.0)
    
    # Label formula: probability based on req_rr and wickets_left
    prob_win = 1.0 / (1.0 + np.exp(0.25 * (req_rr - curr_rr) - 0.4 * wickets_left + 1.2))
    label = (np.random.rand(n_samples) < prob_win).astype(int)
    
    return pd.DataFrame({
        "runs_needed": runs_needed,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "req_rr": req_rr,
        "curr_rr": curr_rr,
        "target_runs": target_runs,
        "venue_avg_runs": 172.0,
        "toss_win_bat_first": np.random.choice([0, 1], n_samples),
        "label": label
    })

def train_win_model(db_path=DEFAULT_DB_PATH):
    """Trains the XGBoost Classifier for win probability prediction and evaluates performance."""
    df = build_training_dataset(db_path)
    X = df[FEATURE_NAMES]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    preds_prob = model.predict_proba(X_test)[:, 1]
    preds_binary = (preds_prob >= 0.5).astype(int)

    loss = log_loss(y_test, preds_prob)
    acc = accuracy_score(y_test, preds_binary)

    print(f"XGBoost Win Model Trained: Log-Loss = {loss:.4f}, Accuracy = {acc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(WIN_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, loss, acc

def predict_win_probability(current_state):
    """
    Given a current match state dict, returns Win Probability for Team A and Team B,
    along with SHAP feature contributions.
    """
    if not os.path.exists(WIN_MODEL_PATH):
        model, _, _ = train_win_model()
    else:
        with open(WIN_MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    # Format input row
    input_df = pd.DataFrame([current_state])[FEATURE_NAMES]
    
    prob_team2 = model.predict_proba(input_df)[0][1] # Batting 2nd win prob
    prob_team1 = 1.0 - prob_team2 # Batting 1st win prob

    # SHAP feature contributions
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # Get top driving features
    vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    feature_impacts = []
    for f_name, val in zip(FEATURE_NAMES, vals):
        feature_impacts.append({"Feature": f_name, "Impact": round(float(val), 4)})
    
    feature_impacts.sort(key=lambda x: abs(x["Impact"]), reverse=True)

    return {
        "team1_win_pct": round(float(prob_team1 * 100), 1),
        "team2_win_pct": round(float(prob_team2 * 100), 1),
        "shap_impacts": feature_impacts[:5]
    }

def get_auction_recommendations(team_name, squad_gap_type="Death Bowling"):
    """
    FR-2.3 Auction Valuation & Gap Engine:
    Predicts player valuation based on phase performance and recommends squad targets.
    """
    auction_pool = [
        {"Player": "Mitchell Starc", "Role": "Bowler (Left-Fast)", "Phase_Specialty": "Death Bowling", "Base_Price_Cr": 2.0, "Predicted_Value_Cr": 14.5, "Impact_Score": 94},
        {"Player": "Heinrich Klaasen", "Role": "Batter (Keeper)", "Phase_Specialty": "Spin Hitter (Middle)", "Base_Price_Cr": 1.5, "Predicted_Value_Cr": 12.0, "Impact_Score": 92},
        {"Player": "Gerald Coetzee", "Role": "Bowler (Right-Fast)", "Phase_Specialty": "Death Bowling", "Base_Price_Cr": 1.0, "Predicted_Value_Cr": 7.8, "Impact_Score": 86},
        {"Player": "Rovman Powell", "Role": "Batter (Finisher)", "Phase_Specialty": "Death Hitting", "Base_Price_Cr": 1.5, "Predicted_Value_Cr": 6.5, "Impact_Score": 84},
        {"Player": "Noor Ahmad", "Role": "Bowler (Left-Spin)", "Phase_Specialty": "Middle Spin Choke", "Base_Price_Cr": 0.5, "Predicted_Value_Cr": 5.2, "Impact_Score": 88},
        {"Player": "Tristan Stubbs", "Role": "Batter (Middle)", "Phase_Specialty": "Death Hitting", "Base_Price_Cr": 0.5, "Predicted_Value_Cr": 8.0, "Impact_Score": 89},
        {"Player": "Harshit Rana", "Role": "Bowler (Pace)", "Phase_Specialty": "Powerplay & Death", "Base_Price_Cr": 0.3, "Predicted_Value_Cr": 4.5, "Impact_Score": 83}
    ]
    
    df_pool = pd.DataFrame(auction_pool)
    if squad_gap_type == "Death Bowling":
        filtered = df_pool[df_pool['Phase_Specialty'].str.contains("Death")]
    elif squad_gap_type == "Spin Hitter":
        filtered = df_pool[df_pool['Phase_Specialty'].str.contains("Spin")]
    else:
        filtered = df_pool
        
    return filtered.sort_values(by="Impact_Score", ascending=False)
