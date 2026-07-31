import time
import pytest
from src.models import train_win_model, predict_win_probability
from src.features import get_h2h_matrix, get_phase_radar_stats
from src.etl import SAMPLE_BATTERS
from src.utils import create_win_prob_donut, create_shap_bar_chart, create_h2h_heatmap, create_phase_radar_chart

def test_xgboost_model_performance():
    """FR-2.1 Model Validation Test: Verifies Log-Loss <= 0.55 and Accuracy >= 70%."""
    model, loss, acc = train_win_model()
    
    assert loss <= 0.55, f"Log-loss requirement failed: {loss:.4f} > 0.55"
    assert acc >= 0.70, f"Accuracy requirement failed: {acc:.4f} < 0.70"

def test_win_probability_and_shap():
    """FR-2.2 Test: Verifies win probability output structure and SHAP feature impacts."""
    sample_state = {
        "runs_needed": 45,
        "balls_left": 30,
        "wickets_left": 6,
        "req_rr": 9.0,
        "curr_rr": 8.5,
        "target_runs": 180,
        "venue_avg_runs": 175.0,
        "toss_win_bat_first": 1
    }
    
    res = predict_win_probability(sample_state)
    
    assert "team1_win_pct" in res
    assert "team2_win_pct" in res
    assert "shap_impacts" in res
    assert abs(res["team1_win_pct"] + res["team2_win_pct"] - 100.0) < 0.2
    assert len(res["shap_impacts"]) > 0

def test_h2h_matrix_latency():
    """FR-3.3 & NFR-1 Latency Benchmark Test: 11x11 matrix generated in < 1 second."""
    team_a = SAMPLE_BATTERS["Chennai Super Kings"]
    team_b = SAMPLE_BATTERS["Mumbai Indians"]
    
    t0 = time.time()
    matrix = get_h2h_matrix(team_a, team_b)
    t1 = time.time()
    
    elapsed = t1 - t0
    assert len(matrix) == 11, f"Expected 11 batters, got {len(matrix)}"
    assert elapsed < 1.0, f"H2H matrix generation latency exceeded 1.0s: {elapsed:.3f}s"

def test_plotly_utils():
    """NFR-2 Test: Verifies Plotly figure generation functions execute without error."""
    team_a = SAMPLE_BATTERS["Chennai Super Kings"]
    team_b = SAMPLE_BATTERS["Mumbai Indians"]
    matrix = get_h2h_matrix(team_a, team_b)
    
    fig_donut = create_win_prob_donut("CSK", 55.0, "MI", 45.0)
    assert fig_donut is not None

    fig_shap = create_shap_bar_chart([{"Feature": "req_rr", "Impact": 0.15}])
    assert fig_shap is not None

    fig_h2h = create_h2h_heatmap(matrix, team_a, team_b)
    assert fig_h2h is not None

    fig_radar = create_phase_radar_chart("CSK", {"Powerplay": 8.0, "Middle": 7.5, "Death": 10.0}, "MI", {"Powerplay": 8.5, "Middle": 8.0, "Death": 11.0})
    assert fig_radar is not None

