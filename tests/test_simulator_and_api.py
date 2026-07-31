import pytest
from fastapi.testclient import TestClient

from src.simulator import run_monte_carlo_simulation, simulate_what_if_scenario
from src.api import app

client = TestClient(app)

def test_monte_carlo_simulation():
    """Verifies Monte Carlo simulation returns expected structure and confidence intervals."""
    sim = run_monte_carlo_simulation(
        target_score=180, 
        current_score=120, 
        balls_remaining=30, 
        wickets_remaining=5, 
        n_simulations=1000
    )
    
    assert sim["n_simulations"] == 1000
    assert sim["mean_score"] > 120
    assert sim["ci_10th"] <= sim["median_score"] <= sim["ci_90th"]
    assert 0.0 <= sim["simulated_win_pct"] <= 100.0

def test_what_if_scenario():
    """Verifies What-If tactical scenario simulation."""
    res = simulate_what_if_scenario(180, 120, 30, 5, "Deploy Death Specialist Bowler")
    assert "baseline" in res
    assert "tactical" in res
    assert "impact_summary" in res

def test_fastapi_health():
    """Tests /health API endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_fastapi_win_prob_endpoint():
    """Tests /api/v1/predict-win-prob API endpoint."""
    payload = {
        "runs_needed": 45,
        "balls_left": 30,
        "wickets_left": 6,
        "req_rr": 9.0,
        "curr_rr": 8.5,
        "target_runs": 180,
        "venue_avg_runs": 175.0,
        "toss_win_bat_first": 1
    }
    response = client.post("/api/v1/predict-win-prob", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "team1_win_pct" in data["data"]

def test_fastapi_simulate_endpoint():
    """Tests /api/v1/simulate-match API endpoint."""
    payload = {
        "target_score": 180,
        "current_score": 135,
        "balls_remaining": 30,
        "wickets_remaining": 6,
        "n_simulations": 1000,
        "tactical_choice": "Deploy Death Specialist Bowler"
    }
    response = client.post("/api/v1/simulate-match", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "tactical" in data["data"]
