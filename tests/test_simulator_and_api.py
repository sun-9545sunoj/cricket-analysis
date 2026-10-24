import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.simulator import phase_outcome_probabilities, run_monte_carlo_simulation, simulate_what_if_scenario

client = TestClient(app)


def test_outcome_probabilities_are_distributions():
    for phase, cfg in phase_outcome_probabilities("T20").items():
        assert abs(cfg["probs"].sum() - 1) < 1e-9
        assert cfg["extras_per_ball"] >= 0


def test_monte_carlo_chase():
    sim = run_monte_carlo_simulation("T20", 180, 120, 30, 6, n_simulations=1000)
    assert sim["n_simulations"] == 1000
    assert sim["mean_score"] > 120
    assert sim["ci_10th"] <= sim["median_score"] <= sim["ci_90th"]
    assert 0 <= sim["simulated_win_pct"] <= 100
    assert sim == run_monte_carlo_simulation("T20", 180, 120, 30, 6, n_simulations=1000), "seeded runs are reproducible"


def test_monte_carlo_first_innings_has_no_win_pct():
    sim = run_monte_carlo_simulation("ODI", None, 0, 300, 10, n_simulations=500, batting_team="India")
    assert sim["simulated_win_pct"] is None
    assert 100 < sim["mean_score"] < 450


def test_death_specialist_lowers_expected_score():
    res = simulate_what_if_scenario("T20", 180, 120, 30, 6, "Deploy Death Specialist Bowler")
    assert res["tactical"]["mean_score"] < res["baseline"]["mean_score"]


def test_health_and_meta():
    assert client.get("/health").json()["status"] == "healthy"
    meta = client.get("/api/v1/meta").json()
    assert set(meta["formats"]) == {"T20", "ODI"}
    assert "India" in meta["formats"]["T20"]["teams"] or meta["formats"]["T20"]["teams"] == []
    assert meta["coverage"]["T20"]["matches"] == 23


def test_head_to_head_and_series_endpoints():
    h2h = client.get("/api/v1/head-to-head", params={"format": "ODI"}).json()
    assert h2h["summary"]["matches"] == 22
    assert h2h["matches"][0]["start_date"] >= h2h["matches"][-1]["start_date"]
    series = client.get("/api/v1/series/latest").json()
    assert series["event"] == "New Zealand tour of India"


def test_player_endpoints():
    bat = client.get("/api/v1/players/batting", params={"format": "T20", "team": "India", "limit": 3}).json()["data"]
    assert len(bat) == 3 and bat[0]["runs"] >= bat[1]["runs"]
    bowl = client.get("/api/v1/players/bowling", params={"format": "ODI", "team": "New Zealand", "limit": 3}).json()["data"]
    assert len(bowl) == 3
    phases = client.get("/api/v1/phase-stats", params={"team": "India", "format": "T20"}).json()["data"]
    assert set(phases) == {"Powerplay", "Middle", "Death"}


def test_predict_and_project_endpoints():
    res = client.post("/api/v1/predict-win-prob",
                      json={"format": "T20", "target": 180, "current_score": 120, "overs": 15.0, "wickets_lost": 4})
    assert res.status_code == 200
    body = res.json()
    assert body["state"]["runs_needed"] == 60
    assert "level" in body["pressure"]
    proj = client.post("/api/v1/project-score", json={"format": "ODI", "current_score": 160, "overs": 30, "wickets_lost": 3})
    assert proj.json()["projected_total"] >= 160


@pytest.mark.parametrize("overs", [15.7, 20.0, 0])
def test_predict_rejects_invalid_overs(overs):
    res = client.post("/api/v1/predict-win-prob",
                      json={"format": "T20", "target": 180, "current_score": 120, "overs": overs, "wickets_lost": 4})
    assert res.status_code == 422


def test_simulate_and_matrix_endpoints():
    sim = client.post("/api/v1/simulate-match", json={"format": "T20", "batting_team": "India",
                                                      "tactical_choice": "Promote Power Hitter Batter", "n_simulations": 500})
    assert sim.status_code == 200 and "impact_summary" in sim.json()
    mu = client.post("/api/v1/h2h-matrix", json={"format": "T20", "team_a": "India", "team_b": "New Zealand"}).json()
    assert len(mu["matrix"]) == len(mu["batters"]) == 11


def test_cluster_and_model_report_endpoints():
    clusters = client.get("/api/v1/batter-clusters", params={"format": "T20", "teams": "India"}).json()
    assert clusters["players"] and all(p["team"] == "India" for p in clusters["players"])
    report = client.get("/api/v1/model-report").json()
    assert set(report) == {"T20", "ODI"}


def test_web_frontend_served():
    page = client.get("/app/")
    assert page.status_code == 200 and "app.js" in page.text
    assert client.get("/app/app.js").status_code == 200
    assert client.get("/", follow_redirects=False).headers["location"] == "/app/"
