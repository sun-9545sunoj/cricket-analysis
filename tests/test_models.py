import pytest

from src.models import cluster_batters, predict_win_probability, project_first_innings_score, scoreboard_state


@pytest.mark.parametrize("fmt", ["T20", "ODI"])
def test_training_report_structure(fixture_database, fmt):
    report = fixture_database["report"][fmt]
    win = report["win_probability"]
    assert set(win["models"]) == {"Logistic Regression", "XGBoost"}
    assert win["selected"] in win["models"]
    assert win["train_matches"] > win["test_matches"] > 0
    assert report["score_projection"]["selected"] in report["score_projection"]["models"]


@pytest.mark.parametrize("fmt,balls", [("T20", 120), ("ODI", 300)])
def test_win_probability_is_sensible(fmt, balls):
    easy = predict_win_probability(fmt, scoreboard_state(fmt, 150, 145, balls - 30, 1))
    hard = predict_win_probability(fmt, scoreboard_state(fmt, 250, 150, balls - 12, 9))
    for res in (easy, hard):
        assert abs(res["chasing_win_pct"] + res["defending_win_pct"] - 100) < 0.2
        assert len(res["shap_impacts"]) == 5
    assert easy["chasing_win_pct"] > hard["chasing_win_pct"]


def test_scoreboard_state():
    state = scoreboard_state("T20", 180, 120, 90, 4)
    assert state == {"runs_needed": 60, "balls_left": 30, "wickets_left": 6, "req_rr": 12.0, "curr_rr": 8.0, "target_runs": 180}


def test_projection_is_at_least_current_score():
    assert project_first_innings_score("T20", 150, 110, 3) >= 150


def test_batter_clusters_have_distinct_names():
    df, info = cluster_batters("T20")
    assert df["style"].nunique() == info["k"] == 4
    assert -1 <= info["silhouette"] <= 1
    assert sum(c["players"] for c in info["clusters"]) == len(df)
