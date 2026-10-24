from src.etl import get_connection
from src.features import (
    batting_stats, bowling_stats, calculate_mpi, get_mpi_interpretation, head_to_head, latest_series,
    phase_stats, xi_matchup,
)


def test_head_to_head_summary_adds_up():
    summary, df = head_to_head("India", "New Zealand", "T20")
    assert summary["matches"] == len(df) == 23
    ties = int((df["result"] == "tie").sum())
    decided_by_super_over = int(((df["result"] == "tie") & (df["winner"] != "")).sum())
    assert summary["team_a_wins"] + summary["team_b_wins"] + summary["no_result"] + ties - decided_by_super_over == 23


def test_latest_bilateral_series_is_january_2026_tour():
    event, series = latest_series("India", "New Zealand")
    assert event == "New Zealand tour of India"
    assert sorted(series["format"].value_counts().items()) == [("ODI", 3), ("T20", 5)]


def test_batting_runs_match_delivery_totals():
    df = batting_stats("ODI", team="India")
    con = get_connection(read_only=True)
    total = con.execute("""
        SELECT SUM(d.runs_batter) FROM deliveries d JOIN matches m USING (match_id)
        WHERE m.format = 'ODI' AND d.batting_team = 'India'
    """).fetchone()[0]
    con.close()
    assert df["runs"].sum() == total


def test_bowling_stats_are_consistent():
    df = bowling_stats("T20", team="New Zealand", min_balls=24)
    assert not df.empty
    assert (df["economy"] == (df["runs_conceded"] / df["balls"] * 6).round(2)).all()


def test_phase_stats_cover_all_phases():
    stats = phase_stats("ODI", "India")
    assert list(stats) == ["Powerplay", "Middle", "Death"]
    assert all(v["run_rate"] > 0 for v in stats.values())


def test_xi_matchup_drops_non_bowlers():
    mu = xi_matchup("India", "New Zealand", "T20")
    assert len(mu["batters"]) == 11
    assert 0 < len(mu["bowlers"]) <= 11
    assert all(any(row[b]["balls"] for row in mu["matrix"]) for b in mu["bowlers"])


def test_match_pressure_index():
    assert calculate_mpi(12.0, 6, 30, "T20") == 8.0
    assert calculate_mpi(6.0, 10, 300, "ODI") == 0.6
    assert get_mpi_interpretation(0.6)["level"] == "Comfortable Control"
    assert get_mpi_interpretation(8.0)["level"] == "Panic Mode"
