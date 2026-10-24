import pytest

from src.config import overs_to_balls, phase_of
from src.etl import get_connection, parse_match_json


def _q(sql):
    con = get_connection(read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_ingest_counts_and_formats(fixture_database):
    assert fixture_database["count"] == 45
    assert dict(_q("SELECT format, COUNT(*) FROM matches GROUP BY format")) == {"T20": 23, "ODI": 22}


def test_every_match_has_two_playing_xis():
    rows = _q("SELECT match_id, team, COUNT(*) FROM match_players GROUP BY match_id, team")
    assert len(rows) == 45 * 2
    assert {n for _, _, n in rows} == {11}


def test_legal_balls_never_exceed_format_limit():
    rows = _q("""
        SELECT m.format, MAX(d.legal_ball_no) FROM deliveries d JOIN matches m USING (match_id) GROUP BY m.format
    """)
    limits = {"T20": 120, "ODI": 300}
    for fmt, max_ball in rows:
        assert max_ball <= limits[fmt]


def _delivery(batter="A", bowler="X", runs=0, extras=None, wicket=None):
    d = {"batter": batter, "bowler": bowler, "non_striker": "B",
         "runs": {"batter": runs, "extras": sum((extras or {}).values()), "total": runs + sum((extras or {}).values())}}
    if extras:
        d["extras"] = extras
    if wicket:
        d["wickets"] = [{"kind": wicket, "player_out": batter}]
    return d


def test_parse_match_handles_extras_wickets_and_super_overs():
    match = {
        "info": {
            "match_type": "T20", "dates": ["2026-01-01"], "teams": ["India", "New Zealand"],
            "outcome": {"result": "tie", "eliminator": "India"},
            "players": {"India": ["A", "B"], "New Zealand": ["X"]},
        },
        "innings": [
            {"team": "India", "overs": [{"over": 0, "deliveries": [
                _delivery(runs=4),
                _delivery(extras={"wides": 1}),
                _delivery(wicket="run out"),
                _delivery(wicket="bowled"),
            ]}]},
            {"team": "New Zealand", "target": {"runs": 6, "overs": 20}, "overs": [{"over": 0, "deliveries": [_delivery(runs=6)]}]},
            {"team": "India", "super_over": True, "overs": [{"over": 0, "deliveries": [_delivery(runs=6)]}]},
        ],
    }
    m, deliveries, players = parse_match_json(match, "test")

    assert m["result"] == "tie" and m["winner"] == "India"
    assert m["target_runs"] == 6
    assert len(deliveries) == 5, "super over innings must be skipped"
    first = [d for d in deliveries if d[1] == 1]
    is_legal = [d[15] for d in first]
    assert is_legal == [True, False, True, True]
    assert [d[6] for d in first] == [1, 1, 2, 3], "legal_ball_no ignores the wide"
    assert first[0][16] is True, "a 4 off the bat is a four"
    assert [d[21] for d in first] == [0, 0, 0, 1], "run outs are not credited to the bowler"
    assert len(players) == 3


@pytest.mark.parametrize("overs,balls", [(0.1, 1), (13.2, 80), (20.0, 120), (49.5, 299)])
def test_overs_to_balls(overs, balls):
    assert overs_to_balls(overs) == balls


def test_overs_to_balls_rejects_invalid_ball_part():
    with pytest.raises(ValueError):
        overs_to_balls(15.7)


def test_phase_boundaries_per_format():
    assert [phase_of("T20", o) for o in (0, 5, 6, 14, 15, 19)] == ["Powerplay", "Powerplay", "Middle", "Middle", "Death", "Death"]
    assert [phase_of("ODI", o) for o in (9, 10, 39, 40)] == ["Powerplay", "Middle", "Middle", "Death"]
