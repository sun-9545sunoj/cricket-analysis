import numpy as np
import pandas as pd

from src.config import FORMATS, phase_case_sql, total_balls
from src.etl import get_connection

PHASE_SQL = phase_case_sql()


def _query(sql, params=None, path=None):
    con = get_connection(path, read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


# ---------------- Teams, head-to-head and series ----------------

def list_teams(fmt, min_matches=20, path=None):
    """Teams with at least `min_matches` matches in the format, most active first."""
    df = _query("""
        SELECT team, COUNT(*) AS matches FROM (
            SELECT team1 AS team FROM matches WHERE format = ?
            UNION ALL
            SELECT team2 FROM matches WHERE format = ?
        ) GROUP BY team HAVING COUNT(*) >= ? ORDER BY matches DESC
    """, [fmt, fmt, min_matches], path)
    return df["team"].tolist()


def head_to_head(team_a, team_b, fmt, path=None):
    """Match-by-match results between two teams plus a win/loss summary."""
    df = _query("""
        SELECT match_id, start_date, event_name, venue, city, winner, result, method,
               win_by_runs, win_by_wickets, player_of_match, toss_winner, toss_decision
        FROM matches
        WHERE format = ? AND ((team1 = ? AND team2 = ?) OR (team1 = ? AND team2 = ?))
        ORDER BY start_date DESC
    """, [fmt, team_a, team_b, team_b, team_a], path)

    def margin(row):
        if row["result"] == "no result":
            return "No result"
        if row["result"] == "tie":
            return f"Tie ({row['winner']} won super over)" if row["winner"] else "Tie"
        by = f"{row['win_by_runs']} runs" if row["win_by_runs"] else f"{row['win_by_wickets']} wickets"
        return f"{row['winner']} won by {by}" + (f" ({row['method']})" if row["method"] else "")

    df["summary"] = df.apply(margin, axis=1) if not df.empty else []
    summary = {
        "matches": int(len(df)),
        "team_a_wins": int((df["winner"] == team_a).sum()),
        "team_b_wins": int((df["winner"] == team_b).sum()),
        "no_result": int((df["result"] == "no result").sum()),
        "first_match": pd.to_datetime(df["start_date"]).min().strftime("%Y-%m-%d") if not df.empty else None,
        "last_match": pd.to_datetime(df["start_date"]).max().strftime("%Y-%m-%d") if not df.empty else None,
    }
    return summary, df


def latest_series(team_a, team_b, fmt=None, bilateral_only=True, path=None):
    """The most recent event where the two teams met (bilateral tours by default), across formats."""
    fmt_clause = "AND format = ?" if fmt else ""
    fmt_clause += " AND event_name LIKE '% tour of %'" if bilateral_only else ""
    params = [team_a, team_b, team_b, team_a] + ([fmt] if fmt else [])
    df = _query(f"""
        SELECT match_id, format, start_date, event_name, match_number, venue, winner, result,
               win_by_runs, win_by_wickets, method, player_of_match
        FROM matches
        WHERE ((team1 = ? AND team2 = ?) OR (team1 = ? AND team2 = ?)) {fmt_clause}
        ORDER BY start_date DESC
    """, params, path)
    if df.empty:
        return None, df
    latest_event = df.iloc[0]["event_name"]
    latest_date = pd.to_datetime(df.iloc[0]["start_date"])
    # Event names repeat across years ("New Zealand tour of India"), so only keep the latest window
    dates = pd.to_datetime(df["start_date"])
    series = df[(df["event_name"] == latest_event) & (dates >= latest_date - pd.Timedelta(days=60))]
    return latest_event, series.sort_values("start_date")


# ---------------- Player aggregates ----------------

def _filters(fmt, team_col, team=None, opponent=None, opp_col=None, since=None):
    clauses, params = ["m.format = ?"], [fmt]
    if team:
        clauses.append(f"{team_col} = ?")
        params.append(team)
    if opponent:
        clauses.append(f"{opp_col} = ?")
        params.append(opponent)
    if since:
        clauses.append("m.start_date >= ?")
        params.append(since)
    return " AND ".join(clauses), params


def batting_stats(fmt, team=None, opponent=None, since=None, min_balls=0, path=None):
    """Per-batter career aggregates (runs, average, strike rate, boundary and dot-ball %)."""
    where, params = _filters(fmt, "d.batting_team", team, opponent, "d.bowling_team", since)
    df = _query(f"""
        SELECT
            d.batter AS player,
            ANY_VALUE(d.batting_team) AS team,
            COUNT(DISTINCT d.match_id) AS innings,
            SUM(d.runs_batter) AS runs,
            COUNT(*) FILTER (WHERE NOT d.is_wide) AS balls,
            SUM(CASE WHEN d.is_four THEN 1 ELSE 0 END) AS fours,
            SUM(CASE WHEN d.is_six THEN 1 ELSE 0 END) AS sixes,
            COUNT(*) FILTER (WHERE NOT d.is_wide AND d.runs_batter = 0) AS dots
        FROM deliveries d JOIN matches m USING (match_id)
        WHERE {where}
        GROUP BY d.batter
        HAVING COUNT(*) FILTER (WHERE NOT d.is_wide) >= ?
    """, params + [min_balls], path)

    outs = _query(f"""
        SELECT d.player_dismissed AS player, COUNT(*) AS outs
        FROM deliveries d JOIN matches m USING (match_id)
        WHERE {where} AND d.player_dismissed IS NOT NULL
          AND d.wicket_type NOT IN ('retired hurt', 'retired not out')
        GROUP BY d.player_dismissed
    """, params, path)
    df = df.merge(outs, on="player", how="left").fillna({"outs": 0})

    count_cols = ["innings", "runs", "balls", "fours", "sixes", "dots", "outs"]
    df[count_cols] = df[count_cols].fillna(0).astype(int)
    df["average"] = np.where(df["outs"] > 0, df["runs"] / df["outs"].clip(lower=1), np.nan).round(2)
    df["strike_rate"] = (df["runs"] / df["balls"].clip(lower=1) * 100).round(2)
    df["boundary_pct"] = ((df["fours"] + df["sixes"]) / df["balls"].clip(lower=1) * 100).round(2)
    df["dot_pct"] = (df["dots"] / df["balls"].clip(lower=1) * 100).round(2)
    return df.sort_values("runs", ascending=False).reset_index(drop=True)


def bowling_stats(fmt, team=None, opponent=None, since=None, min_balls=0, path=None):
    """Per-bowler aggregates (wickets, economy, average, strike rate, dot-ball %)."""
    where, params = _filters(fmt, "d.bowling_team", team, opponent, "d.batting_team", since)
    df = _query(f"""
        SELECT
            d.bowler AS player,
            ANY_VALUE(d.bowling_team) AS team,
            COUNT(DISTINCT d.match_id) AS innings,
            COUNT(*) FILTER (WHERE d.is_legal) AS balls,
            SUM(d.runs_batter + CASE WHEN d.is_wide OR d.is_noball THEN d.runs_extras ELSE 0 END) AS runs_conceded,
            SUM(d.is_bowler_wicket) AS wickets,
            COUNT(*) FILTER (WHERE d.is_legal AND d.total_runs = 0) AS dots
        FROM deliveries d JOIN matches m USING (match_id)
        WHERE {where}
        GROUP BY d.bowler
        HAVING COUNT(*) FILTER (WHERE d.is_legal) >= ?
    """, params + [min_balls], path)
    count_cols = ["innings", "balls", "runs_conceded", "wickets", "dots"]
    df[count_cols] = df[count_cols].fillna(0).astype(int)
    df["economy"] = (df["runs_conceded"] / df["balls"].clip(lower=1) * 6).round(2)
    df["average"] = np.where(df["wickets"] > 0, df["runs_conceded"] / df["wickets"].clip(lower=1), np.nan).round(2)
    df["strike_rate"] = np.where(df["wickets"] > 0, df["balls"] / df["wickets"].clip(lower=1), np.nan).round(2)
    df["dot_pct"] = (df["dots"] / df["balls"].clip(lower=1) * 100).round(2)
    return df.sort_values(["wickets", "economy"], ascending=[False, True]).reset_index(drop=True)


def phase_stats(fmt, team, opponent=None, since=None, path=None):
    """Batting run rate and wickets lost per innings in each phase for a team."""
    where, params = _filters(fmt, "d.batting_team", team, opponent, "d.bowling_team", since)
    df = _query(f"""
        SELECT {PHASE_SQL} AS phase,
               SUM(d.total_runs) AS runs,
               COUNT(*) FILTER (WHERE d.is_legal) AS balls,
               SUM(d.is_wicket) AS wickets,
               COUNT(DISTINCT d.match_id || '-' || d.innings) AS innings
        FROM deliveries d JOIN matches m USING (match_id)
        WHERE {where} AND d.innings <= 2
        GROUP BY phase
    """, params, path)
    out = {}
    for _, _, name in FORMATS[fmt]["phases"]:
        row = df[df["phase"] == name]
        if row.empty or row["balls"].iloc[0] == 0:
            out[name] = {"run_rate": 0.0, "wickets_per_innings": 0.0}
        else:
            r = row.iloc[0]
            out[name] = {
                "run_rate": round(float(r["runs"] / r["balls"] * 6), 2),
                "wickets_per_innings": round(float(r["wickets"] / max(r["innings"], 1)), 2),
            }
    return out


def latest_xi(team, fmt, opponent=None, path=None):
    """Playing XI from the team's most recent match in the format (optionally against `opponent`)."""
    opp_clause = "AND ((m.team1 = ? AND m.team2 = ?) OR (m.team1 = ? AND m.team2 = ?))" if opponent else ""
    params = [team, fmt] + ([team, opponent, opponent, team] if opponent else [])
    df = _query(f"""
        WITH last_match AS (
            SELECT m.match_id FROM matches m JOIN match_players p USING (match_id)
            WHERE p.team = ? AND m.format = ? {opp_clause}
            ORDER BY m.start_date DESC LIMIT 1
        )
        SELECT p.player FROM match_players p JOIN last_match USING (match_id) WHERE p.team = ?
    """, params + [team], path)
    return df["player"].tolist()


# ---------------- Matchups ----------------

def get_h2h_matrix(batters, bowlers, fmt, path=None):
    """Batter-vs-bowler records across all internationals in the format."""
    if not batters or not bowlers:
        return []
    df_raw = _query(f"""
        SELECT d.batter, d.bowler,
               SUM(d.runs_batter) AS runs,
               COUNT(*) FILTER (WHERE NOT d.is_wide) AS balls,
               SUM(d.is_bowler_wicket) AS dismissals
        FROM deliveries d JOIN matches m USING (match_id)
        WHERE m.format = ?
          AND d.batter IN ({', '.join('?' * len(batters))})
          AND d.bowler IN ({', '.join('?' * len(bowlers))})
        GROUP BY d.batter, d.bowler
    """, [fmt] + list(batters) + list(bowlers), path)
    lookup = {(r.batter, r.bowler): r for r in df_raw.itertuples()}

    matrix = []
    for batter in batters:
        row = {"Batter": batter}
        for bowler in bowlers:
            rec = lookup.get((batter, bowler))
            runs, balls, dismissals = (int(rec.runs), int(rec.balls), int(rec.dismissals)) if rec else (0, 0, 0)
            sr = round(runs / balls * 100, 1) if balls else 0.0
            row[bowler] = {"runs": runs, "balls": balls, "dismissals": dismissals, "sr": sr,
                           "summary": f"{runs}r ({balls}b), {dismissals}W, SR: {sr}"}
        matrix.append(row)
    return matrix


def get_tactical_alerts(matrix_data, bowlers, fmt="T20"):
    """Flags matchups where the bowler has dominated: repeated dismissals or a strike rate well below par."""
    par_sr = 105.0 if fmt == "T20" else 70.0
    alerts = []
    for row in matrix_data:
        for bowler in bowlers:
            d = row[bowler]
            if d["balls"] >= 12 and (d["dismissals"] >= 2 or d["sr"] < par_sr):
                alerts.append({"batter": row["Batter"], "bowler": bowler, **{k: d[k] for k in ("runs", "balls", "dismissals", "sr")}})
    return sorted(alerts, key=lambda a: (-a["dismissals"], a["sr"]))


def xi_matchup(batting_team, bowling_team, fmt, path=None):
    """Latest XIs from the teams' last meeting: batter-vs-bowler matrix plus tactical alerts."""
    batters = latest_xi(batting_team, fmt, bowling_team, path) or latest_xi(batting_team, fmt, path=path)
    bowlers = latest_xi(bowling_team, fmt, batting_team, path) or latest_xi(bowling_team, fmt, path=path)
    matrix = get_h2h_matrix(batters, bowlers, fmt, path)
    # Drop players in the XI who have not bowled to any of these batters (e.g. specialist batters)
    bowlers = [b for b in bowlers if any(row[b]["balls"] for row in matrix)]
    matrix = [{"Batter": row["Batter"], **{b: row[b] for b in bowlers}} for row in matrix]
    return {"batters": batters, "bowlers": bowlers, "matrix": matrix,
            "alerts": get_tactical_alerts(matrix, bowlers, fmt)}


# ---------------- Match pressure ----------------

def calculate_mpi(req_run_rate, wickets_remaining, balls_remaining, fmt="T20"):
    """
    Match Pressure Index:
    MPI = (Required Run Rate / Wickets Remaining) * (Total Balls / Balls Remaining)
    """
    w_rem = max(wickets_remaining, 1)
    b_rem = max(balls_remaining, 1)
    return round(float((req_run_rate / w_rem) * (total_balls(fmt) / b_rem)), 3)


def get_mpi_interpretation(mpi):
    """Translates the Match Pressure Index into a status and plain-English explanation."""
    if mpi < 1.0:
        return {"level": "Comfortable Control", "emoji": "🟢", "color": "#38ef7d",
                "desc": "Batting side in full control. Standard run rate needed with plenty of wickets in hand."}
    if mpi < 2.5:
        return {"level": "Moderate Pressure", "emoji": "🟡", "color": "#00c6ff",
                "desc": "Evenly poised. Regular boundaries needed to stay on target."}
    if mpi < 4.0:
        return {"level": "High Pressure", "emoji": "🟠", "color": "#ffaa00",
                "desc": "Chase getting tight. A wicket now would sharply reduce the chances."}
    return {"level": "Panic Mode", "emoji": "🔴", "color": "#ff4b4b",
            "desc": "Extreme pressure. Needs boundaries almost every over."}
