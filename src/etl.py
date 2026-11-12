import os
import json
import zipfile
import requests
import duckdb
import pandas as pd

from src.config import ARCHIVES, CRICSHEET_BASE_URL, NON_BOWLER_DISMISSALS, db_path, raw_dir

MATCH_COLUMNS = [
    "match_id", "format", "start_date", "season", "event_name", "match_number", "city", "venue",
    "team1", "team2", "toss_winner", "toss_decision", "winner", "result", "method",
    "win_by_runs", "win_by_wickets", "player_of_match", "scheduled_overs", "target_runs", "target_overs",
]

DELIVERY_COLUMNS = [
    "match_id", "innings", "batting_team", "bowling_team", "over", "ball", "legal_ball_no",
    "batter", "bowler", "non_striker", "runs_batter", "runs_extras", "total_runs",
    "is_wide", "is_noball", "is_legal", "is_four", "is_six",
    "wicket_type", "player_dismissed", "is_wicket", "is_bowler_wicket",
]

PLAYER_COLUMNS = ["match_id", "team", "player"]

BATCH_SIZE = 400


def get_connection(path=None, read_only=False):
    path = path or db_path()
    if not read_only:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return duckdb.connect(path, read_only=read_only)


def download_archives(formats=None, refresh=False, dest_dir=None):
    """Downloads Cricsheet JSON archives (one zip per format) into data/raw."""
    dest_dir = dest_dir or raw_dir()
    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for fmt in formats or ARCHIVES:
        filename = ARCHIVES[fmt]
        path = os.path.join(dest_dir, filename)
        if refresh or not os.path.exists(path):
            print(f"Downloading {filename} from Cricsheet...")
            resp = requests.get(f"{CRICSHEET_BASE_URL}/{filename}", timeout=300)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
        paths.append(path)
    return paths


def init_db(path=None):
    """Creates the matches, deliveries and match_players tables (dropping any existing data)."""
    con = get_connection(path)
    con.execute("DROP TABLE IF EXISTS matches; DROP TABLE IF EXISTS deliveries; DROP TABLE IF EXISTS match_players;")
    con.execute("""
    CREATE TABLE matches (
        match_id VARCHAR PRIMARY KEY,
        format VARCHAR,
        start_date DATE,
        season VARCHAR,
        event_name VARCHAR,
        match_number INTEGER,
        city VARCHAR,
        venue VARCHAR,
        team1 VARCHAR,
        team2 VARCHAR,
        toss_winner VARCHAR,
        toss_decision VARCHAR,
        winner VARCHAR,
        result VARCHAR,
        method VARCHAR,
        win_by_runs INTEGER,
        win_by_wickets INTEGER,
        player_of_match VARCHAR,
        scheduled_overs INTEGER,
        target_runs INTEGER,
        target_overs DOUBLE
    );
    """)
    con.execute("""
    CREATE TABLE deliveries (
        match_id VARCHAR,
        innings INTEGER,
        batting_team VARCHAR,
        bowling_team VARCHAR,
        over INTEGER,
        ball INTEGER,
        legal_ball_no INTEGER,
        batter VARCHAR,
        bowler VARCHAR,
        non_striker VARCHAR,
        runs_batter INTEGER,
        runs_extras INTEGER,
        total_runs INTEGER,
        is_wide BOOLEAN,
        is_noball BOOLEAN,
        is_legal BOOLEAN,
        is_four BOOLEAN,
        is_six BOOLEAN,
        wicket_type VARCHAR,
        player_dismissed VARCHAR,
        is_wicket INTEGER,
        is_bowler_wicket INTEGER
    );
    """)
    con.execute("""
    CREATE TABLE match_players (
        match_id VARCHAR,
        team VARCHAR,
        player VARCHAR
    );
    """)
    con.close()


def parse_match_json(json_data, match_id):
    """Parses one Cricsheet JSON match into (match record, delivery tuples, player tuples)."""
    info = json_data.get("info", {})
    teams = info.get("teams", ["Team A", "Team B"])
    toss = info.get("toss", {})
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    event = info.get("event", {})
    dates = info.get("dates", [])

    # Super overs are stored as extra innings; they are not part of the match proper
    innings_data = [inn for inn in json_data.get("innings", []) if not inn.get("super_over")]

    target_runs, target_overs = 0, None
    if len(innings_data) > 1:
        target = innings_data[1].get("target")
        if target:
            target_runs = target.get("runs", 0)
            target_overs = target.get("overs")
        else:
            first_total = sum(
                d.get("runs", {}).get("total", 0)
                for o in innings_data[0].get("overs", []) for d in o.get("deliveries", [])
            )
            target_runs = first_total + 1

    if outcome.get("result"):
        result = outcome["result"]
    elif outcome.get("winner"):
        result = "normal"
    else:
        result = "unknown"

    match_record = {
        "match_id": str(match_id),
        "format": info.get("match_type"),
        "start_date": dates[0] if dates else None,
        "season": str(info.get("season", "")),
        "event_name": event.get("name", ""),
        "match_number": event.get("match_number"),
        "city": info.get("city", "Unknown"),
        "venue": info.get("venue", "Unknown Venue"),
        "team1": teams[0] if len(teams) > 0 else "Team A",
        "team2": teams[1] if len(teams) > 1 else "Team B",
        "toss_winner": toss.get("winner", ""),
        "toss_decision": toss.get("decision", ""),
        # Tied matches decided by a super over record the winner as the eliminator
        "winner": outcome.get("winner") or outcome.get("eliminator") or "",
        "result": result,
        "method": outcome.get("method", ""),
        "win_by_runs": int(by.get("runs", 0)),
        "win_by_wickets": int(by.get("wickets", 0)),
        "player_of_match": ", ".join(info.get("player_of_match", [])),
        "scheduled_overs": info.get("overs"),
        "target_runs": target_runs,
        "target_overs": target_overs,
    }

    deliveries = []
    for inn_idx, inn in enumerate(innings_data, start=1):
        batting_team = inn.get("team", "")
        bowling_team = next((t for t in teams if t != batting_team), "")
        legal_ball_no = 0
        for over_obj in inn.get("overs", []):
            over_num = over_obj.get("over", 0)
            for ball_idx, deliv in enumerate(over_obj.get("deliveries", []), start=1):
                runs = deliv.get("runs", {})
                extras = deliv.get("extras", {})
                wickets = deliv.get("wickets", [])

                is_wide = "wides" in extras
                is_noball = "noballs" in extras
                is_legal = not (is_wide or is_noball)
                if is_legal:
                    legal_ball_no += 1

                wicket_type = wickets[0].get("kind") if wickets else None
                runs_batter = runs.get("batter", 0)
                # Boundaries are the batter's runs only when not run (Cricsheet flags "non_boundary" otherwise)
                boundary = not runs.get("non_boundary", False)

                deliveries.append((
                    str(match_id), inn_idx, batting_team, bowling_team, over_num, ball_idx, legal_ball_no,
                    deliv.get("batter", "Unknown"), deliv.get("bowler", "Unknown"), deliv.get("non_striker", "Unknown"),
                    runs_batter, runs.get("extras", 0), runs.get("total", 0),
                    is_wide, is_noball, is_legal,
                    runs_batter == 4 and boundary, runs_batter == 6 and boundary,
                    wicket_type,
                    wickets[0].get("player_out") if wickets else None,
                    len(wickets),
                    sum(1 for w in wickets if w.get("kind") not in NON_BOWLER_DISMISSALS),
                ))

    players = [
        (str(match_id), team, player)
        for team, roster in info.get("players", {}).items() for player in roster
    ]
    return match_record, deliveries, players


def _iter_match_files(sources):
    """Yields (match_id, parsed JSON) from zip archives and/or directories of JSON files."""
    for source in sources:
        if source.endswith(".zip"):
            with zipfile.ZipFile(source) as zf:
                for name in zf.namelist():
                    if name.endswith(".json"):
                        yield os.path.splitext(os.path.basename(name))[0], json.loads(zf.read(name))
        elif os.path.isdir(source):
            for name in sorted(os.listdir(source)):
                if name.endswith(".json"):
                    with open(os.path.join(source, name), encoding="utf-8") as f:
                        yield os.path.splitext(name)[0], json.load(f)


def _flush(con, matches, deliveries, players):
    if not matches:
        return
    df_m = pd.DataFrame(matches, columns=MATCH_COLUMNS)
    df_d = pd.DataFrame(deliveries, columns=DELIVERY_COLUMNS)
    df_p = pd.DataFrame(players, columns=PLAYER_COLUMNS)
    con.register("df_m", df_m)
    con.register("df_d", df_d)
    con.register("df_p", df_p)
    con.execute(f"INSERT INTO matches SELECT {', '.join(MATCH_COLUMNS)} FROM df_m")
    con.execute(f"INSERT INTO deliveries SELECT {', '.join(DELIVERY_COLUMNS)} FROM df_d")
    con.execute("INSERT INTO match_players SELECT * FROM df_p")
    for name in ("df_m", "df_d", "df_p"):
        con.unregister(name)
    matches.clear()
    deliveries.clear()
    players.clear()


def ingest(sources=None, path=None, formats=None):
    """Full reload: rebuilds the database from Cricsheet archives/directories. Returns match count."""
    if sources is None:
        sources = [os.path.join(raw_dir(), ARCHIVES[f]) for f in (formats or ARCHIVES)]
    allowed = set(formats or ARCHIVES)

    init_db(path)
    con = get_connection(path)
    matches, deliveries, players = [], [], []
    for match_id, data in _iter_match_files(sources):
        info = data.get("info", {})
        if info.get("match_type") not in allowed or info.get("gender", "male") != "male":
            continue
        try:
            m_rec, d_recs, p_recs = parse_match_json(data, match_id)
        except Exception as e:
            print(f"Skipping {match_id}: {e}")
            continue
        matches.append(m_rec)
        deliveries.extend(d_recs)
        players.extend(p_recs)
        if len(matches) >= BATCH_SIZE:
            _flush(con, matches, deliveries, players)
    _flush(con, matches, deliveries, players)

    count = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    con.close()
    return count


def database_ready(path=None):
    path = path or db_path()
    if not os.path.exists(path):
        return False
    con = get_connection(path, read_only=True)
    try:
        return con.execute("SELECT COUNT(*) FROM matches").fetchone()[0] > 0
    except duckdb.Error:
        return False
    finally:
        con.close()
