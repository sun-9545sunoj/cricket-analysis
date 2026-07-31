import os
import json
import zipfile
import random
import requests
import duckdb
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ipl.duckdb")
DEFAULT_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

IPL_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Punjab Kings",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

IPL_VENUES = [
    "Wankhede Stadium, Mumbai",
    "M. Chinnaswamy Stadium, Bengaluru",
    "MA Chidambaram Stadium, Chepauk, Chennai",
    "Eden Gardens, Kolkata",
    "Arun Jaitley Stadium, Delhi",
    "Narendra Modi Stadium, Ahmedabad",
    "Rajiv Gandhi International Stadium, Hyderabad",
    "HPCA Stadium, Dharamshala"
]

SAMPLE_BATTERS = {
    "Chennai Super Kings": ["Ruturaj Gaikwad", "Devon Conway", "Shivam Dube", "MS Dhoni", "Ravindra Jadeja", "Daryl Mitchell", "Sameer Rizvi", "Moeen Ali", "Deepak Chahar", "Tushar Deshpande", "Matheesha Pathirana"],
    "Mumbai Indians": ["Rohit Sharma", "Ishan Kishan", "Suryakumar Yadav", "Tilak Varma", "Hardik Pandya", "Tim David", "Naman Dhir", "Mohammad Nabi", "Piyush Chawla", "Jasprit Bumrah", "Gerald Coetzee"],
    "Royal Challengers Bengaluru": ["Virat Kohli", "Faf du Plessis", "Rajat Patidar", "Glenn Maxwell", "Cameron Green", "Dinesh Karthik", "Mahipal Lomror", "Karn Sharma", "Mohammed Siraj", "Yash Dayal", "Lockie Ferguson"],
    "Kolkata Knight Riders": ["Sunil Narine", "Phil Salt", "Venkatesh Iyer", "Shreyas Iyer", "Rinku Singh", "Andre Russell", "Ramandeep Singh", "Mitchell Starc", "Harshit Rana", "Varun Chakaravarthy", "Vaibhav Arora"],
    "Delhi Capitals": ["Prithvi Shaw", "Jake Fraser-McGurk", "Abishek Porel", "Rishabh Pant", "Tristan Stubbs", "Axar Patel", "Kumar Kushagra", "Kuldeep Yadav", "Mukesh Kumar", "Khaleel Ahmed", "Ishant Sharma"],
    "Rajasthan Royals": ["Yashasvi Jaiswal", "Jos Buttler", "Sanju Samson", "Riyan Parag", "Dhruv Jurel", "Shimron Hetmyer", "Rovman Powell", "Ravichandran Ashwin", "Trent Boult", "Avesh Khan", "Yuzvendra Chahal"],
    "Sunrisers Hyderabad": ["Travis Head", "Abhishek Sharma", "Aiden Markram", "Heinrich Klaasen", "Nitish Kumar Reddy", "Abdul Samad", "Shahbaz Ahmed", "Pat Cummins", "Bhuvaneshwar Kumar", "Jaydev Unadkat", "T Natarajan"],
    "Gujarat Titans": ["Shubman Gill", "Sai Sudharsan", "Shahrukh Khan", "David Miller", "Rahul Tewatia", "Rashid Khan", "Vijay Shankar", "Noor Ahmad", "Mohit Sharma", "Umesh Yadav", "Spencer Johnson"]
}

def get_connection(db_path=DEFAULT_DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return duckdb.connect(db_path)

def setup_db_if_needed(db_path=DEFAULT_DB_PATH, raw_dir=DEFAULT_RAW_DIR):
    """Checks if database exists; if not, initializes tables, generates sample data, and ingests JSON files."""
    if not os.path.exists(db_path):
        init_db(db_path)
        if not os.path.exists(raw_dir) or len([f for f in os.listdir(raw_dir) if f.endswith('.json')]) == 0:
            generate_sample_data(100, raw_dir)
        ingest_json_files(raw_dir, db_path)
    return True

def init_db(db_path=DEFAULT_DB_PATH):
    """Create normalized database tables matches and deliveries."""
    con = get_connection(db_path)
    con.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id VARCHAR PRIMARY KEY,
        season VARCHAR,
        city VARCHAR,
        venue VARCHAR,
        team1 VARCHAR,
        team2 VARCHAR,
        toss_winner VARCHAR,
        toss_decision VARCHAR,
        winner VARCHAR,
        win_by_runs INTEGER,
        win_by_wickets INTEGER,
        target_runs INTEGER
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS deliveries (
        match_id VARCHAR,
        innings INTEGER,
        over INTEGER,
        ball INTEGER,
        batter VARCHAR,
        bowler VARCHAR,
        non_striker VARCHAR,
        runs_batter INTEGER,
        runs_extras INTEGER,
        total_runs INTEGER,
        is_wide BOOLEAN,
        is_noball BOOLEAN,
        wicket_type VARCHAR,
        player_dismissed VARCHAR,
        is_wicket INTEGER
    );
    """)
    con.close()

def parse_match_json(json_data, match_id):
    """Parses a single Cricsheet JSON match object into matches and deliveries data structures."""
    info = json_data.get("info", {})
    teams = info.get("teams", ["Team A", "Team B"])
    toss = info.get("toss", {})
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})

    target_runs = None
    innings_data = json_data.get("innings", [])
    if len(innings_data) > 1:
        # Target for 2nd innings is total of 1st innings + 1
        first_inn_runs = 0
        for over_obj in innings_data[0].get("overs", []):
            for deliv in over_obj.get("deliveries", []):
                first_inn_runs += deliv.get("runs", {}).get("total", 0)
        target_runs = first_inn_runs + 1

    match_record = {
        "match_id": str(match_id),
        "season": str(info.get("season", "2024")),
        "city": info.get("city", "Unknown"),
        "venue": info.get("venue", "Unknown Venue"),
        "team1": teams[0] if len(teams) > 0 else "Team A",
        "team2": teams[1] if len(teams) > 1 else "Team B",
        "toss_winner": toss.get("winner", ""),
        "toss_decision": toss.get("decision", ""),
        "winner": outcome.get("winner", ""),
        "win_by_runs": int(by.get("runs", 0)),
        "win_by_wickets": int(by.get("wickets", 0)),
        "target_runs": target_runs if target_runs is not None else 0
    }

    deliveries_list = []
    for inn_idx, inn in enumerate(innings_data, start=1):
        for over_obj in inn.get("overs", []):
            over_num = over_obj.get("over", 0)
            for ball_idx, deliv in enumerate(over_obj.get("deliveries", []), start=1):
                runs = deliv.get("runs", {})
                extras = deliv.get("extras", {})
                wickets = deliv.get("wickets", [])

                wicket_type = wickets[0].get("kind") if wickets else None
                player_dismissed = wickets[0].get("player_out") if wickets else None

                deliveries_list.append({
                    "match_id": str(match_id),
                    "innings": inn_idx,
                    "over": over_num,
                    "ball": ball_idx,
                    "batter": deliv.get("batter", "Unknown"),
                    "bowler": deliv.get("bowler", "Unknown"),
                    "non_striker": deliv.get("non_striker", "Unknown"),
                    "runs_batter": runs.get("batter", 0),
                    "runs_extras": runs.get("extras", 0),
                    "total_runs": runs.get("total", 0),
                    "is_wide": "wides" in extras,
                    "is_noball": "noballs" in extras,
                    "wicket_type": wicket_type,
                    "player_dismissed": player_dismissed,
                    "is_wicket": 1 if wickets else 0
                })

    return match_record, deliveries_list

def ingest_json_files(raw_dir=DEFAULT_RAW_DIR, db_path=DEFAULT_DB_PATH):
    """Reads all JSON files from raw_dir and inserts into DuckDB."""
    init_db(db_path)
    con = get_connection(db_path)

    matches_list = []
    all_deliveries = []

    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist.")
        return 0

    json_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    for filename in json_files:
        filepath = os.path.join(raw_dir, filename)
        match_id = os.path.splitext(filename)[0]
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                m_rec, d_recs = parse_match_json(data, match_id)
                matches_list.append(m_rec)
                all_deliveries.extend(d_recs)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if matches_list:
        df_matches = pd.DataFrame(matches_list)
        df_deliveries = pd.DataFrame(all_deliveries)

        # Clear existing data for fresh ingestion
        con.execute("DELETE FROM matches;")
        con.execute("DELETE FROM deliveries;")

        con.register("df_matches", df_matches)
        con.execute("INSERT INTO matches SELECT * FROM df_matches")
        
        con.register("df_deliveries", df_deliveries)
        con.execute("INSERT INTO deliveries SELECT * FROM df_deliveries")
        
    count = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    con.close()
    return count

def generate_sample_data(num_matches=100, raw_dir=DEFAULT_RAW_DIR):
    """Generates realistic synthetic Cricsheet-style JSON match archives for robust modeling & demo."""
    os.makedirs(raw_dir, exist_ok=True)
    teams_list = list(SAMPLE_BATTERS.keys())

    for m_idx in range(1, num_matches + 1):
        match_id = f"ipl_{2024000 + m_idx}"
        team1, team2 = random.sample(teams_list, 2)
        venue = random.choice(IPL_VENUES)
        city = venue.split(",")[-1].strip()
        toss_winner = random.choice([team1, team2])
        toss_decision = random.choice(["bat", "field"])

        roster1 = SAMPLE_BATTERS[team1]
        roster2 = SAMPLE_BATTERS[team2]

        batters_inn1 = roster1 if toss_decision == "bat" and toss_winner == team1 else (roster2 if toss_decision == "bat" else roster1)
        bowlers_inn1 = roster2 if batters_inn1 == roster1 else roster1

        batters_inn2 = bowlers_inn1
        bowlers_inn2 = batters_inn1

        # Simulate Innings 1
        inn1_overs = []
        tot_runs_inn1 = 0
        wickets_inn1 = 0
        b_idx = 0
        for ov in range(20):
            if wickets_inn1 >= 10:
                break
            bowler = random.choice(bowlers_inn1[-5:]) # Main bowlers
            delivs = []
            for b in range(1, 7):
                if wickets_inn1 >= 10:
                    break
                batter = batters_inn1[b_idx % 11]
                non_striker = batters_inn1[(b_idx + 1) % 11]
                
                # Distribution of runs
                r = random.choices([0, 1, 2, 4, 6, "W"], weights=[0.35, 0.38, 0.08, 0.11, 0.05, 0.03])[0]
                if r == "W":
                    runs_b, runs_ex, total_r = 0, 0, 0
                    wickets_list = [{"kind": random.choice(["caught", "bowled", "lbw"]), "player_out": batter}]
                    wickets_inn1 += 1
                    b_idx += 1
                else:
                    runs_b, runs_ex, total_r = r, 0, r
                    wickets_list = []
                
                tot_runs_inn1 += total_r
                delivs.append({
                    "batter": batter,
                    "bowler": bowler,
                    "non_striker": non_striker,
                    "runs": {"batter": runs_b, "extras": runs_ex, "total": total_r},
                    "wickets": wickets_list
                })
            inn1_overs.append({"over": ov, "deliveries": delivs})

        target_runs = tot_runs_inn1 + 1

        # Simulate Innings 2
        inn2_overs = []
        tot_runs_inn2 = 0
        wickets_inn2 = 0
        b_idx_2 = 0
        winner = None

        for ov in range(20):
            if tot_runs_inn2 >= target_runs or wickets_inn2 >= 10:
                break
            bowler = random.choice(bowlers_inn2[-5:])
            delivs = []
            for b in range(1, 7):
                if tot_runs_inn2 >= target_runs or wickets_inn2 >= 10:
                    break
                batter = batters_inn2[b_idx_2 % 11]
                non_striker = batters_inn2[(b_idx_2 + 1) % 11]

                r = random.choices([0, 1, 2, 4, 6, "W"], weights=[0.33, 0.39, 0.08, 0.12, 0.05, 0.03])[0]
                if r == "W":
                    runs_b, runs_ex, total_r = 0, 0, 0
                    wickets_list = [{"kind": random.choice(["caught", "bowled", "lbw"]), "player_out": batter}]
                    wickets_inn2 += 1
                    b_idx_2 += 1
                else:
                    runs_b, runs_ex, total_r = r, 0, r
                    wickets_list = []

                tot_runs_inn2 += total_r
                delivs.append({
                    "batter": batter,
                    "bowler": bowler,
                    "non_striker": non_striker,
                    "runs": {"batter": runs_b, "extras": runs_ex, "total": total_r},
                    "wickets": wickets_list
                })
            inn2_overs.append({"over": ov, "deliveries": delivs})

        if tot_runs_inn2 >= target_runs:
            winner = (team1 if batters_inn2 == roster1 else team2)
            win_by_wickets = 10 - wickets_inn2
            win_by_runs = 0
        else:
            winner = (team1 if batters_inn1 == roster1 else team2)
            win_by_runs = target_runs - 1 - tot_runs_inn2
            win_by_wickets = 0

        match_json = {
            "info": {
                "season": random.choice(["2023", "2024"]),
                "city": city,
                "venue": venue,
                "teams": [team1, team2],
                "toss": {"winner": toss_winner, "decision": toss_decision},
                "outcome": {
                    "winner": winner,
                    "by": {"runs": win_by_runs} if win_by_runs > 0 else {"wickets": win_by_wickets}
                },
                "players": {
                    team1: roster1,
                    team2: roster2
                }
            },
            "innings": [
                {"team": (team1 if batters_inn1 == roster1 else team2), "overs": inn1_overs},
                {"team": (team1 if batters_inn2 == roster1 else team2), "overs": inn2_overs}
            ]
        }

        filepath = os.path.join(raw_dir, f"{match_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(match_json, f, indent=2)

    print(f"Successfully generated {num_matches} match JSON files in {raw_dir}.")

if __name__ == "__main__":
    generate_sample_data(100)
    count = ingest_json_files()
    print(f"ETL Complete. Total matches in database: {count}")
