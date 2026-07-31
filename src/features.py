import numpy as np
import pandas as pd
import duckdb
from src.etl import DEFAULT_DB_PATH, get_connection

def get_phase(over_num):
    """Categorize over into Powerplay, Middle, or Death phase."""
    if over_num < 6:
        return "Powerplay"
    elif over_num < 15:
        return "Middle"
    else:
        return "Death"

def calculate_mpi(req_run_rate, wickets_remaining, balls_remaining):
    """
    Computes Match Pressure Index (MPI):
    MPI = (Required Run Rate / Wickets Remaining) * (Balls Remaining / 120)^(-1)
    """
    w_rem = max(wickets_remaining, 1)
    b_rem = max(balls_remaining, 1)
    
    mpi = (req_run_rate / w_rem) * (120.0 / b_rem)
    return round(float(mpi), 3)

def get_mpi_interpretation(mpi):
    """Translates Match Pressure Index (MPI) into human-friendly status & plain English explanation."""
    if mpi < 1.0:
        return {
            "level": "Comfortable Control",
            "emoji": "🟢",
            "color": "#38ef7d",
            "desc": "Batting side in full control. Standard run rate needed with plenty of wickets in hand."
        }
    elif mpi < 2.5:
        return {
            "level": "Moderate Pressure",
            "emoji": "🟡",
            "color": "#00c6ff",
            "desc": "Equilibrium match state. Boundaries needed every over to stay on target."
        }
    elif mpi < 4.0:
        return {
            "level": "High Pressure",
            "emoji": "🟠",
            "color": "#ffaa00",
            "desc": "Chase getting tight! A lost wicket now will drastically reduce win chances."
        }
    else:
        return {
            "level": "Panic Mode",
            "emoji": "🔴",
            "color": "#ff4b4b",
            "desc": "Extreme pressure! Requires aggressive boundary hitting on almost every ball."
        }

def get_sr_true_interpretation(sr_true):
    """Translates True Strike Rate (SR_true) into plain English human benchmark."""
    if sr_true >= 1.2:
        return {
            "badge": "🚀 High Impact Scorer",
            "color": "#38ef7d",
            "desc": f"Scoring {round((sr_true-1.0)*100, 1)}% faster than typical batters at this venue in this phase."
        }
    elif sr_true >= 0.9:
        return {
            "badge": "🎯 On Ground Benchmark",
            "color": "#00c6ff",
            "desc": "Scoring at standard expected venue speed for this phase."
        }
    else:
        return {
            "badge": "⚠️ Below Venue Average",
            "color": "#ff4b4b",
            "desc": f"Scoring {round((1.0-sr_true)*100, 1)}% slower than average batters at this ground."
        }

def compute_venue_phase_averages(db_path=DEFAULT_DB_PATH):
    """Computes average strike rate across all batters at each venue and phase."""
    con = get_connection(db_path)
    query = """
    SELECT 
        m.venue,
        CASE 
            WHEN d.over < 6 THEN 'Powerplay'
            WHEN d.over < 15 THEN 'Middle'
            ELSE 'Death'
        END as phase,
        SUM(d.runs_batter) as total_runs,
        COUNT(CASE WHEN d.is_wide = False THEN 1 END) as total_balls
    FROM deliveries d
    JOIN matches m ON d.match_id = m.match_id
    GROUP BY m.venue, phase
    """
    df = con.execute(query).df()
    con.close()
    
    df['avg_sr'] = np.where(df['total_balls'] > 0, (df['total_runs'] / df['total_balls']) * 100.0, 120.0)
    return df

def calculate_true_strike_rate(batter_name, venue, phase, player_runs, player_balls, venue_phase_df):
    """
    Computes True Strike Rate (SR_true):
    SR_true = Player SR at Venue in Phase / Average SR of All Batters at Venue in Same Phase
    """
    if player_balls == 0:
        return 1.0

    player_sr = (player_runs / player_balls) * 100.0
    
    # Lookup venue & phase average
    matched = venue_phase_df[(venue_phase_df['venue'] == venue) & (venue_phase_df['phase'] == phase)]
    if not matched.empty and matched['avg_sr'].values[0] > 0:
        venue_phase_sr = matched['avg_sr'].values[0]
    else:
        venue_phase_sr = 125.0  # Fallback baseline T20 SR

    sr_true = player_sr / venue_phase_sr
    return round(float(sr_true), 3)

def get_h2h_matrix(team_a_batters, team_b_bowlers, db_path=DEFAULT_DB_PATH):
    """
    Generates an 11x11 Head-to-Head matchup matrix comparing Team A batters vs Team B bowlers.
    Returns a pandas DataFrame indexed by Batter with Bowler columns containing metrics dictionary.
    """
    con = get_connection(db_path)
    
    batters_str = ", ".join([f"'{b}'" for b in team_a_batters])
    bowlers_str = ", ".join([f"'{b}'" for b in team_b_bowlers])
    
    query = f"""
    SELECT 
        batter,
        bowler,
        SUM(runs_batter) as runs,
        COUNT(CASE WHEN is_wide = False THEN 1 END) as balls,
        SUM(is_wicket) as dismissals
    FROM deliveries
    WHERE batter IN ({batters_str}) AND bowler IN ({bowlers_str})
    GROUP BY batter, bowler
    """
    df_raw = con.execute(query).df()
    con.close()

    # Pre-populate empty 11x11 matrix
    matrix_data = []
    for batter in team_a_batters:
        row = {"Batter": batter}
        for bowler in team_b_bowlers:
            match_row = df_raw[(df_raw['batter'] == batter) & (df_raw['bowler'] == bowler)]
            if not match_row.empty:
                runs = int(match_row['runs'].values[0])
                balls = int(match_row['balls'].values[0])
                dismissals = int(match_row['dismissals'].values[0])
                sr = round((runs / balls * 100.0), 1) if balls > 0 else 0.0
                out_str = f"{runs}r ({balls}b), {dismissals}W, SR: {sr}"
            else:
                runs, balls, dismissals, sr = 0, 0, 0, 0.0
                out_str = "0r (0b), 0W"
            
            row[bowler] = {
                "runs": runs,
                "balls": balls,
                "dismissals": dismissals,
                "sr": sr,
                "summary": out_str
            }
        matrix_data.append(row)
        
    return matrix_data

def get_tactical_alerts(matrix_data, team_a_batters, team_b_bowlers):
    """Identifies high-risk tactical mismatches for alert cards (e.g. SR < 100 or >=2 dismissals)."""
    alerts = []
    for row in matrix_data:
        batter = row["Batter"]
        for bowler in team_b_bowlers:
            data = row[bowler]
            if data["balls"] >= 5:
                if data["dismissals"] >= 2 or (data["sr"] < 105.0 and data["balls"] >= 8):
                    alerts.append({
                        "batter": batter,
                        "bowler": bowler,
                        "dismissals": data["dismissals"],
                        "sr": data["sr"],
                        "balls": data["balls"],
                        "runs": data["runs"],
                        "message": f"CRITICAL MISMATCH: {batter} vs {bowler} (SR {data['sr']}, {data['dismissals']} Dismissals in {data['balls']} balls)"
                    })
    return alerts

def get_phase_radar_stats(team_name, db_path=DEFAULT_DB_PATH):
    """Calculates phase capabilities (Powerplay, Middle, Death) for a team."""
    con = get_connection(db_path)
    query = f"""
    SELECT 
        CASE 
            WHEN d.over < 6 THEN 'Powerplay'
            WHEN d.over < 15 THEN 'Middle'
            ELSE 'Death'
        END as phase,
        SUM(d.total_runs) as total_runs,
        COUNT(CASE WHEN d.is_wide = False THEN 1 END) as total_balls,
        SUM(d.is_wicket) as wickets_lost
    FROM deliveries d
    JOIN matches m ON d.match_id = m.match_id
    WHERE m.team1 = '{team_name}' OR m.team2 = '{team_name}'
    GROUP BY phase
    """
    df = con.execute(query).df()
    con.close()
    
    stats = {"Powerplay": 8.2, "Middle": 7.8, "Death": 10.5} # Baselines
    for _, row in df.iterrows():
        phase = row['phase']
        balls = row['total_balls']
        runs = row['total_runs']
        if balls > 0:
            stats[phase] = round((runs / balls) * 6.0, 2)
            
    return stats
