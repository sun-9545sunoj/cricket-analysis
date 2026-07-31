import os
import numpy as np
import pandas as pd
import streamlit as st

from src.etl import setup_db_if_needed, DEFAULT_DB_PATH, IPL_TEAMS, IPL_VENUES, SAMPLE_BATTERS
from src.features import (
    calculate_mpi, 
    get_mpi_interpretation, 
    calculate_true_strike_rate, 
    get_sr_true_interpretation, 
    compute_venue_phase_averages, 
    get_h2h_matrix, 
    get_tactical_alerts, 
    get_phase_radar_stats
)
from src.models import train_win_model, predict_win_probability, get_auction_recommendations
from src.simulator import run_monte_carlo_simulation, simulate_what_if_scenario
from src.utils import (
    create_win_prob_donut, 
    create_shap_bar_chart, 
    create_h2h_heatmap, 
    create_phase_radar_chart,
    create_win_prob_curve_chart,
    create_monte_carlo_hist
)

# Page Configuration
st.set_page_config(
    page_title="IPL Edge Analytics | Match Center",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Modern CSS & Friendly Tooltip Styles
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .metric-badge-team1 {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 12px;
        padding: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(56, 239, 125, 0.2);
    }
    .metric-badge-team2 {
        background: linear-gradient(135deg, #005c97, #00c6ff);
        border-radius: 12px;
        padding: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 198, 255, 0.2);
    }
    .summary-card {
        background: linear-gradient(135deg, rgba(30, 34, 45, 0.9), rgba(20, 24, 33, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-left: 5px solid #38ef7d;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
    }
    .alert-card {
        background: rgba(255, 75, 75, 0.15);
        border-left: 4px solid #ff4b4b;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .info-card {
        background: rgba(0, 198, 255, 0.12);
        border-left: 4px solid #00c6ff;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def setup_environment():
    """Initializes database and trains initial ML win model if needed."""
    from src.etl import init_db, generate_sample_data, ingest_json_files
    if not os.path.exists(DEFAULT_DB_PATH):
        generate_sample_data(100)
        ingest_json_files()
        train_win_model()
    return True

setup_environment()

# Header & Title
st.title("🏏 IPL Match Center & Smart Predictive Dashboard")
st.caption("Human-Friendly Cricket Analytics • Live Win Probabilities • Tactical Mismatches • Match Simulation")

# Sidebar - Global Match Parameters
st.sidebar.header("⚙️ Match Setup & Live Situation")
st.sidebar.caption("Adjust match settings below to see instant predictive updates.")

team1 = st.sidebar.selectbox("Team A (Batting / Home)", IPL_TEAMS, index=0)
available_teams_b = [t for t in IPL_TEAMS if t != team1]
team2 = st.sidebar.selectbox("Team B (Bowling / Visitor)", available_teams_b, index=0)

venue = st.sidebar.selectbox("Venue / Stadium", IPL_VENUES, index=0)
toss_winner = st.sidebar.radio("Toss Winner", [team1, team2], index=0)
toss_decision = st.sidebar.radio("Toss Decision", ["Bat First", "Field First"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Live 2nd Innings Scorecard")
target_score = st.sidebar.number_input("Target Score (To Win)", min_value=50, max_value=300, value=175, step=1)
current_score = st.sidebar.number_input("Current Score", min_value=0, max_value=300, value=115, step=1)
overs_bowled = st.sidebar.slider("Overs Bowled (Completed)", min_value=0.1, max_value=19.5, value=13.2, step=0.1)
wickets_lost = st.sidebar.slider("Wickets Down", min_value=0, max_value=9, value=3, step=1)

# Derived live match variables
balls_bowled = int(overs_bowled) * 6 + int(round((overs_bowled % 1) * 10))
balls_remaining = max(120 - balls_bowled, 1)
runs_needed = max(target_score - current_score, 0)
wickets_remaining = max(10 - wickets_lost, 1)

curr_rr = round((current_score / max(balls_bowled, 1)) * 6.0, 2)
req_rr = round((runs_needed / balls_remaining) * 6.0, 2)
mpi_value = calculate_mpi(req_rr, wickets_remaining, balls_remaining)
mpi_info = get_mpi_interpretation(mpi_value)

match_state = {
    "runs_needed": runs_needed,
    "balls_left": balls_remaining,
    "wickets_left": wickets_remaining,
    "req_rr": req_rr,
    "curr_rr": curr_rr,
    "target_runs": target_score,
    "venue_avg_runs": 172.0,
    "toss_win_bat_first": 1 if (toss_winner == team2 and toss_decision == "Field First") else 0
}

win_results = predict_win_probability(match_state)

# Plain English Executive Match Summary Banner
favored_team = team1 if win_results['team1_win_pct'] >= 50.0 else team2
favored_pct = max(win_results['team1_win_pct'], win_results['team2_win_pct'])

st.markdown(f"""
<div class="summary-card">
    <div style="font-size: 18px; font-weight: bold; color: #38ef7d;">
        📢 Plain English Match Verdict: {favored_team} is Favored to Win ({favored_pct}%)
    </div>
    <div style="font-size: 14px; margin-top: 6px; color: #dddddd;">
        <strong>Situation Summary:</strong> {team1} needs <strong>{runs_needed} runs</strong> off <strong>{balls_remaining} balls</strong> 
        (Required Rate: <strong>{req_rr} RPO</strong>) with <strong>{wickets_remaining} wickets</strong> in hand. 
        Match pressure status is <strong>{mpi_info['emoji']} {mpi_info['level']}</strong> ({mpi_info['desc']}).
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Win Probability & Match Drivers", 
    "⚔️ 22-Player Tactical Mismatches", 
    "🎲 Match Simulator & What-If Sandbox", 
    "💰 Auction Recommendations & API"
])

# ----------------------------------------------------
# TAB 1: Human-Friendly Win Probability & Key Drivers
# ----------------------------------------------------
with tab1:
    with st.expander("ℹ️ How to Read This Tab (Simple Guide)", expanded=False):
        st.write("""
        - **Win Probability**: Shows each team's calculated chance of winning the match right now based on machine learning trained on thousands of historical IPL delivery states.
        - **Pressure Gauge**: Evaluates match tension into clear categories (*Comfortable, Moderate, High, Panic Mode*).
        - **Top Key Drivers**: Highlights the top reasons why the prediction favors one team (e.g., wickets remaining or required run rate).
        """)

    st.subheader("Win Chances & Match Pressure Gauge")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-badge-team1">
            <div style="font-size: 13px; text-transform: uppercase;">{team1} Win Chance</div>
            <div style="font-size: 34px; font-weight: 800;">{win_results['team1_win_pct']}%</div>
            <div style="font-size: 12px; opacity: 0.9;">Chasing Target</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-badge-team2">
            <div style="font-size: 13px; text-transform: uppercase;">{team2} Win Chance</div>
            <div style="font-size: 34px; font-weight: 800;">{win_results['team2_win_pct']}%</div>
            <div style="font-size: 12px; opacity: 0.9;">Defending Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.metric(
            label="Match Pressure Gauge", 
            value=f"{mpi_info['emoji']} {mpi_info['level']}", 
            delta=f"MPI Score: {mpi_value}"
        )
    with col_m4:
        st.metric(
            label="Scoring Speed Needed", 
            value=f"{req_rr} RPO", 
            delta=f"Current Rate: {curr_rr} RPO"
        )

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        donut_fig = create_win_prob_donut(team1, win_results['team1_win_pct'], team2, win_results['team2_win_pct'])
        st.plotly_chart(donut_fig, use_container_width=True)

    with c2:
        shap_fig = create_shap_bar_chart(win_results['shap_impacts'])
        st.plotly_chart(shap_fig, use_container_width=True)

    # 40-Over Dynamic Win Timeline
    st.markdown("---")
    st.subheader("📈 40-Over Ball-by-Ball Match Momentum Timeline")
    st.caption("Tracks how match control shifts ball-by-ball across overs.")
    
    balls_arr = np.arange(1, balls_bowled + 1)
    start_p = 50.0
    end_p = win_results['team1_win_pct']
    curve_p = np.linspace(start_p, end_p, len(balls_arr)) + np.random.normal(0, 2.5, len(balls_arr))
    curve_p = np.clip(curve_p, 5.0, 95.0)

    curve_fig = create_win_prob_curve_chart(balls_arr, curve_p, team1, team2)
    st.plotly_chart(curve_fig, use_container_width=True)

    # Venue-Adjusted True SR Evaluator
    st.markdown("---")
    st.subheader("🎯 Ground Benchmark Evaluator (True Strike Rate)")
    st.caption("Compares individual batter performance against historic venue standards for the same phase.")

    venue_df = compute_venue_phase_averages()
    
    col_ts1, col_ts2, col_ts3 = st.columns(3)
    with col_ts1:
        p_runs = st.number_input("Batter Runs Scored", min_value=1, max_value=150, value=45)
    with col_ts2:
        p_balls = st.number_input("Balls Faced", min_value=1, max_value=100, value=25)
    with col_ts3:
        p_phase = st.selectbox("Innings Phase", ["Powerplay", "Middle", "Death"], index=2)
        
    sr_true = calculate_true_strike_rate("Sample Batter", venue, p_phase, p_runs, p_balls, venue_df)
    sr_interp = get_sr_true_interpretation(sr_true)

    st.markdown(f"""
    <div class="info-card">
        <strong>{sr_interp['badge']} (True SR: {sr_true}x Benchmark)</strong><br>
        {sr_interp['desc']} (Actual Strike Rate: {round((p_runs/p_balls)*100, 1)}).
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 2: 22-Player Head-to-Head & Matchup Matrix
# ----------------------------------------------------
with tab2:
    with st.expander("ℹ️ How to Read The Tactical Matrix", expanded=False):
        st.write("""
        - **11x11 Grid**: Each cell shows how a specific batter fares against a specific bowler (Runs scored, Balls faced, Dismissals, and Strike Rate).
        - **Color Intensity**: Darker colors highlight higher scoring rates.
        - **Tactical Alerts**: Automatically flags high-vulnerability matchups where a bowler consistently dominates a batter.
        """)

    st.subheader("22-Player Tactical Matchup Heatmap & Mismatches")

    team_a_roster = SAMPLE_BATTERS.get(team1, SAMPLE_BATTERS["Chennai Super Kings"])
    team_b_roster = SAMPLE_BATTERS.get(team2, SAMPLE_BATTERS["Mumbai Indians"])

    h2h_data = get_h2h_matrix(team_a_roster, team_b_roster)
    heatmap_fig = create_h2h_heatmap(h2h_data, team_a_roster, team_b_roster)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.markdown("---")
    c_alert, c_radar = st.columns([1, 1])
    with c_alert:
        st.subheader("🚨 Key Tactical Mismatch Cards")
        alerts = get_tactical_alerts(h2h_data, team_a_roster, team_b_roster)
        if alerts:
            for alt in alerts[:4]:
                st.markdown(f"""
                <div class="alert-card">
                    <strong>⚠️ High Mismatch Risk</strong>: <strong>{alt['batter']}</strong> vs <strong>{alt['bowler']}</strong><br>
                    <span style="font-size: 13px; color: #ddd;">
                    • Record: {alt['runs']} runs off {alt['balls']} balls ({alt['dismissals']} Out)<br>
                    • Strike Rate: {alt['sr']} (Below Average)<br>
                    💡 <strong>Tactical Advice:</strong> Introduce {alt['bowler']} when {alt['batter']} comes out to bat.
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card">
                <strong>✅ Balanced Matchup Profile</strong><br>
                No extreme individual batter vs bowler vulnerabilities detected in past delivery records.
            </div>
            """, unsafe_allow_html=True)

    with c_radar:
        stats_a = get_phase_radar_stats(team1)
        stats_b = get_phase_radar_stats(team2)
        radar_fig = create_phase_radar_chart(team1, stats_a, team2, stats_b)
        st.plotly_chart(radar_fig, use_container_width=True)

# ----------------------------------------------------
# TAB 3: Monte Carlo Match Simulator & What-If Sandbox
# ----------------------------------------------------
with tab3:
    with st.expander("ℹ️ What is a Monte Carlo Simulation?", expanded=False):
        st.write("""
        Instead of guessing the outcome, a **Monte Carlo Simulation** uses computer algorithms to play out the rest of the match **10,000 times ball-by-ball**, taking into account realistic boundary hit rates, dot balls, and wicket risks.
        - **Why it helps**: It shows the most likely score range (90% confidence interval) and allows captains/analysts to test 'What-If' strategies before making live decisions!
        """)

    st.subheader("🎲 10,000-Match Computer Simulation & 'What-If' Scenario Tester")

    col_sim1, col_sim2 = st.columns([1, 2])
    with col_sim1:
        n_sims = st.slider("Simulated Matches Count", min_value=1000, max_value=10000, value=5000, step=1000)
        tactic_choice = st.selectbox(
            "Test a 'What-If' Tactical Strategy", 
            ["Baseline Tactic", "Deploy Death Specialist Bowler", "Promote Power Hitter Batter"]
        )
        
        sim_results = simulate_what_if_scenario(
            target_score, current_score, balls_remaining, wickets_remaining, tactic_choice
        )
        
        st.markdown("### 📊 Simulation Findings in Simple Terms")
        baseline = sim_results["baseline"]
        tactic = sim_results["tactical"]
        
        st.metric(
            label="Simulated Win Probability", 
            value=f"{tactic['simulated_win_pct']}%", 
            delta=f"{tactic['simulated_win_pct'] - baseline['simulated_win_pct']:+.1f}% vs Standard Strategy"
        )
        st.metric(
            label="Expected Final Score", 
            value=f"{tactic['mean_score']} runs", 
            delta=f"{tactic['mean_score'] - baseline['mean_score']:+.1f} runs"
        )
        st.metric(
            label="90% Score Range (Most Likely)", 
            value=f"{tactic['ci_10th']} to {tactic['ci_90th']} runs"
        )
        
        st.info(f"💡 **Plain English Summary**: {sim_results['impact_summary']}")

    with col_sim2:
        hist_fig = create_monte_carlo_hist(tactic)
        st.plotly_chart(hist_fig, use_container_width=True)

# ----------------------------------------------------
# TAB 4: Auction Recommendations & REST API Docs
# ----------------------------------------------------
with tab4:
    st.subheader("💰 Smart Player Auction Recommendations & Developer API")

    col_auc, col_api = st.columns([1, 1])

    with col_auc:
        st.markdown("### 🏷️ Target Auction Player Recommendations")
        st.caption("Predicts player auction market values based on phase performance metrics.")
        gap_type = st.radio("Team Vulnerability Need", ["Death Bowling", "Spin Hitter", "All-Round Depth"], index=0)
        rec_df = get_auction_recommendations(team1, gap_type)
        st.dataframe(
            rec_df.style.format({
                "Base_Price_Cr": "₹{:.1f} Cr",
                "Predicted_Value_Cr": "₹{:.1f} Cr",
                "Impact_Score": "{:d}/100"
            }).background_gradient(subset=["Impact_Score"], cmap="Greens"),
            use_container_width=True
        )

    with col_api:
        st.markdown("### ⚡ Developer API Endpoints")
        st.markdown("""
        All predictive models and simulations are accessible via REST API (`src/api.py`).
        
        - **`GET /docs`**: Interactive Swagger API Playground
        - **`GET /health`**: Health Check
        - **`POST /api/v1/predict-win-prob`**: Win Probability API
        - **`POST /api/v1/simulate-match`**: Monte Carlo Simulator API
        """)
        st.success("🔗 Run `uvicorn src.api:app --reload` to access API docs at `http://localhost:8000/docs`.")

# Footer
st.markdown("---")
st.caption("IPL Edge Analytics Platform • Human-Centered Cricket Decision Support System")
