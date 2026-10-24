import pandas as pd
import streamlit as st

from src.config import DEFAULT_TEAM_A, DEFAULT_TEAM_B, FORMATS, overs_to_balls, total_balls
from src.etl import database_ready, get_connection
from src.features import (
    batting_stats, bowling_stats, calculate_mpi, get_mpi_interpretation, head_to_head, latest_series,
    list_teams, phase_stats, xi_matchup,
)
from src.models import (
    cluster_batters, model_report, predict_win_probability, project_first_innings_score, scoreboard_state,
)
from src.simulator import TACTICS, simulate_what_if_scenario
from src.utils import (
    create_calibration_chart, create_cluster_scatter, create_h2h_heatmap, create_monte_carlo_hist,
    create_phase_bar_chart, create_shap_bar_chart, create_win_prob_donut,
)

st.set_page_config(page_title="Cricket Analytics Hub", page_icon="🏏", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .metric-badge-team1 { background: linear-gradient(135deg, #11998e, #38ef7d); border-radius: 12px; padding: 18px; color: white; text-align: center; }
    .metric-badge-team2 { background: linear-gradient(135deg, #005c97, #00c6ff); border-radius: 12px; padding: 18px; color: white; text-align: center; }
    .summary-card { background: linear-gradient(135deg, rgba(30, 34, 45, 0.9), rgba(20, 24, 33, 0.95)); border: 1px solid rgba(255, 255, 255, 0.15);
                    border-left: 5px solid #38ef7d; border-radius: 12px; padding: 18px; margin-bottom: 20px; }
    .alert-card { background: rgba(255, 75, 75, 0.15); border-left: 4px solid #ff4b4b; border-radius: 10px; padding: 14px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

if not database_ready():
    st.error("No data yet. Run `python -m src.pipeline` to download Cricsheet data and train the models, then reload.")
    st.stop()


@st.cache_data
def coverage():
    con = get_connection(read_only=True)
    try:
        return con.execute("SELECT format, COUNT(*), MIN(start_date), MAX(start_date) FROM matches GROUP BY format").df()
    finally:
        con.close()


def result_text(m):
    if m["result"] == "no result":
        return "No result"
    if m["result"] == "tie":
        return f"Tie ({m['winner']} won super over)" if m["winner"] else "Tie"
    by = f"{m['win_by_runs']} runs" if m["win_by_runs"] else f"{m['win_by_wickets']} wkts"
    return f"{m['winner']} won by {by}" + (f" ({m['method']})" if m["method"] else "")


cov = coverage()
st.title("🏏 Cricket Analytics Hub")
st.caption("Real ball-by-ball international data from Cricsheet • " + " • ".join(
    f"{r['count_star()']:,} {r['format']}s ({str(r['min(start_date)'])[:4]}–{str(r['max(start_date)'])[:10]})"
    for _, r in cov.iterrows()
))

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Matchup")
fmt = st.sidebar.selectbox("Format", list(FORMATS), format_func=lambda f: {"T20": "T20 International", "ODI": "One Day International"}[f])
teams = list_teams(fmt)
team_a = st.sidebar.selectbox("Team A", teams, index=teams.index(DEFAULT_TEAM_A) if DEFAULT_TEAM_A in teams else 0)
teams_b = [t for t in teams if t != team_a]
team_b = st.sidebar.selectbox("Team B", teams_b, index=teams_b.index(DEFAULT_TEAM_B) if DEFAULT_TEAM_B in teams_b else 0)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Live Scoreboard")
st.sidebar.caption("Enter the score of a match in progress for the predictor and simulator.")
batting = st.sidebar.radio("Batting side", [team_a, team_b])
bowling = team_b if batting == team_a else team_a
chasing = st.sidebar.radio("Innings", ["Chasing (2nd innings)", "Setting a total (1st innings)"]) == "Chasing (2nd innings)"
max_score = 500 if fmt == "ODI" else 350
target = st.sidebar.number_input("Target", 1, max_score, 180 if fmt == "T20" else 290) if chasing else None
current = st.sidebar.number_input("Current score", 0, max_score, 120 if fmt == "T20" else 150)
overs = st.sidebar.number_input("Overs bowled (e.g. 15.2)", 0.1, FORMATS[fmt]["overs"] - 0.5, 15.0 if fmt == "T20" else 30.0, 0.1, format="%.1f")
wickets_lost = st.sidebar.number_input("Wickets down", 0, 9, 4 if fmt == "T20" else 3)

try:
    balls_bowled = overs_to_balls(overs)
except ValueError as e:
    st.sidebar.error(str(e))
    st.stop()

tab_series, tab_live, tab_matchups, tab_sim, tab_styles, tab_models = st.tabs([
    "🤝 Series & Head-to-Head", "📊 Live Predictor", "⚔️ Player Matchups",
    "🎲 Simulator", "🧬 Player Styles (ML)", "🧪 Model Report",
])

# ---------------- Series & head-to-head ----------------
with tab_series:
    event, series = latest_series(team_a, team_b)
    if event:
        tally = series.groupby("format")["winner"].value_counts().unstack(fill_value=0)
        tally_text = " • ".join(
            f"{f}s: {team_a} {tally.loc[f].get(team_a, 0)}–{tally.loc[f].get(team_b, 0)} {team_b}" for f in tally.index
        )
        results = " | ".join(
            f"{r['format']} {pd.to_datetime(r['start_date']):%d %b %Y}: {result_text(r)}" for _, r in series.iterrows()
        )
        st.markdown(f"""
        <div class="summary-card">
            <div style="font-size: 18px; font-weight: bold; color: #38ef7d;">📅 Latest bilateral series: {event}</div>
            <div style="margin-top: 6px;">{tally_text}</div>
            <div style="font-size: 13px; margin-top: 6px; color: #ddd;">{results}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Matches from a series in progress appear once Cricsheet publishes them — run `python -m src.pipeline --refresh`.")
    else:
        st.info("No bilateral series between these teams in the data.")

    summary, h2h = head_to_head(team_a, team_b, fmt)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{fmt} matches", summary["matches"], f"{summary['first_match']} → {summary['last_match']}", delta_color="off")
    c2.metric(f"{team_a} wins", summary["team_a_wins"])
    c3.metric(f"{team_b} wins", summary["team_b_wins"])
    decided = summary["team_a_wins"] + summary["team_b_wins"]
    c4.metric(f"{team_a} win rate", f"{round(summary['team_a_wins'] / decided * 100) if decided else 0}%",
              f"{summary['no_result']} no result", delta_color="off")

    left, right = st.columns(2)
    with left:
        st.subheader("Recent results")
        recent = h2h.head(10).copy()
        recent["Date"] = pd.to_datetime(recent["start_date"]).dt.strftime("%d %b %Y")
        recent["Result"] = recent["summary"]
        st.dataframe(recent[["Date", "event_name", "Result", "player_of_match"]].rename(
            columns={"event_name": "Event", "player_of_match": "Player of match"}), hide_index=True, width="stretch")
    with right:
        since = (pd.to_datetime(cov.loc[cov["format"] == fmt, "max(start_date)"].iloc[0]) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
        st.plotly_chart(create_phase_bar_chart(
            [p[2] for p in FORMATS[fmt]["phases"]],
            team_a, phase_stats(fmt, team_a, since=since), team_b, phase_stats(fmt, team_b, since=since),
        ), width="stretch")

    bat_cols = {"player": "Batter", "innings": "Inns", "runs": "Runs", "average": "Avg", "strike_rate": "SR"}
    bowl_cols = {"player": "Bowler", "innings": "Inns", "wickets": "Wkts", "economy": "Econ", "average": "Avg"}
    for col, (team, opp) in zip(st.columns(2), [(team_a, team_b), (team_b, team_a)]):
        with col:
            st.markdown(f"**🏏 {team} top batters vs {opp}**")
            st.dataframe(batting_stats(fmt, team, opp, min_balls=30).head(6)[list(bat_cols)].rename(columns=bat_cols),
                         hide_index=True, width="stretch")
            st.markdown(f"**🎯 {team} top bowlers vs {opp}**")
            st.dataframe(bowling_stats(fmt, team, opp, min_balls=30).head(6)[list(bowl_cols)].rename(columns=bowl_cols),
                         hide_index=True, width="stretch")

# ---------------- Live predictor ----------------
with tab_live:
    if chasing:
        state = scoreboard_state(fmt, target, current, balls_bowled, wickets_lost)
        if state["runs_needed"] == 0:
            st.success(f"{batting} have reached the target.")
        else:
            pred = predict_win_probability(fmt, state)
            mpi = calculate_mpi(state["req_rr"], state["wickets_left"], state["balls_left"], fmt)
            pressure = get_mpi_interpretation(mpi)
            favored = batting if pred["chasing_win_pct"] >= 50 else bowling
            st.markdown(f"""
            <div class="summary-card">
                <div style="font-size: 18px; font-weight: bold; color: #38ef7d;">📢 {favored} favoured ({max(pred['chasing_win_pct'], pred['defending_win_pct'])}%)</div>
                <div style="font-size: 14px; margin-top: 6px; color: #ddd;">
                    {batting} need <strong>{state['runs_needed']}</strong> off <strong>{state['balls_left']}</strong> balls
                    (required <strong>{state['req_rr']} RPO</strong>) with <strong>{state['wickets_left']}</strong> wickets in hand.
                    Pressure: <strong>{pressure['emoji']} {pressure['level']}</strong> — {pressure['desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("The model reads the scoreboard only (runs, balls, wickets, target) — it does not rate the two teams.")

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-badge-team1"><div style="font-size: 13px;">{batting.upper()} (CHASING)</div>'
                        f'<div style="font-size: 34px; font-weight: 800;">{pred["chasing_win_pct"]}%</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-badge-team2"><div style="font-size: 13px;">{bowling.upper()} (DEFENDING)</div>'
                        f'<div style="font-size: 34px; font-weight: 800;">{pred["defending_win_pct"]}%</div></div>', unsafe_allow_html=True)
            c3.metric("Match pressure", f"{pressure['emoji']} {pressure['level']}", f"MPI {mpi}", delta_color="off")
            c4.metric("Required rate", f"{state['req_rr']} RPO", f"Current {state['curr_rr']} RPO", delta_color="off")

            left, right = st.columns(2)
            left.plotly_chart(create_win_prob_donut(batting, pred["chasing_win_pct"], bowling, pred["defending_win_pct"]), width="stretch")
            right.plotly_chart(create_shap_bar_chart(pred["shap_impacts"]), width="stretch")
    else:
        projected = project_first_innings_score(fmt, current, balls_bowled, wickets_lost)
        sim = simulate_what_if_scenario(fmt, None, current, total_balls(fmt) - balls_bowled, 10 - wickets_lost,
                                        "Baseline Tactic", batting_team=batting)["baseline"]
        st.markdown(f"""
        <div class="summary-card">
            <div style="font-size: 18px; font-weight: bold; color: #38ef7d;">📢 {batting} are heading for about {projected}</div>
            <div style="font-size: 14px; margin-top: 6px; color: #ddd;">{current}/{wickets_lost} after {overs} overs. The ML model projects {projected};
            a Monte Carlo simulation using {batting}'s recent scoring puts the likely range at {sim['ci_10th']:.0f}–{sim['ci_90th']:.0f}.</div>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("ML projected total", projected)
        c2.metric("Simulated likely range", f"{sim['ci_10th']:.0f} – {sim['ci_90th']:.0f}", "10th–90th percentile", delta_color="off")
        c3.metric("Current run rate", f"{current / max(balls_bowled, 1) * 6:.2f} RPO")

# ---------------- Player matchups ----------------
with tab_matchups:
    st.caption(f"{batting} batters vs {bowling} bowlers — XIs from their latest {fmt} meeting, records across all {fmt} internationals. "
               "Change \"Batting side\" to flip.")
    mu = xi_matchup(batting, bowling, fmt)
    st.plotly_chart(create_h2h_heatmap(mu["matrix"], mu["batters"], mu["bowlers"], batting, bowling), width="stretch")
    st.subheader("🚨 Matchups where the bowler has dominated")
    if mu["alerts"]:
        for col, a in zip(st.columns(3) * 2, mu["alerts"][:6]):
            col.markdown(f"""<div class="alert-card"><strong>{a['bowler']}</strong> vs <strong>{a['batter']}</strong><br>
                {a['runs']} runs off {a['balls']} balls • {a['dismissals']} dismissals • SR {a['sr']}</div>""", unsafe_allow_html=True)
    else:
        st.info("No bowler has a dominant record (12+ balls) against these batters.")

# ---------------- Simulator ----------------
with tab_sim:
    left, right = st.columns([1, 2])
    with left:
        tactic = st.selectbox("What-if strategy", list(TACTICS))
        n_sims = st.select_slider("Simulations", [1000, 5000, 10000], value=5000)
        use_profile = st.checkbox("Use batting side's last-3-years scoring profile", value=True)
        res = simulate_what_if_scenario(fmt, target, current, total_balls(fmt) - balls_bowled, 10 - wickets_lost,
                                        tactic, n_sims, batting if use_profile else None)
        base, tac = res["baseline"], res["tactical"]
        is_baseline = tactic == "Baseline Tactic"
        if chasing:
            st.metric(f"{batting} chase win chance", f"{tac['simulated_win_pct']}%",
                      None if is_baseline else f"{tac['simulated_win_pct'] - base['simulated_win_pct']:+.1f} pts vs baseline")
        st.metric("Expected final score", tac["mean_score"],
                  None if is_baseline else f"{tac['mean_score'] - base['mean_score']:+.1f} vs baseline")
        st.metric("Likely range (10th–90th pct)", f"{tac['ci_10th']:.0f} – {tac['ci_90th']:.0f}")
        st.info(f"💡 {res['impact_summary']}")
    with right:
        st.plotly_chart(create_monte_carlo_hist(tac, batting, target), width="stretch")

# ---------------- Player styles ----------------
with tab_styles:
    clusters, info = cluster_batters(fmt)
    st.caption(f"KMeans (k={info['k']}) on {info['players']} {fmt} batters with {info['min_balls']}+ balls faced, using strike rate, "
               f"average, boundary %, dot-ball % and the share of balls faced in the powerplay and death overs. "
               f"Silhouette score {info['silhouette']}. {team_a} and {team_b} players are outlined.")
    st.plotly_chart(create_cluster_scatter(clusters, {team_a, team_b}), width="stretch")
    focus = clusters[clusters["team"].isin([team_a, team_b])].sort_values("runs", ascending=False).head(40)
    st.dataframe(focus[["player", "team", "style", "runs", "strike_rate", "average", "boundary_pct", "powerplay_share", "death_share"]]
                 .round(2), hide_index=True, width="stretch")

# ---------------- Model report ----------------
with tab_models:
    report = model_report().get(fmt)
    if not report:
        st.warning("No trained models; run `python -m src.pipeline`.")
    else:
        win, score, clus = report["win_probability"], report["score_projection"], report["batter_clusters"]
        st.subheader(f"1. {win['task']}")
        st.caption(f"Trained on {win['train_matches']:,} matches ({win['train_rows']:,} ball states); tested on the "
                   f"{win['test_matches']} most recent matches from {win['test_from']} onward. Selected: {win['selected']} (XGBoost must beat the baseline by >0.005 log loss).")
        st.dataframe(pd.DataFrame(win["models"]).T, width="stretch")
        st.plotly_chart(create_calibration_chart(win["calibration"], win["selected"], fmt), width="stretch")

        st.subheader(f"2. {score['task']}")
        st.caption(f"Trained on {score['train_matches']:,} first innings; tested on {score['test_matches']} recent ones. "
                   f"Errors in runs. Selected: {score['selected']} (lowest test MAE).")
        st.dataframe(pd.DataFrame(score["models"]).T, width="stretch")

        st.subheader(f"3. {clus['task']}")
        st.caption(f"{clus['players']} batters, k={clus['k']}, silhouette {clus['silhouette']}.")
        st.dataframe(pd.DataFrame(clus["clusters"]), hide_index=True, width="stretch")

st.markdown("---")
st.caption("Data: [Cricsheet](https://cricsheet.org) • API: `uvicorn src.api:app` → /docs")
