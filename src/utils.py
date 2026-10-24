import plotly.graph_objects as go

COLOR_TEAM1 = "#38ef7d"
COLOR_TEAM2 = "#00c6ff"
COLOR_DANGER = "#ff4b4b"
STYLE_COLORS = ["#38ef7d", "#00c6ff", "#ffaa00", "#c77dff", "#ff4b4b"]

FEATURE_LABELS = {
    "runs_needed": "Runs needed",
    "balls_left": "Balls remaining",
    "wickets_left": "Wickets in hand",
    "req_rr": "Required run rate",
    "curr_rr": "Current run rate",
    "target_runs": "Target",
}

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    margin=dict(t=40, b=40, l=50, r=20),
)


def create_win_prob_donut(batting_team, chase_pct, bowling_team, defend_pct):
    fig = go.Figure(go.Pie(
        labels=[batting_team, bowling_team],
        values=[chase_pct, defend_pct],
        hole=0.65,
        sort=False,
        marker=dict(colors=[COLOR_TEAM1, COLOR_TEAM2], line=dict(color="#0e1117", width=3)),
        textinfo="label+percent",
    ))
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(t=20, b=40, l=20, r=20)},
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        annotations=[dict(text="<b>WIN PROBABILITY</b>", x=0.5, y=0.5, showarrow=False)],
    )
    return fig


def create_shap_bar_chart(shap_impacts):
    fig = go.Figure(go.Bar(
        x=[d["Impact"] for d in shap_impacts],
        y=[FEATURE_LABELS.get(d["Feature"], d["Feature"]) for d in shap_impacts],
        orientation="h",
        marker_color=[COLOR_TEAM1 if d["Impact"] > 0 else COLOR_DANGER for d in shap_impacts],
        text=[f"{d['Impact']:+.2f}" for d in shap_impacts],
        textposition="auto",
    ))
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(t=40, b=50, l=130, r=20)},
        title=dict(text="Why: SHAP contributions vs an average chase", font=dict(size=15)),
        xaxis=dict(title="Effect on chase win log-odds", gridcolor="#333", zerolinecolor="#666"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_phase_bar_chart(phases, team_a, stats_a, team_b, stats_b):
    fig = go.Figure([
        go.Bar(name=team_a, x=phases, y=[stats_a[p]["run_rate"] for p in phases], marker_color=COLOR_TEAM1),
        go.Bar(name=team_b, x=phases, y=[stats_b[p]["run_rate"] for p in phases], marker_color=COLOR_TEAM2),
    ])
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Batting run rate by phase (last 3 years, all opponents)", font=dict(size=15)),
        yaxis=dict(title="Runs per over", gridcolor="#333"),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def create_h2h_heatmap(matrix_data, batters, bowlers, team_a, team_b):
    z = [[row[b]["sr"] if row[b]["balls"] else None for b in bowlers] for row in matrix_data]
    text = [[f"{row[b]['runs']}r/{row[b]['balls']}b<br>{row[b]['dismissals']}W" if row[b]["balls"] else ""
             for b in bowlers] for row in matrix_data]
    fig = go.Figure(go.Heatmap(
        z=z, x=bowlers, y=batters, text=text, texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}<br>SR %{z}<extra></extra>",
        colorscale="Viridis", colorbar=dict(title=dict(text="Strike rate")), hoverongaps=False,
    ))
    fig.update_layout(
        **{**BASE_LAYOUT, "margin": dict(t=20, b=130, l=140, r=40)},
        xaxis=dict(title=f"Bowlers ({team_b})", tickangle=-45),
        yaxis=dict(title=f"Batters ({team_a})", autorange="reversed"),
        height=650,
    )
    return fig


def create_monte_carlo_hist(sim_data, batting_team, target=None):
    fig = go.Figure(go.Histogram(x=sim_data["score_distribution"], nbinsx=30, marker_color=COLOR_TEAM2))
    fig.add_vline(x=sim_data["mean_score"], line_dash="dash", line_color=COLOR_TEAM1)
    fig.add_vline(x=sim_data["ci_10th"], line_dash="dash", line_color="#888")
    fig.add_vline(x=sim_data["ci_90th"], line_dash="dash", line_color="#888")
    if target is not None:
        fig.add_vline(x=target, line_dash="dash", line_color=COLOR_DANGER, annotation_text="Target")
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=f"{sim_data['n_simulations']:,} simulated finishes for {batting_team}", font=dict(size=15)),
        xaxis=dict(title="Final score", gridcolor="#333"),
        yaxis=dict(title="Simulations (sample of 1,000)", gridcolor="#333"),
    )
    return fig


def create_cluster_scatter(df, focus_teams):
    fig = go.Figure()
    for i, style in enumerate(sorted(df["style"].unique())):
        pts = df[df["style"] == style]
        focus = pts["team"].isin(focus_teams)
        fig.add_trace(go.Scatter(
            x=pts["strike_rate"], y=pts["average"], mode="markers", name=style,
            text=pts["player"] + " (" + pts["team"] + ")",
            hovertemplate="%{text}<br>SR %{x:.1f} • Avg %{y:.1f}<extra>" + style + "</extra>",
            marker=dict(
                color=STYLE_COLORS[i % len(STYLE_COLORS)],
                size=focus.map({True: 12, False: 6}).tolist(),
                opacity=focus.map({True: 1.0, False: 0.35}).tolist(),
                line=dict(color="white", width=focus.map({True: 1.5, False: 0}).tolist()),
            ),
        ))
    fig.update_layout(
        **BASE_LAYOUT,
        xaxis=dict(title="Strike rate", gridcolor="#333"),
        yaxis=dict(title="Batting average (capped at 80)", gridcolor="#333"),
        legend=dict(orientation="h", y=-0.12),
        height=600,
    )
    return fig


def create_calibration_chart(calibration, model_name, fmt):
    fig = go.Figure([
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color="#666", dash="dash")),
        go.Scatter(
            x=[c["predicted"] for c in calibration], y=[c["observed"] for c in calibration],
            mode="lines+markers", name=model_name, marker=dict(color=COLOR_TEAM1, size=8), line=dict(color=COLOR_TEAM1),
        ),
    ])
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=f"Win-probability calibration on held-out {fmt} matches", font=dict(size=15)),
        xaxis=dict(title="Predicted chase win probability", range=[0, 1], gridcolor="#333"),
        yaxis=dict(title="Observed win rate", range=[0, 1], gridcolor="#333"),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig
