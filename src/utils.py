import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Color Palette
COLOR_TEAM1 = "#38ef7d" # Neon Green / Mint
COLOR_TEAM2 = "#00c6ff" # Electric Cyan
COLOR_BG = "#0e1117"
COLOR_CARD = "#1e222d"

def create_win_prob_donut(team1_name, team1_pct, team2_name, team2_pct):
    """Creates an interactive Plotly Donut chart for Win Probabilities."""
    fig = go.Figure(data=[go.Pie(
        labels=[team1_name, team2_name],
        values=[team1_pct, team2_pct],
        hole=0.65,
        marker=dict(colors=[COLOR_TEAM1, COLOR_TEAM2], line=dict(color='#0e1117', width=3)),
        textinfo='label+percent',
        textfont=dict(size=14, color='white'),
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color="white")),
        margin=dict(t=20, b=40, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text='WIN PROBABILITY', x=0.5, y=0.5, font_size=14, font_color="white", font_weight="bold", showarrow=False)]
    )
    return fig

def create_shap_bar_chart(shap_impacts):
    """Creates a horizontal bar chart visualizing SHAP feature impacts."""
    df = pd.DataFrame(shap_impacts)
    df['Color'] = df['Impact'].apply(lambda x: '#38ef7d' if x > 0 else '#ff4b4b')
    
    # Human readable feature names
    name_map = {
        "runs_needed": "Runs Needed",
        "balls_left": "Balls Remaining",
        "wickets_left": "Wickets Remaining",
        "req_rr": "Required Run Rate",
        "curr_rr": "Current Run Rate",
        "venue_avg_runs": "Venue Historical Avg",
        "target_runs": "Target Total",
        "toss_win_bat_first": "Toss & Pitch Factor"
    }
    df['Feature_Label'] = df['Feature'].map(lambda x: name_map.get(x, x))

    fig = go.Figure(go.Bar(
        x=df['Impact'],
        y=df['Feature_Label'],
        orientation='h',
        marker_color=df['Color'],
        text=df['Impact'].apply(lambda x: f"{x:+.3f}"),
        textposition='auto'
    ))

    fig.update_layout(
        title=dict(text="SHAP Decision Factor Impact (Top Drivers)", font=dict(color="white", size=16)),
        xaxis=dict(title="SHAP Value (Contribution to Win Probability)", color="white", gridcolor="#333"),
        yaxis=dict(color="white", autorange="reversed"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=40, l=120, r=20)
    )
    return fig

def create_h2h_heatmap(matrix_data, team_a_batters, team_b_bowlers):
    """Generates an 11x11 Head-to-Head Strike Rate Heatmap matrix."""
    sr_matrix = []
    text_matrix = []

    for row in matrix_data:
        sr_row = []
        text_row = []
        for bowler in team_b_bowlers:
            cell = row[bowler]
            sr_row.append(cell["sr"])
            text_row.append(f"SR: {cell['sr']}<br>{cell['runs']}r/{cell['balls']}b<br>{cell['dismissals']}W")
        sr_matrix.append(sr_row)
        text_matrix.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=sr_matrix,
        x=team_b_bowlers,
        y=team_a_batters,
        text=text_matrix,
        texttemplate="%{text}",
        colorscale="Viridis",
        colorbar=dict(title=dict(text="Strike Rate", font=dict(color="white")), tickfont=dict(color="white")),
        hoverongaps=False
    ))

    fig.update_layout(
        title=dict(text="11x11 Head-to-Head Batter vs Bowler Matchup Matrix (Strike Rate & Dismissals)", font=dict(color="white", size=16)),
        xaxis=dict(title="Bowlers (Team B)", color="white", tickangle=-45),
        yaxis=dict(title="Batters (Team A)", color="white", autorange="reversed"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=100, l=120, r=40),
        height=650
    )
    return fig

def create_phase_radar_chart(team_a_name, stats_a, team_b_name, stats_b):
    """Generates a Radar chart comparing team run-rate capabilities across phases."""
    categories = ['Powerplay', 'Middle', 'Death']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[stats_a['Powerplay'], stats_a['Middle'], stats_a['Death']],
        theta=categories,
        fill='toself',
        name=team_a_name,
        line_color=COLOR_TEAM1
    ))
    fig.add_trace(go.Scatterpolar(
        r=[stats_b['Powerplay'], stats_b['Middle'], stats_b['Death']],
        theta=categories,
        fill='toself',
        name=team_b_name,
        line_color=COLOR_TEAM2
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 14], color="white"),
            angularaxis=dict(color="white")
        ),
        showlegend=True,
        legend=dict(font=dict(color="white")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text="Phase Capability Radar (Run Rate per Over)", font=dict(color="white", size=16))
    )
    return fig

def create_win_prob_curve_chart(balls_series, team1_win_series, team1_name, team2_name):
    """Plots dynamic ball-by-ball win probability curve over the course of a match."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=balls_series,
        y=team1_win_series,
        mode='lines',
        name=f"{team1_name} Win %",
        line=dict(color=COLOR_TEAM1, width=3),
        fill='tozeroy',
        fillcolor='rgba(56, 239, 125, 0.15)'
    ))

    # 50% equilibrium line
    fig.add_shape(
        type="line",
        x0=0, x1=max(balls_series) if len(balls_series)>0 else 120,
        y0=50, y1=50,
        line=dict(color="rgba(255, 255, 255, 0.4)", width=1.5, dash="dash")
    )

    fig.update_layout(
        title=dict(text=f"Ball-by-Ball Live Win Probability Curve ({team1_name} vs {team2_name})", font=dict(color="white", size=16)),
        xaxis=dict(title="Balls Bowled in Match", color="white", gridcolor="#333"),
        yaxis=dict(title=f"{team1_name} Win Probability (%)", range=[0, 100], color="white", gridcolor="#333"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=40, l=40, r=20)
    )
    return fig

def create_monte_carlo_hist(sim_data):
    """Plots Monte Carlo 10,000 simulation score distribution histogram."""
    scores = sim_data["score_distribution"]
    
    fig = px.histogram(
        x=scores,
        nbins=25,
        labels={"x": "Simulated Final Target Score"},
        color_discrete_sequence=[COLOR_TEAM2]
    )

    fig.add_vline(x=sim_data["mean_score"], line_dash="dash", line_color="#38ef7d", annotation_text=f"Mean: {sim_data['mean_score']}")
    fig.add_vline(x=sim_data["ci_10th"], line_dash="dot", line_color="#ff4b4b", annotation_text=f"10th%: {sim_data['ci_10th']}")
    fig.add_vline(x=sim_data["ci_90th"], line_dash="dot", line_color="#00c6ff", annotation_text=f"90th%: {sim_data['ci_90th']}")

    fig.update_layout(
        title=dict(text=f"Monte Carlo 10,000-Match Score Distribution ({sim_data['n_simulations']} Simulations)", font=dict(color="white", size=16)),
        xaxis=dict(color="white", gridcolor="#333"),
        yaxis=dict(title="Frequency", color="white", gridcolor="#333"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=40, l=40, r=20)
    )
    return fig

