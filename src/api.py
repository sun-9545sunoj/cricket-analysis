import os
import json
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import DEFAULT_TEAM_A, DEFAULT_TEAM_B, FORMATS, PROJECT_ROOT, overs_to_balls, total_balls
from src.etl import get_connection
from src.features import (
    batting_stats, bowling_stats, calculate_mpi, get_mpi_interpretation, head_to_head, latest_series,
    list_teams, phase_stats, xi_matchup,
)
from src.models import (
    cluster_batters, model_report, predict_win_probability, project_first_innings_score, scoreboard_state,
)
from src.simulator import TACTICS, simulate_what_if_scenario

WEB_DIR = os.path.join(PROJECT_ROOT, "web")

Format = Literal["T20", "ODI"]

app = FastAPI(
    title="Cricket Analytics API",
    description="International cricket analytics on Cricsheet ball-by-ball data: head-to-head records, "
                "player stats, win probability, score projection, Monte Carlo simulation and player clustering.",
    version="2.0.0",
)


def records(df):
    """DataFrame -> JSON-safe list of dicts (NaN -> null, dates -> YYYY-MM-DD)."""
    df = df.copy()
    if "start_date" in df:
        df["start_date"] = pd.to_datetime(df["start_date"]).dt.strftime("%Y-%m-%d")
    return json.loads(df.to_json(orient="records"))


def _balls(overs, fmt):
    try:
        balls = overs_to_balls(overs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not 0 < balls < total_balls(fmt):
        raise HTTPException(status_code=422, detail=f"Overs must be between 0.1 and {FORMATS[fmt]['overs'] - 1}.5")
    return balls


class ChaseRequest(BaseModel):
    format: Format = "T20"
    target: int = Field(..., ge=1, examples=[180])
    current_score: int = Field(..., ge=0, examples=[120])
    overs: float = Field(..., examples=[15.0])
    wickets_lost: int = Field(..., ge=0, le=9, examples=[4])


class FirstInningsRequest(BaseModel):
    format: Format = "T20"
    current_score: int = Field(..., ge=0, examples=[95])
    overs: float = Field(..., examples=[11.0])
    wickets_lost: int = Field(..., ge=0, le=9, examples=[2])


class SimulationRequest(BaseModel):
    format: Format = "T20"
    target_score: Optional[int] = Field(180, description="Omit to simulate a first innings")
    current_score: int = Field(120, ge=0)
    overs: float = Field(15.0)
    wickets_lost: int = Field(4, ge=0, le=9)
    batting_team: Optional[str] = Field(None, description="Use this team's recent scoring profile")
    tactical_choice: str = Field("Baseline Tactic", examples=["Deploy Death Specialist Bowler"])
    n_simulations: int = Field(5000, ge=100, le=20000)


class MatchupRequest(BaseModel):
    format: Format = "T20"
    team_a: str = DEFAULT_TEAM_A
    team_b: str = DEFAULT_TEAM_B


@app.get("/")
def root():
    return RedirectResponse(url="/app/")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Cricket Analytics API", "version": app.version}


@app.get("/api/v1/meta")
def api_meta():
    """Formats, teams with enough history, data coverage and front-end defaults."""
    con = get_connection(read_only=True)
    try:
        coverage = con.execute(
            "SELECT format, COUNT(*), MIN(start_date), MAX(start_date) FROM matches GROUP BY format"
        ).fetchall()
    finally:
        con.close()
    return {
        "formats": {fmt: {"overs": cfg["overs"], "phases": [p[2] for p in cfg["phases"]], "teams": list_teams(fmt)}
                    for fmt, cfg in FORMATS.items()},
        "coverage": {fmt: {"matches": n, "from": str(lo), "to": str(hi)} for fmt, n, lo, hi in coverage},
        "defaults": {"team_a": DEFAULT_TEAM_A, "team_b": DEFAULT_TEAM_B},
        "tactics": list(TACTICS),
    }


@app.get("/api/v1/head-to-head")
def api_head_to_head(team_a: str = DEFAULT_TEAM_A, team_b: str = DEFAULT_TEAM_B, format: Format = "T20"):
    summary, df = head_to_head(team_a, team_b, format)
    return {"summary": summary, "matches": records(df)}


@app.get("/api/v1/series/latest")
def api_latest_series(team_a: str = DEFAULT_TEAM_A, team_b: str = DEFAULT_TEAM_B, bilateral_only: bool = True):
    event, df = latest_series(team_a, team_b, bilateral_only=bilateral_only)
    return {"event": event, "matches": records(df)}


@app.get("/api/v1/players/batting")
def api_batting(format: Format = "T20", team: Optional[str] = None, opponent: Optional[str] = None,
                since: Optional[str] = None, min_balls: int = 30, limit: int = Query(15, le=200)):
    return {"data": records(batting_stats(format, team, opponent, since, min_balls).head(limit))}


@app.get("/api/v1/players/bowling")
def api_bowling(format: Format = "T20", team: Optional[str] = None, opponent: Optional[str] = None,
                since: Optional[str] = None, min_balls: int = 30, limit: int = Query(15, le=200)):
    return {"data": records(bowling_stats(format, team, opponent, since, min_balls).head(limit))}


@app.get("/api/v1/phase-stats")
def api_phase_stats(team: str, format: Format = "T20", opponent: Optional[str] = None, since: Optional[str] = None):
    return {"team": team, "data": phase_stats(format, team, opponent, since)}


@app.post("/api/v1/predict-win-prob")
def api_predict_win_prob(req: ChaseRequest):
    balls = _balls(req.overs, req.format)
    state = scoreboard_state(req.format, req.target, req.current_score, balls, req.wickets_lost)
    if state["runs_needed"] == 0:
        raise HTTPException(status_code=422, detail="Target already reached.")
    mpi = calculate_mpi(state["req_rr"], state["wickets_left"], state["balls_left"], req.format)
    return {
        "state": state,
        "prediction": predict_win_probability(req.format, state),
        "pressure": {"mpi": mpi, **get_mpi_interpretation(mpi)},
    }


@app.post("/api/v1/project-score")
def api_project_score(req: FirstInningsRequest):
    balls = _balls(req.overs, req.format)
    return {"projected_total": project_first_innings_score(req.format, req.current_score, balls, req.wickets_lost)}


@app.post("/api/v1/simulate-match")
def api_simulate_match(req: SimulationRequest):
    balls = _balls(req.overs, req.format)
    return simulate_what_if_scenario(
        req.format, req.target_score, req.current_score, total_balls(req.format) - balls,
        10 - req.wickets_lost, req.tactical_choice, req.n_simulations, req.batting_team,
    )


@app.post("/api/v1/h2h-matrix")
def api_h2h_matrix(req: MatchupRequest):
    """Latest XI of team A (batting) against latest XI of team B (bowling), from their last meeting."""
    return {"team_a": req.team_a, "team_b": req.team_b, **xi_matchup(req.team_a, req.team_b, req.format)}


@app.get("/api/v1/batter-clusters")
def api_batter_clusters(format: Format = "T20", teams: Optional[str] = Query(None, description="Comma-separated")):
    df, info = cluster_batters(format)
    if teams:
        df = df[df["team"].isin([t.strip() for t in teams.split(",")])]
    cols = ["player", "team", "style", "runs", "balls", "strike_rate", "average", "boundary_pct", "dot_pct",
            "powerplay_share", "death_share"]
    return {"info": info, "players": records(df[cols].round(3))}


@app.get("/api/v1/model-report")
def api_model_report():
    report = model_report()
    if not report:
        raise HTTPException(status_code=404, detail="No trained models; run `python -m src.pipeline`.")
    return report


# HTML/JS front end (web/) served at /app
app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="web")
