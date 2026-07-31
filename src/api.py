from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

from src.models import predict_win_probability, get_auction_recommendations
from src.simulator import run_monte_carlo_simulation, simulate_what_if_scenario
from src.features import get_h2h_matrix, get_tactical_alerts
from src.etl import SAMPLE_BATTERS

app = FastAPI(
    title="IPL Edge Analytics & Predictive API",
    description="Enterprise REST API for Real-Time Win Probability, Monte Carlo Match Simulations, and H2H Tactical Analytics",
    version="1.0.0"
)

class MatchStateRequest(BaseModel):
    runs_needed: int = Field(..., example=45)
    balls_left: int = Field(..., example=30)
    wickets_left: int = Field(..., example=6)
    req_rr: float = Field(..., example=9.0)
    curr_rr: float = Field(..., example=8.5)
    target_runs: int = Field(..., example=180)
    venue_avg_runs: float = Field(175.0, example=175.0)
    toss_win_bat_first: int = Field(1, example=1)

class SimulationRequest(BaseModel):
    target_score: int = Field(180, example=180)
    current_score: int = Field(135, example=135)
    balls_remaining: int = Field(30, example=30)
    wickets_remaining: int = Field(6, example=6)
    n_simulations: int = Field(5000, example=5000)
    tactical_choice: Optional[str] = Field("Deploy Death Specialist Bowler", example="Deploy Death Specialist Bowler")

class H2HRequest(BaseModel):
    team_a: str = Field("Chennai Super Kings", example="Chennai Super Kings")
    team_b: str = Field("Mumbai Indians", example="Mumbai Indians")

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "IPL Edge Analytics API", "version": "1.0.0"}

@app.post("/api/v1/predict-win-prob")
def api_predict_win_prob(req: MatchStateRequest):
    try:
        results = predict_win_probability(req.model_dump())
        return {"success": True, "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/simulate-match")
def api_simulate_match(req: SimulationRequest):
    try:
        res = simulate_what_if_scenario(
            req.target_score, 
            req.current_score, 
            req.balls_remaining, 
            req.wickets_remaining, 
            req.tactical_choice
        )
        return {"success": True, "data": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/h2h-matrix")
def api_h2h_matrix(req: H2HRequest):
    try:
        roster_a = SAMPLE_BATTERS.get(req.team_a, SAMPLE_BATTERS["Chennai Super Kings"])
        roster_b = SAMPLE_BATTERS.get(req.team_b, SAMPLE_BATTERS["Mumbai Indians"])
        matrix = get_h2h_matrix(roster_a, roster_b)
        alerts = get_tactical_alerts(matrix, roster_a, roster_b)
        return {
            "success": True, 
            "team_a": req.team_a, 
            "team_b": req.team_b, 
            "matrix": matrix, 
            "alerts": alerts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/auction-recommendations")
def api_auction_recommendations(team: str = "Chennai Super Kings", gap_type: str = "Death Bowling"):
    try:
        df_rec = get_auction_recommendations(team, gap_type)
        return {"success": True, "data": df_rec.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
