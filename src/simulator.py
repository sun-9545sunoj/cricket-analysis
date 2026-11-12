from functools import lru_cache

import numpy as np

from src.config import FORMATS, phase_case_sql, phase_of, total_balls
from src.etl import get_connection

# Outcomes per legal ball; -1 is a wicket. Extras from wides/no-balls are added as a per-ball run rate.
OUTCOMES = np.array([0, 1, 2, 3, 4, 6, -1])

TACTICS = {
    "Baseline Tactic": {},
    "Deploy Death Specialist Bowler": {"bowler_economy_mod": 0.85},
    "Promote Power Hitter Batter": {"batter_sr_mod": 1.20},
}


@lru_cache(maxsize=64)
def phase_outcome_probabilities(fmt, team=None):
    """
    Empirical outcome distribution of a legal ball in each phase. With `team`, uses that team's
    batting over the last three years of data, so the profile reflects the current side.
    """
    team_clause = (
        "AND d.batting_team = ? AND m.start_date >= (SELECT MAX(start_date) FROM matches) - INTERVAL 3 YEAR"
        if team else ""
    )
    con = get_connection(read_only=True)
    try:
        df = con.execute(f"""
            SELECT {phase_case_sql()} AS phase,
                   CASE WHEN d.is_wicket > 0 THEN -1 ELSE LEAST(d.runs_batter + d.runs_extras, 6) END AS outcome,
                   COUNT(*) AS n
            FROM deliveries d JOIN matches m USING (match_id)
            WHERE m.format = ? AND d.is_legal AND d.innings <= 2 {team_clause}
            GROUP BY phase, outcome
        """, [fmt] + ([team] if team else [])).df()
        extras = con.execute(f"""
            SELECT {phase_case_sql()} AS phase,
                   SUM(CASE WHEN NOT d.is_legal THEN d.total_runs ELSE 0 END) * 1.0
                       / NULLIF(COUNT(*) FILTER (WHERE d.is_legal), 0) AS extra_runs_per_ball
            FROM deliveries d JOIN matches m USING (match_id)
            WHERE m.format = ? AND d.innings <= 2 {team_clause}
            GROUP BY phase
        """, [fmt] + ([team] if team else [])).df()
    finally:
        con.close()

    result = {}
    for _, _, phase in FORMATS[fmt]["phases"]:
        counts = df[df["phase"] == phase].set_index("outcome")["n"]
        # 5s (overthrows) are folded into 4s
        counts.index = [4 if o == 5 else o for o in counts.index]
        counts = counts.groupby(level=0).sum()
        probs = np.array([counts.get(o, 0) for o in OUTCOMES], dtype=float)
        probs = probs / probs.sum() if probs.sum() else np.full(len(OUTCOMES), 1 / len(OUTCOMES))
        extra = extras.loc[extras["phase"] == phase, "extra_runs_per_ball"]
        result[phase] = {"probs": probs, "extras_per_ball": float(extra.iloc[0]) if len(extra) else 0.0}
    return result


def _apply_modifiers(probs, bowler_economy_mod, batter_sr_mod):
    p = probs.copy()
    if batter_sr_mod > 1.0:
        p[4] *= 1 + (batter_sr_mod - 1) * 0.5
        p[5] *= 1 + (batter_sr_mod - 1) * 0.5
    elif batter_sr_mod < 1.0:
        p[0] *= 1 + (1 - batter_sr_mod) * 0.4
    if bowler_economy_mod < 1.0:
        p[0] *= 1 + (1 - bowler_economy_mod) * 0.5
        p[4] *= bowler_economy_mod
        p[5] *= bowler_economy_mod
    return p / p.sum()


def run_monte_carlo_simulation(
    fmt,
    target_score,
    current_score,
    balls_remaining,
    wickets_remaining,
    n_simulations=5000,
    bowler_economy_mod=1.0,
    batter_sr_mod=1.0,
    batting_team=None,
    seed=42,
):
    """
    Simulates the rest of an innings ball by ball, `n_simulations` times in parallel, using real
    per-phase outcome rates (the batting team's own rates when `batting_team` is given).
    `target_score=None` simulates a first innings (no chase).
    """
    rng = np.random.default_rng(seed)
    phase_probs = phase_outcome_probabilities(fmt, batting_team)
    n_balls = total_balls(fmt)

    scores = np.full(n_simulations, float(current_score))
    wickets = np.full(n_simulations, 10 - wickets_remaining)
    active = np.ones(n_simulations, dtype=bool)

    for b in range(balls_remaining):
        if not active.any():
            break
        ball_no = n_balls - balls_remaining + b
        cfg = phase_probs[phase_of(fmt, ball_no // 6)]
        probs = _apply_modifiers(cfg["probs"], bowler_economy_mod, batter_sr_mod)
        outcome = rng.choice(OUTCOMES, size=n_simulations, p=probs)

        is_wicket = (outcome == -1) & active
        runs = np.where(outcome > 0, outcome, 0) + cfg["extras_per_ball"]
        scores += np.where(active & ~is_wicket, runs, 0)
        wickets += is_wicket
        active &= wickets < 10
        if target_score is not None:
            active &= scores < target_score

    scores = np.round(scores)
    wins = int((scores >= target_score).sum()) if target_score is not None else 0
    return {
        "n_simulations": n_simulations,
        "mean_score": round(float(scores.mean()), 1),
        "median_score": round(float(np.median(scores)), 1),
        "ci_10th": round(float(np.percentile(scores, 10)), 1),
        "ci_90th": round(float(np.percentile(scores, 90)), 1),
        "simulated_win_pct": round(wins / n_simulations * 100, 1) if target_score is not None else None,
        "score_distribution": scores[:1000].tolist(),
    }


def simulate_what_if_scenario(fmt, target_score, current_score, balls_remaining, wickets_remaining,
                              tactical_choice, n_simulations=5000, batting_team=None):
    """Compares a tactical change against the baseline simulation of the same match state."""
    args = (fmt, target_score, current_score, balls_remaining, wickets_remaining, n_simulations)
    baseline = run_monte_carlo_simulation(*args, batting_team=batting_team)
    modifiers = TACTICS.get(tactical_choice, {})
    tactic = run_monte_carlo_simulation(*args, batting_team=batting_team, **modifiers) if modifiers else baseline

    score_delta = tactic["mean_score"] - baseline["mean_score"]
    if not modifiers:
        impact = "Standard baseline tactic selected."
    else:
        impact = f"{tactical_choice} changes the expected score by {score_delta:+.1f} runs"
        if target_score is not None:
            impact += f" (chase win chance {tactic['simulated_win_pct'] - baseline['simulated_win_pct']:+.1f} pts)"
        impact += "."
    return {"baseline": baseline, "tactical": tactic, "tactical_choice": tactical_choice, "impact_summary": impact}
