import numpy as np
import pandas as pd

def run_monte_carlo_simulation(
    target_score, 
    current_score, 
    balls_remaining, 
    wickets_remaining, 
    n_simulations=5000, 
    bowler_economy_mod=1.0, 
    batter_sr_mod=1.0
):
    """
    Executes a Monte Carlo stochastic simulation of the remaining match overs.
    Returns simulated final scores, win probability %, and confidence intervals.
    """
    np.random.seed(42)
    
    final_scores = []
    wins = 0

    # Base ball run outcomes and probabilities under standard conditions
    # 0, 1, 2, 4, 6, Wicket
    base_outcomes = np.array([0, 1, 2, 4, 6, -1])
    base_probs = np.array([0.34, 0.38, 0.08, 0.11, 0.05, 0.04])
    
    # Apply bowler & batter skill modifiers
    modified_probs = base_probs.copy()
    
    # High batter SR increases 4s and 6s
    if batter_sr_mod > 1.0:
        modified_probs[3] *= (1.0 + (batter_sr_mod - 1.0) * 0.5) # 4s
        modified_probs[4] *= (1.0 + (batter_sr_mod - 1.0) * 0.5) # 6s
    elif batter_sr_mod < 1.0:
        modified_probs[0] *= (1.0 + (1.0 - batter_sr_mod) * 0.4) # Dots

    # Low bowler economy decreases boundaries, increases dots
    if bowler_economy_mod < 1.0:
        modified_probs[0] *= (1.0 + (1.0 - bowler_economy_mod) * 0.5)
        modified_probs[3] *= bowler_economy_mod
        modified_probs[4] *= bowler_economy_mod

    # Normalize probabilities
    modified_probs = modified_probs / np.sum(modified_probs)

    for _ in range(n_simulations):
        score = current_score
        wickets = 10 - wickets_remaining
        
        # Simulate ball by ball
        for b in range(balls_remaining):
            if wickets >= 10:
                break
            if target_score is not None and score >= target_score:
                break

            # Over phase multiplier (Death overs have higher boundary & wicket variance)
            balls_bowled_in_inn = (120 - balls_remaining) + b
            if balls_bowled_in_inn >= 90: # Death overs (16-20)
                phase_probs = modified_probs.copy()
                phase_probs[4] *= 1.4 # More 6s
                phase_probs[5] *= 1.3 # More wickets
                phase_probs = phase_probs / np.sum(phase_probs)
                outcome = np.random.choice(base_outcomes, p=phase_probs)
            else:
                outcome = np.random.choice(base_outcomes, p=modified_probs)

            if outcome == -1:
                wickets += 1
            else:
                score += outcome

        final_scores.append(score)
        if target_score is not None and score >= target_score:
            wins += 1

    final_scores = np.array(final_scores)
    
    mean_score = float(np.mean(final_scores))
    median_score = float(np.median(final_scores))
    p10 = float(np.percentile(final_scores, 10))
    p90 = float(np.percentile(final_scores, 90))
    win_pct = round(float((wins / n_simulations) * 100.0), 1) if target_score is not None else 0.0

    return {
        "n_simulations": n_simulations,
        "mean_score": round(mean_score, 1),
        "median_score": round(median_score, 1),
        "ci_10th": round(p10, 1),
        "ci_90th": round(p90, 1),
        "simulated_win_pct": win_pct,
        "score_distribution": final_scores[:1000].tolist() # Sample for histogram
    }

def simulate_what_if_scenario(target_score, current_score, balls_remaining, wickets_remaining, tactical_choice):
    """
    Simulates tactical 'What-If' scenarios:
    1. Baseline Tactic
    2. Deploy Death Specialist Bowler (Econ -15%)
    3. Promote Power Hitter Batter (SR +20%)
    """
    baseline = run_monte_carlo_simulation(target_score, current_score, balls_remaining, wickets_remaining)

    if tactical_choice == "Deploy Death Specialist Bowler":
        tactic = run_monte_carlo_simulation(target_score, current_score, balls_remaining, wickets_remaining, bowler_economy_mod=0.85)
        impact_msg = f"Deploying Death Specialist reduces expected score by {baseline['mean_score'] - tactic['mean_score']:.1f} runs (Win Prob impact: {tactic['simulated_win_pct'] - baseline['simulated_win_pct']:+.1f}%)."
    elif tactical_choice == "Promote Power Hitter Batter":
        tactic = run_monte_carlo_simulation(target_score, current_score, balls_remaining, wickets_remaining, batter_sr_mod=1.20)
        impact_msg = f"Promoting Power Hitter increases expected score by {tactic['mean_score'] - baseline['mean_score']:.1f} runs (Win Prob impact: {tactic['simulated_win_pct'] - baseline['simulated_win_pct']:+.1f}%)."
    else:
        tactic = baseline
        impact_msg = "Standard baseline tactic selected."

    return {
        "baseline": baseline,
        "tactical": tactic,
        "tactical_choice": tactical_choice,
        "impact_summary": impact_msg
    }
