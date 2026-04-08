"""
monte_carlo.py
--------------
Monte Carlo Simulation Engine for F1 Cost Cap Compliance.

Methodology:
- Models monthly spend as: actual_spend * (1 + epsilon)
  where epsilon ~ N(0, sigma) — normally distributed random noise
- sigma estimated from historical month-to-month variance in dataset
- Runs N simulations (default 10,000) to build a probability distribution
- Outputs: breach probability, expected final spend, confidence intervals

This is the same methodology used in financial risk management (Value at Risk).
"""

import numpy as np
import pandas as pd
from src.compliance_engine import run_compliance_report, SEASON_CAPS

np.random.seed(42)  # reproducibility


def estimate_sigma(df, team, season):
    """
    Estimate monthly spending volatility (sigma) for a team-season.
    Uses coefficient of variation of monthly totals.
    If insufficient data, falls back to global average sigma.
    """
    CATEGORIES = [
        'personnel', 'chassis_aero', 'operations',
        'tooling', 'catering', 'it_sim', 'other'
    ]
    tdf = df[(df['team'] == team) & (df['season'] == season)].copy()
    tdf['monthly_total'] = tdf[CATEGORIES].sum(axis=1)

    if len(tdf) < 3:
        return 0.02  # fallback: 2% volatility

    # Coefficient of variation of month-to-month changes
    changes = tdf['monthly_total'].pct_change().dropna()
    sigma = changes.std()

    # Cap sigma between 1% and 8% — realistic for F1 spending
    return float(np.clip(sigma, 0.01, 0.08))


def run_monte_carlo(
    df,
    team,
    season,
    current_month,
    n_simulations=10_000
):
    """
    Run Monte Carlo simulation for a team-season given spend up to current_month.

    Parameters:
    -----------
    df            : full spending dataframe
    team          : team name (e.g. 'RedBull')
    season        : season year (e.g. 2021)
    current_month : month number we're simulating from (1-12)
    n_simulations : number of Monte Carlo iterations

    Returns:
    --------
    dict with:
        - simulated_totals     : array of N simulated full-season totals
        - breach_probability   : % of simulations that exceed the cap
        - expected_total       : mean of simulated totals
        - percentile_5         : 5th percentile (optimistic scenario)
        - percentile_95        : 95th percentile (pessimistic scenario)
        - cap                  : season cap
        - spend_to_date        : confirmed spend up to current_month
        - remaining_months     : months left to simulate
        - sigma                : volatility used
    """
    CATEGORIES = [
        'personnel', 'chassis_aero', 'operations',
        'tooling', 'catering', 'it_sim', 'other'
    ]

    cap = SEASON_CAPS[season]
    tdf = df[(df['team'] == team) & (df['season'] == season)].sort_values('month')

    # Confirmed spend up to current_month
    confirmed = tdf[tdf['month'] <= current_month].copy()
    confirmed['monthly_total'] = confirmed[CATEGORIES].sum(axis=1)
    spend_to_date = confirmed['monthly_total'].sum()

    # Average monthly spend so far
    avg_monthly = spend_to_date / current_month
    remaining_months = 12 - current_month

    # Estimate volatility
    sigma = estimate_sigma(df, team, season)

    # Run simulations
    simulated_totals = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Simulate remaining months with random noise
        noise = np.random.normal(0, sigma, remaining_months)
        simulated_remaining = avg_monthly * (1 + noise)
        # No negative spending
        simulated_remaining = np.maximum(simulated_remaining, 0)
        simulated_totals[i] = spend_to_date + simulated_remaining.sum()

    breach_probability = (simulated_totals > cap).mean() * 100

    return {
        'team': team,
        'season': season,
        'current_month': current_month,
        'simulated_totals': simulated_totals,
        'breach_probability': round(breach_probability, 2),
        'expected_total': round(simulated_totals.mean(), 2),
        'percentile_5': round(np.percentile(simulated_totals, 5), 2),
        'percentile_95': round(np.percentile(simulated_totals, 95), 2),
        'cap': cap,
        'spend_to_date': round(spend_to_date, 2),
        'remaining_months': remaining_months,
        'sigma': round(sigma, 4),
    }


def run_all_teams_simulation(df, season, current_month, n_simulations=10_000):
    """Run Monte Carlo for all 5 teams for a given season and month."""
    teams = ['RedBull', 'Mercedes', 'Ferrari', 'McLaren', 'Alpine']
    results = []
    for team in teams:
        result = run_monte_carlo(df, team, season, current_month, n_simulations)
        results.append({
            'team': team,
            'spend_to_date': result['spend_to_date'],
            'expected_total': result['expected_total'],
            'cap': result['cap'],
            'breach_probability': result['breach_probability'],
            'p5': result['percentile_5'],
            'p95': result['percentile_95'],
            'sigma': result['sigma'],
        })
    return pd.DataFrame(results)


if __name__ == '__main__':
    df, summary = run_compliance_report()

    print('\n=== Monte Carlo Simulation — RedBull 2021 (after Month 6) ===\n')
    result = run_monte_carlo(df, 'RedBull', 2021, current_month=6)

    print(f"Spend to date (M1-M6):     ${result['spend_to_date']}M")
    print(f"Expected full season:       ${result['expected_total']}M")
    print(f"5th percentile (optimistic): ${result['percentile_5']}M")
    print(f"95th percentile (pessimistic): ${result['percentile_95']}M")
    print(f"Season cap:                ${result['cap']}M")
    print(f"Breach probability:        {result['breach_probability']}%")
    print(f"Volatility (sigma):        {result['sigma']}")

    print('\n=== All Teams — 2021 Season after Month 6 ===\n')
    all_results = run_all_teams_simulation(df, 2021, current_month=6)
    print(all_results.to_string(index=False))