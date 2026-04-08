"""
intervention_engine.py
----------------------
Intervention Calculator for F1 Cost Cap Compliance.

Methodology:
- If a team is projected to breach the cap, this engine calculates
  the minimum spending cuts needed across categories to return to compliance
- Uses linear programming (scipy.optimize.linprog) — real operations research
- Prioritises cuts in lower-impact categories first:
  Priority order (least to most critical):
  1. catering      (lowest sporting impact)
  2. it_sim        (can defer simulation work)
  3. other         (miscellaneous)
  4. tooling       (manufacturing tools)
  5. operations    (race operations — harder to cut mid-season)
  6. chassis_aero  (core performance — significant impact)
  7. personnel     (hardest to cut — contracts, redundancy costs)

Real-world context:
- This mirrors the decisions a Cost Analysis team makes mid-season
- Teams cannot cut personnel easily (contracts) so it's last resort
- Catering was exactly where Red Bull's 2021 breach was partly found
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from src.compliance_engine import run_compliance_report, CATEGORIES, SEASON_CAPS


# Priority order for cuts — index 0 = cut first, index 6 = cut last
CUT_PRIORITY = [
    'catering',      # 0 — cut first, lowest sporting impact
    'it_sim',        # 1
    'other',         # 2
    'tooling',       # 3
    'operations',    # 4
    'chassis_aero',  # 5
    'personnel',     # 6 — cut last, highest impact
]

# Maximum % cut allowed per category (realistic constraints)
MAX_CUT_PCT = {
    'catering':     0.30,   # can cut up to 30%
    'it_sim':       0.25,
    'other':        0.30,
    'tooling':      0.20,
    'operations':   0.15,
    'chassis_aero': 0.12,
    'personnel':    0.08,   # hardest to cut — contracts
}

# Sporting impact score per category (for reporting)
SPORTING_IMPACT = {
    'catering':     'Minimal',
    'it_sim':       'Low',
    'other':        'Low',
    'tooling':      'Medium',
    'operations':   'Medium-High',
    'chassis_aero': 'High',
    'personnel':    'Very High',
}


def calculate_intervention(
    df,
    team,
    season,
    current_month,
    safety_buffer_pct=0.005
):
    """
    Calculate minimum cuts needed to bring a team under the cap.

    Parameters:
    -----------
    df                : full spending dataframe
    team              : team name
    season            : season year
    current_month     : month we're calculating from
    safety_buffer_pct : target to be this % BELOW cap (default 0.5%)

    Returns:
    --------
    dict with:
        - status           : 'intervention_needed' or 'compliant'
        - projected_total  : current projected full-season spend
        - cap              : season cap
        - overspend        : how much over cap
        - cuts             : dict of recommended cuts per category
        - total_cut        : total $ to cut
        - new_projected    : projected total after cuts
        - feasible         : whether cuts are achievable within constraints
        - interventions    : list of intervention steps in priority order
    """
    cap = SEASON_CAPS[season]
    target = cap * (1 - safety_buffer_pct)  # target spend with safety buffer

    tdf = df[(df['team'] == team) & (df['season'] == season)].sort_values('month')
    tdf = tdf.copy()
    tdf['monthly_total'] = tdf[CATEGORIES].sum(axis=1)

    # Confirmed spend to date
    confirmed = tdf[tdf['month'] <= current_month]
    spend_to_date = confirmed['monthly_total'].sum()

    # Linear projection for remaining months
    avg_monthly = spend_to_date / current_month
    remaining_months = 12 - current_month
    projected_remaining = avg_monthly * remaining_months
    projected_total = spend_to_date + projected_remaining

    if projected_total <= cap:
        return {
            'status': 'compliant',
            'team': team,
            'season': season,
            'current_month': current_month,
            'projected_total': round(projected_total, 2),
            'cap': cap,
            'overspend': 0,
            'headroom': round(cap - projected_total, 2),
            'cuts': {},
            'total_cut': 0,
            'new_projected': round(projected_total, 2),
            'feasible': True,
            'interventions': [],
        }

    overspend = projected_total - target
    total_cut_needed = overspend

    # Average remaining monthly spend per category
    avg_category_spend = {}
    for cat in CATEGORIES:
        avg_category_spend[cat] = confirmed[cat].mean()

    remaining_category_spend = {
        cat: avg_category_spend[cat] * remaining_months
        for cat in CATEGORIES
    }

    # Linear programming:
    # Minimise: sum of cuts (weighted by priority)
    # Subject to: total cuts >= overspend
    #             each cut <= max_cut_pct * remaining_spend_in_category

    n = len(CATEGORIES)
    priority_weights = {cat: CUT_PRIORITY.index(cat) + 1 for cat in CATEGORIES}

    # Objective: minimise weighted cuts (higher weight = cut later)
    c = [priority_weights[cat] for cat in CATEGORIES]

    # Inequality constraints: -sum(cuts) <= -total_cut_needed
    A_ub = [[-1] * n]
    b_ub = [-total_cut_needed]

    # Bounds: 0 <= cut <= max_cut for each category
    bounds = []
    for cat in CATEGORIES:
        max_cut = MAX_CUT_PCT[cat] * remaining_category_spend[cat]
        bounds.append((0, max_cut))

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub,
        bounds=bounds, method='highs'
    )

    feasible = result.success
    cuts = {}
    total_cut = 0

    if feasible:
        for i, cat in enumerate(CATEGORIES):
            cut_amount = result.x[i]
            if cut_amount > 0.01:  # ignore negligible cuts
                cuts[cat] = round(cut_amount, 3)
                total_cut += cut_amount
    else:
        # Fallback: greedy cuts in priority order
        remaining_needed = total_cut_needed
        for cat in CUT_PRIORITY:
            if remaining_needed <= 0:
                break
            max_cut = MAX_CUT_PCT[cat] * remaining_category_spend[cat]
            cut = min(max_cut, remaining_needed)
            if cut > 0.01:
                cuts[cat] = round(cut, 3)
                total_cut += cut
                remaining_needed -= cut
        feasible = remaining_needed <= 0

    new_projected = projected_total - total_cut

    # Build intervention steps
    interventions = []
    for cat in CUT_PRIORITY:
        if cat in cuts:
            monthly_cut = cuts[cat] / remaining_months if remaining_months > 0 else 0
            interventions.append({
                'category': cat,
                'total_cut': cuts[cat],
                'monthly_cut': round(monthly_cut, 3),
                'sporting_impact': SPORTING_IMPACT[cat],
                'remaining_months': remaining_months,
            })

    return {
        'status': 'intervention_needed',
        'team': team,
        'season': season,
        'current_month': current_month,
        'projected_total': round(projected_total, 2),
        'cap': cap,
        'target': round(target, 2),
        'overspend': round(overspend, 2),
        'cuts': cuts,
        'total_cut': round(total_cut, 2),
        'new_projected': round(new_projected, 2),
        'feasible': feasible,
        'interventions': interventions,
    }


def run_all_interventions(df, season, current_month):
    """Run intervention calculator for all teams."""
    teams = ['RedBull', 'Mercedes', 'Ferrari', 'McLaren', 'Alpine']
    results = []
    for team in teams:
        r = calculate_intervention(df, team, season, current_month)
        results.append({
            'team': team,
            'projected': r['projected_total'],
            'cap': r['cap'],
            'overspend': r['overspend'],
            'total_cut_needed': r['total_cut'],
            'feasible': r['feasible'],
            'status': r['status'],
        })
    return pd.DataFrame(results)


if __name__ == '__main__':
    df, summary = run_compliance_report()

    print('\n=== Intervention Calculator — RedBull 2021 (Month 9) ===\n')
    result = calculate_intervention(df, 'RedBull', 2021, current_month=9)

    print(f"Status:            {result['status']}")
    print(f"Projected total:   ${result['projected_total']}M")
    print(f"Cap:               ${result['cap']}M")
    print(f"Overspend:         ${result['overspend']}M")
    print(f"Target (w/ buffer): ${result['target']}M")
    print(f"Feasible:          {result['feasible']}")
    print(f"Total cuts needed: ${result['total_cut']}M")
    print(f"New projected:     ${result['new_projected']}M")

    print('\n--- Recommended Interventions (in priority order) ---\n')
    if result['interventions']:
        print(f"{'Category':<15} {'Total Cut':>12} {'Monthly Cut':>13} "
              f"{'Months Left':>13} {'Impact'}")
        print('-' * 70)
        for step in result['interventions']:
            print(
                f"{step['category']:<15} "
                f"${step['total_cut']:>10.3f}M "
                f"${step['monthly_cut']:>11.3f}M "
                f"{step['remaining_months']:>13} "
                f"  {step['sporting_impact']}"
            )
    else:
        print('No interventions needed.')

    print('\n=== All Teams — 2021 Season at Month 9 ===\n')
    all_results = run_all_interventions(df, 2021, current_month=9)
    print(all_results.to_string(index=False))

    print('\n=== RedBull 2021 — Month by Month Intervention Tracker ===\n')
    print(f"{'Month':<8} {'Projected':>12} {'Overspend':>11} "
          f"{'Cut Needed':>12} {'Feasible'}")
    print('-' * 55)
    for month in range(6, 12):
        r = calculate_intervention(df, 'RedBull', 2021, current_month=month)
        print(
            f"  M{month:<6} "
            f"${r['projected_total']:>10.2f}M "
            f"${r['overspend']:>9.2f}M "
            f"${r['total_cut']:>10.2f}M "
            f"  {r['feasible']}"
        )