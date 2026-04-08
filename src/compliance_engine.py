"""
compliance_engine.py
--------------------
F1 Cost Cap Compliance Model — Core Engine
Handles 3 seasons (2021-2023), 5 teams, 7 spending categories.

Data note:
- RedBull 2021: FIA_confirmed (actual breach, $149.8M vs $147.4M cap)
- All other figures: estimated based on FIA Financial Regulations structure
  and The Race / Motor Sport Magazine reporting. Actual submissions are
  confidential to the FIA Cost Cap Administration.
"""

import pandas as pd

CATEGORIES = [
    'personnel', 'chassis_aero', 'operations',
    'tooling', 'catering', 'it_sim', 'other'
]

# Real cap figures per season (FIA confirmed)
# 2021: $147.4M | 2022: $142.4M (inflation adj) | 2023: $135M base + $2.4M (23 races)
SEASON_CAPS = {
    2021: 147.4,
    2022: 142.4,
    2023: 137.4,
}

def load_data(filepath='data/team_budgets.csv'):
    df = pd.read_csv(filepath)
    df['season'] = df['season'].astype(int)
    df['month'] = df['month'].astype(int)
    return df

def compute_cumulative(df):
    """Add monthly total and cumulative spend columns per team per season."""
    df = df.copy()
    df['monthly_total'] = df[CATEGORIES].sum(axis=1)
    df = df.sort_values(['team', 'season', 'month'])
    df['cumulative_spend'] = df.groupby(['team', 'season'])['monthly_total'].cumsum()
    return df

def project_full_season(df):
    """
    For each team-season, project full year spend linearly
    from months recorded so far.
    """
    results = []
    for (team, season), group in df.groupby(['team', 'season']):
        group = group.sort_values('month')
        months_recorded = group['month'].max()
        spend_so_far = group['cumulative_spend'].max()
        cap = SEASON_CAPS[season]
        source = group['data_source'].iloc[0]

        if months_recorded == 0:
            projected = 0
        else:
            monthly_avg = spend_so_far / months_recorded
            projected = monthly_avg * 12

        results.append({
            'team': team,
            'season': season,
            'months_recorded': months_recorded,
            'spend_so_far': round(spend_so_far, 2),
            'projected_full_season': round(projected, 2),
            'season_cap': cap,
            'headroom': round(cap - projected, 2),
            'pct_of_cap': round((projected / cap) * 100, 1),
            'data_source': source,
        })

    return pd.DataFrame(results)

def assign_risk(row):
    """
    Risk tiers based on % of cap used:
    GREEN : < 95%   — comfortable headroom
    AMBER : 95-100% — approaching limit
    RED   : > 100%  — breach
    """
    pct = row['pct_of_cap']
    if pct > 100:
        return 'RED'
    elif pct >= 95:
        return 'AMBER'
    else:
        return 'GREEN'

def run_compliance_report(filepath='data/team_budgets.csv'):
    df = load_data(filepath)
    df = compute_cumulative(df)
    summary = project_full_season(df)
    summary['risk'] = summary.apply(assign_risk, axis=1)
    summary = summary.sort_values(
        ['season', 'pct_of_cap'], ascending=[True, False]
    ).reset_index(drop=True)
    return df, summary

if __name__ == '__main__':
    df, summary = run_compliance_report()

    for season in sorted(summary['season'].unique()):
        s = summary[summary['season'] == season]
        cap = SEASON_CAPS[season]
        print(f'\n=== {season} Season | Cap: ${cap}M ===')
        print(f"{'Team':<12} {'Projected':>12} {'Cap':>8} {'Headroom':>10} {'% of Cap':>10} {'Risk':<8} {'Source'}")
        print('-' * 75)
        for _, row in s.iterrows():
            print(
                f"{row['team']:<12} "
                f"${row['projected_full_season']:>10.1f}M "
                f"${row['season_cap']:>6.1f}M "
                f"${row['headroom']:>8.1f}M "
                f"{row['pct_of_cap']:>9.1f}% "
                f"{row['risk']:<8} "
                f"{row['data_source']}"
            )

    print('\n=== Summary Across All Seasons ===')
    print(f"Total RED   (breach): {(summary['risk'] == 'RED').sum()}")
    print(f"Total AMBER (at risk): {(summary['risk'] == 'AMBER').sum()}")
    print(f"Total GREEN (compliant): {(summary['risk'] == 'GREEN').sum()}")