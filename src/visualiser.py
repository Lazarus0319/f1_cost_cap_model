"""
visualiser.py
-------------
Charts for the F1 Cost Cap Compliance Model.
Generates 3 plots:
  1. % of cap used per team per season (bar chart)
  2. Cumulative spend vs cap over months (line chart) — 2021 only
  3. Category breakdown for RedBull 2021 (breach case study)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from src.compliance_engine import run_compliance_report, CATEGORIES, SEASON_CAPS

# Colour scheme
RISK_COLOURS = {'RED': '#E8002D', 'AMBER': '#FFA500', 'GREEN': '#00A550'}
TEAM_COLOURS = {
    'RedBull':  '#3671C6',
    'Mercedes': '#27F4D2',
    'Ferrari':  '#E8002D',
    'McLaren':  '#FF8000',
    'Alpine':   '#FF87BC',
}


def plot_pct_of_cap(summary, save_path='data/chart_pct_of_cap.png'):
    """Bar chart: % of cap used per team, grouped by season."""
    seasons = sorted(summary['season'].unique())
    teams = ['RedBull', 'Mercedes', 'Ferrari', 'McLaren', 'Alpine']
    x = np.arange(len(teams))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, season in enumerate(seasons):
        s = summary[summary['season'] == season].set_index('team')
        pcts = [s.loc[t, 'pct_of_cap'] if t in s.index else 0 for t in teams]
        colours = [
            RISK_COLOURS[s.loc[t, 'risk']] if t in s.index else '#ccc'
            for t in teams
        ]
        bars = ax.bar(x + i * width, pcts, width, label=str(season),
                      color=colours, edgecolor='white', linewidth=0.5)

    # Cap line at 100%
    ax.axhline(100, color='black', linewidth=1.5, linestyle='--', label='Cap (100%)')

    ax.set_xlabel('Team', fontsize=12)
    ax.set_ylabel('% of Season Cap Used', fontsize=12)
    ax.set_title('F1 Cost Cap Usage by Team & Season (2021–2023)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(teams, fontsize=11)
    ax.set_ylim(60, 110)
    ax.legend(title='Season', fontsize=10)

    # Risk legend
    patches = [mpatches.Patch(color=c, label=r) for r, c in RISK_COLOURS.items()]
    ax.legend(handles=patches + [
        mpatches.Patch(color='white', label=''),
    ], title='Risk', loc='lower right', fontsize=9)

    # Add season legend separately
    season_patches = [mpatches.Patch(color='grey', alpha=0.4+i*0.3, label=str(s))
                      for i, s in enumerate(seasons)]
    ax.add_artist(ax.legend(handles=season_patches, title='Season',
                            loc='upper left', fontsize=9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_cumulative_spend(df, season=2021, save_path='data/chart_cumulative_2021.png'):
    """Line chart: cumulative spend vs cap over 12 months for a given season."""
    cap = SEASON_CAPS[season]
    season_df = df[df['season'] == season]

    fig, ax = plt.subplots(figsize=(12, 6))

    for team, colour in TEAM_COLOURS.items():
        tdf = season_df[season_df['team'] == team].sort_values('month')
        if tdf.empty:
            continue
        ax.plot(tdf['month'], tdf['cumulative_spend'],
                label=team, color=colour, linewidth=2.5, marker='o', markersize=4)

    # Cap line
    ax.axhline(cap, color='black', linewidth=2, linestyle='--', label=f'Cap (${cap}M)')

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Cumulative Spend ($M)', fontsize=12)
    ax.set_title(f'Cumulative Spend vs Cap — {season} Season', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_category_breakdown(df, team='RedBull', season=2021,
                            save_path='data/chart_category_breakdown.png'):
    """Stacked bar: monthly category breakdown for a team-season."""
    tdf = df[(df['team'] == team) & (df['season'] == season)].sort_values('month')
    cap = SEASON_CAPS[season]

    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(12)
    cat_colours = ['#003f88', '#0066cc', '#3399ff', '#66b2ff', '#99ccff', '#cce5ff', '#e6f2ff']

    for i, cat in enumerate(CATEGORIES):
        vals = tdf[cat].values
        ax.bar(tdf['month'], vals, bottom=bottom,
               label=cat.replace('_', ' ').title(),
               color=cat_colours[i], edgecolor='white', linewidth=0.3)
        bottom += vals

    ax.axhline(cap / 12, color='red', linewidth=2, linestyle='--',
               label=f'Monthly cap average (${cap/12:.1f}M)')

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Monthly Spend ($M)', fontsize=12)
    ax.set_title(f'{team} {season} — Monthly Spend by Category\n(FIA Cost Cap Breach Case Study)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def generate_all_charts(df, summary):
    print('\nGenerating charts...')
    plot_pct_of_cap(summary)
    plot_cumulative_spend(df, season=2021)
    plot_category_breakdown(df, team='RedBull', season=2021)
    print('All charts saved to data/')


if __name__ == '__main__':
    df, summary = run_compliance_report()
    generate_all_charts(df, summary)