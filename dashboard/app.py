"""
app.py
------
F1 Cost Cap Compliance Model — Interactive Streamlit Dashboard
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.compliance_engine import run_compliance_report, CATEGORIES, SEASON_CAPS

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title='F1 Cost Cap Compliance Model',
    page_icon='🏎️',
    layout='wide'
)

# ── Load data ─────────────────────────────────────────────────────
@st.cache_data
def load():
    return run_compliance_report()

df, summary = load()

# ── Header ────────────────────────────────────────────────────────
st.title('🏎️ F1 Cost Cap Compliance Model')
st.markdown(
    'Tracking team spending against the FIA Financial Regulations (2021–2023). '
    'RedBull 2021 data is **FIA confirmed**. All other figures are **estimated** '
    'based on public reporting — actual submissions are confidential to the FIA CCA.'
)
st.divider()

# ── Sidebar filters ───────────────────────────────────────────────
st.sidebar.header('Filters')
selected_season = st.sidebar.selectbox('Season', [2021, 2022, 2023], index=0)
selected_teams = st.sidebar.multiselect(
    'Teams',
    ['RedBull', 'Mercedes', 'Ferrari', 'McLaren', 'Alpine'],
    default=['RedBull', 'Mercedes', 'Ferrari', 'McLaren', 'Alpine']
)

season_summary = summary[
    (summary['season'] == selected_season) &
    (summary['team'].isin(selected_teams))
].copy()

cap = SEASON_CAPS[selected_season]

# ── KPI cards ─────────────────────────────────────────────────────
st.subheader(f'📊 {selected_season} Season Overview  |  Cap: ${cap}M')
col1, col2, col3, col4 = st.columns(4)

col1.metric('Season Cap', f'${cap}M')
col2.metric(
    '🔴 Breach (RED)',
    int((season_summary['risk'] == 'RED').sum())
)
col3.metric(
    '🟡 At Risk (AMBER)',
    int((season_summary['risk'] == 'AMBER').sum())
)
col4.metric(
    '🟢 Compliant (GREEN)',
    int((season_summary['risk'] == 'GREEN').sum())
)

st.divider()

# ── Compliance table ──────────────────────────────────────────────
st.subheader('Compliance Report')

def colour_risk(val):
    colours = {'RED': '#ffcccc', 'AMBER': '#fff3cc', 'GREEN': '#ccffcc'}
    return f'background-color: {colours.get(val, "white")}'

season_summary = season_summary.round(2)

display = season_summary[[
    'team', 'projected_full_season', 'season_cap',
    'headroom', 'pct_of_cap', 'risk', 'data_source'
]].copy()
display['projected_full_season'] = display['projected_full_season'].map(lambda x: f'{x:.2f}')
display['season_cap'] = display['season_cap'].map(lambda x: f'{x:.2f}')
display['headroom'] = display['headroom'].map(lambda x: f'{x:.2f}')
display['pct_of_cap'] = display['pct_of_cap'].map(lambda x: f'{x:.2f}%')
display = display.rename(columns={
    'team': 'Team',
    'projected_full_season': 'Projected ($M)',
    'season_cap': 'Cap ($M)',
    'headroom': 'Headroom ($M)',
    'pct_of_cap': '% of Cap',
    'risk': 'Risk',
    'data_source': 'Source',
})

for col in ['Projected ($M)', 'Cap ($M)', 'Headroom ($M)', '% of Cap']:
    display[col] = display[col].round(2)

st.dataframe(
    display.style.map(colour_risk, subset=['Risk']),
    use_container_width=True,
    hide_index=True
)

st.divider()

# ── Charts row ────────────────────────────────────────────────────
left, right = st.columns(2)

# Chart 1: % of cap bar chart
with left:
    st.subheader('% of Cap Used')
    risk_colours = {'RED': '#E8002D', 'AMBER': '#FFA500', 'GREEN': '#00A550'}
    fig1 = go.Figure()
    for _, row in season_summary.iterrows():
        fig1.add_trace(go.Bar(
            x=[row['team']],
            y=[row['pct_of_cap']],
            name=row['team'],
            marker_color=risk_colours[row['risk']],
            text=f"{row['pct_of_cap']}%",
            textposition='outside'
        ))
    fig1.add_hline(y=100, line_dash='dash', line_color='black',
                   annotation_text='Cap (100%)')
    fig1.update_layout(
        yaxis_range=[60, 115],
        yaxis_title='% of Cap',
        showlegend=False,
        height=400,
        plot_bgcolor='white'
    )
    st.plotly_chart(fig1, use_container_width=True)

# Chart 2: cumulative spend line chart
with right:
    st.subheader('Cumulative Spend vs Cap')
    team_colours = {
        'RedBull': '#3671C6', 'Mercedes': '#00D2BE',
        'Ferrari': '#E8002D', 'McLaren': '#FF8000', 'Alpine': '#FF87BC'
    }
    fig2 = go.Figure()
    season_df = df[
        (df['season'] == selected_season) &
        (df['team'].isin(selected_teams))
    ]
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    for team in selected_teams:
        tdf = season_df[season_df['team'] == team].sort_values('month')
        if tdf.empty:
            continue
        fig2.add_trace(go.Scatter(
            x=months[:len(tdf)],
            y=tdf['cumulative_spend'].round(1),
            name=team,
            mode='lines+markers',
            line=dict(color=team_colours.get(team, 'grey'), width=2.5)
        ))
    fig2.add_hline(y=cap, line_dash='dash', line_color='black',
                   annotation_text=f'Cap (${cap}M)')
    fig2.update_layout(
        yaxis_title='Cumulative Spend ($M)',
        height=400,
        plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Category breakdown ────────────────────────────────────────────
st.subheader('Spending Category Breakdown')
selected_team_detail = st.selectbox('Select team', selected_teams)

tdf = df[
    (df['team'] == selected_team_detail) &
    (df['season'] == selected_season)
].sort_values('month')

cat_colours = px.colors.sequential.Blues[1:]
fig3 = go.Figure()
for i, cat in enumerate(CATEGORIES):
    fig3.add_trace(go.Bar(
        x=months,
        y=tdf[cat].round(2),
        name=cat.replace('_', ' ').title(),
        marker_color=cat_colours[i % len(cat_colours)]
    ))
fig3.add_hline(y=cap / 12, line_dash='dash', line_color='red',
               annotation_text=f'Monthly avg cap (${cap/12:.1f}M)')
fig3.update_layout(
    barmode='stack',
    yaxis_title='Monthly Spend ($M)',
    height=400,
    plot_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02)
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Multi-season view ─────────────────────────────────────────────
st.subheader('📈 Multi-Season Compliance Trend')
fig4 = px.line(
    summary[summary['team'].isin(selected_teams)],
    x='season', y='pct_of_cap',
    color='team',
    markers=True,
    color_discrete_map=team_colours,
    labels={'pct_of_cap': '% of Cap Used', 'season': 'Season'},
)
fig4.add_hline(y=100, line_dash='dash', line_color='black',
               annotation_text='Cap (100%)')
fig4.update_layout(height=400, plot_bgcolor='white')
st.plotly_chart(fig4, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    'Data sources: FIA Financial Regulations | Motor Sport Magazine | The Race | '
    'RedBull 2021 breach: FIA Accepted Breach Agreement (Oct 2022). '
    'Built by Lakshya Agarwal — F1 Cost Cap Compliance Model'
)