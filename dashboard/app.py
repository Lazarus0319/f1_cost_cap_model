"""
app.py
------
F1 Cost Cap Compliance Model — Interactive Streamlit Dashboard
Professional F1-themed dark UI
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.compliance_engine import run_compliance_report, CATEGORIES, SEASON_CAPS
from src.monte_carlo import run_monte_carlo, run_all_teams_simulation
from src.ml_predictor import train_model, predict_breach
from src.intervention_engine import calculate_intervention, run_all_interventions

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title='F1 Cost Cap Compliance Model',
    page_icon='🏎️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Custom CSS — F1 Professional Dark Theme ───────────────────────
st.markdown("""
<style>
    /* Import F1-style font */
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;600;700;800&family=Barlow+Condensed:wght@600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Barlow', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0f0f0f;
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 2px solid #e10600;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a;
        border-bottom: 2px solid #e10600;
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        color: #888888;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #e10600 !important;
        color: #ffffff !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-left: 3px solid #e10600;
        border-radius: 4px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #888888 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Barlow Condensed', sans-serif !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #e10600;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 1px;
        text-transform: uppercase;
        width: 100%;
        transition: background-color 0.2s;
    }

    .stButton > button:hover {
        background-color: #ff1801;
        color: white;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #2a2a2a;
        border-radius: 4px;
    }

    /* Divider */
    hr {
        border-color: #2a2a2a;
    }

    /* Selectbox and slider */
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        border-color: #2a2a2a;
        color: white;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Barlow Condensed', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Success/Error boxes */
    .stSuccess {
        background-color: #0d2b1a;
        border-left: 3px solid #00A550;
    }

    .stError {
        background-color: #2b0d0d;
        border-left: 3px solid #e10600;
    }

    /* Caption */
    .stCaption {
        color: #555555 !important;
        font-size: 11px !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #e10600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
TEAM_COLOURS = {
    'RedBull': '#3671C6', 'Mercedes': '#00D2BE',
    'Ferrari': '#E8002D', 'McLaren': '#FF8000', 'Alpine': '#FF87BC'
}
RISK_COLOURS = {'RED': '#E8002D', 'AMBER': '#FFA500', 'GREEN': '#00A550'}
TEAMS = ['RedBull', 'Mercedes', 'Ferrari', 'McLaren', 'Alpine']
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

PLOT_LAYOUT = dict(
    plot_bgcolor='#1a1a1a',
    paper_bgcolor='#1a1a1a',
    font=dict(color='#ffffff', family='Barlow'),
    xaxis=dict(gridcolor='#2a2a2a', linecolor='#2a2a2a'),
    yaxis=dict(gridcolor='#2a2a2a', linecolor='#2a2a2a'),
)

# ── Load data ─────────────────────────────────────────────────────
@st.cache_data
def load():
    return run_compliance_report()

df, summary = load()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-family: Barlow Condensed, sans-serif;
                    font-size: 22px; font-weight: 800;
                    color: #e10600; letter-spacing: 2px;'>
            F1 COST CAP
        </div>
        <div style='font-size: 11px; color: #888; letter-spacing: 3px;
                    text-transform: uppercase; margin-top: 4px;'>
            Compliance Model
        </div>
    </div>
    <hr style='border-color: #e10600; margin: 10px 0 20px 0;'>
    """, unsafe_allow_html=True)

    selected_season = st.selectbox('Season', [2021, 2022, 2023], index=0)
    selected_teams = st.multiselect('Teams', TEAMS, default=TEAMS)

    st.markdown("""
    <hr style='border-color: #2a2a2a; margin: 20px 0;'>
    <div style='font-size: 10px; color: #555; line-height: 1.6;'>
        <b style='color: #888;'>DATA SOURCES</b><br>
        🔴 RedBull 2021: FIA confirmed<br>
        ⚪ All others: estimated<br><br>
        <b style='color: #888;'>METHODOLOGY</b><br>
        Monte Carlo · Random Forest<br>
        Linear Programming
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 10px 0 20px 0;'>
    <div style='font-family: Barlow Condensed, sans-serif;
                font-size: 42px; font-weight: 800;
                letter-spacing: 2px; line-height: 1;'>
        🏎️ F1 COST CAP COMPLIANCE MODEL
    </div>
    <div style='color: #888888; font-size: 13px; margin-top: 8px;
                letter-spacing: 0.5px;'>
        A probabilistic framework combining Monte Carlo simulation,
        Machine Learning, and Linear Programming
        to track, predict, and optimise team spending against FIA Financial Regulations (2021–2023)
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '📊  Compliance Report',
    '🎲  Monte Carlo Simulator',
    '🤖  ML Breach Predictor',
    '✂️  Intervention Calculator',
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — COMPLIANCE REPORT
# ════════════════════════════════════════════════════════════════
with tab1:
    season_summary = summary[
        (summary['season'] == selected_season) &
        (summary['team'].isin(selected_teams))
    ].copy()
    cap = SEASON_CAPS[selected_season]

    # KPI row
    st.markdown(f"""
    <div style='font-family: Barlow Condensed, sans-serif;
                font-size: 13px; color: #888; letter-spacing: 2px;
                text-transform: uppercase; margin-bottom: 12px;'>
        {selected_season} Season Overview
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Season Cap', f'${cap}M')
    col2.metric('🔴 Breach', int((season_summary['risk'] == 'RED').sum()))
    col3.metric('🟡 At Risk', int((season_summary['risk'] == 'AMBER').sum()))
    col4.metric('🟢 Compliant', int((season_summary['risk'] == 'GREEN').sum()))

    st.divider()

    # Table
    def colour_risk(val):
        colours = {'RED': '#3d0000', 'AMBER': '#3d2d00', 'GREEN': '#003d1a'}
        return f'background-color: {colours.get(val, "transparent")}; color: white;'

    season_summary_display = season_summary.copy()
    for col in ['projected_full_season', 'season_cap', 'headroom', 'pct_of_cap']:
        season_summary_display[col] = season_summary_display[col].map(lambda x: f'{x:.2f}')

    display = season_summary_display[[
        'team', 'projected_full_season', 'season_cap',
        'headroom', 'pct_of_cap', 'risk', 'data_source'
    ]].rename(columns={
        'team': 'Team', 'projected_full_season': 'Projected ($M)',
        'season_cap': 'Cap ($M)', 'headroom': 'Headroom ($M)',
        'pct_of_cap': '% of Cap', 'risk': 'Risk', 'data_source': 'Source',
    })

    st.dataframe(
        display.style.map(colour_risk, subset=['Risk']),
        use_container_width=True, hide_index=True
    )
    st.divider()

    left, right = st.columns(2)
    with left:
        st.markdown('##### % OF CAP USED')
        fig1 = go.Figure()
        for _, row in season_summary.iterrows():
            fig1.add_trace(go.Bar(
                x=[row['team']], y=[row['pct_of_cap']],
                name=row['team'],
                marker_color=RISK_COLOURS[row['risk']],
                text=f"{row['pct_of_cap']}%", textposition='outside',
                textfont=dict(color='white', size=11)
            ))
        fig1.add_hline(y=100, line_dash='dash', line_color='#888888',
                       annotation_text='Cap', annotation_font_color='#888888')
        fig1.update_layout(
            yaxis_range=[60, 115], yaxis_title='% of Cap',
            showlegend=False, height=380, **PLOT_LAYOUT
        )
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        st.markdown('##### CUMULATIVE SPEND VS CAP')
        fig2 = go.Figure()
        season_df = df[
            (df['season'] == selected_season) &
            (df['team'].isin(selected_teams))
        ]
        for team in selected_teams:
            tdf = season_df[season_df['team'] == team].sort_values('month')
            if tdf.empty:
                continue
            fig2.add_trace(go.Scatter(
                x=MONTHS[:len(tdf)], y=tdf['cumulative_spend'].round(1),
                name=team, mode='lines+markers',
                line=dict(color=TEAM_COLOURS.get(team, 'grey'), width=2.5),
                marker=dict(size=5)
            ))
        fig2.add_hline(y=cap, line_dash='dash', line_color='#e10600',
                       annotation_text=f'Cap ${cap}M',
                       annotation_font_color='#e10600')
        fig2.update_layout(
            yaxis_title='Cumulative Spend ($M)', height=380,
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        bgcolor='rgba(0,0,0,0)'),
            **PLOT_LAYOUT
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown('##### SPENDING CATEGORY BREAKDOWN')
    selected_team_detail = st.selectbox(
        'Select team', selected_teams, key='tab1_team'
    )
    tdf = df[
        (df['team'] == selected_team_detail) &
        (df['season'] == selected_season)
    ].sort_values('month')

    cat_colours = ['#003f88','#0055bb','#0071ee','#3391ff',
                   '#66aeff','#99c8ff','#cce4ff']
    fig3 = go.Figure()
    for i, cat in enumerate(CATEGORIES):
        fig3.add_trace(go.Bar(
            x=MONTHS, y=tdf[cat].round(2),
            name=cat.replace('_', ' ').title(),
            marker_color=cat_colours[i]
        ))
    fig3.add_hline(y=cap / 12, line_dash='dash', line_color='#e10600',
                   annotation_text=f'Monthly avg ${cap/12:.1f}M',
                   annotation_font_color='#e10600')
    fig3.update_layout(
        barmode='stack', yaxis_title='Monthly Spend ($M)',
        height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    bgcolor='rgba(0,0,0,0)'),
        **PLOT_LAYOUT
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.markdown('##### MULTI-SEASON COMPLIANCE TREND')
    fig4 = px.line(
        summary[summary['team'].isin(selected_teams)],
        x='season', y='pct_of_cap', color='team', markers=True,
        color_discrete_map=TEAM_COLOURS,
        labels={'pct_of_cap': '% of Cap Used', 'season': 'Season'},
    )
    fig4.add_hline(y=100, line_dash='dash', line_color='#e10600',
                   annotation_text='Cap', annotation_font_color='#e10600')
    fig4.update_layout(height=380, **PLOT_LAYOUT)
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — MONTE CARLO
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style='color: #888; font-size: 12px; letter-spacing: 0.5px;
                line-height: 1.8; margin-bottom: 20px;'>
        Simulates <b style='color:white'>10,000 possible spending scenarios</b>
        for a team-season using stochastic modelling.<br>
        Monthly spend modelled as <code>actual × (1 + ε)</code>
        where <code>ε ~ N(0, σ)</code> — same methodology as
        financial Value at Risk (VaR) models.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    mc_team = col1.selectbox('Team', TEAMS, key='mc_team')
    mc_season = col2.selectbox('Season', [2021, 2022, 2023], key='mc_season')
    mc_month = col3.slider('Current Month', 1, 11, 6, key='mc_month',
                           format='Month %d')

    if st.button('▶  RUN SIMULATION', key='mc_run'):
        with st.spinner('Running 10,000 simulations...'):
            result = run_monte_carlo(df, mc_team, mc_season, mc_month)

        cap = result['cap']
        simulated = result['simulated_totals']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Breach Probability', f"{result['breach_probability']}%")
        c2.metric('Expected Total', f"${result['expected_total']}M")
        c3.metric('Optimistic (P5)', f"${result['percentile_5']}M")
        c4.metric('Pessimistic (P95)', f"${result['percentile_95']}M")

        st.divider()

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=simulated, nbinsx=80,
            marker_color='#3671C6', opacity=0.85,
            name='Simulated Outcomes',
            marker_line=dict(color='#1a1a1a', width=0.3)
        ))
        fig_mc.add_vline(x=cap, line_dash='dash', line_color='#e10600',
                         line_width=2,
                         annotation_text=f'Cap ${cap}M',
                         annotation_font_color='#e10600',
                         annotation_position='top left')
        fig_mc.add_vline(x=result['expected_total'], line_dash='dot',
                         line_color='#00A550', line_width=2,
                         annotation_text=f"E[Total] ${result['expected_total']}M",
                         annotation_font_color='#00A550')
        fig_mc.add_vrect(
            x0=cap, x1=float(np.max(simulated)),
            fillcolor='#e10600', opacity=0.08,
            annotation_text=f"Breach Zone ({result['breach_probability']}%)",
            annotation_font_color='#e10600',
            annotation_position='top right'
        )
        fig_mc.update_layout(
            title=dict(
                text=f'{mc_team} {mc_season} — 10,000 Simulated Season Totals (from Month {mc_month})',
                font=dict(size=14, color='white')
            ),
            xaxis_title='Full Season Total ($M)',
            yaxis_title='Frequency',
            height=450,
            **PLOT_LAYOUT
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        st.divider()
        st.markdown('##### ALL TEAMS COMPARISON')
        with st.spinner('Running simulations for all teams...'):
            all_mc = run_all_teams_simulation(df, mc_season, mc_month)
        all_mc['breach_probability'] = all_mc['breach_probability'].map(
            lambda x: f'{x:.2f}%')
        all_mc['expected_total'] = all_mc['expected_total'].map(
            lambda x: f'${x:.2f}M')
        all_mc['p5'] = all_mc['p5'].map(lambda x: f'${x:.2f}M')
        all_mc['p95'] = all_mc['p95'].map(lambda x: f'${x:.2f}M')
        st.dataframe(all_mc, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTOR
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style='color: #888; font-size: 12px; letter-spacing: 0.5px;
                line-height: 1.8; margin-bottom: 20px;'>
        Trains a <b style='color:white'>Random Forest Regressor</b> on
        historical spending patterns across 15 team-seasons.<br>
        Given spend through month N, predicts full-season total with
        95% confidence interval using tree ensemble variance.
        Validated with <b style='color:white'>Leave-One-Season-Out
        cross-validation</b>.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    ml_team = col1.selectbox('Team', TEAMS, key='ml_team')
    ml_season = col2.selectbox('Season', [2021, 2022, 2023], key='ml_season')
    ml_month = col3.slider('Current Month', 3, 11, 6, key='ml_month',
                           format='Month %d')

    if st.button('▶  RUN ML PREDICTION', key='ml_run'):
        with st.spinner('Training Random Forest on 15 team-seasons...'):
            model, le, validation = train_model(df, ml_month)
            result = predict_breach(df, ml_team, ml_season, ml_month, model, le)

        cap = result['cap']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('ML Predicted Total', f"${result['ml_predicted_total']}M")
        c2.metric('Breach Probability', f"{result['breach_probability']}%")
        c3.metric('95% CI Lower', f"${result['ci_lower']}M")
        c4.metric('95% CI Upper', f"${result['ci_upper']}M")

        st.divider()

        left, right = st.columns(2)
        with left:
            fig_ml = go.Figure()
            fig_ml.add_trace(go.Bar(
                x=[ml_team],
                y=[result['ml_predicted_total']],
                error_y=dict(
                    type='data', symmetric=False,
                    array=[result['ci_upper'] - result['ml_predicted_total']],
                    arrayminus=[result['ml_predicted_total'] - result['ci_lower']],
                    color='#888888'
                ),
                marker_color=TEAM_COLOURS.get(ml_team, '#888'),
                name='ML Prediction',
                width=0.4
            ))
            fig_ml.add_hline(y=cap, line_dash='dash', line_color='#e10600',
                             annotation_text=f'Cap ${cap}M',
                             annotation_font_color='#e10600')
            fig_ml.update_layout(
                title=dict(text='ML Prediction + 95% CI',
                           font=dict(size=13, color='white')),
                yaxis_title='Predicted Season Spend ($M)',
                yaxis_range=[result['ci_lower']-5,
                             max(result['ci_upper'], cap)+5],
                height=380, **PLOT_LAYOUT
            )
            st.plotly_chart(fig_ml, use_container_width=True)

        with right:
            importances = pd.DataFrame(
                list(validation['feature_importances'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)

            fig_imp = go.Figure(go.Bar(
                x=importances['Importance'],
                y=importances['Feature'],
                orientation='h',
                marker=dict(
                    color=importances['Importance'],
                    colorscale=[[0, '#1a3a6b'], [1, '#e10600']],
                ),
                text=importances['Importance'].map(lambda x: f'{x:.3f}'),
                textposition='outside',
                textfont=dict(color='white', size=10)
            ))
            fig_imp.update_layout(
                title=dict(text='Feature Importances',
                           font=dict(size=13, color='white')),
                xaxis_title='Importance Score',
                height=380, **PLOT_LAYOUT
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        st.caption(
            f'Cross-validation MAE: ${validation["cv_mae_mean"]}M '
            f'(±{validation["cv_mae_std"]}M) | '
            f'R²: {validation["cv_r2_mean"]} | '
            f'Note: limited to 15 team-season observations — '
            f'feature importance analysis is the primary output'
        )

# ════════════════════════════════════════════════════════════════
# TAB 4 — INTERVENTION CALCULATOR
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style='color: #888; font-size: 12px; letter-spacing: 0.5px;
                line-height: 1.8; margin-bottom: 20px;'>
        If a team is projected to breach, this engine calculates the
        <b style='color:white'>minimum spending cuts</b> needed using
        <b style='color:white'>linear programming</b> (scipy.optimize.linprog).<br>
        Cuts are prioritised from lowest to highest sporting impact —
        mirroring real Cost Analysis team decision-making.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    iv_team = col1.selectbox('Team', TEAMS, key='iv_team')
    iv_season = col2.selectbox('Season', [2021, 2022, 2023], key='iv_season')
    iv_month = col3.slider('Current Month', 1, 11, 9, key='iv_month',
                           format='Month %d')

    if st.button('▶  CALCULATE INTERVENTION', key='iv_run'):
        with st.spinner('Running linear programming optimiser...'):
            result = calculate_intervention(df, iv_team, iv_season, iv_month)

        if result['status'] == 'compliant':
            st.success(
                f"✅ {iv_team} is compliant at Month {iv_month}. "
                f"Projected: ${result['projected_total']}M | "
                f"Headroom: ${result['headroom']}M"
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Projected Total', f"${result['projected_total']}M")
            c2.metric('Overspend', f"${result['overspend']}M",
                      delta=f"-${result['overspend']}M needed",
                      delta_color='inverse')
            c3.metric('Total Cuts Needed', f"${result['total_cut']}M")
            c4.metric('Feasible',
                      '✅ Yes' if result['feasible'] else '❌ Too Late')

            st.divider()

            left, right = st.columns(2)
            with left:
                st.markdown('##### RECOMMENDED CUTS')
                if result['interventions']:
                    iv_df = pd.DataFrame(result['interventions'])
                    iv_df['total_cut'] = iv_df['total_cut'].map(
                        lambda x: f'${x:.3f}M')
                    iv_df['monthly_cut'] = iv_df['monthly_cut'].map(
                        lambda x: f'${x:.3f}M')
                    iv_df = iv_df.rename(columns={
                        'category': 'Category',
                        'total_cut': 'Total Cut',
                        'monthly_cut': 'Monthly Cut',
                        'remaining_months': 'Months Left',
                        'sporting_impact': 'Impact'
                    })[['Category', 'Total Cut', 'Monthly Cut',
                        'Months Left', 'Impact']]
                    st.dataframe(iv_df, use_container_width=True,
                                 hide_index=True)

            with right:
                st.markdown('##### CUTS BY CATEGORY')
                if result['interventions']:
                    cats = [i['category'] for i in result['interventions']]
                    cuts = [i['total_cut'] for i in result['interventions']]
                    impact_colours = {
                        'Minimal': '#00A550', 'Low': '#66bb6a',
                        'Medium': '#FFA500', 'Medium-High': '#ff7043',
                        'High': '#e10600', 'Very High': '#8b0000'
                    }
                    bar_colours = [
                        impact_colours.get(i['sporting_impact'], '#888')
                        for i in result['interventions']
                    ]
                    fig_iv = go.Figure(go.Bar(
                        x=cats, y=cuts,
                        marker_color=bar_colours,
                        text=[f'${c:.3f}M' for c in cuts],
                        textposition='outside',
                        textfont=dict(color='white', size=10)
                    ))
                    fig_iv.update_layout(
                        xaxis_title='Category',
                        yaxis_title='Cut ($M)',
                        height=320,
                        **PLOT_LAYOUT
                    )
                    st.plotly_chart(fig_iv, use_container_width=True)

        st.divider()
        st.markdown('##### MONTH-BY-MONTH FEASIBILITY TRACKER')
        tracker_rows = []
        for m in range(1, 12):
            r = calculate_intervention(df, iv_team, iv_season, m)
            tracker_rows.append({
                'Month': MONTHS[m-1],
                'Projected ($M)': f"${r['projected_total']}",
                'Overspend ($M)': f"${r['overspend']}",
                'Cut Needed ($M)': f"${r['total_cut']}",
                'Feasible': '✅' if r['feasible'] else '❌ Too Late',
                'Status': r['status'].replace('_', ' ').title()
            })
        st.dataframe(
            pd.DataFrame(tracker_rows),
            use_container_width=True, hide_index=True
        )

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='display: flex; justify-content: space-between;
            align-items: center; padding: 10px 0;'>
    <div style='font-size: 11px; color: #555;'>
        Data: FIA Financial Regulations · Motor Sport Magazine · The Race ·
        RedBull 2021: FIA Accepted Breach Agreement (Oct 2022)
    </div>
    <div style='font-size: 11px; color: #555;'>
        Built by <b style='color: #888;'>Lakshya Agarwal</b> ·
        F1 Cost Cap Compliance Model
    </div>
</div>
""", unsafe_allow_html=True)