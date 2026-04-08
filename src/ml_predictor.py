"""
ml_predictor.py
---------------
Machine Learning Breach Predictor for F1 Cost Cap Compliance.

Methodology:
- Trains a Random Forest Regressor on monthly spending patterns
- Given spend through month N, predicts full-season total
- Uses Leave-One-Season-Out cross validation (robust with small dataset)
- Outputs: predicted final spend + confidence interval + breach probability

Features used:
- team (encoded)
- season
- current_month
- cumulative_spend_to_date
- spend_velocity (average monthly spend rate)
- pct_of_cap_used (how much of cap consumed so far)
- category_ratios (what % of spend is personnel vs chassis etc)

Why Random Forest:
- Handles non-linear relationships (spend doesn't scale linearly)
- Works with small datasets (15 team-seasons)
- Naturally provides feature importance — tells us WHAT drives breaches
- More robust than linear regression for this use case
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from src.compliance_engine import run_compliance_report, CATEGORIES, SEASON_CAPS


def build_features(df, current_month):
    """
    Build feature matrix for ML model.
    Each row = one team-season, observed up to current_month.

    Features:
    - team_encoded        : integer encoding of team name
    - season_norm         : normalised season (2021=0, 2022=0.5, 2023=1)
    - current_month       : which month we're predicting from
    - cumulative_spend    : total spend M1 to current_month
    - spend_velocity      : average monthly spend rate
    - pct_cap_used        : % of cap consumed so far
    - personnel_ratio     : personnel as % of total spend so far
    - chassis_ratio       : chassis_aero as % of total spend so far
    - operations_ratio    : operations as % of total spend so far
    - late_season         : binary flag if current_month >= 9
    """
    le = LabelEncoder()
    teams = sorted(df['team'].unique())
    le.fit(teams)

    rows = []
    targets = []

    for (team, season), group in df.groupby(['team', 'season']):
        group = group.sort_values('month')
        cap = SEASON_CAPS[season]

        # Full season total (target)
        group['monthly_total'] = group[CATEGORIES].sum(axis=1)
        full_season_total = group['monthly_total'].sum()

        # Observed data up to current_month
        observed = group[group['month'] <= current_month]
        if observed.empty:
            continue

        cumulative_spend = observed['monthly_total'].sum()
        spend_velocity = cumulative_spend / current_month
        pct_cap_used = (cumulative_spend / cap) * 100

        # Category ratios
        personnel_ratio = observed['personnel'].sum() / cumulative_spend
        chassis_ratio = observed['chassis_aero'].sum() / cumulative_spend
        operations_ratio = observed['operations'].sum() / cumulative_spend

        # Season normalisation
        season_norm = (season - 2021) / 2.0

        features = {
            'team_encoded': le.transform([team])[0],
            'season_norm': season_norm,
            'current_month': current_month,
            'cumulative_spend': cumulative_spend,
            'spend_velocity': spend_velocity,
            'pct_cap_used': pct_cap_used,
            'personnel_ratio': personnel_ratio,
            'chassis_ratio': chassis_ratio,
            'operations_ratio': operations_ratio,
            'late_season': int(current_month >= 9),
        }

        rows.append(features)
        targets.append(full_season_total)

    X = pd.DataFrame(rows)
    y = np.array(targets)
    return X, y, le


def train_model(df, current_month):
    """
    Train Random Forest on all available team-season data.
    Uses Leave-One-Season-Out cross validation.

    Returns trained model + validation metrics.
    """
    X, y, le = build_features(df, current_month)

    if len(X) < 5:
        raise ValueError(f"Not enough data to train model at month {current_month}")

    # Leave-One-Season-Out Cross Validation
    seasons = [2021, 2022, 2023]
    cv_maes = []
    cv_r2s = []

    for test_season in seasons:
        # Split: train on other seasons, test on this season
        season_norms = {2021: 0.0, 2022: 0.5, 2023: 1.0}
        test_mask = X['season_norm'] == season_norms[test_season]
        train_mask = ~test_mask

        if train_mask.sum() < 3:
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else float('nan')
        cv_maes.append(mae)
        cv_r2s.append(r2)

    # Train final model on ALL data
    final_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42
    )
    final_model.fit(X, y)

    validation = {
        'cv_mae_mean': round(np.mean(cv_maes), 3),
        'cv_mae_std': round(np.std(cv_maes), 3),
        'cv_r2_mean': round(np.nanmean(cv_r2s), 3),
        'feature_names': list(X.columns),
        'feature_importances': dict(zip(
            X.columns,
            final_model.feature_importances_.round(4)
        )),
    }

    return final_model, le, validation


def predict_breach(df, team, season, current_month, model=None, le=None):
    """
    Predict full-season spend and breach probability for a team-season
    given data up to current_month.

    Uses Random Forest prediction + Monte Carlo uncertainty quantification.
    """
    from src.monte_carlo import estimate_sigma

    if model is None or le is None:
        model, le, _ = train_model(df, current_month)

    cap = SEASON_CAPS[season]
    group = df[(df['team'] == team) & (df['season'] == season)].sort_values('month')
    group = group.copy()
    group['monthly_total'] = group[CATEGORIES].sum(axis=1)

    observed = group[group['month'] <= current_month]
    cumulative_spend = observed['monthly_total'].sum()
    spend_velocity = cumulative_spend / current_month
    pct_cap_used = (cumulative_spend / cap) * 100
    season_norm = (season - 2021) / 2.0

    personnel_ratio = observed['personnel'].sum() / cumulative_spend
    chassis_ratio = observed['chassis_aero'].sum() / cumulative_spend
    operations_ratio = observed['operations'].sum() / cumulative_spend

    features = pd.DataFrame([{
        'team_encoded': le.transform([team])[0],
        'season_norm': season_norm,
        'current_month': current_month,
        'cumulative_spend': cumulative_spend,
        'spend_velocity': spend_velocity,
        'pct_cap_used': pct_cap_used,
        'personnel_ratio': personnel_ratio,
        'chassis_ratio': chassis_ratio,
        'operations_ratio': operations_ratio,
        'late_season': int(current_month >= 9),
    }])

    # ML point prediction
    ml_prediction = model.predict(features)[0]

    # Uncertainty via individual tree predictions
    tree_predictions = np.array([
        tree.predict(features)[0]
        for tree in model.estimators_
    ])

    prediction_std = tree_predictions.std()
    ci_lower = ml_prediction - 1.96 * prediction_std
    ci_upper = ml_prediction + 1.96 * prediction_std

    # Breach probability from tree distribution
    breach_probability = (tree_predictions > cap).mean() * 100

    return {
        'team': team,
        'season': season,
        'current_month': current_month,
        'cumulative_spend': round(cumulative_spend, 2),
        'ml_predicted_total': round(ml_prediction, 2),
        'ci_lower': round(ci_lower, 2),
        'ci_upper': round(ci_upper, 2),
        'prediction_std': round(prediction_std, 3),
        'breach_probability': round(breach_probability, 2),
        'cap': cap,
        'headroom': round(cap - ml_prediction, 2),
        'pct_of_cap': round((ml_prediction / cap) * 100, 2),
    }


if __name__ == '__main__':
    df, summary = run_compliance_report()

    print('\n=== ML Predictor — Training & Validation (Month 6) ===\n')
    model, le, validation = train_model(df, current_month=6)

    print(f"Cross-validation MAE: ${validation['cv_mae_mean']}M "
          f"(±{validation['cv_mae_std']}M)")
    print(f"Cross-validation R²:  {validation['cv_r2_mean']}")

    print('\nFeature Importances (what drives full-season spend):')
    importances = sorted(
        validation['feature_importances'].items(),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in importances:
        bar = '█' * int(imp * 40)
        print(f"  {feat:<22} {imp:.4f}  {bar}")

    print('\n=== Predictions for All Teams — 2021 Season at Month 6 ===\n')
    print(f"{'Team':<12} {'Predicted':>12} {'CI Lower':>10} "
          f"{'CI Upper':>10} {'Cap':>8} {'Breach%':>9} {'Headroom':>10}")
    print('-' * 80)

    for team in ['RedBull', 'Mercedes', 'Ferrari', 'McLaren', 'Alpine']:
        result = predict_breach(df, team, 2021, 6, model, le)
        print(
            f"{result['team']:<12} "
            f"${result['ml_predicted_total']:>10.2f}M "
            f"${result['ci_lower']:>8.2f}M "
            f"${result['ci_upper']:>8.2f}M "
            f"${result['cap']:>6.1f}M "
            f"{result['breach_probability']:>8.2f}% "
            f"${result['headroom']:>8.2f}M"
        )

    print('\n=== Month-by-Month Breach Probability — RedBull 2021 ===\n')
    print(f"{'Month':<8} {'Predicted Total':>16} {'Breach Prob':>13} {'Status'}")
    print('-' * 50)
    for month in range(3, 13):
        m, le_m, _ = train_model(df, current_month=month)
        r = predict_breach(df, 'RedBull', 2021, month, m, le_m)
        status = '🔴 BREACH RISK' if r['breach_probability'] > 50 else \
                 '🟡 WATCH' if r['breach_probability'] > 20 else '🟢 OK'
        print(f"  M{month:<6} ${r['ml_predicted_total']:>14.2f}M "
              f"{r['breach_probability']:>12.2f}%  {status}")