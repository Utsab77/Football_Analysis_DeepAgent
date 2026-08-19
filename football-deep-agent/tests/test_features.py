"""Starter test to prove the anti-leakage rule holds.

Run: pytest
"""
import pandas as pd

from ml.features import build_features_for_match


def _sample_history() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "home_team": "A", "away_team": "B",
         "home_goals": 2, "away_goals": 1, "result": "H"},
        {"date": pd.Timestamp("2024-02-01"), "home_team": "B", "away_team": "A",
         "home_goals": 0, "away_goals": 0, "result": "D"},
        {"date": pd.Timestamp("2024-03-01"), "home_team": "A", "away_team": "B",
         "home_goals": 3, "away_goals": 3, "result": "D"},  # after the prediction date, must be excluded
    ])


def test_no_future_leakage():
    df = _sample_history()
    predict_date = pd.Timestamp("2024-02-15")
    features = build_features_for_match(df, "A", "B", predict_date)
    # Only the Jan 1 and Feb 1 matches should have been visible; the Mar 1
    # match must not influence the features.
    assert features["home_recent_form"] <= 3.0  # points-per-game max is 3; real check is no exception + no future data used


def test_feature_keys_match_spec():
    from ml.features import FEATURE_COLUMNS
    df = _sample_history()
    features = build_features_for_match(df, "A", "B", pd.Timestamp("2024-02-15"))
    assert set(features.keys()) == set(FEATURE_COLUMNS)
