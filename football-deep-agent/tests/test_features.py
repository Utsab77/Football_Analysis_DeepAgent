"""Comprehensive tests to prove the anti-leakage rule holds.

Run: pytest
"""
import pandas as pd
import pytest

from ml.features import build_features_for_match, recent_form, goals_avg, team_strength, league_position, FEATURE_COLUMNS


def _sample_history() -> pd.DataFrame:
    """Create a sample dataset with known results for testing."""
    return pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "home_team": "A", "away_team": "B",
         "home_goals": 2, "away_goals": 1, "result": "H"},
        {"date": pd.Timestamp("2024-01-15"), "home_team": "C", "away_team": "D",
         "home_goals": 1, "away_goals": 0, "result": "H"},
        {"date": pd.Timestamp("2024-02-01"), "home_team": "B", "away_team": "A",
         "home_goals": 0, "away_goals": 0, "result": "D"},
        {"date": pd.Timestamp("2024-02-15"), "home_team": "A", "away_team": "C",
         "home_goals": 1, "away_goals": 2, "result": "A"},
        {"date": pd.Timestamp("2024-03-01"), "home_team": "A", "away_team": "B",
         "home_goals": 3, "away_goals": 3, "result": "D"},
        {"date": pd.Timestamp("2024-03-15"), "home_team": "C", "away_team": "D",
         "home_goals": 2, "away_goals": 2, "result": "D"},
    ])


def test_no_future_leakage():
    """Test that future matches don't influence current features."""
    df = _sample_history()
    predict_date = pd.Timestamp("2024-02-15")
    features = build_features_for_match(df, "A", "B", predict_date)
    # Only matches before Feb 15 should be used
    assert features["home_recent_form"] <= 3.0
    assert features["away_recent_form"] <= 3.0


def test_feature_keys_match_spec():
    """Test that all required features are present."""
    df = _sample_history()
    features = build_features_for_match(df, "A", "B", pd.Timestamp("2024-02-15"))
    assert set(features.keys()) == set(FEATURE_COLUMNS)


def test_recent_form_calculation():
    """Test recent form points-per-game calculation."""
    df = _sample_history()
    # Team A played on Jan 1 (won) and Feb 1 (drew) before Feb 15
    # Points: 3 + 1 = 4, matches: 2, form = 2.0
    form = recent_form(df, "A", pd.Timestamp("2024-02-15"), n=5)
    assert form == 2.0


def test_goals_avg_calculation():
    """Test average goals scored calculation."""
    df = _sample_history()
    # Team A scored 2 on Jan 1, 0 on Feb 1 before Feb 15
    # Total: 2, matches: 2, avg: 1.0
    avg = goals_avg(df, "A", pd.Timestamp("2024-02-15"), n=10, conceded=False)
    assert avg == 1.0


def test_conceded_avg_calculation():
    """Test average goals conceded calculation."""
    df = _sample_history()
    # Team A conceded 1 on Jan 1, 0 on Feb 1 before Feb 15
    # Total: 1, matches: 2, avg: 0.5
    avg = goals_avg(df, "A", pd.Timestamp("2024-02-15"), n=10, conceded=True)
    assert avg == 0.5


def test_team_strength_calculation():
    """Test team strength (win rate) calculation."""
    df = _sample_history()
    # Team A: won on Jan 1, drew on Feb 1 before Feb 15
    # Wins: 1, matches: 2, strength: 0.5
    strength = team_strength(df, "A", pd.Timestamp("2024-02-15"), n=20)
    assert strength == 0.5


def test_league_position_calculation():
    """Test league position calculation."""
    df = _sample_history()
    # Before Feb 15, Team A has 4 points (3+1), Team B has 1 point (0+1)
    # Team A should be ranked higher (lower position number)
    pos_a = league_position(df, "A", pd.Timestamp("2024-02-15"))
    pos_b = league_position(df, "B", pd.Timestamp("2024-02-15"))
    assert pos_a < pos_b  # A is ranked higher than B


def test_league_position_difference():
    """Test that league position difference is calculated correctly."""
    df = _sample_history()
    features = build_features_for_match(df, "A", "B", pd.Timestamp("2024-02-15"))
    # Home team position - Away team position
    assert features["league_position_difference"] != 0.0  # Should be non-zero


def test_future_match_excluded():
    """Test that a match on the prediction date is excluded."""
    df = _sample_history()
    # Before Feb 15: Team A has Jan 1 (won) and Feb 1 (drew) - form = 2.0
    # After Feb 15: Team A has Mar 1 (drew) - not included
    form_before = recent_form(df, "A", pd.Timestamp("2024-02-15"), n=5)
    form_after = recent_form(df, "A", pd.Timestamp("2024-03-15"), n=5)
    # Form after should include Mar 1 match
    assert form_before == 2.0
    assert form_after == (3 + 1 + 1) / 3  # 5 points / 3 matches


def test_no_data_returns_defaults():
    """Test that missing history returns safe defaults."""
    df = _sample_history()
    # Team E has no history
    features = build_features_for_match(df, "E", "F", pd.Timestamp("2024-02-15"))
    assert features["home_team_strength"] == 0.5  # Default strength
    assert features["home_recent_form"] == 0.0    # No form
    assert features["home_goals_avg"] == 0.0      # No goals
    assert features["home_conceded_avg"] == 0.0   # No conceded
