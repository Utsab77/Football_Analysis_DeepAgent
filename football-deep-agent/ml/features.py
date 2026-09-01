"""Feature engineering for pre-match prediction.

Every function here must only use information available *before kickoff*
for the given match's date. That's the anti-leakage rule for this whole
module -- when in doubt, filter on `date < match_date` before aggregating.
"""
from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "home_team_strength",
    "away_team_strength",
    "home_recent_form",
    "away_recent_form",
    "home_goals_avg",
    "away_goals_avg",
    "home_conceded_avg",
    "away_conceded_avg",
    "home_advantage",
    "league_position_difference",
]


def _team_matches_before(df: pd.DataFrame, team: str, before_date, n: int | None = None) -> pd.DataFrame:
    """All matches (home or away) for `team` strictly before `before_date`."""
    mask = ((df["home_team"] == team) | (df["away_team"] == team)) & (df["date"] < before_date)
    prior = df.loc[mask].sort_values("date")
    return prior.tail(n) if n else prior


def recent_form(df: pd.DataFrame, team: str, before_date, n: int = 5) -> float:
    """Points-per-game over the last `n` matches before `before_date`. 0.0 if no history."""
    recent = _team_matches_before(df, team, before_date, n=n)
    if recent.empty:
        return 0.0
    points = 0
    for _, row in recent.iterrows():
        is_home = row["home_team"] == team
        result = row["result"]
        won = (is_home and result == "H") or (not is_home and result == "A")
        drew = result == "D"
        points += 3 if won else (1 if drew else 0)
    return points / len(recent)


def goals_avg(df: pd.DataFrame, team: str, before_date, n: int = 10, conceded: bool = False) -> float:
    """Average goals scored (or conceded, if conceded=True) over last `n` matches."""
    recent = _team_matches_before(df, team, before_date, n=n)
    if recent.empty:
        return 0.0
    total = 0.0
    for _, row in recent.iterrows():
        is_home = row["home_team"] == team
        if conceded:
            total += row["away_goals"] if is_home else row["home_goals"]
        else:
            total += row["home_goals"] if is_home else row["away_goals"]
    return total / len(recent)


def team_strength(df: pd.DataFrame, team: str, before_date, n: int = 20) -> float:
    """Placeholder rating: win rate over last `n` matches. Swap in Elo/xG-based rating later."""
    recent = _team_matches_before(df, team, before_date, n=n)
    if recent.empty:
        return 0.5
    wins = 0
    for _, row in recent.iterrows():
        is_home = row["home_team"] == team
        result = row["result"]
        if (is_home and result == "H") or (not is_home and result == "A"):
            wins += 1
    return wins / len(recent)


def league_position(df: pd.DataFrame, team: str, before_date) -> int:
    """Calculate league position based on points accumulated before `before_date`.
    
    Returns position from 1 (top) to N (bottom). If no history, returns mid-table (10).
    """
    # Get all teams that have played before this date
    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    
    # Calculate points for each team
    team_points = {}
    for t in all_teams:
        team_matches = _team_matches_before(df, t, before_date)
        points = 0
        for _, row in team_matches.iterrows():
            is_home = row["home_team"] == t
            result = row["result"]
            won = (is_home and result == "H") or (not is_home and result == "A")
            drew = result == "D"
            points += 3 if won else (1 if drew else 0)
        team_points[t] = points
    
    # Sort teams by points (descending), then alphabetically for ties
    sorted_teams = sorted(team_points.keys(), key=lambda t: (-team_points[t], t))
    
    # Return position of the requested team (1-indexed)
    if team in sorted_teams:
        return sorted_teams.index(team) + 1
    return 10  # Default to mid-table if team not found


def build_features_for_match(df: pd.DataFrame, home_team: str, away_team: str, match_date) -> dict:
    """Build the full feature dict for one upcoming match.

    `df` should be the full historical match table; this function filters
    to matches before `match_date` internally, so it's safe to pass in
    the whole processed dataset.
    """
    home_pos = league_position(df, home_team, match_date)
    away_pos = league_position(df, away_team, match_date)
    
    return {
        "home_team_strength": team_strength(df, home_team, match_date),
        "away_team_strength": team_strength(df, away_team, match_date),
        "home_recent_form": recent_form(df, home_team, match_date),
        "away_recent_form": recent_form(df, away_team, match_date),
        "home_goals_avg": goals_avg(df, home_team, match_date, conceded=False),
        "away_goals_avg": goals_avg(df, away_team, match_date, conceded=False),
        "home_conceded_avg": goals_avg(df, home_team, match_date, conceded=True),
        "away_conceded_avg": goals_avg(df, away_team, match_date, conceded=True),
        "home_advantage": 1.0,
        "league_position_difference": home_pos - away_pos,
    }


def build_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature row for every match in `df`, for model training."""
    rows = []
    for _, match in df.iterrows():
        feats = build_features_for_match(df, match["home_team"], match["away_team"], match["date"])
        feats["result"] = match["result"]
        rows.append(feats)
    return pd.DataFrame(rows)
