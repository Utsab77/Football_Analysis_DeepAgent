"""Tool: get_league_position() - get team's league position before a date."""
from __future__ import annotations

import pandas as pd

from ml.features import league_position
from ml.preprocessing import PROCESSED_DIR


def get_league_position(team: str, before_date) -> dict:
    """Get the league position of a team as of a given date.
    
    Args:
        team: Team name
        before_date: Calculate position based on matches before this date
        
    Returns:
        dict with team name and position (1 = top)
    """
    df = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    position = league_position(df, team, before_date)
    return {
        "team": team,
        "position": position,
        "before_date": str(before_date),
    }


def get_full_league_table(before_date) -> list[dict]:
    """Get the full league table as of a given date.
    
    Args:
        before_date: Calculate standings based on matches before this date
        
    Returns:
        list of dicts with team, points, and position
    """
    df = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    
    # Get all teams
    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    
    # Calculate points for each team
    team_stats = []
    for team in all_teams:
        team_matches = df[
            ((df["home_team"] == team) | (df["away_team"] == team)) & 
            (df["date"] < before_date)
        ]
        
        points = 0
        wins = 0
        draws = 0
        losses = 0
        goals_scored = 0
        goals_conceded = 0
        
        for _, row in team_matches.iterrows():
            is_home = row["home_team"] == team
            result = row["result"]
            
            if is_home:
                goals_scored += row["home_goals"]
                goals_conceded += row["away_goals"]
            else:
                goals_scored += row["away_goals"]
                goals_conceded += row["home_goals"]
            
            won = (is_home and result == "H") or (not is_home and result == "A")
            drew = result == "D"
            
            if won:
                wins += 1
                points += 3
            elif drew:
                draws += 1
                points += 1
            else:
                losses += 1
        
        team_stats.append({
            "team": team,
            "played": wins + draws + losses,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "goal_difference": goals_scored - goals_conceded,
            "points": points,
        })
    
    # Sort by points, then goal difference, then goals scored
    team_stats.sort(key=lambda x: (-x["points"], -x["goal_difference"], -x["goals_scored"]))
    
    # Add position
    for i, team in enumerate(team_stats):
        team["position"] = i + 1
    
    return team_stats
