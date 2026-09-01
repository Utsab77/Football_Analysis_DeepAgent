"""Sub-agent: Team Analysis Agent.

Phase 4 (Weeks 7-8):
Analyzes a single team's recent performance, home/away splits, goals, and strength.
Returns compact result for aggregation by the main agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ml.features import recent_form, goals_avg, team_strength, league_position
from ml.preprocessing import PROCESSED_DIR


@dataclass
class TeamAnalysisResult:
    """Result from team analysis sub-agent."""
    team: str
    form_points_per_game: float
    goals_scored_avg: float
    goals_conceded_avg: float
    strength: float
    league_position: int
    home_form: float = 0.0
    away_form: float = 0.0
    recent_matches: list[dict] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "form_points_per_game": self.form_points_per_game,
            "goals_scored_avg": self.goals_scored_avg,
            "goals_conceded_avg": self.goals_conceded_avg,
            "strength": self.strength,
            "league_position": self.league_position,
            "home_form": self.home_form,
            "away_form": self.away_form,
            "recent_matches": self.recent_matches,
            "summary": self.summary,
        }


def run(context: dict) -> dict:
    """Run team analysis sub-agent.
    
    Args:
        context: {
            "team": str - team name,
            "before_date": str - ISO date string,
            "n_matches": int - number of recent matches to analyze (default: 10)
        }
    
    Returns:
        TeamAnalysisResult as dict
    """
    team = context["team"]
    before_date = pd.Timestamp(context["before_date"])
    n_matches = context.get("n_matches", 10)
    
    # Load match data
    df = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    
    # Calculate metrics
    form = recent_form(df, team, before_date, n=5)
    goals = goals_avg(df, team, before_date, n=n_matches, conceded=False)
    conceded = goals_avg(df, team, before_date, n=n_matches, conceded=True)
    strength = team_strength(df, team, before_date, n=20)
    position = league_position(df, team, before_date)
    
    # Calculate home/away splits
    home_matches = df[
        (df["home_team"] == team) & (df["date"] < before_date)
    ].tail(5)
    away_matches = df[
        (df["away_team"] == team) & (df["date"] < before_date)
    ].tail(5)
    
    home_form = _calculate_form_from_matches(home_matches, team, is_home=True)
    away_form = _calculate_form_from_matches(away_matches, team, is_home=False)
    
    # Get recent matches for context
    recent = df[
        ((df["home_team"] == team) | (df["away_team"] == team)) & 
        (df["date"] < before_date)
    ].sort_values("date").tail(n_matches)
    
    recent_matches = []
    for _, match in recent.iterrows():
        is_home = match["home_team"] == team
        recent_matches.append({
            "date": str(match["date"].date()),
            "opponent": match["away_team"] if is_home else match["home_team"],
            "venue": "Home" if is_home else "Away",
            "goals_scored": match["home_goals"] if is_home else match["away_goals"],
            "goals_conceded": match["away_goals"] if is_home else match["home_goals"],
            "result": match["result"],
        })
    
    # Generate summary
    summary = _generate_team_summary(team, form, goals, conceded, strength, position)
    
    result = TeamAnalysisResult(
        team=team,
        form_points_per_game=form,
        goals_scored_avg=goals,
        goals_conceded_avg=conceded,
        strength=strength,
        league_position=position,
        home_form=home_form,
        away_form=away_form,
        recent_matches=recent_matches,
        summary=summary,
    )
    
    return result.to_dict()


def _calculate_form_from_matches(matches: pd.DataFrame, team: str, is_home: bool) -> float:
    """Calculate points per game from a set of matches."""
    if matches.empty:
        return 0.0
    
    points = 0
    for _, row in matches.iterrows():
        result = row["result"]
        if is_home:
            won = result == "H"
            drew = result == "D"
        else:
            won = result == "A"
            drew = result == "D"
        
        points += 3 if won else (1 if drew else 0)
    
    return points / len(matches)


def _generate_team_summary(
    team: str, form: float, goals: float, conceded: float, 
    strength: float, position: int
) -> str:
    """Generate a human-readable summary of team analysis."""
    form_desc = "excellent" if form >= 2.0 else "good" if form >= 1.5 else "poor" if form < 1.0 else "average"
    attack_desc = "strong" if goals >= 1.5 else "decent" if goals >= 1.0 else "weak"
    defense_desc = "solid" if conceded <= 0.8 else "leaky" if conceded >= 1.5 else "average"
    
    return (
        f"{team} are in {form_desc} form ({form:.1f} PPG), "
        f"ranked {position}th in the league. "
        f"They have a {attack_desc} attack ({goals:.1f} goals/game) "
        f"and {defense_desc} defense ({conceded:.1f} conceded/game). "
        f"Overall strength rating: {strength:.2f}."
    )
