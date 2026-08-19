"""Sub-agent: Historical Analysis Agent.

Phase 4 (Weeks 7-8):
Analyzes head-to-head records and historical trends between two teams.
Returns compact result for aggregation by the main agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ml.preprocessing import PROCESSED_DIR


@dataclass
class HistoricalAnalysisResult:
    """Result from historical analysis sub-agent."""
    team_a: str
    team_b: str
    total_matches: int
    team_a_wins: int
    team_b_wins: int
    draws: int
    team_a_goals_avg: float
    team_b_goals_avg: float
    recent_trend: str  # "team_a_dominant", "team_b_dominant", "balanced"
    notable_results: list[dict] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> dict:
        return {
            "team_a": self.team_a,
            "team_b": self.team_b,
            "total_matches": self.total_matches,
            "team_a_wins": self.team_a_wins,
            "team_b_wins": self.team_b_wins,
            "draws": self.draws,
            "team_a_goals_avg": self.team_a_goals_avg,
            "team_b_goals_avg": self.team_b_goals_avg,
            "recent_trend": self.recent_trend,
            "notable_results": self.notable_results,
            "summary": self.summary,
        }


def run(context: dict) -> dict:
    """Run historical analysis sub-agent.
    
    Args:
        context: {
            "team_a": str - first team name,
            "team_b": str - second team name,
            "before_date": str - ISO date string,
            "n_matches": int - number of H2H matches to analyze (default: 20)
        }
    
    Returns:
        HistoricalAnalysisResult as dict
    """
    team_a = context["team_a"]
    team_b = context["team_b"]
    before_date = pd.Timestamp(context["before_date"])
    n_matches = context.get("n_matches", 20)
    
    # Load match data
    df = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    
    # Get head-to-head matches
    h2h_mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    ) & (df["date"] < before_date)
    
    h2h_matches = df[h2h_mask].sort_values("date").tail(n_matches)
    
    # Calculate statistics
    total_matches = len(h2h_matches)
    team_a_wins = 0
    team_b_wins = 0
    draws = 0
    team_a_goals = 0
    team_b_goals = 0
    
    notable_results = []
    
    for _, match in h2h_matches.iterrows():
        is_a_home = match["home_team"] == team_a
        
        # Get goals for each team
        a_goals = match["home_goals"] if is_a_home else match["away_goals"]
        b_goals = match["away_goals"] if is_a_home else match["home_goals"]
        
        team_a_goals += a_goals
        team_b_goals += b_goals
        
        # Determine winner
        if match["result"] == "D":
            draws += 1
        elif (is_a_home and match["result"] == "H") or (not is_a_home and match["result"] == "A"):
            team_a_wins += 1
        else:
            team_b_wins += 1
        
        # Track notable results (high-scoring or significant wins)
        if abs(a_goals - b_goals) >= 2 or (a_goals + b_goals) >= 4:
            notable_results.append({
                "date": str(match["date"].date()),
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "score": f"{match['home_goals']}-{match['away_goals']}",
                "winner": team_a if (
                    (is_a_home and match["result"] == "H") or 
                    (not is_a_home and match["result"] == "A")
                ) else team_b,
            })
    
    # Calculate averages
    team_a_goals_avg = team_a_goals / total_matches if total_matches > 0 else 0.0
    team_b_goals_avg = team_b_goals / total_matches if total_matches > 0 else 0.0
    
    # Determine recent trend
    recent_trend = _determine_trend(team_a_wins, team_b_wins, draws)
    
    # Generate summary
    summary = _generate_h2h_summary(
        team_a, team_b, total_matches, team_a_wins, team_b_wins, 
        draws, team_a_goals_avg, team_b_goals_avg, recent_trend
    )
    
    result = HistoricalAnalysisResult(
        team_a=team_a,
        team_b=team_b,
        total_matches=total_matches,
        team_a_wins=team_a_wins,
        team_b_wins=team_b_wins,
        draws=draws,
        team_a_goals_avg=team_a_goals_avg,
        team_b_goals_avg=team_b_goals_avg,
        recent_trend=recent_trend,
        notable_results=notable_results[-5:],  # Last 5 notable results
        summary=summary,
    )
    
    return result.to_dict()


def _determine_trend(a_wins: int, b_wins: int, draws: int) -> str:
    """Determine the recent trend in head-to-head matches."""
    if a_wins > b_wins * 1.5:
        return "team_a_dominant"
    elif b_wins > a_wins * 1.5:
        return "team_b_dominant"
    else:
        return "balanced"


def _generate_h2h_summary(
    team_a: str, team_b: str, total: int, a_wins: int, b_wins: int,
    draws: int, a_goals: float, b_goals: float, trend: str
) -> str:
    """Generate a human-readable summary of H2H analysis."""
    if total == 0:
        return f"No previous matches found between {team_a} and {team_b}."
    
    trend_desc = {
        "team_a_dominant": f"{team_a} have dominated recent meetings",
        "team_b_dominant": f"{team_b} have dominated recent meetings",
        "balanced": "The rivalry has been closely contested",
    }[trend]
    
    return (
        f"In {total} meetings: {team_a} won {a_wins}, {team_b} won {b_wins}, "
        f"{draws} draws. {team_a} average {a_goals:.1f} goals/game, "
        f"{team_b} average {b_goals:.1f} goals/game. {trend_desc}."
    )
