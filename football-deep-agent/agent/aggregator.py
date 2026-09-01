"""Result Aggregator: combines outputs from multiple sub-agents into one candidate answer.

Phase 4 (Weeks 7-8):
Aggregates Team Analysis, Historical Analysis, and Prediction outputs
before passing to the Critic Agent for review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AggregatedResult:
    """Combined result from all sub-agents."""
    task: str
    home_team: str
    away_team: str
    prediction: dict  # {home_win, draw, away_win}
    team_analysis_home: dict | None = None
    team_analysis_away: dict | None = None
    historical_analysis: dict | None = None
    scenario_analysis: dict | None = None
    confidence_score: float = 0.0
    key_factors: list[str] = field(default_factory=list)
    summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "prediction": self.prediction,
            "team_analysis_home": self.team_analysis_home,
            "team_analysis_away": self.team_analysis_away,
            "historical_analysis": self.historical_analysis,
            "scenario_analysis": self.scenario_analysis,
            "confidence_score": self.confidence_score,
            "key_factors": self.key_factors,
            "summary": self.summary,
            "timestamp": self.timestamp,
        }


def aggregate_results(
    task: str,
    home_team: str,
    away_team: str,
    prediction: dict,
    team_analysis_home: dict = None,
    team_analysis_away: dict = None,
    historical_analysis: dict = None,
    scenario_analysis: dict = None,
) -> AggregatedResult:
    """Aggregate results from all sub-agents into a single candidate answer.
    
    Args:
        task: Original task description
        home_team: Home team name
        away_team: Away team name
        prediction: ML model prediction {home_win, draw, away_win}
        team_analysis_home: Team analysis for home team (optional)
        team_analysis_away: Team analysis for away team (optional)
        historical_analysis: H2H analysis (optional)
        scenario_analysis: Scenario analysis (optional)
    
    Returns:
        AggregatedResult with combined analysis
    """
    # Extract key factors from analyses
    key_factors = []
    confidence_factors = []
    
    # From home team analysis
    if team_analysis_home:
        home_form = team_analysis_home.get("form_points_per_game", 0)
        home_pos = team_analysis_home.get("league_position", 10)
        key_factors.append(f"{home_team} form: {home_form:.1f} PPG, Position: {home_pos}")
        confidence_factors.append(0.8 if home_form > 1.5 else 0.6)
    
    # From away team analysis
    if team_analysis_away:
        away_form = team_analysis_away.get("form_points_per_game", 0)
        away_pos = team_analysis_away.get("league_position", 10)
        key_factors.append(f"{away_team} form: {away_form:.1f} PPG, Position: {away_pos}")
        confidence_factors.append(0.8 if away_form > 1.5 else 0.6)
    
    # From historical analysis
    if historical_analysis:
        total_matches = historical_analysis.get("total_matches", 0)
        if total_matches > 0:
            a_wins = historical_analysis.get("team_a_wins", 0)
            b_wins = historical_analysis.get("team_b_wins", 0)
            trend = historical_analysis.get("recent_trend", "balanced")
            key_factors.append(f"H2H: {total_matches} matches, Trend: {trend}")
            confidence_factors.append(0.7 if total_matches >= 5 else 0.5)
    
    # From scenario analysis (if any)
    if scenario_analysis:
        scenario_desc = scenario_analysis.get("scenario_description", "")
        if scenario_desc:
            key_factors.append(f"Scenario: {scenario_desc}")
            confidence_factors.append(scenario_analysis.get("confidence", 0.6))
    
    # Calculate overall confidence
    confidence_score = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    # Generate summary
    summary = _generate_aggregated_summary(
        home_team, away_team, prediction, key_factors, confidence_score
    )
    
    return AggregatedResult(
        task=task,
        home_team=home_team,
        away_team=away_team,
        prediction=prediction,
        team_analysis_home=team_analysis_home,
        team_analysis_away=team_analysis_away,
        historical_analysis=historical_analysis,
        scenario_analysis=scenario_analysis,
        confidence_score=confidence_score,
        key_factors=key_factors,
        summary=summary,
    )


def _generate_aggregated_summary(
    home_team: str,
    away_team: str,
    prediction: dict,
    key_factors: list[str],
    confidence: float,
) -> str:
    """Generate a comprehensive summary of the aggregated analysis."""
    # Determine most likely outcome
    most_likely = max(prediction, key=prediction.get)
    outcome_names = {
        "home_win": f"{home_team} win",
        "draw": "Draw",
        "away_win": f"{away_team} win",
    }
    
    prob = prediction[most_likely]
    
    # Confidence level description
    if confidence >= 0.8:
        conf_desc = "high"
    elif confidence >= 0.6:
        conf_desc = "moderate"
    else:
        conf_desc = "low"
    
    # Build summary
    parts = [
        f"Analysis Complete: {home_team} vs {away_team}",
        f"Most Likely Outcome: {outcome_names[most_likely]} ({prob:.1%})",
        f"Confidence: {conf_desc} ({confidence:.0%})",
    ]
    
    if key_factors:
        parts.append("Key Factors:")
        for factor in key_factors[:5]:  # Limit to top 5
            parts.append(f"  - {factor}")
    
    return "\n".join(parts)


def validate_aggregation(result: AggregatedResult) -> list[str]:
    """Validate the aggregated result for completeness and consistency.
    
    Returns list of issues found (empty if valid).
    """
    issues = []
    
    # Check prediction probabilities sum to ~1
    pred = result.prediction
    total = pred.get("home_win", 0) + pred.get("draw", 0) + pred.get("away_win", 0)
    if abs(total - 1.0) > 0.01:
        issues.append(f"Prediction probabilities sum to {total}, expected ~1.0")
    
    # Check required fields
    if not result.home_team or not result.away_team:
        issues.append("Missing team names")
    
    if not result.prediction:
        issues.append("Missing prediction")
    
    # Check for reasonable probabilities
    for outcome in ["home_win", "draw", "away_win"]:
        prob = pred.get(outcome, 0)
        if prob < 0 or prob > 1:
            issues.append(f"Invalid probability for {outcome}: {prob}")
    
    return issues
