"""Sub-agent: Scenario Agent.

Phase 4 (Weeks 7-8):
Evaluates hypothetical changes (player absence, formation change) 
and describes expected impact on match outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ml.features import build_features_for_match
from ml.predict import predict_match
from ml.preprocessing import PROCESSED_DIR


@dataclass
class ScenarioResult:
    """Result from scenario analysis sub-agent."""
    scenario_description: str
    base_prediction: dict
    adjusted_prediction: dict
    impact_analysis: dict
    confidence: float  # 0-1, how confident we are in the impact assessment
    summary: str = ""
    
    def to_dict(self) -> dict:
        return {
            "scenario_description": self.scenario_description,
            "base_prediction": self.base_prediction,
            "adjusted_prediction": self.adjusted_prediction,
            "impact_analysis": self.impact_analysis,
            "confidence": self.confidence,
            "summary": self.summary,
        }


# Scenario impact weights (simplified model)
SCENARIO_IMPACTS = {
    "key_player_absent": {
        "home_win": -0.08,
        "draw": 0.03,
        "away_win": 0.05,
        "description": "Key player absence weakens attacking threat",
    },
    "goalkeeper_absent": {
        "home_win": -0.12,
        "draw": 0.02,
        "away_win": 0.10,
        "description": "Goalkeeper absence significantly weakens defense",
    },
    "formation_change_defensive": {
        "home_win": -0.03,
        "draw": 0.08,
        "away_win": -0.05,
        "description": "Defensive formation reduces goal expectancy",
    },
    "formation_change_attacking": {
        "home_win": 0.05,
        "draw": -0.05,
        "away_win": 0.0,
        "description": "Attacking formation increases goal expectancy",
    },
    "home_crowd_boost": {
        "home_win": 0.04,
        "draw": 0.01,
        "away_win": -0.05,
        "description": "Strong home crowd provides morale boost",
    },
    "fatigue": {
        "home_win": -0.04,
        "draw": 0.02,
        "away_win": 0.02,
        "description": "Fatigue from recent fixtures affects performance",
    },
}


def run(context: dict) -> dict:
    """Run scenario analysis sub-agent.
    
    Args:
        context: {
            "home_team": str,
            "away_team": str,
            "match_date": str - ISO date string,
            "scenario_type": str - key from SCENARIO_IMPACTS,
            "custom_impact": dict - optional custom impact overrides
        }
    
    Returns:
        ScenarioResult as dict
    """
    home_team = context["home_team"]
    away_team = context["away_team"]
    match_date = pd.Timestamp(context["match_date"])
    scenario_type = context.get("scenario_type", "key_player_absent")
    custom_impact = context.get("custom_impact", None)
    
    # Get base prediction
    base_prediction = predict_match(home_team, away_team, match_date)
    
    # Get scenario impact
    if custom_impact:
        impact = custom_impact
        scenario_desc = context.get("scenario_description", "Custom scenario")
        confidence = 0.6  # Lower confidence for custom scenarios
    elif scenario_type in SCENARIO_IMPACTS:
        impact = SCENARIO_IMPACTS[scenario_type]
        scenario_desc = impact["description"]
        confidence = 0.7  # Moderate confidence for predefined scenarios
    else:
        # Default to no impact
        impact = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        scenario_desc = f"Unknown scenario type: {scenario_type}"
        confidence = 0.3
    
    # Apply impact to base prediction
    adjusted_prediction = {
        "home_win": max(0, min(1, base_prediction["home_win"] + impact.get("home_win", 0))),
        "draw": max(0, min(1, base_prediction["draw"] + impact.get("draw", 0))),
        "away_win": max(0, min(1, base_prediction["away_win"] + impact.get("away_win", 0))),
    }
    
    # Renormalize probabilities to sum to 1
    total = sum(adjusted_prediction.values())
    if total > 0:
        adjusted_prediction = {k: v / total for k, v in adjusted_prediction.items()}
    
    # Analyze impact
    impact_analysis = _analyze_impact(base_prediction, adjusted_prediction, scenario_desc)
    
    # Generate summary
    summary = _generate_scenario_summary(
        home_team, away_team, scenario_desc, base_prediction, adjusted_prediction, impact_analysis
    )
    
    result = ScenarioResult(
        scenario_description=scenario_desc,
        base_prediction=base_prediction,
        adjusted_prediction=adjusted_prediction,
        impact_analysis=impact_analysis,
        confidence=confidence,
        summary=summary,
    )
    
    return result.to_dict()


def _analyze_impact(base: dict, adjusted: dict, scenario_desc: str) -> dict:
    """Analyze the impact of the scenario on prediction."""
    changes = {}
    
    for outcome in ["home_win", "draw", "away_win"]:
        diff = adjusted[outcome] - base[outcome]
        changes[outcome] = {
            "base": base[outcome],
            "adjusted": adjusted[outcome],
            "change": diff,
            "direction": "increased" if diff > 0 else "decreased" if diff < 0 else "unchanged",
        }
    
    # Determine most likely outcome shift
    max_change = max(changes.values(), key=lambda x: abs(x["change"]))
    
    return {
        "changes": changes,
        "most_affected_outcome": max_change,
        "scenario_impact": scenario_desc,
    }


def _generate_scenario_summary(
    home_team: str, away_team: str, scenario_desc: str,
    base: dict, adjusted: dict, analysis: dict
) -> str:
    """Generate a human-readable summary of scenario analysis."""
    # Find most likely outcome before and after
    base_likely = max(base, key=base.get)
    adjusted_likely = max(adjusted, key=adjusted.get)
    
    base_prob = base[base_likely]
    adjusted_prob = adjusted[adjusted_likely]
    
    outcome_names = {
        "home_win": f"{home_team} win",
        "draw": "Draw",
        "away_win": f"{away_team} win",
    }
    
    if base_likely == adjusted_likely:
        return (
            f"Scenario: {scenario_desc}. "
            f"Most likely outcome remains {outcome_names[adjusted_likely]} "
            f"({adjusted_prob:.1%} from {base_prob:.1%}). "
            f"Confidence: {analysis.get('confidence', 0.7):.0%}."
        )
    else:
        return (
            f"Scenario: {scenario_desc}. "
            f"Most likely outcome shifts from {outcome_names[base_likely]} "
            f"({base_prob:.1%}) to {outcome_names[adjusted_likely]} "
            f"({adjusted_prob:.1%}). "
            f"Confidence: {analysis.get('confidence', 0.7):.0%}."
        )
