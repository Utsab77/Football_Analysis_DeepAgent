"""Tool: run_prediction_model() -- the agent's bridge into ml/predict.py.

Keep this a thin wrapper; all real logic lives in ml/.
"""
from __future__ import annotations

from ml.predict import predict_match


def run_prediction_model(home_team: str, away_team: str, match_date=None) -> dict:
    return predict_match(home_team, away_team, match_date)
