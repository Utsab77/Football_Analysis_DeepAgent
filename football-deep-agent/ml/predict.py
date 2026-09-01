"""Single entry point for match prediction. This is the function the
agent's `run_prediction_model()` tool calls -- keep this interface
stable even as the internals evolve.
"""
from __future__ import annotations

from ml import ensemble, logistic_model, random_forest, xgboost_model
from ml.features import build_features_for_match
from ml.preprocessing import PROCESSED_DIR

import pandas as pd


def _load_history() -> pd.DataFrame:
    path = PROCESSED_DIR / "matches_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No processed data at {path}. Run `python ml/preprocessing.py` first."
        )
    return pd.read_parquet(path)


def predict_match(home_team: str, away_team: str, match_date=None) -> dict:
    """Predict Home/Draw/Away probabilities for an upcoming match.

    Returns: {"home_win": float, "draw": float, "away_win": float}
    """
    history = _load_history()
    match_date = match_date or pd.Timestamp.now()
    features = build_features_for_match(history, home_team, away_team, match_date)

    log_model = logistic_model.load()
    rf_model = random_forest.load()
    xgb_model = xgboost_model.load()

    proba = ensemble.average_ensemble(
        logistic_model.predict_proba(log_model, features),
        random_forest.predict_proba(rf_model, features),
        xgboost_model.predict_proba(xgb_model, features),
    )

    return {
        "home_win": proba.get("H", 0.0),
        "draw": proba.get("D", 0.0),
        "away_win": proba.get("A", 0.0),
    }


if __name__ == "__main__":
    print(predict_match("Arsenal", "Chelsea"))
