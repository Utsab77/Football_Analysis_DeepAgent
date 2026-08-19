# ML Prediction Engine (Phase 1, Weeks 1-2)

Build this fully, standalone, before writing a single line of agent code.
It must work as a plain function call:

```python
from ml.predict import predict_match
predict_match(home_team="Arsenal", away_team="Chelsea")
# -> {"home_win": 0.58, "draw": 0.24, "away_win": 0.18}
```

## Scope

- One competition to start (Premier League recommended).
- Historical pre-match prediction only — no live data yet.
- Every feature must be computable *before kickoff*. If a feature needs
  information from the match itself, it's leakage — throw it out.

## Files

- `preprocessing.py` — load raw data, clean it, build the match-level table
- `features.py` — feature engineering (see FEATURES list below)
- `logistic_model.py`, `random_forest.py`, `xgboost_model.py` — baseline models
- `ensemble.py` — combine the baselines
- `evaluation.py` — accuracy, precision/recall/F1, log loss, Brier score, calibration
- `predict.py` — the single entry point everything else (agent, API) calls

## Required features (minimum set)

```
home_team_strength      away_team_strength
home_recent_form        away_recent_form
home_goals_avg          away_goals_avg
home_conceded_avg       away_conceded_avg
home_advantage          league_position_difference
```

## Validation rule

Use **time-based** train/test splits (train on earlier matches, test on
later ones). Never randomly shuffle matches across train/test — that
leaks future information into training.
