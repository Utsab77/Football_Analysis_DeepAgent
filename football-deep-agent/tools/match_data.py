"""Tool: fetch raw match data for a team (schedule, results).

Phase 2 (Weeks 3-4) -- this is one of the first real tools the minimal
agent calls. Wraps ml.preprocessing history for now; swap in a live
data source later.
"""
from __future__ import annotations

import pandas as pd

from ml.preprocessing import PROCESSED_DIR


def get_matches(team: str, before_date=None) -> list[dict]:
    """Return this team's historical matches, optionally before a given date."""
    path = PROCESSED_DIR / "matches_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed data at {path}. Run ml/preprocessing.py first.")
    df = pd.read_parquet(path)
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    if before_date is not None:
        mask &= df["date"] < before_date
    return df.loc[mask].to_dict(orient="records")
