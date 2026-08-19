"""Tool: get_recent_form() and get_head_to_head()."""
from __future__ import annotations

import pandas as pd

from ml.features import recent_form
from ml.preprocessing import PROCESSED_DIR


def get_recent_form(team: str, before_date, n: int = 5) -> dict:
    df = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    return {"team": team, "form_points_per_game": recent_form(df, team, before_date, n=n)}


def get_head_to_head(team_a: str, team_b: str, before_date, n: int = 10) -> list[dict]:
    df = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b))
        | ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    ) & (df["date"] < before_date)
    return df.loc[mask].sort_values("date").tail(n).to_dict(orient="records")
