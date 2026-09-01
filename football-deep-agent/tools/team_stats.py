"""Tool: get_team_stats() -- summary stats for a team as of a given date."""
from __future__ import annotations

from ml.features import goals_avg, team_strength


def get_team_stats(team: str, before_date) -> dict:
    from ml.preprocessing import PROCESSED_DIR
    import pandas as pd

    df = pd.read_parquet(PROCESSED_DIR / "matches_clean.parquet")
    return {
        "team": team,
        "strength": team_strength(df, team, before_date),
        "goals_avg": goals_avg(df, team, before_date, conceded=False),
        "conceded_avg": goals_avg(df, team, before_date, conceded=True),
    }
