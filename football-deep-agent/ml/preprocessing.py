"""Load raw match data and build a clean, match-level DataFrame.

Phase 1 (Weeks 1-2). No agent code depends on internals here — only on
the public functions, so you're free to change the implementation later.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]


def load_raw_matches(filename: str = "matches.csv") -> pd.DataFrame:
    """Load the raw match results CSV from data/raw/.

    TODO: point this at your chosen dataset/data source. See
    docs/architecture.md section on recommended starting scope
    (one competition, e.g. Premier League).
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No raw data at {path}. Drop your dataset there first, "
            "or update `filename`."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")
    return df.sort_values("date").reset_index(drop=True)


def derive_result(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `result` column: 'H', 'D', or 'A' from the goal columns."""
    df = df.copy()
    conditions = [
        df["home_goals"] > df["away_goals"],
        df["home_goals"] == df["away_goals"],
    ]
    df["result"] = pd.Series(
        pd.NA, index=df.index, dtype="object"
    )
    df.loc[conditions[0], "result"] = "H"
    df.loc[conditions[1], "result"] = "D"
    df.loc[~(conditions[0] | conditions[1]), "result"] = "A"
    return df


def save_processed(df: pd.DataFrame, filename: str = "matches_clean.parquet") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    raw = load_raw_matches()
    clean = derive_result(raw)
    out = save_processed(clean)
    print(f"Wrote {len(clean)} matches to {out}")
