"""Download and prepare match data from Kaggle.

Downloads the 'Club Football Match Data (2000-2025)' dataset,
renames columns to match our preprocessing schema, filters to
Premier League only, and saves to data/raw/matches.csv.
"""
from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Kaggle dataset identifier
KAGGLE_DATASET = "adamgbor/club-football-match-data-2000-2025"

# Column mapping from Kaggle dataset to our schema
COLUMN_MAP = {
    "MatchDate": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHome": "home_goals",
    "FTAway": "away_goals",
    "FTResult": "result",
    "Division": "division",
}

# Premier League division code in this dataset
EPL_DIVISION = "E0"


def download_kaggle_data() -> Path:
    """Download dataset from Kaggle and return path to downloaded files."""
    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Downloaded to: {path}")
    return Path(path)


def prepare_matches(kaggle_path: Path, division: str = EPL_DIVISION) -> pd.DataFrame:
    """Load Matches.csv, rename columns, filter to specified division."""
    matches_file = kaggle_path / "Matches.csv"
    if not matches_file.exists():
        # Try alternate casing
        matches_file = kaggle_path / "matches.csv"
    if not matches_file.exists():
        raise FileNotFoundError(f"Matches.csv not found in {kaggle_path}")

    df = pd.read_csv(matches_file)

    # Rename columns to our schema
    df = df.rename(columns=COLUMN_MAP)

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Filter to Premier League if specified
    if division and "division" in df.columns:
        before = len(df)
        df = df[df["division"] == division].copy()
        print(f"Filtered to {division}: {len(df)} matches (from {before})")

    # Keep only required columns for preprocessing
    required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    # Drop rows with missing essential data
    df = df.dropna(subset=required)
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Final dataset: {len(df)} matches from {df['date'].min()} to {df['date'].max()}")
    return df[required]


def save_raw(df: pd.DataFrame, filename: str = "matches.csv") -> Path:
    """Save prepared data to data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} matches to {out_path}")
    return out_path


def load_or_download() -> Path:
    """Main entry: download if not cached, prepare, save. Returns path to raw CSV."""
    raw_path = RAW_DIR / "matches.csv"
    if raw_path.exists():
        print(f"Raw data already exists at {raw_path}")
        return raw_path

    kaggle_path = download_kaggle_data()
    df = prepare_matches(kaggle_path)
    return save_raw(df)


if __name__ == "__main__":
    load_or_download()
