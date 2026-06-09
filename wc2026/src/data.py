"""
data.py — load and prepare the two datasets that power the model.

DATASET 1: Historical international results
  Kaggle: "International football results from 1872 to present" (Mart Jürisoo).
  Download results.csv and place it in data/results.csv
  Columns: date, home_team, away_team, home_score, away_score, tournament,
           city, country, neutral

DATASET 2: World Football Elo Ratings
  eloratings.net publishes current ratings. The simplest path: save a CSV with
  columns [team, elo] into data/elo.csv. (You can scrape or hand-build it; for
  ~48 teams a manual CSV is totally fine.)

Both are optional to *run* — if elo.csv is missing, the model falls back to
Elo ratings derived from the historical results themselves (see model.py).
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
ELO_CSV = os.path.join(DATA_DIR, "elo.csv")


def load_results():
    """Return a cleaned DataFrame of historical international matches."""
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(
            f"Missing {RESULTS_CSV}. Download the Kaggle 'International football "
            f"results 1872-present' dataset and save results.csv into data/."
        )
    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


def load_elo():
    """Return dict team -> elo rating, or None if no elo.csv present."""
    if not os.path.exists(ELO_CSV):
        return None
    df = pd.read_csv(ELO_CSV)
    return dict(zip(df["team"], df["elo"]))


def build_alias_map(aliases):
    """Invert the config alias map: dataset_name -> canonical_name."""
    inv = {}
    for canonical, variants in aliases.items():
        for v in variants:
            inv[v] = canonical
    return inv


def normalize_team_names(df, alias_map):
    """Rename dataset team spellings to the canonical config spellings."""
    df = df.copy()
    df["home_team"] = df["home_team"].replace(alias_map)
    df["away_team"] = df["away_team"].replace(alias_map)
    return df
