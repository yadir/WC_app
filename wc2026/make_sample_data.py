"""
make_sample_data.py — generate a synthetic results.csv so you can run the whole
pipeline immediately, before downloading the real Kaggle dataset.

This invents plausible-but-fake international results for the 48 teams using
rough strength tiers. It is ONLY for wiring up and demoing the app. Replace
data/results.csv with the real Kaggle file before showing colleagues real
numbers.

Usage:  python make_sample_data.py
"""

import numpy as np
import pandas as pd
from itertools import combinations
from src.config import ALL_TEAMS

rng = np.random.default_rng(7)

# crude strength tiers just to make the demo look sensible
STRONG = {"Brazil", "France", "Argentina", "Spain", "England", "Germany",
          "Portugal", "Netherlands", "Belgium"}
MID = {"Croatia", "Uruguay", "Morocco", "Switzerland", "Senegal", "Japan",
       "United States", "Mexico", "Colombia", "Ecuador", "South Korea",
       "Australia", "Sweden", "Austria", "Norway", "Egypt", "Ivory Coast"}


def strength(t):
    if t in STRONG: return 1.9
    if t in MID: return 1.4
    return 1.0


rows = []
start = pd.Timestamp("2018-01-01")
# round-robin-ish friendly history, a few passes
for season in range(7):
    for a, b in combinations(ALL_TEAMS, 2):
        if rng.random() > 0.18:   # only a subset of pairs each season
            continue
        date = start + pd.Timedelta(days=int(rng.integers(0, 365)) + season * 365)
        la, lb = strength(a), strength(b)
        ga = rng.poisson(la * 0.9)
        gb = rng.poisson(lb * 0.9)
        rows.append({"date": date, "home_team": a, "away_team": b,
                     "home_score": ga, "away_score": gb,
                     "tournament": "Friendly", "city": "", "country": "",
                     "neutral": True})

df = pd.DataFrame(rows).sort_values("date")
df.to_csv("data/results.csv", index=False)
print(f"Wrote data/results.csv with {len(df)} synthetic matches.")
print("NOTE: synthetic demo data. Replace with the real Kaggle results.csv.")
