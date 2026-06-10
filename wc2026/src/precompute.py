"""
precompute.py — run the heavy computation offline, save lightweight outputs.

Usage:
    python -m src.precompute

This fits the model on data/results.csv (+ optional data/elo.csv), runs the
Monte Carlo simulation, and writes:
    data/sim_results.json   (title odds, group-win, advance probabilities)
    data/model_params.json  (team strengths, for the live match predictor)

The Streamlit app reads ONLY these JSON files, so it stays fast and never has
to run 10k simulations on a web server. Re-run this script after each match day
to refresh the numbers, then redeploy (or have the app read from a stored file).
"""

import json
import os

from .data import (load_results, load_elo, load_market_odds, build_alias_map,
                   normalize_team_names)
from .config import TEAM_ALIASES, ALL_TEAMS
from . import model as M
from . import simulate as S

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main(n_sims=10000):
    print("Loading data...")
    results = load_results()
    alias_map = build_alias_map(TEAM_ALIASES)
    results = normalize_team_names(results, alias_map)
    elo = load_elo()
    if elo:
        elo = { (alias_map.get(k, k)): v for k, v in elo.items() }

    # Anchor team strengths toward the betting market's outright odds.
    market = load_market_odds()
    if elo and market:
        market = { (alias_map.get(k, k)): v for k, v in market.items() }
        elo = M.blend_market_elo(elo, market)
        print(f"Anchored {sum(1 for t in market if t in elo)} teams to market odds "
              f"(weight={M.MARKET_BLEND}).")

    print("Fitting model...")
    params = M.fit(results, elo)

    # Warn about any tournament team missing from the data
    missing = [t for t in ALL_TEAMS if t not in params.attack]
    if missing:
        print(f"WARNING: no historical data for: {missing}. "
              f"They'll use baseline strength. Check TEAM_ALIASES in config.py.")

    print(f"Running {n_sims} simulations...")
    sim = S.simulate(params, n_sims=n_sims)

    with open(os.path.join(DATA_DIR, "sim_results.json"), "w") as f:
        json.dump(sim, f, indent=2)

    # Save params the match predictor needs (attack/defence/elo/base/home_adv)
    params_out = {
        "attack": params.attack,
        "defence": params.defence,
        "home_adv": params.home_adv,
        "base_goals": params.base_goals,
        "elo": params.elo,
    }
    with open(os.path.join(DATA_DIR, "model_params.json"), "w") as f:
        json.dump(params_out, f, indent=2)

    print("Done. Wrote sim_results.json and model_params.json")


if __name__ == "__main__":
    main()
