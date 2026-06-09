"""
model.py — the prediction model.

Two layers:
  1. A Poisson attack/defence model fit on recent international results, with
     time-decay weighting (a light Dixon-Coles flavour).
  2. An Elo blend: each team's Elo (from elo.csv or derived) nudges the
     expected goals, so teams with little recent data still get sane numbers.

The public surface is:
  - fit(results_df, elo_dict) -> ModelParams
  - match_goal_expectations(params, home, away, neutral=True) -> (xg_home, xg_away)
  - match_outcome_probs(params, home, away) -> dict with win/draw/loss + score grid
"""

import math
import numpy as np
import pandas as pd
from collections import defaultdict

# ---- tunables -------------------------------------------------------------
HALF_LIFE_DAYS = 365 * 2          # recent 2 years weigh ~2x vs 4 years ago
MAX_GOALS = 8                     # truncate the score grid here
ELO_BLEND = 0.70                  # 0 = pure Poisson, 1 = pure Elo-implied.
                                  # Weighted toward Elo so favourites track the
                                  # market/Opta rather than raw goal tallies
                                  # (which over-rate teams that pad goals on
                                  # weak opposition).
ELO_MULT_SPREAD = 1.30            # width of the Elo goal-multiplier band: the
                                  # multiplier runs (1 - SPREAD/2)..(1 + SPREAD/2)
                                  # from huge underdog to huge favourite. Wider =
                                  # favourites dominate more decisively (closer to
                                  # Opta's concentrated odds).
RECENT_CUTOFF_YEARS = 8           # ignore matches older than this for fitting


class ModelParams:
    def __init__(self, attack, defence, home_adv, base_goals, elo):
        self.attack = attack          # dict team -> attack strength (log)
        self.defence = defence        # dict team -> defence strength (log)
        self.home_adv = home_adv      # scalar log home advantage
        self.base_goals = base_goals  # league-average goals baseline
        self.elo = elo                # dict team -> elo rating


def _time_weights(dates, ref_date):
    age_days = (ref_date - dates).dt.days.clip(lower=0)
    return np.power(0.5, age_days / HALF_LIFE_DAYS)


def fit(results_df, elo_dict=None):
    """Fit attack/defence strengths via weighted Poisson means.

    This is a fast, robust approximation of a full Dixon-Coles MLE: instead of
    numerically maximising the likelihood, we use weighted average goals for and
    against, which is stable and good enough for a fun office predictor.
    """
    df = results_df.copy()
    ref = df["date"].max()
    df = df[df["date"] >= ref - pd.Timedelta(days=365 * RECENT_CUTOFF_YEARS)]
    df["w"] = _time_weights(df["date"], ref)

    # baseline average goals per team per match (weighted)
    total_goals = (df["home_score"] * df["w"]).sum() + (df["away_score"] * df["w"]).sum()
    total_w = df["w"].sum() * 2
    base_goals = total_goals / total_w

    # weighted goals scored / conceded per team (home & away pooled)
    gf = defaultdict(float); ga = defaultdict(float); wsum = defaultdict(float)
    for _, r in df.iterrows():
        h, a, w = r["home_team"], r["away_team"], r["w"]
        gf[h] += r["home_score"] * w; ga[h] += r["away_score"] * w; wsum[h] += w
        gf[a] += r["away_score"] * w; ga[a] += r["home_score"] * w; wsum[a] += w

    attack, defence = {}, {}
    for t in wsum:
        if wsum[t] <= 0:
            continue
        avg_for = gf[t] / wsum[t]
        avg_against = ga[t] / wsum[t]
        # log-strength relative to baseline; small epsilon to avoid log(0)
        attack[t] = math.log(max(avg_for, 0.05) / base_goals)
        defence[t] = math.log(max(avg_against, 0.05) / base_goals)

    # home advantage (weighted): mean home goals vs away goals
    hg = (df["home_score"] * df["w"]).sum() / df["w"].sum()
    ag = (df["away_score"] * df["w"]).sum() / df["w"].sum()
    home_adv = math.log(max(hg, 0.05) / max(ag, 0.05))

    # Elo: use provided dict, else derive a crude Elo from attack-defence.
    if elo_dict is None:
        elo_dict = {}
        for t in attack:
            strength = attack.get(t, 0) - defence.get(t, 0)
            elo_dict[t] = 1500 + strength * 250  # rough mapping to Elo scale

    return ModelParams(attack, defence, home_adv, base_goals, elo_dict)


def _elo_implied_xg(params, home, away):
    """Translate an Elo gap into an expected-goals tilt."""
    eh = params.elo.get(home, 1500)
    ea = params.elo.get(away, 1500)
    # logistic expected score from Elo, then map to a goals multiplier
    exp_home = 1 / (1 + 10 ** ((ea - eh) / 400))
    # exp_home in (0,1); centered so an even matchup gives a ~1.0 multiplier.
    mult_home = 1.0 + ELO_MULT_SPREAD * (exp_home - 0.5)
    mult_away = 1.0 + ELO_MULT_SPREAD * ((1 - exp_home) - 0.5)
    return mult_home, mult_away


def match_goal_expectations(params, home, away, neutral=True):
    """Expected goals for each side. neutral=True for World Cup venues."""
    a_h = params.attack.get(home, 0.0)
    d_a = params.defence.get(away, 0.0)
    a_a = params.attack.get(away, 0.0)
    d_h = params.defence.get(home, 0.0)

    log_home = math.log(params.base_goals) + a_h + d_a
    log_away = math.log(params.base_goals) + a_a + d_h
    if not neutral:
        log_home += params.home_adv

    xg_home = math.exp(log_home)
    xg_away = math.exp(log_away)

    # blend with Elo-implied tilt
    mh, ma = _elo_implied_xg(params, home, away)
    xg_home = (1 - ELO_BLEND) * xg_home + ELO_BLEND * (params.base_goals * mh)
    xg_away = (1 - ELO_BLEND) * xg_away + ELO_BLEND * (params.base_goals * ma)

    return max(xg_home, 0.15), max(xg_away, 0.15)


def _poisson_pmf(lmbda, k):
    return math.exp(-lmbda) * lmbda ** k / math.factorial(k)


def score_grid(params, home, away, neutral=True):
    xg_h, xg_a = match_goal_expectations(params, home, away, neutral)
    ph = [_poisson_pmf(xg_h, i) for i in range(MAX_GOALS + 1)]
    pa = [_poisson_pmf(xg_a, j) for j in range(MAX_GOALS + 1)]
    grid = np.outer(ph, pa)
    grid = grid / grid.sum()
    return grid


def match_outcome_probs(params, home, away, neutral=True):
    grid = score_grid(params, home, away, neutral)
    p_home = np.tril(grid, -1).sum()   # home goals > away goals
    p_away = np.triu(grid, 1).sum()    # away goals > home goals
    p_draw = np.trace(grid)
    # most likely scorelines
    flat = [(i, j, grid[i, j]) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    flat.sort(key=lambda x: -x[2])
    return {
        "home_win": float(p_home),
        "draw": float(p_draw),
        "away_win": float(p_away),
        "xg_home": float(match_goal_expectations(params, home, away, neutral)[0]),
        "xg_away": float(match_goal_expectations(params, home, away, neutral)[1]),
        "top_scorelines": [(i, j, float(p)) for i, j, p in flat[:5]],
        "grid": grid,
    }
