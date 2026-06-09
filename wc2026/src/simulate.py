"""
simulate.py — Monte Carlo simulation of the full 2026 tournament, using the
REAL knockout bracket (src/bracket.py) transcribed from the FIFA schedule.

Flow per simulation:
  1. Play all group matches (sampling scorelines from the model).
  2. Rank each group with FIFA tiebreakers -> winners, runners-up, thirds.
  3. Select the 8 best third-place teams across all 12 groups.
  4. Assign those 8 thirds to their exact R32 slots, respecting the per-match
     candidate-group constraints from the schedule (constraint-based matching).
  5. Play the exact knockout tree (M73..M104) to a champion.

Aggregate over N simulations for: group-win, advance, reach SF/final, title.
"""

import numpy as np
from collections import defaultdict
from itertools import combinations

from . import model as M
from .config import (GROUPS, ADVANCE_PER_GROUP, BEST_THIRD_PLACED,
                     POINTS_WIN, POINTS_DRAW)
from .bracket import R32_SLOTS, TREE, FINAL_MATCH, ROUND_OF


def _sample_score(params, a, b, rng):
    grid = M.score_grid(params, a, b, neutral=True)
    flat = grid.flatten()
    idx = rng.choice(len(flat), p=flat)
    n = grid.shape[1]
    return idx // n, idx % n


def _play_group(params, teams, rng):
    pts = defaultdict(int); gf = defaultdict(int); ga = defaultdict(int)
    for a, b in combinations(teams, 2):
        ga_, gb_ = _sample_score(params, a, b, rng)
        gf[a] += ga_; ga[a] += gb_; gf[b] += gb_; ga[b] += ga_
        if ga_ > gb_:
            pts[a] += POINTS_WIN
        elif gb_ > ga_:
            pts[b] += POINTS_WIN
        else:
            pts[a] += POINTS_DRAW; pts[b] += POINTS_DRAW
    def key(t):
        return (pts[t], gf[t] - ga[t], gf[t], rng.random())
    ranked = sorted(teams, key=key, reverse=True)
    standings = {t: {"pts": pts[t], "gd": gf[t] - ga[t], "gf": gf[t]} for t in teams}
    return ranked, standings


def _best_thirds(thirds_with_stats, rng):
    """Return list of (group_id, team) for the 8 best third-place teams."""
    def key(item):
        gid, t, s = item
        return (s["pts"], s["gd"], s["gf"], rng.random())
    ranked = sorted(thirds_with_stats, key=key, reverse=True)
    return [(gid, t) for gid, t, _ in ranked[:BEST_THIRD_PLACED]]


def _assign_thirds(third_slots, qualifying_thirds, rng):
    """Constraint-based assignment of the 8 qualifying third-place teams to the
    8 THIRD slots in the R32.

    third_slots: list of (match_id, candidate_groups[list])
    qualifying_thirds: list of (group_id, team)

    Returns dict match_id -> team. Uses a simple backtracking matcher so every
    third team lands in a slot whose candidate list contains its group (this is
    exactly the guarantee FIFA's Annex C table provides).
    """
    # order slots by fewest candidates first (most constrained) for efficiency
    slots = sorted(third_slots, key=lambda s: len(s[1]))
    thirds_by_group = {gid: team for gid, team in qualifying_thirds}
    qualifying_groups = set(thirds_by_group.keys())

    assignment = {}
    used_groups = set()

    def backtrack(i):
        if i == len(slots):
            return True
        mid, cands = slots[i]
        # candidate groups that actually qualified and aren't used yet
        options = [g for g in cands if g in qualifying_groups and g not in used_groups]
        rng.shuffle(options)
        for g in options:
            used_groups.add(g); assignment[mid] = thirds_by_group[g]
            if backtrack(i + 1):
                return True
            used_groups.discard(g); del assignment[mid]
        return False

    ok = backtrack(0)
    if not ok:
        # extremely rare fallback: assign leftovers arbitrarily
        leftover = [g for g in qualifying_groups if g not in used_groups]
        for (mid, _), g in zip([s for s in slots if s[0] not in assignment], leftover):
            assignment[mid] = thirds_by_group[g]
    return assignment


def _resolve_r32(winners, runners, thirds_assignment):
    """Build match_id -> (teamA, teamB) for the 16 R32 matches."""
    matches = {}
    for mid, (sa, sb) in R32_SLOTS.items():
        matches[mid] = (_slot_team(sa, winners, runners, thirds_assignment, mid),
                        _slot_team(sb, winners, runners, thirds_assignment, mid))
    return matches


def _slot_team(slot, winners, runners, thirds_assignment, mid):
    kind = slot[0]
    if kind == "WIN":
        return winners[slot[1]]
    if kind == "RUN":
        return runners[slot[1]]
    if kind == "THIRD":
        return thirds_assignment[mid]
    raise ValueError(slot)


def _play_match(params, a, b, rng):
    ga_, gb_ = _sample_score(params, a, b, rng)
    if ga_ == gb_:
        return a if rng.random() < 0.5 else b  # penalties
    return a if ga_ > gb_ else b


def _play_knockout(params, r32_matches, rng):
    """Play the exact tree. Returns champion and dict team -> furthest round.

    'reached' records the furthest round each team REACHED (i.e. played in).
    Losing the final still counts as reaching "F"; the champion is tagged
    "Champion".
    """
    reached = {}
    results = {}   # match_id -> winning team

    # Round of 32: everyone here reached at least R32
    for mid, (a, b) in r32_matches.items():
        reached[a] = "R32"; reached[b] = "R32"
        results[mid] = _play_match(params, a, b, rng)

    # Remaining rounds in tree order; both teams in a match reached that round
    for mid, (fa, fb) in TREE.items():
        a, b = results[fa], results[fb]
        rnd = ROUND_OF[mid]
        reached[a] = rnd; reached[b] = rnd
        results[mid] = _play_match(params, a, b, rng)

    champ = results[FINAL_MATCH]
    reached[champ] = "Champion"
    return champ, reached


def simulate(params, n_sims=10000, seed=42):
    rng = np.random.default_rng(seed)

    title = defaultdict(int)
    group_win = defaultdict(int)
    advance = defaultdict(int)
    reach_final = defaultdict(int)
    reach_sf = defaultdict(int)

    # which R32 matches carry THIRD slots, with their candidate groups
    third_slots = []
    for mid, (sa, sb) in R32_SLOTS.items():
        for s in (sa, sb):
            if s[0] == "THIRD":
                third_slots.append((mid, s[1]))

    for _ in range(n_sims):
        winners, runners = {}, {}
        thirds_stats = []
        for gid, teams in GROUPS.items():
            ranked, standings = _play_group(params, teams, rng)
            winners[gid] = ranked[0]
            runners[gid] = ranked[1]
            thirds_stats.append((gid, ranked[2], standings[ranked[2]]))
            group_win[ranked[0]] += 1
            for t in ranked[:ADVANCE_PER_GROUP]:
                advance[t] += 1

        qualifying_thirds = _best_thirds(thirds_stats, rng)
        for _, t in qualifying_thirds:
            advance[t] += 1

        thirds_assignment = _assign_thirds(third_slots, qualifying_thirds, rng)
        r32 = _resolve_r32(winners, runners, thirds_assignment)
        champ, reached = _play_knockout(params, r32, rng)

        title[champ] += 1
        # A team "reached" a round if its furthest round is that round or later.
        order = {"R32": 0, "R16": 1, "QF": 2, "SF": 3, "F": 4, "Champion": 5}
        for t, r in reached.items():
            lvl = order[r]
            if lvl >= order["F"]:
                reach_final[t] += 1
            if lvl >= order["SF"]:
                reach_sf[t] += 1

    def to_prob(d):
        return {t: c / n_sims for t, c in d.items()}

    return {
        "title": to_prob(title),
        "group_win": to_prob(group_win),
        "advance": to_prob(advance),
        "reach_final": to_prob(reach_final),
        "reach_sf": to_prob(reach_sf),
        "n_sims": n_sims,
    }
