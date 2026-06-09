"""
config.py — 2026 FIFA World Cup tournament configuration.

The group draw below is the REAL confirmed draw (Final Draw held 5 Dec 2025,
completed after the March 2026 playoffs). Source: multiple outlets reporting
FIFA's confirmed groups (NBC Sports, ESPN, Yahoo Sports), retrieved June 2026.

If anything here ever looks stale, this is the single file to edit — the model
and the app both read the tournament structure from here.
"""

# ---------------------------------------------------------------------------
# The 12 groups (4 teams each, 48 total).
# Team names are chosen to match the naming used in the historical results
# dataset where possible. A few need aliasing (see TEAM_ALIASES below).
# ---------------------------------------------------------------------------
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Map config names -> the spelling used in the Kaggle / Elo datasets.
# (The datasets use "Korea Republic", "IR Iran", etc. in some versions; adjust
#  here once you load your data and see exactly which spellings appear.)
TEAM_ALIASES = {
    "South Korea": ["Korea Republic", "Korea, South"],
    "Turkey": ["Türkiye", "Turkiye"],
    "Czechia": ["Czech Republic"],
    "Curacao": ["Curaçao"],
    "Ivory Coast": ["Côte d'Ivoire", "Cote d'Ivoire"],
    "Cape Verde": ["Cabo Verde"],
    "DR Congo": ["Congo DR", "Democratic Republic of the Congo"],
    "United States": ["USA"],
    "Iran": ["IR Iran"],
    "Bosnia and Herzegovina": ["Bosnia-Herzegovina"],
}

# ---------------------------------------------------------------------------
# Format constants (confirmed 48-team format).
# ---------------------------------------------------------------------------
TEAMS_PER_GROUP = 4
GROUPS_COUNT = 12
ADVANCE_PER_GROUP = 2          # top two automatically advance
BEST_THIRD_PLACED = 8          # eight best third-place teams also advance
KNOCKOUT_SIZE = 32             # Round of 32

POINTS_WIN = 3
POINTS_DRAW = 1
POINTS_LOSS = 0

# Group-stage tiebreaker order (FIFA, simplified for simulation — we use the
# "all matches" criteria; head-to-head and conduct/fair-play are approximated
# since simulated games don't generate cards). Order applied:
#   1. Points
#   2. Goal difference (all matches)
#   3. Goals scored (all matches)
#   4. (tie broken randomly as a stand-in for conduct score / FIFA ranking)
#
# Third-place ranking across groups uses: points, GD, goals scored, then random.

FINAL_DATE = "2026-07-19"      # MetLife Stadium, New Jersey
OPENING_DATE = "2026-06-11"    # Estadio Azteca, Mexico City

ALL_TEAMS = [t for g in GROUPS.values() for t in g]
assert len(ALL_TEAMS) == 48, f"Expected 48 teams, got {len(ALL_TEAMS)}"
