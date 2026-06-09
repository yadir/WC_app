"""
app.py — MIE World Cup 2026 Prediction Explorer (v1, exploratory).

Run locally:   streamlit run app.py
Deploy:        push repo to GitHub -> Streamlit Community Cloud -> point at app.py

This app is read-only and exploratory. It shows the model's view of the
tournament to help colleagues make informed picks in the work league. No
picks-submission, no leaderboard — by design.

It reads two precomputed files (run `python -m src.precompute` first):
    data/sim_results.json
    data/model_params.json
"""

import json
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.config import GROUPS, ALL_TEAMS
from src.schedule import GROUP_STAGE, KNOCKOUTS

DATA_DIR = "data"

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(page_title="MIE World Cup 2026 Explorer",
                   page_icon="⚽", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@500;600;700&family=Archivo:wght@400;600;800&display=swap');

:root {
  --pitch: #0b3d2e;
  --pitch-deep: #07261d;
  --line: #1e5c47;
  --chalk: #f2f5f1;
  --flood: #f4c84a;       /* floodlight amber accent */
  --muted: #9bb3a8;
}
.stApp {
  background: linear-gradient(180deg, var(--pitch-deep) 0%, var(--pitch) 100%);
  color: var(--chalk);
  font-family: 'Archivo', sans-serif;
}
h1, h2, h3 { font-family: 'Archivo Narrow', sans-serif; letter-spacing: .01em; }
h1 { font-weight: 700; font-size: 2.4rem; text-transform: uppercase; }
.eyebrow {
  font-family: 'Archivo Narrow', sans-serif; text-transform: uppercase;
  letter-spacing: .22em; color: var(--flood); font-size: .8rem; font-weight: 600;
}
[data-testid="stSidebar"] { background: var(--pitch-deep); border-right: 1px solid var(--line); }
.stDataFrame { background: rgba(255,255,255,.02); }
.kpi {
  border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
  background: rgba(255,255,255,.03);
}
.kpi .label { color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .12em; }
.kpi .val { font-family: 'Archivo Narrow', sans-serif; font-size: 1.6rem; font-weight: 700; }
hr { border-color: var(--line); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PITCH = "#0b3d2e"; FLOOD = "#f4c84a"; CHALK = "#f2f5f1"; MUTED = "#9bb3a8"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_sim():
    path = os.path.join(DATA_DIR, "sim_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_params():
    path = os.path.join(DATA_DIR, "model_params.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


sim = load_sim()
params = load_params()

if sim is None or params is None:
    st.error("No precomputed data found. Run `python -m src.precompute` first to "
             "generate data/sim_results.json and data/model_params.json.")
    st.stop()


# ---- model helpers reused from the params (kept inline so the app is standalone) ----
MAX_GOALS = 8
ELO_BLEND = 0.35


def _elo_mult(home, away):
    eh = params["elo"].get(home, 1500); ea = params["elo"].get(away, 1500)
    exp_home = 1 / (1 + 10 ** ((ea - eh) / 400))
    return 0.6 + 0.8 * exp_home, 0.6 + 0.8 * (1 - exp_home)


def xg(home, away):
    base = params["base_goals"]
    a_h = params["attack"].get(home, 0.0); d_a = params["defence"].get(away, 0.0)
    a_a = params["attack"].get(away, 0.0); d_h = params["defence"].get(home, 0.0)
    xh = math.exp(math.log(base) + a_h + d_a)
    xa = math.exp(math.log(base) + a_a + d_h)
    mh, ma = _elo_mult(home, away)
    xh = (1 - ELO_BLEND) * xh + ELO_BLEND * (base * mh)
    xa = (1 - ELO_BLEND) * xa + ELO_BLEND * (base * ma)
    return max(xh, .15), max(xa, .15)


def _pmf(l, k):
    return math.exp(-l) * l ** k / math.factorial(k)


def outcome(home, away):
    xh, xa = xg(home, away)
    ph = [_pmf(xh, i) for i in range(MAX_GOALS + 1)]
    pa = [_pmf(xa, j) for j in range(MAX_GOALS + 1)]
    grid = np.outer(ph, pa); grid /= grid.sum()
    return {
        "home_win": float(np.tril(grid, -1).sum()),
        "draw": float(np.trace(grid)),
        "away_win": float(np.triu(grid, 1).sum()),
        "xg_home": xh, "xg_away": xa, "grid": grid,
    }


# ---------------------------------------------------------------------------
# Sidebar nav
# ---------------------------------------------------------------------------
st.sidebar.markdown("<div class='eyebrow'>MIE Predictions League</div>",
                    unsafe_allow_html=True)
st.sidebar.markdown("## World Cup 2026")
page = st.sidebar.radio("Go to", [
    "Tournament Overview", "Group Explorer", "Match Predictor", "Fixtures",
    "How this works"
])
st.sidebar.markdown("---")
st.sidebar.caption(f"Model view from {sim['n_sims']:,} simulated tournaments. "
                   "For exploration — make your official picks in the league.")


def bar(df, x, y, color_val=FLOOD, height=None):
    ch = (alt.Chart(df).mark_bar(color=color_val)
          .encode(
              x=alt.X(f"{x}:Q", title=None, axis=alt.Axis(format="%", grid=False)),
              y=alt.Y(f"{y}:N", sort="-x", title=None),
              tooltip=[alt.Tooltip(f"{x}:Q", format=".1%"), f"{y}:N"])
          .properties(height=height or (28 * len(df))))
    return ch.configure_view(strokeWidth=0).configure_axis(
        labelColor=CHALK, titleColor=CHALK, labelFontSize=12)


# ---------------------------------------------------------------------------
# PAGE 1 — Tournament Overview
# ---------------------------------------------------------------------------
if page == "Tournament Overview":
    st.markdown("<div class='eyebrow'>The model's view</div>", unsafe_allow_html=True)
    st.markdown("# Who wins the World Cup?")
    st.write("Every number here comes from simulating the whole tournament "
             f"{sim['n_sims']:,} times. Use it to sense-check your gut before you "
             "lock in your league picks.")

    title = (pd.DataFrame(sim["title"].items(), columns=["team", "p"])
             .sort_values("p", ascending=False))
    top = title.head(3).to_dict("records")
    cols = st.columns(3)
    labels = ["Favourite", "Second favourite", "Third favourite"]
    for c, rec, lab in zip(cols, top, labels):
        c.markdown(f"<div class='kpi'><div class='label'>{lab}</div>"
                   f"<div class='val'>{rec['team']}</div>"
                   f"<div style='color:var(--flood)'>{rec['p']:.1%} to win</div></div>",
                   unsafe_allow_html=True)

    st.markdown("### Title odds — all contenders")
    showing = title[title["p"] > 0.004].head(20)
    st.altair_chart(bar(showing, "p", "team"), use_container_width=True)

    # a "dark horse" callout: highest title prob outside the top 6
    dark = title.iloc[6:].head(1).to_dict("records")
    if dark:
        st.info(f"Dark horse to watch: **{dark[0]['team']}** at {dark[0]['p']:.1%} — "
                "outside the headline favourites but with a real puncher's chance.")


# ---------------------------------------------------------------------------
# PAGE 2 — Group Explorer
# ---------------------------------------------------------------------------
elif page == "Group Explorer":
    st.markdown("<div class='eyebrow'>Group stage</div>", unsafe_allow_html=True)
    st.markdown("# Group Explorer")
    gid = st.selectbox("Pick a group", list(GROUPS.keys()),
                       format_func=lambda g: f"Group {g}")
    teams = GROUPS[gid]
    rows = []
    for t in teams:
        rows.append({
            "Team": t,
            "Win group": sim["group_win"].get(t, 0.0),
            "Advance to R32": sim["advance"].get(t, 0.0),
        })
    df = pd.DataFrame(rows).sort_values("Win group", ascending=False)

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"### Group {gid} — chance to win the group")
        st.altair_chart(bar(df.rename(columns={"Win group": "p"}), "p", "Team"),
                        use_container_width=True)
    with right:
        st.markdown("### Chance to reach the knockouts")
        st.write("Top two advance automatically, plus the eight best third-place "
                 "teams across all groups — so 'advance' can exceed the obvious two.")
        disp = df.copy()
        disp["Win group"] = (disp["Win group"] * 100).round(1).astype(str) + "%"
        disp["Advance to R32"] = (disp["Advance to R32"] * 100).round(1).astype(str) + "%"
        st.dataframe(disp, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 3 — Match Predictor
# ---------------------------------------------------------------------------
elif page == "Match Predictor":
    st.markdown("<div class='eyebrow'>Head to head</div>", unsafe_allow_html=True)
    st.markdown("# Match Predictor")
    st.write("Pick any two teams to see how the model expects the game to go. "
             "Neutral venue, as at a World Cup.")

    c1, c2 = st.columns(2)
    home = c1.selectbox("Team 1", sorted(ALL_TEAMS), index=sorted(ALL_TEAMS).index("Brazil"))
    away = c2.selectbox("Team 2", sorted(ALL_TEAMS), index=sorted(ALL_TEAMS).index("France"))

    if home == away:
        st.warning("Pick two different teams.")
    else:
        o = outcome(home, away)
        prob_df = pd.DataFrame({
            "result": [f"{home} win", "Draw", f"{away} win"],
            "p": [o["home_win"], o["draw"], o["away_win"]],
        })
        st.markdown("### Result probability")
        ch = (alt.Chart(prob_df).mark_bar()
              .encode(
                  x=alt.X("p:Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
                  color=alt.Color("result:N",
                                  scale=alt.Scale(range=[FLOOD, MUTED, "#7fa8d0"]),
                                  legend=alt.Legend(title=None, orient="bottom")),
                  tooltip=[alt.Tooltip("p:Q", format=".1%"), "result:N"])
              .properties(height=70))
        st.altair_chart(ch.configure_view(strokeWidth=0)
                        .configure_axis(labelColor=CHALK), use_container_width=True)

        a, b = st.columns(2)
        a.markdown(f"<div class='kpi'><div class='label'>{home} expected goals</div>"
                   f"<div class='val'>{o['xg_home']:.2f}</div></div>", unsafe_allow_html=True)
        b.markdown(f"<div class='kpi'><div class='label'>{away} expected goals</div>"
                   f"<div class='val'>{o['xg_away']:.2f}</div></div>", unsafe_allow_html=True)

        st.markdown("### Most likely scorelines")
        grid = o["grid"][:6, :6]
        sl = []
        for i in range(6):
            for j in range(6):
                sl.append({"h": i, "a": j, "p": float(grid[i, j])})
        sldf = pd.DataFrame(sl)
        heat = (alt.Chart(sldf).mark_rect()
                .encode(
                    x=alt.X("a:O", title=f"{away} goals"),
                    y=alt.Y("h:O", title=f"{home} goals"),
                    color=alt.Color("p:Q", scale=alt.Scale(scheme="yellowgreen"),
                                    legend=None),
                    tooltip=[alt.Tooltip("p:Q", format=".1%")])
                .properties(height=260))
        st.altair_chart(heat.configure_axis(labelColor=CHALK, titleColor=CHALK),
                        use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 4 — Fixtures
# ---------------------------------------------------------------------------
elif page == "Fixtures":
    st.markdown("<div class='eyebrow'>The calendar</div>", unsafe_allow_html=True)
    st.markdown("# Fixtures")
    st.write("All 104 matches, 11 June to 19 July. Times are US Eastern (UTC-4) "
             "as published. Knockout rows show the bracket slots until teams are known.")

    view = st.radio("Show", ["Group stage", "Knockout stage"], horizontal=True)

    if view == "Group stage":
        df = pd.DataFrame(GROUP_STAGE,
                          columns=["Date", "Time (ET)", "Group", "Match", "Venue"])
        groups_pick = st.multiselect("Filter by group", sorted(GROUPS.keys()),
                                     default=[])
        if groups_pick:
            df = df[df["Group"].isin(groups_pick)]
        st.dataframe(df, hide_index=True, use_container_width=True, height=560)
    else:
        df = pd.DataFrame(KNOCKOUTS,
                          columns=["Date", "Time (ET)", "Round", "Matchup", "Venue"])
        st.dataframe(df, hide_index=True, use_container_width=True, height=560)
        st.caption("Round of 32 begins 28 June. Final: 19 July, New York/NJ "
                   "(MetLife Stadium).")


# ---------------------------------------------------------------------------
# PAGE 5 — How this works
# ---------------------------------------------------------------------------
else:
    st.markdown("<div class='eyebrow'>Under the hood</div>", unsafe_allow_html=True)
    st.markdown("# How this works")
    st.markdown("""
**The model.** Each team gets an attacking and defensive strength estimated from
recent international results, weighted so newer games count for more. We blend
that with World Football Elo ratings so even teams with thin recent records get
sensible numbers. From those strengths we get expected goals for any matchup,
and model goals as Poisson-distributed to read off win / draw / loss and likely
scorelines.

**The tournament numbers.** We simulate the entire 2026 World Cup many thousands
of times — every group game, the eight-best-third-place rule, and the knockout
bracket through to the final on 19 July. Counting how often each team wins its
group or lifts the trophy gives the probabilities you see.

**What it is and isn't.** This is a guide, not a guarantee — football is famously
upset-prone, which is exactly why the league is fun. Treat the model as a smart
second opinion next to your own knowledge.

**The bracket is the real one.** The knockout tree (Round of 32 through the
final) follows the official FIFA schedule exactly, including the rule that the
eight best third-place teams slot into specific Round-of-32 matches. So the
knockout path a team faces in the simulation matches the actual tournament
structure.
""")
