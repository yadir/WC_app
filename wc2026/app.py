"""
app.py — World Cup 2026 Explorer.

A dark, editorial-style dashboard for an office predictions league: title odds,
a supercomputer-style projections table, group breakdowns, a head-to-head match
predictor, and the full fixture calendar — all driven by an offline Monte Carlo
model (see src/precompute.py).

Run locally:   streamlit run app.py
It reads two precomputed files (run `python -m src.precompute` first):
    data/sim_results.json   data/model_params.json
"""

import json
import os
import math
import numpy as np
import pandas as pd
import streamlit as st

from src.config import GROUPS, ALL_TEAMS
from src.schedule import GROUP_STAGE, KNOCKOUTS
from src.flags import flag_url, canonical

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}

st.set_page_config(page_title="Known World Cup 2026 Predictions Portal",
                   page_icon="🟠", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Theme — dark + orange, editorial. All custom; no default Streamlit chrome.
# ---------------------------------------------------------------------------
ACCENT = "#ff6a1f"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
  --bg:#0a0a0b; --panel:#0e0e11; --card:#15151a; --card2:#1c1c23;
  --border:#272730; --text:#fafafa; --muted:#92929e;
  --accent:#ff6a1f; --accent2:#ff9352; --accent-dim:rgba(255,106,31,.14);
}

/* Keep Streamlit's native header (and its sidebar collapse/expand buttons)
   fully working. Only hide the clutter items, leaving the toggle untouched. */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"]{display:none !important;}
[data-testid="stHeader"]{background:transparent !important;}
.block-container{padding-top:2.6rem; padding-bottom:4rem; max-width:1180px;}

html, body, .stApp{
  background:
    radial-gradient(1100px 520px at 82% -8%, rgba(255,106,31,.10), transparent 60%),
    radial-gradient(900px 600px at -5% 0%, rgba(255,106,31,.05), transparent 55%),
    var(--bg);
  color:var(--text);
  font-family:'Inter',system-ui,sans-serif;
}

h1,h2,h3,h4{font-family:'Space Grotesk','Inter',sans-serif; color:var(--text); letter-spacing:-.01em;}
h1{font-weight:700; font-size:2.6rem; line-height:1.04; margin:.1rem 0 .2rem;}
a{color:var(--accent2);}

/* sidebar — locked permanently open (no collapsing) */
[data-testid="stSidebar"]{background:#08080a; border-right:1px solid var(--border);
  transform:none !important; visibility:visible !important;
  width:250px !important; min-width:250px !important; max-width:250px !important;
  margin-left:0 !important; left:0 !important;}
[data-testid="stSidebar"][aria-expanded="false"]{transform:none !important; margin-left:0 !important;}
[data-testid="stSidebar"] .block-container{padding-top:1.6rem;}
[data-testid="stSidebar"] *{color:var(--text);}
/* remove every collapse / expand affordance */
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stExpandSidebarButton"],
[data-testid="baseButton-headerNoPadding"]{display:none !important;}

/* eyebrow / kickers */
.kicker{font:600 .72rem/1 'Space Grotesk',sans-serif; text-transform:uppercase;
  letter-spacing:.28em; color:var(--accent); margin-bottom:.5rem;}
.sub{color:var(--muted); font-size:.97rem; max-width:60ch;}

/* brand block in sidebar */
.brand{display:flex; align-items:center; gap:.6rem; margin-bottom:.2rem;}
.brand .dot{width:12px;height:12px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 14px 2px rgba(255,106,31,.6);}
.brand .bt{font:700 1.35rem/1 'Space Grotesk',sans-serif; letter-spacing:.12em;}
.brand-sub{color:var(--muted); font-size:.7rem; letter-spacing:.16em; line-height:1.5;
  text-transform:uppercase; margin:.35rem 0 1.1rem 1.35rem;}

/* cards */
.card{background:linear-gradient(180deg,var(--card),#101015);
  border:1px solid var(--border); border-radius:16px; padding:1.1rem 1.2rem;}
.fav{position:relative; overflow:hidden;}
.fav .lab{font:600 .68rem/1 'Space Grotesk'; text-transform:uppercase;
  letter-spacing:.22em; color:var(--muted);}
.fav .team{display:flex; align-items:center; gap:.6rem; margin:.7rem 0 .3rem;}
.fav .team b{font:700 1.5rem/1 'Space Grotesk';}
.fav .pct{font:700 2.5rem/1 'Space Grotesk'; color:var(--accent);}
.fav .pl{color:var(--muted); font-size:.8rem; margin-top:.25rem;}
.fav .glow{position:absolute; right:-40px; top:-40px; width:140px; height:140px;
  border-radius:50%; background:radial-gradient(circle,rgba(255,106,31,.28),transparent 70%);}

.flag{border-radius:3px; box-shadow:0 0 0 1px rgba(255,255,255,.08);
  object-fit:cover; vertical-align:middle;}

/* horizontal odds rows */
.rows{display:flex; flex-direction:column; gap:.34rem; margin-top:.4rem;}
.r{display:grid; grid-template-columns:22px 30px 152px 1fr 58px; align-items:center;
  gap:.55rem; padding:.18rem .2rem; border-radius:8px;}
.r:hover{background:rgba(255,255,255,.03);}
.r .rk{color:var(--muted); font:600 .82rem 'Space Grotesk'; text-align:right;}
.r .nm{font-weight:600; font-size:.92rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.r .track{height:9px; border-radius:6px; background:var(--card2); overflow:hidden;}
.r .fill{height:100%; border-radius:6px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));}
.r .vv{text-align:right; font:700 .9rem 'Space Grotesk'; color:var(--text);}

/* group chip */
.chip{display:inline-block; min-width:20px; text-align:center; font:700 .68rem 'Space Grotesk';
  color:var(--accent); background:var(--accent-dim); border:1px solid rgba(255,106,31,.3);
  border-radius:6px; padding:.1rem .34rem;}

/* projections table */
.ptbl{width:100%; border-collapse:separate; border-spacing:0 0; font-size:.86rem;}
.ptbl thead th{position:sticky; top:0; background:#0c0c0f; color:var(--muted);
  font:600 .68rem 'Space Grotesk'; text-transform:uppercase; letter-spacing:.1em;
  padding:.7rem .5rem; text-align:center; border-bottom:1px solid var(--border);}
.ptbl thead th.l{text-align:left;}
.ptbl td{padding:.42rem .5rem; text-align:center; border-bottom:1px solid #161620;}
.ptbl td.team{text-align:left; white-space:nowrap;}
.ptbl tr:hover td{background:rgba(255,255,255,.022);}
.ptbl .tnm{font-weight:600; margin-left:.5rem; vertical-align:middle;}
.ptbl .cell{border-radius:7px; font:700 .82rem 'Space Grotesk'; padding:.34rem 0; display:block;}

/* VS predictor */
.vs{display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:1rem;
  background:linear-gradient(180deg,var(--card),#101015); border:1px solid var(--border);
  border-radius:18px; padding:1.4rem 1.2rem;}
.vs .side{display:flex; flex-direction:column; align-items:center; gap:.5rem;}
.vs .side b{font:700 1.15rem 'Space Grotesk'; text-align:center;}
.vs .mid{font:700 1rem 'Space Grotesk'; color:var(--muted);}
.seg{display:flex; height:38px; border-radius:10px; overflow:hidden; margin:.2rem 0 .1rem;
  border:1px solid var(--border);}
.seg span{display:flex; align-items:center; justify-content:center;
  font:700 .8rem 'Space Grotesk'; color:#0a0a0b;}
.legend{display:flex; gap:1.1rem; color:var(--muted); font-size:.78rem; margin-top:.4rem;}
.legend i{display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:.35rem;}

/* scoreline heatmap grid */
.heat{display:grid; gap:3px;}
.heat .h{font:600 .7rem 'Space Grotesk'; color:var(--muted); text-align:center;}
.heat .c{aspect-ratio:1/1; border-radius:5px; display:flex; align-items:center;
  justify-content:center; font:600 .68rem 'Space Grotesk';}

/* fixtures */
.fx{display:flex; flex-direction:column; gap:.3rem;}
.fxr{display:grid; grid-template-columns:96px 64px 30px 1fr 130px; align-items:center;
  gap:.6rem; padding:.5rem .7rem; border:1px solid var(--border); border-radius:10px;
  background:var(--card); font-size:.86rem;}
.fxr .dt{color:var(--text); font-weight:600;}
.fxr .tm{color:var(--muted); font-size:.8rem;}
.fxr .mu{display:flex; align-items:center; gap:.45rem; flex-wrap:wrap;}
.fxr .vn{color:var(--muted); font-size:.8rem; text-align:right;}
.fxr.kn{grid-template-columns:96px 64px 1fr 130px;}
.rnd{display:inline-block; font:700 .64rem 'Space Grotesk'; color:var(--accent);
  background:var(--accent-dim); border-radius:5px; padding:.08rem .4rem; margin-right:.4rem;}

/* streamlit widget polish */
[data-testid="stWidgetLabel"] p{color:var(--muted); font-size:.8rem; font-weight:600;
  text-transform:uppercase; letter-spacing:.08em;}
div[data-baseweb="select"]>div{background:var(--card); border-color:var(--border);}
.stRadio [role="radiogroup"]{gap:.2rem;}
hr{border-color:var(--border);}
.note{background:var(--accent-dim); border:1px solid rgba(255,106,31,.3);
  border-radius:12px; padding:.85rem 1rem; color:var(--text); font-size:.9rem;}

/* how-it-works: stat tiles */
.stats{display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; margin:.4rem 0 1.6rem;}
.stat{background:linear-gradient(180deg,var(--card),#101015); border:1px solid var(--border);
  border-radius:14px; padding:1rem 1.1rem; position:relative; overflow:hidden;}
.stat .num{font:700 1.9rem/1 'Space Grotesk'; color:var(--accent);}
.stat .lab{color:var(--muted); font-size:.78rem; margin-top:.35rem; line-height:1.35;}
.stat::after{content:''; position:absolute; right:-30px; bottom:-30px; width:90px; height:90px;
  border-radius:50%; background:radial-gradient(circle,rgba(255,106,31,.16),transparent 70%);}

/* how-it-works: info cards */
.info{background:linear-gradient(180deg,var(--card),#101015); border:1px solid var(--border);
  border-radius:16px; padding:1.2rem 1.3rem; height:100%;}
.info h4{display:flex; align-items:center; gap:.55rem; margin:0 0 .55rem; font-size:1.08rem;}
.info .ic{width:30px; height:30px; border-radius:9px; background:var(--accent-dim);
  border:1px solid rgba(255,106,31,.35); display:flex; align-items:center; justify-content:center;
  font-size:1rem;}
.info p{color:var(--muted); font-size:.9rem; line-height:1.6; margin:.4rem 0;}
.info .tag{color:var(--accent2); font-weight:600;}

/* how-it-works: numbered steps */
.step{display:flex; gap:.9rem; padding:1rem 1.1rem; border:1px solid var(--border);
  border-radius:14px; background:var(--card); margin-bottom:.6rem;}
.step .bdg{flex:0 0 34px; height:34px; border-radius:10px; font:700 1.05rem 'Space Grotesk';
  color:#120a04; background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex; align-items:center; justify-content:center;}
.step .bd b{font-family:'Space Grotesk'; font-size:1rem;}
.step .bd p{color:var(--muted); font-size:.9rem; line-height:1.55; margin:.25rem 0 0;}
.sec{font:600 .72rem/1 'Space Grotesk'; text-transform:uppercase; letter-spacing:.2em;
  color:var(--muted); margin:1.8rem 0 .9rem; padding-bottom:.5rem; border-bottom:1px solid var(--border);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data
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
    st.error("No precomputed data found. Run `python -m src.precompute` first.")
    st.stop()


# ---------------------------------------------------------------------------
# Model helpers (inline so the app is standalone). Keep in sync with src/model.py
# ---------------------------------------------------------------------------
MAX_GOALS = 8
ELO_BLEND = 0.70
ELO_MULT_SPREAD = 1.30
HOST_TEAMS = {"United States", "Canada", "Mexico"}   # keep in sync with src/model.py
HOST_ELO_BONUS = 50


def _elo_mult(home, away):
    eh = params["elo"].get(home, 1500) + (HOST_ELO_BONUS if home in HOST_TEAMS else 0)
    ea = params["elo"].get(away, 1500) + (HOST_ELO_BONUS if away in HOST_TEAMS else 0)
    exp_home = 1 / (1 + 10 ** ((ea - eh) / 400))
    return (1.0 + ELO_MULT_SPREAD * (exp_home - 0.5),
            1.0 + ELO_MULT_SPREAD * ((1 - exp_home) - 0.5))


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
# Presentation helpers
# ---------------------------------------------------------------------------
def flag(team, h=18):
    url = flag_url(team)
    if not url:
        return ""
    return f"<img class='flag' src='{url}' style='height:{h}px'>"


def heat(v):
    """Background + text colour for a probability cell (0..1), orange scale."""
    a = max(0.0, min(1.0, float(v)))
    bg = f"rgba(255,106,31,{0.05 + 0.8 * a})"
    fg = "#120a04" if a > 0.5 else ("var(--text)" if a > 0.06 else "var(--muted)")
    return bg, fg


def bar_rows(pairs, max_value=None):
    """pairs: list of (team, value). Render ranked horizontal bars."""
    mx = max_value or (max(v for _, v in pairs) if pairs else 1) or 1
    out = ["<div class='rows'>"]
    for i, (team, v) in enumerate(pairs, 1):
        w = max(2.0, 100 * v / mx)
        out.append(
            f"<div class='r'><div class='rk'>{i}</div>{flag(team,18)}"
            f"<div class='nm'>{team}</div>"
            f"<div class='track'><div class='fill' style='width:{w:.1f}%'></div></div>"
            f"<div class='vv'>{v*100:.1f}%</div></div>")
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    "<div class='brand'><span class='dot'></span>"
    "<span class='bt'>KNOWN</span></div>"
    "<div class='brand-sub'>World Cup 2026<br>Predictions Portal</div>",
    unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "Overview", "Predictions Table", "Group Explorer",
    "Match Predictor", "Fixtures", "How it works",
], label_visibility="collapsed", key="nav_page")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.caption(f"Model view from {sim['n_sims']:,} simulated tournaments. "
                   "Built for exploration — make your official picks in the league.")


def header(kicker, title, sub=None):
    st.markdown(f"<div class='kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if sub:
        st.markdown(f"<div class='sub'>{sub}</div>", unsafe_allow_html=True)
    st.write("")


title_sorted = sorted(sim["title"].items(), key=lambda x: -x[1])


# ===========================================================================
# PAGE — Overview
# ===========================================================================
if page == "Overview":
    header("The model's view", "Who wins the World Cup?",
           f"Every number is drawn from simulating the entire tournament "
           f"{sim['n_sims']:,} times. Sense-check your gut before you lock in picks.")

    cols = st.columns(3)
    labels = ["Favourite", "Second favourite", "Third favourite"]
    for col, (team, p), lab in zip(cols, title_sorted[:3], labels):
        col.markdown(
            f"<div class='card fav'><div class='glow'></div>"
            f"<div class='lab'>{lab}</div>"
            f"<div class='team'>{flag(team,30)}<b>{team}</b></div>"
            f"<div class='pct'>{p*100:.1f}%</div>"
            f"<div class='pl'>to lift the trophy · {GROUP_OF[team]} group</div></div>",
            unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown("### Title odds — all contenders")
        contenders = [(t, p) for t, p in title_sorted if p > 0.004][:22]
        st.markdown(bar_rows(contenders), unsafe_allow_html=True)
    with right:
        st.markdown("### Deepest runs")
        st.markdown("<div class='sub'>Chance to reach the final.</div>",
                    unsafe_allow_html=True)
        fin = sorted(sim["reach_final"].items(), key=lambda x: -x[1])[:10]
        st.markdown(bar_rows(fin), unsafe_allow_html=True)
        dark = title_sorted[6:7]
        if dark:
            t, p = dark[0]
            st.markdown(
                f"<div class='note' style='margin-top:1rem'>Dark horse "
                f"{flag(t,16)} <b>{t}</b> — {p*100:.1f}% to win it all, just "
                f"outside the headline favourites.</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE — Predictions Table (supercomputer grid)
# ===========================================================================
elif page == "Predictions Table":
    header("The supercomputer", "Predictions Table",
           "Every team's projected run. Each cell is the share of "
           f"{sim['n_sims']:,} simulations in which the team reaches that stage.")

    stages = [("Win group", "group_win"), ("Reach R32", "advance"),
              ("Reach R16", "reach_r16"), ("Reach QF", "reach_qf"),
              ("Reach SF", "reach_sf"), ("Final", "reach_final"),
              ("Win cup", "title")]

    c1, c2 = st.columns([1, 1])
    sort_label = c1.selectbox("Sort by", [s[0] for s in stages], index=6, key="pt_sort")
    pick = c2.multiselect("Filter by group", sorted(GROUPS.keys()), default=[], key="pt_groups")
    sort_key = dict(stages)[sort_label]

    rows = []
    for t in ALL_TEAMS:
        if pick and GROUP_OF[t] not in pick:
            continue
        rows.append((t, {k: sim.get(k, {}).get(t, 0.0) for _, k in stages}))
    rows.sort(key=lambda x: -x[1][sort_key])

    head = ("<table class='ptbl'><thead><tr>"
            "<th class='l'>#</th><th class='l'>Team</th><th>Grp</th>"
            + "".join(f"<th>{lab}</th>" for lab, _ in stages)
            + "</tr></thead><tbody>")
    body = []
    for i, (t, d) in enumerate(rows, 1):
        cells = []
        for _, k in stages:
            bg, fg = heat(d[k])
            cells.append(f"<td><span class='cell' style='background:{bg};color:{fg}'>"
                         f"{d[k]*100:.1f}</span></td>")
        body.append(
            f"<tr><td class='l' style='color:var(--muted)'>{i}</td>"
            f"<td class='team'>{flag(t,18)}<span class='tnm'>{t}</span></td>"
            f"<td><span class='chip'>{GROUP_OF[t]}</span></td>"
            + "".join(cells) + "</tr>")
    st.markdown(head + "".join(body) + "</tbody></table>", unsafe_allow_html=True)
    st.caption("Values are percentages. 'Reach R32' = survive the group "
               "(top two, or one of the eight best third-placed teams).")


# ===========================================================================
# PAGE — Group Explorer
# ===========================================================================
elif page == "Group Explorer":
    header("Group stage", "Group Explorer",
           "Top two of each group advance automatically, plus the eight best "
           "third-placed teams — so a team's chance to advance can beat the odds "
           "of finishing top two.")

    gid = st.selectbox("Choose a group", list(GROUPS.keys()),
                       format_func=lambda g: f"Group {g}", key="ge_group")
    teams = GROUPS[gid]
    gw = sorted([(t, sim["group_win"].get(t, 0.0)) for t in teams], key=lambda x: -x[1])
    adv = sorted([(t, sim["advance"].get(t, 0.0)) for t in teams], key=lambda x: -x[1])

    left, right = st.columns(2)
    with left:
        st.markdown(f"### Win Group {gid}")
        st.markdown(bar_rows(gw, max_value=1.0), unsafe_allow_html=True)
    with right:
        st.markdown("### Reach the knockouts")
        st.markdown(bar_rows(adv, max_value=1.0), unsafe_allow_html=True)


# ===========================================================================
# PAGE — Match Predictor
# ===========================================================================
elif page == "Match Predictor":
    header("Head to head", "Match Predictor",
           "Pick any two teams to see how the model expects the game to go. "
           "Neutral venue — except the co-hosts (USA, Canada, Mexico), who carry "
           "a slight home advantage.")

    c1, c2 = st.columns(2)
    teams_sorted = sorted(ALL_TEAMS)
    home = c1.selectbox("Team 1", teams_sorted, index=teams_sorted.index("Spain"), key="mp_home")
    away = c2.selectbox("Team 2", teams_sorted, index=teams_sorted.index("France"), key="mp_away")

    if home == away:
        st.warning("Pick two different teams.")
    else:
        o = outcome(home, away)
        hw, dr, aw = o["home_win"], o["draw"], o["away_win"]
        st.write("")
        st.markdown(
            f"<div class='vs'>"
            f"<div class='side'>{flag(home,46)}<b>{home}</b></div>"
            f"<div class='mid'>VS</div>"
            f"<div class='side'>{flag(away,46)}<b>{away}</b></div></div>",
            unsafe_allow_html=True)

        st.write("")
        st.markdown("### Result probability")
        st.markdown(
            f"<div class='seg'>"
            f"<span style='width:{hw*100:.1f}%;background:linear-gradient(90deg,#ff6a1f,#ff9352)'>"
            f"{hw*100:.0f}%</span>"
            f"<span style='width:{dr*100:.1f}%;background:#3a3a44;color:#fafafa'>{dr*100:.0f}%</span>"
            f"<span style='width:{aw*100:.1f}%;background:linear-gradient(90deg,#3d7ea6,#5aa0cc)'>"
            f"{aw*100:.0f}%</span></div>"
            f"<div class='legend'>"
            f"<span><i style='background:#ff6a1f'></i>{home} win</span>"
            f"<span><i style='background:#3a3a44'></i>Draw</span>"
            f"<span><i style='background:#3d7ea6'></i>{away} win</span></div>",
            unsafe_allow_html=True)

        st.write("")
        a, b = st.columns(2)
        a.markdown(f"<div class='card'><div class='fav lab'>{home} expected goals</div>"
                   f"<div class='pct' style='font-size:2rem;margin-top:.3rem'>{o['xg_home']:.2f}</div></div>",
                   unsafe_allow_html=True)
        b.markdown(f"<div class='card'><div class='fav lab'>{away} expected goals</div>"
                   f"<div class='pct' style='font-size:2rem;margin-top:.3rem'>{o['xg_away']:.2f}</div></div>",
                   unsafe_allow_html=True)

        st.write("")
        st.markdown("### Most likely scorelines")
        n = 6
        g = o["grid"][:n, :n]
        gmax = g.max()
        cells = ["<div class='heat' style='grid-template-columns:30px repeat(%d,1fr)'>" % n]
        cells.append("<div class='h'></div>")
        for j in range(n):
            cells.append(f"<div class='h'>{j}</div>")
        for i in range(n):
            cells.append(f"<div class='h' style='display:flex;align-items:center;justify-content:center'>{i}</div>")
            for j in range(n):
                v = g[i, j] / gmax if gmax else 0
                bg, fg = heat(v)
                cells.append(f"<div class='c' style='background:{bg};color:{fg}'>{g[i,j]*100:.0f}</div>")
        cells.append("</div>")
        st.markdown("".join(cells), unsafe_allow_html=True)
        st.caption(f"Rows = {home} goals, columns = {away} goals. Cell values are % chance "
                   "of that exact scoreline.")


# ===========================================================================
# PAGE — Fixtures
# ===========================================================================
elif page == "Fixtures":
    header("The calendar", "Fixtures",
           "All 104 matches, 11 June to 19 July. Times are US Eastern (UTC-4). "
           "Knockout rows show bracket slots until teams are decided.")

    view = st.radio("View", ["Group stage", "Knockout stage"], horizontal=True,
                    label_visibility="collapsed", key="fx_view")

    def matchup_html(text):
        parts = text.split(" v ")
        if len(parts) == 2:
            a, b = parts
            return (f"{flag(a,15)} <span>{a.strip()}</span>"
                    f"<span style='color:var(--muted);margin:0 .25rem'>v</span>"
                    f"{flag(b,15)} <span>{b.strip()}</span>")
        return f"<span>{text}</span>"

    if view == "Group stage":
        gpick = st.multiselect("Filter by group", sorted(GROUPS.keys()), default=[], key="fx_groups")
        out = ["<div class='fx'>"]
        for date, time, grp, match, venue in GROUP_STAGE:
            if gpick and grp not in gpick:
                continue
            out.append(
                f"<div class='fxr'><div class='dt'>{date}</div>"
                f"<div class='tm'>{time}</div><div><span class='chip'>{grp}</span></div>"
                f"<div class='mu'>{matchup_html(match)}</div>"
                f"<div class='vn'>{venue}</div></div>")
        out.append("</div>")
        st.markdown("".join(out), unsafe_allow_html=True)
    else:
        out = ["<div class='fx'>"]
        for date, time, rnd, match, venue in KNOCKOUTS:
            label = rnd.split()[0]
            out.append(
                f"<div class='fxr kn'><div class='dt'>{date}</div>"
                f"<div class='tm'>{time}</div>"
                f"<div class='mu'><span class='rnd'>{label}</span>"
                f"<span style='color:var(--muted)'>{match}</span></div>"
                f"<div class='vn'>{venue}</div></div>")
        out.append("</div>")
        st.markdown("".join(out), unsafe_allow_html=True)
        st.caption("Round of 32 begins 28 June. Final: 19 July, New York/NJ (MetLife Stadium).")


# ===========================================================================
# PAGE — How it works
# ===========================================================================
else:
    header("Under the hood", "How it works",
           "No black box. Here's exactly what data feeds the portal and how it "
           "turns into the probabilities you see on every page.")

    st.markdown(
        "<div class='stats'>"
        "<div class='stat'><div class='num'>49k+</div>"
        "<div class='lab'>international matches, 1872–2026</div></div>"
        "<div class='stat'><div class='num'>48</div>"
        "<div class='lab'>teams, each with a live Elo rating</div></div>"
        "<div class='stat'><div class='num'>10,000</div>"
        "<div class='lab'>full tournaments simulated</div></div>"
        "<div class='stat'><div class='num'>104</div>"
        "<div class='lab'>matches modelled, group to final</div></div>"
        "</div>", unsafe_allow_html=True)

    # ---- The data ----
    st.markdown("<div class='sec'>The data — what goes in</div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    d1.markdown(
        "<div class='info'><h4><span class='ic'>📊</span>Match results</h4>"
        "<p>Every men's full international since the first ever match in 1872 — "
        "<span class='tag'>~49,000 games</span> with date, teams, score and venue.</p>"
        "<p><b>What it tells us:</b> how many goals a team tends to score and concede "
        "against real opposition, and whether they're trending up or down. Recent "
        "matches count for much more than old ones, so current form dominates.</p></div>",
        unsafe_allow_html=True)
    d2.markdown(
        "<div class='info'><h4><span class='ic'>🌍</span>Elo ratings</h4>"
        "<p>A single <span class='tag'>strength number</span> per team from the World "
        "Football Elo Ratings — the same family of rating used for chess.</p>"
        "<p><b>What it tells us:</b> overall quality on one scale, so a team is judged "
        "by <i>who</i> they beat, not just how many goals they piled up on weaker sides. "
        "This keeps minnows who thrash other minnows from looking like contenders.</p></div>",
        unsafe_allow_html=True)

    # ---- The model ----
    st.markdown("<div class='sec'>The model — how it works</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='step'><div class='bdg'>1</div><div class='bd'>"
        "<b>Rate every team</b>"
        "<p>From the results we estimate each team's attacking and defensive strength "
        "(time-weighted toward recent games), then blend it heavily with their Elo "
        "rating. The blend is the key dial — leaning on Elo keeps the favourites "
        "realistic instead of over-rating goal-padding. The three co-hosts (USA, "
        "Canada, Mexico) get a slight home-soil bonus on top.</p></div></div>"

        "<div class='step'><div class='bdg'>2</div><div class='bd'>"
        "<b>Predict a single match</b>"
        "<p>For any two teams we compute the goals each is expected to score, then treat "
        "goals as random (a Poisson distribution). That gives the full spread of "
        "scorelines — and from it, win / draw / loss odds. It's exactly what powers the "
        "Match Predictor.</p></div></div>"

        "<div class='step'><div class='bdg'>3</div><div class='bd'>"
        "<b>Simulate the whole tournament — 10,000 times</b>"
        "<p>We play out all 104 matches: every group game with the real FIFA tiebreakers "
        "and the eight-best-third-place rule, then the actual knockout bracket through the "
        "final on 19 July. Run that 10,000 times and count how often each team tops its "
        "group, advances, or lifts the trophy. Those frequencies are the percentages you "
        "see everywhere in the portal.</p></div></div>",
        unsafe_allow_html=True)

    # ---- Reading it ----
    st.markdown("<div class='sec'>Reading the numbers</div>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    r1.markdown(
        "<div class='info'><h4><span class='ic'>🎯</span>Calibrated to the field</h4>"
        "<p>The model is tuned so its favourites resemble the big public supercomputer "
        "projections — but it's an <b>independent</b> model. Expect the same shape, not "
        "identical decimals.</p></div>", unsafe_allow_html=True)
    r2.markdown(
        "<div class='info'><h4><span class='ic'>🎲</span>A guide, not a guarantee</h4>"
        "<p>Football is gloriously upset-prone — that's the whole fun of the league. "
        "Treat these as a smart second opinion next to your own gut.</p></div>",
        unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='note'>Sources — results: open <b>martj42</b> "
                "international-results dataset · ratings: <b>eloratings.net</b> · "
                "flags: <b>flagcdn.com</b>.</div>", unsafe_allow_html=True)
