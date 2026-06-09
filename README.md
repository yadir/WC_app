# Known World Cup 2026 Predictions Portal

An exploratory dashboard that shows a model's view of the 2026 World Cup —
title odds, group-winner probabilities, and head-to-head match predictions — to
help colleagues make informed picks in the office predictions league.

It is **read-only and exploratory by design**: no picks submission, no
leaderboard. People explore here, then enter their official picks in the league.

## What's inside

```
wc2026/
├── app.py                 # Streamlit app (5 pages)
├── make_sample_data.py    # generates synthetic results.csv for an instant demo
├── requirements.txt
├── data/                  # CSV inputs + precomputed JSON outputs
└── src/
    ├── config.py          # the REAL confirmed 2026 group draw + format rules
    ├── bracket.py         # the REAL knockout tree (M73-M104) + R32 slot pairings
    ├── schedule.py        # the full 104-match schedule (dates, venues, times)
    ├── data.py            # load historical results + Elo
    ├── model.py           # Poisson attack/defence + Elo blend
    ├── simulate.py        # Monte Carlo sim using the exact bracket
    └── precompute.py      # fit model, run sims, write JSON for the app
```

The app has five pages: Tournament Overview (title odds), Group Explorer
(per-group probabilities), Match Predictor (any two teams head-to-head),
Fixtures (the full real schedule), and How this works.

## Quickstart (demo, ~2 minutes)

```bash
pip install -r requirements.txt
python make_sample_data.py        # synthetic data so it runs immediately
python -m src.precompute          # fit + simulate -> data/*.json
streamlit run app.py
```

Open the local URL Streamlit prints. You'll see real probabilities computed from
(synthetic) data, with the real 2026 groups.

## Using real data

1. **Historical results** — download the Kaggle dataset *"International football
   results from 1872 to present"* and save it as `data/results.csv`.
2. **Elo ratings** (optional but recommended) — save a CSV `data/elo.csv` with
   columns `team,elo` from World Football Elo Ratings (eloratings.net). If you
   skip this, the model derives a rough Elo from the results.
3. Check team-name spellings: open `src/config.py` and make sure each tournament
   team matches the dataset's spelling, adding any needed entries to
   `TEAM_ALIASES`. The precompute step prints a warning listing any team it
   couldn't find data for.
4. Re-run `python -m src.precompute`, then `streamlit run app.py`.

## Deploy (free)

**Streamlit Community Cloud** is the easiest:

1. Push this folder to a GitHub repo (include the precomputed `data/*.json` so
   the deployed app doesn't need to run simulations).
2. Go to share.streamlit.io, connect the repo, set `app.py` as the entry point.
3. It's live on a public URL in a couple of minutes.

The app only reads the precomputed JSON, so it stays fast and never times out.
After a match day, re-run `precompute` locally, commit the new JSON, and the
deployed app updates.

> If you'd rather keep it internal-only, check with IT about hosting behind SSO
> or on internal infra instead of a public Streamlit URL.

## Notes and honest caveats

- **The knockout bracket is the real one.** The Round-of-32 pairings and the
  full tree (R32 → R16 → QF → SF → Final, matches M73–M104) are transcribed from
  the official FIFA schedule in `src/bracket.py`. The eight best third-place
  teams are assigned to their specific R32 slots with a constraint solver that
  respects each slot's eligible groups, matching FIFA's Annex C guarantee.
- **The Poisson fit is a fast approximation** (weighted goal averages) rather
  than a full Dixon-Coles maximum-likelihood fit. It's stable and good for a fun
  predictor; upgrade `model.fit()` if you want more rigour.
- It's a guide, not a guarantee. Upsets are the whole point of the tournament.

## The model in one line

Historical results → team attack/defence strengths (time-weighted) → blended
with Elo → Poisson match probabilities → Monte Carlo the 48-team bracket many
thousands of times → read off every probability the app shows.
