# MLB Daily Prediction System

Predicts three betting markets for every game on today's MLB slate:

| Market | Model | Output |
|---|---|---|
| Moneyline | XGBClassifier | Winner + confidence % (`predict_proba`) |
| Total runs | 3x XGBRegressor | Away runs, home runs, combined total |
| NRFI / YRFI | XGBClassifier | Call + confidence % (`predict_proba`) |

## Setup

```bash
pip install -r requirements.txt
python mlb_predictor.py            
python mlb_predictor.py --date 2026-07-12
```

If `xgboost` is not installed the script silently falls back to
scikit-learn Random Forests.

## Data pipeline

1. **Schedule** — `statsapi.schedule()` gives game_pk, teams, probable
   starters; `statsapi.lookup_player()` resolves MLBAM pitcher IDs;
   the live boxscore gives batting orders when lineups are posted.
2. **Moneyline features** — last-30-day K/9, BB/9, GB% from
   `pybaseball.statcast_pitcher()`; season xFIP from FanGraphs
   `pitching_stats()`; team wRC+/OBP from `team_batting()`.
3. **Totals features** — starters' expected runs allowed (xFIP proxy),
   last-14-day bullpen ERA from Baseball-Reference
   `pitching_stats_range()` (relievers only, grouped by team), and park
   factors (embedded FanGraphs-style constants — pybaseball has no
   stable park-factor endpoint).
4. **NRFI features** — 1st-inning-only Statcast slices: RA/9 from
   score deltas per pitch, 1st-inning walk rate, plus season OBP/ISO
   for batters 1–3 of each posted lineup.

Every fetch has a league-average fallback so an unannounced starter or
an API timeout never kills the slate.

## Training data: `mlb_history_2024_2025.csv`

One row per historical game. Required columns:

**Features**
`home_sp_k9, home_sp_bb9, home_sp_xfip, home_sp_gb_pct,
away_sp_k9, away_sp_bb9, away_sp_xfip, away_sp_gb_pct,
home_team_wrc_plus, home_team_obp, away_team_wrc_plus, away_team_obp,
home_sp_xra9, away_sp_xra9, combined_sp_xra9,
home_bullpen_era14, away_bullpen_era14, park_factor,
home_sp_fi_ra9, home_sp_fi_bb_rate, away_sp_fi_ra9, away_sp_fi_bb_rate,
home_top3_obp, home_top3_iso, away_top3_obp, away_top3_iso`

**Targets**
`home_win` (0/1), `home_runs`, `away_runs`, `total_runs`,
`first_inning_run` (0/1 — 1 means YRFI)

Feature values must be computed **as of game date** (no lookahead).
If the CSV is missing, the script trains on synthetic data and prints a
loud warning — useful for testing the pipeline, useless for betting.

## Known limitations

- FanGraphs advanced stats (xFIP, wRC+) are season-level; pybaseball
  cannot split them by arbitrary date range or batter handedness, so
  "vs. opposing starter handedness" uses season aggregates.
- FanGraphs Cloudflare-blocks some networks (HTTP 403). When that
  happens the script substitutes a Statcast-derived 30-day xFIP proxy
  for pitchers and MLB Stats API OBP + OPS-indexed wRC+ for teams.
- 1st-inning "ERA" is really RA/9 (unearned runs included) computed
  from Statcast score deltas.
- Cold-cache Statcast pulls take ~30–60 s per pitcher; `pybaseball`
  caching is enabled so subsequent runs are fast.
- This is a modeling exercise, not betting advice.
