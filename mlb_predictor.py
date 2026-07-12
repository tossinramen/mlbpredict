"""
mlb_predictor.py
================
End-to-end MLB daily prediction system.

Markets covered (separate model per market):
  1. Moneyline (outright winner)          -> XGBClassifier  (.predict_proba)
  2. Total runs (away / home / combined)  -> 3x XGBRegressor
  3. NRFI / YRFI (run in the 1st inning)  -> XGBClassifier  (.predict_proba)

Data sources:
  - MLB-StatsAPI (statsapi):  today's schedule, probable pitchers, lineups,
    player season splits (OBP / ISO).
  - pybaseball:  Statcast pitch-level data (last-30-day pitcher form,
    1st-inning splits), FanGraphs season stats (xFIP, wRC+, OBP),
    Baseball-Reference daily logs (last-14-day bullpen ERA).

Training data:
  Expects ./mlb_history_2024_2025.csv (schema documented in README.md).
  If the file is missing the script trains on SYNTHETIC data so the full
  pipeline can still be exercised -- a loud warning is printed.

Usage:
  python mlb_predictor.py                # today's slate
  python mlb_predictor.py --date 2026-07-12
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# Windows consoles often default to cp1252; keep accented names readable.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------
# Third-party data / ML libraries (graceful degradation where possible)
# --------------------------------------------------------------------------
import statsapi  # MLB-StatsAPI wrapper

import pybaseball
from pybaseball import (
    pitching_stats,           # FanGraphs season pitching (xFIP, ...)
    pitching_stats_range,     # Baseball-Reference daily logs (bullpen form)
    statcast_pitcher,         # Statcast pitch-level data by MLBAM id
    team_batting,             # FanGraphs team batting (wRC+, OBP)
)

pybaseball.cache.enable()  # cache Statcast/FanGraphs pulls between runs

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAVE_XGB = True
except ImportError:  # fall back to sklearn if xgboost is not installed
    from sklearn.ensemble import (
        RandomForestClassifier as XGBClassifier,
        RandomForestRegressor as XGBRegressor,
    )
    _HAVE_XGB = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# ==========================================================================
# Constants
# ==========================================================================
HISTORY_CSV = "mlb_history_2024_2025.csv"

# League-average fallbacks used whenever a specific stat cannot be fetched
# (unannounced starter, API timeout, rookie with no sample, ...).
LEAGUE_AVG = {
    "k9": 8.6, "bb9": 3.2, "xfip": 4.10, "gb_pct": 0.435,
    "wrc_plus": 100.0, "obp": 0.315,
    "bullpen_era14": 4.20, "park_factor": 1.00,
    "fi_ra9": 4.40, "fi_bb_rate": 0.085,   # 1st-inning run env is hotter
    "top3_obp": 0.335, "top3_iso": 0.165,  # top of order > league avg
}

# statsapi full team name -> FanGraphs abbreviation (for team_batting joins)
TEAM_TO_FG = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "OAK",
    "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG", "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}

# Runs park factors (FanGraphs 3-yr basic, 100 = neutral -> stored /100).
# pybaseball has no stable park-factor endpoint, so these are embedded and
# refreshed via _try_fetch_park_factors() when FanGraphs is reachable.
PARK_FACTORS = {
    "Arizona Diamondbacks": 1.03, "Atlanta Braves": 1.02,
    "Baltimore Orioles": 0.98, "Boston Red Sox": 1.06,
    "Chicago Cubs": 0.99, "Chicago White Sox": 1.01,
    "Cincinnati Reds": 1.07, "Cleveland Guardians": 0.97,
    "Colorado Rockies": 1.12, "Detroit Tigers": 0.96,
    "Houston Astros": 1.00, "Kansas City Royals": 1.02,
    "Los Angeles Angels": 0.99, "Los Angeles Dodgers": 1.01,
    "Miami Marlins": 0.97, "Milwaukee Brewers": 1.00,
    "Minnesota Twins": 0.99, "New York Mets": 0.96,
    "New York Yankees": 1.02, "Athletics": 0.98,
    "Oakland Athletics": 0.98, "Philadelphia Phillies": 1.02,
    "Pittsburgh Pirates": 0.97, "San Diego Padres": 0.96,
    "San Francisco Giants": 0.95, "Seattle Mariners": 0.93,
    "St. Louis Cardinals": 0.99, "Tampa Bay Rays": 0.98,
    "Texas Rangers": 1.01, "Toronto Blue Jays": 1.00,
    "Washington Nationals": 1.00,
}

# Feature schemas -- the training CSV must contain these exact columns.
ML_FEATURES = [
    "home_sp_k9", "home_sp_bb9", "home_sp_xfip", "home_sp_gb_pct",
    "away_sp_k9", "away_sp_bb9", "away_sp_xfip", "away_sp_gb_pct",
    "home_team_wrc_plus", "home_team_obp",
    "away_team_wrc_plus", "away_team_obp",
]
TOTALS_FEATURES = [
    "home_sp_xra9", "away_sp_xra9", "combined_sp_xra9",
    "home_bullpen_era14", "away_bullpen_era14",
    "home_team_wrc_plus", "away_team_wrc_plus",
    "park_factor",
]
NRFI_FEATURES = [
    "home_sp_fi_ra9", "home_sp_fi_bb_rate",
    "away_sp_fi_ra9", "away_sp_fi_bb_rate",
    "home_top3_obp", "home_top3_iso",
    "away_top3_obp", "away_top3_iso",
]

TARGET_ML = "home_win"            # 1 if home team won
TARGET_HOME_RUNS = "home_runs"
TARGET_AWAY_RUNS = "away_runs"
TARGET_TOTAL_RUNS = "total_runs"
TARGET_NRFI = "first_inning_run"  # 1 if a run scored in the 1st (YRFI)


# ==========================================================================
# STEP 1 -- The Daily Data Loader
# ==========================================================================
@dataclass
class GameInfo:
    game_pk: int
    away_team: str
    home_team: str
    away_pitcher_name: str
    home_pitcher_name: str
    away_pitcher_id: Optional[int] = None
    home_pitcher_id: Optional[int] = None
    away_lineup_ids: list = field(default_factory=list)  # MLBAM batter ids 1-9
    home_lineup_ids: list = field(default_factory=list)


def _lookup_player_id(name: str) -> Optional[int]:
    """Resolve a player name to an MLBAM id via statsapi."""
    if not name or name.upper() in ("TBD", "TBA"):
        return None
    try:
        hits = statsapi.lookup_player(name)
        if hits:
            return int(hits[0]["id"])
    except Exception as exc:
        print(f"    [warn] player lookup failed for '{name}': {exc}")
    return None


def _fetch_lineups(game_pk: int) -> tuple[list, list]:
    """Batting-order MLBAM ids (away, home) if lineups are posted."""
    try:
        game = statsapi.get("game", {"gamePk": game_pk})
        box = game["liveData"]["boxscore"]["teams"]
        away = [int(pid) for pid in box["away"].get("battingOrder", [])]
        home = [int(pid) for pid in box["home"].get("battingOrder", [])]
        return away, home
    except Exception:
        return [], []


def load_todays_games(date_str: str) -> list[GameInfo]:
    """STEP 1: pull today's schedule, game_pks, probable starters, lineups."""
    print(f"\n[Step 1] Loading schedule for {date_str} ...")
    games: list[GameInfo] = []
    try:
        sched = statsapi.schedule(date=date_str)
    except Exception as exc:
        print(f"  [error] statsapi.schedule failed: {exc}")
        return games

    for g in sched:
        if g.get("game_type") not in ("R", "F", "D", "L", "W"):  # skip exhib.
            continue
        info = GameInfo(
            game_pk=g["game_id"],
            away_team=g["away_name"],
            home_team=g["home_name"],
            away_pitcher_name=g.get("away_probable_pitcher") or "TBD",
            home_pitcher_name=g.get("home_probable_pitcher") or "TBD",
        )
        info.away_pitcher_id = _lookup_player_id(info.away_pitcher_name)
        info.home_pitcher_id = _lookup_player_id(info.home_pitcher_name)
        info.away_lineup_ids, info.home_lineup_ids = _fetch_lineups(info.game_pk)
        games.append(info)
        print(f"  {info.away_team} @ {info.home_team}  "
              f"({info.away_pitcher_name} vs {info.home_pitcher_name})")

    print(f"  -> {len(games)} game(s) found.")
    return games


# ==========================================================================
# Shared fetch helpers (Statcast / FanGraphs / B-Ref), all with fallbacks
# ==========================================================================
_EVENT_OUTS = {
    "strikeout": 1, "strikeout_double_play": 2, "field_out": 1,
    "force_out": 1, "grounded_into_double_play": 2, "double_play": 2,
    "sac_fly": 1, "sac_bunt": 1, "fielders_choice_out": 1,
    "sac_fly_double_play": 2, "sac_bunt_double_play": 2, "triple_play": 3,
    "other_out": 1, "caught_stealing_2b": 1, "caught_stealing_3b": 1,
    "caught_stealing_home": 1, "pickoff_1b": 1, "pickoff_2b": 1,
    "pickoff_3b": 1,
}


def _statcast_pitcher_window(pid: int, start: dt.date, end: dt.date) -> Optional[pd.DataFrame]:
    try:
        df = statcast_pitcher(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), pid)
        return df if df is not None and len(df) else None
    except Exception as exc:
        print(f"    [warn] statcast pull failed for {pid}: {exc}")
        return None


def pitcher_form_last30(pid: Optional[int], asof: dt.date) -> dict:
    """K/9, BB/9, GB% and an xFIP proxy from the pitcher's last 30 days of
    Statcast data. The xFIP proxy uses the standard formula with expected HR
    = fly balls x league HR/FB (10.5%); it backs up FanGraphs, which some
    networks block (Cloudflare 403)."""
    out = {"k9": LEAGUE_AVG["k9"], "bb9": LEAGUE_AVG["bb9"],
           "gb_pct": LEAGUE_AVG["gb_pct"], "xfip_proxy": None}
    if pid is None:
        return out
    df = _statcast_pitcher_window(pid, asof - dt.timedelta(days=30), asof)
    if df is None:
        return out

    ev = df.dropna(subset=["events"])
    if ev.empty:
        return out
    outs = ev["events"].map(_EVENT_OUTS).fillna(0).sum()
    ip = outs / 3.0
    if ip < 3:  # sample too small to trust
        return out
    k = ev["events"].str.contains("strikeout").sum()
    bb = ev["events"].isin(["walk", "intent_walk"]).sum()
    out["k9"] = round(9.0 * k / ip, 2)
    out["bb9"] = round(9.0 * bb / ip, 2)

    bip = df.dropna(subset=["bb_type"])
    if len(bip) >= 10:
        out["gb_pct"] = round((bip["bb_type"] == "ground_ball").mean(), 3)

    fb = (bip["bb_type"] == "fly_ball").sum()
    hbp = ev["events"].eq("hit_by_pitch").sum()
    out["xfip_proxy"] = round(
        (13 * fb * 0.105 + 3 * (bb + hbp) - 2 * k) / ip + 3.10, 2)
    return out


_FG_PITCHING_CACHE: Optional[pd.DataFrame] = None


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z ]", "", (name or "").lower()).strip()


def pitcher_xfip(name: str, season: int,
                 proxy: Optional[float] = None) -> float:
    """Season xFIP from FanGraphs (name-matched); pybaseball can't split
    FanGraphs advanced stats by arbitrary date range, so season is used.
    Falls back to the Statcast-derived 30-day proxy when FanGraphs is
    unreachable, then to league average."""
    global _FG_PITCHING_CACHE
    if _FG_PITCHING_CACHE is None:
        try:
            _FG_PITCHING_CACHE = pitching_stats(season, season, qual=0)
        except Exception as exc:
            print(f"    [warn] FanGraphs pitching_stats failed ({exc}); "
                  f"using Statcast xFIP proxy.")
            _FG_PITCHING_CACHE = pd.DataFrame()
    df = _FG_PITCHING_CACHE
    fallback = proxy if proxy is not None else LEAGUE_AVG["xfip"]
    if df.empty or "xFIP" not in df.columns:
        return fallback
    match = df[df["Name"].map(_norm_name) == _norm_name(name)]
    if match.empty:
        return fallback
    val = match.iloc[0]["xFIP"]
    return float(val) if pd.notna(val) else fallback


_FG_TEAM_BATTING_CACHE: Optional[pd.DataFrame] = None
_MLB_TEAM_HITTING_CACHE: Optional[dict] = None


def _mlb_api_team_hitting(season: int) -> dict:
    """All 30 teams' season hitting stats straight from the MLB Stats API
    (one call, no auth, not Cloudflare-gated). Returns
    {team_name: {"obp": float, "ops": float}} plus "_league_ops"."""
    global _MLB_TEAM_HITTING_CACHE
    if _MLB_TEAM_HITTING_CACHE is not None:
        return _MLB_TEAM_HITTING_CACHE
    out: dict = {}
    try:
        url = (f"https://statsapi.mlb.com/api/v1/teams/stats"
               f"?season={season}&group=hitting&stats=season&sportIds=1")
        splits = requests.get(url, timeout=30).json()["stats"][0]["splits"]
        ops_vals = []
        for s in splits:
            obp, ops = float(s["stat"]["obp"]), float(s["stat"]["ops"])
            out[s["team"]["name"]] = {"obp": obp, "ops": ops}
            ops_vals.append(ops)
        out["_league_ops"] = float(np.mean(ops_vals)) if ops_vals else 0.715
    except Exception as exc:
        print(f"    [warn] MLB API team hitting failed: {exc}")
    _MLB_TEAM_HITTING_CACHE = out
    return out


def team_offense(team_name: str, season: int) -> dict:
    """Season team wRC+ and OBP -- FanGraphs team_batting first, MLB Stats
    API second (with wRC+ approximated by OPS indexed to league average).

    NOTE: pybaseball does not expose FanGraphs L/R platoon splits, so season
    aggregates stand in for 'vs opposing starter handedness'. The feature
    names stay handedness-agnostic so a richer source can drop in later.
    """
    global _FG_TEAM_BATTING_CACHE
    out = {"wrc_plus": LEAGUE_AVG["wrc_plus"], "obp": LEAGUE_AVG["obp"]}
    if _FG_TEAM_BATTING_CACHE is None:
        try:
            _FG_TEAM_BATTING_CACHE = team_batting(season, season)
        except Exception as exc:
            print(f"    [warn] FanGraphs team_batting failed ({exc}); "
                  f"falling back to MLB Stats API.")
            _FG_TEAM_BATTING_CACHE = pd.DataFrame()
    df = _FG_TEAM_BATTING_CACHE
    abbr = TEAM_TO_FG.get(team_name)
    if not df.empty and abbr is not None and "Team" in df.columns:
        row = df[df["Team"] == abbr]
        if not row.empty:
            if "wRC+" in row.columns and pd.notna(row.iloc[0]["wRC+"]):
                out["wrc_plus"] = float(row.iloc[0]["wRC+"])
            if "OBP" in row.columns and pd.notna(row.iloc[0]["OBP"]):
                out["obp"] = float(row.iloc[0]["OBP"])
            return out

    mlb = _mlb_api_team_hitting(season)
    if team_name in mlb:
        out["obp"] = mlb[team_name]["obp"]
        league_ops = mlb.get("_league_ops", 0.715)
        out["wrc_plus"] = round(100.0 * mlb[team_name]["ops"] / league_ops, 1)
    return out


_BULLPEN_CACHE: Optional[pd.DataFrame] = None


def bullpen_era_last14(team_name: str, asof: dt.date) -> float:
    """Team bullpen ERA over the last 14 days.

    Uses Baseball-Reference daily logs (pitching_stats_range), keeping only
    pitchers with 0 games started in the window, grouped by team.
    """
    global _BULLPEN_CACHE
    if _BULLPEN_CACHE is None:
        start = (asof - dt.timedelta(days=14)).strftime("%Y-%m-%d")
        try:
            df = pitching_stats_range(start, asof.strftime("%Y-%m-%d"))
            df = df[pd.to_numeric(df["GS"], errors="coerce").fillna(0) == 0]
            df["ER"] = pd.to_numeric(df["ER"], errors="coerce").fillna(0)
            df["IP"] = pd.to_numeric(df["IP"], errors="coerce").fillna(0)
            _BULLPEN_CACHE = df
        except Exception as exc:
            print(f"    [warn] B-Ref bullpen pull failed: {exc}")
            _BULLPEN_CACHE = pd.DataFrame()
    df = _BULLPEN_CACHE
    if df.empty or "Tm" not in df.columns:
        return LEAGUE_AVG["bullpen_era14"]

    # B-Ref 'Tm' formats vary (abbrev or city); match loosely on both.
    abbr = TEAM_TO_FG.get(team_name, "")
    mask = df["Tm"].astype(str).str.contains(abbr, case=False, na=False) | \
        df["Tm"].astype(str).apply(lambda t: str(t).lower() in team_name.lower())
    grp = df[mask]
    ip, er = grp["IP"].sum(), grp["ER"].sum()
    if ip < 10:
        return LEAGUE_AVG["bullpen_era14"]
    return round(9.0 * er / ip, 2)


def get_park_factor(home_team: str) -> float:
    return PARK_FACTORS.get(home_team, LEAGUE_AVG["park_factor"])


# ==========================================================================
# STEP 2 -- Feature Engineering: Moneyline
# ==========================================================================
def build_moneyline_features(game: GameInfo, asof: dt.date, season: int) -> dict:
    """Last-30-day starter form + season xFIP + team offense for both sides."""
    h_form = pitcher_form_last30(game.home_pitcher_id, asof)
    a_form = pitcher_form_last30(game.away_pitcher_id, asof)
    h_off = team_offense(game.home_team, season)
    a_off = team_offense(game.away_team, season)
    return {
        "home_sp_k9": h_form["k9"], "home_sp_bb9": h_form["bb9"],
        "home_sp_xfip": pitcher_xfip(game.home_pitcher_name, season,
                                     proxy=h_form["xfip_proxy"]),
        "home_sp_gb_pct": h_form["gb_pct"],
        "away_sp_k9": a_form["k9"], "away_sp_bb9": a_form["bb9"],
        "away_sp_xfip": pitcher_xfip(game.away_pitcher_name, season,
                                     proxy=a_form["xfip_proxy"]),
        "away_sp_gb_pct": a_form["gb_pct"],
        "home_team_wrc_plus": h_off["wrc_plus"], "home_team_obp": h_off["obp"],
        "away_team_wrc_plus": a_off["wrc_plus"], "away_team_obp": a_off["obp"],
    }


# ==========================================================================
# STEP 3 -- Feature Engineering: Totals
# ==========================================================================
def build_totals_features(game: GameInfo, ml_feats: dict, asof: dt.date) -> dict:
    """Starter expected runs allowed (xFIP as RA/9 proxy), 14-day bullpen
    ERA, park factor, plus each offense's wRC+."""
    home_xra9 = ml_feats["home_sp_xfip"]
    away_xra9 = ml_feats["away_sp_xfip"]
    return {
        "home_sp_xra9": home_xra9,
        "away_sp_xra9": away_xra9,
        "combined_sp_xra9": round(home_xra9 + away_xra9, 2),
        "home_bullpen_era14": bullpen_era_last14(game.home_team, asof),
        "away_bullpen_era14": bullpen_era_last14(game.away_team, asof),
        "home_team_wrc_plus": ml_feats["home_team_wrc_plus"],
        "away_team_wrc_plus": ml_feats["away_team_wrc_plus"],
        "park_factor": get_park_factor(game.home_team),
    }


# ==========================================================================
# STEP 4 -- Feature Engineering: NRFI / YRFI (1st-inning only)
# ==========================================================================
def pitcher_first_inning(pid: Optional[int], asof: dt.date) -> dict:
    """1st-inning RA/9 (run-average, computed from Statcast score deltas)
    and 1st-inning walk rate over the pitcher's last ~90 days of starts."""
    out = {"fi_ra9": LEAGUE_AVG["fi_ra9"], "fi_bb_rate": LEAGUE_AVG["fi_bb_rate"]}
    if pid is None:
        return out
    df = _statcast_pitcher_window(pid, asof - dt.timedelta(days=90), asof)
    if df is None:
        return out
    fi = df[df["inning"] == 1]
    if fi.empty:
        return out

    starts = fi["game_pk"].nunique()
    if starts < 3:  # too few first innings to be meaningful
        return out
    # Runs allowed on each pitch = batting team's score delta on that pitch.
    runs = (fi["post_bat_score"] - fi["bat_score"]).clip(lower=0).sum()
    out["fi_ra9"] = round(9.0 * runs / starts, 2)

    ev = fi.dropna(subset=["events"])
    pa = len(ev)
    if pa >= 10:
        bb = ev["events"].isin(["walk", "intent_walk"]).sum()
        out["fi_bb_rate"] = round(bb / pa, 3)
    return out


def _batter_obp_iso(pid: int) -> Optional[tuple[float, float]]:
    """Season OBP and ISO for one batter via statsapi season splits."""
    try:
        data = statsapi.player_stat_data(pid, group="hitting", type="season")
        stats = data["stats"][0]["stats"]
        obp = float(stats["obp"])
        iso = float(stats["slg"]) - float(stats["avg"])
        return obp, iso
    except Exception:
        return None


def top3_lineup_stats(lineup_ids: list) -> dict:
    """Average OBP / ISO for batters 1-3. Falls back to league-average
    top-of-order numbers when the lineup isn't posted yet."""
    out = {"top3_obp": LEAGUE_AVG["top3_obp"], "top3_iso": LEAGUE_AVG["top3_iso"]}
    vals = [v for pid in lineup_ids[:3] if (v := _batter_obp_iso(pid))]
    if vals:
        out["top3_obp"] = round(float(np.mean([v[0] for v in vals])), 3)
        out["top3_iso"] = round(float(np.mean([v[1] for v in vals])), 3)
    return out


def build_nrfi_features(game: GameInfo, asof: dt.date) -> dict:
    h_fi = pitcher_first_inning(game.home_pitcher_id, asof)
    a_fi = pitcher_first_inning(game.away_pitcher_id, asof)
    h_top = top3_lineup_stats(game.home_lineup_ids)
    a_top = top3_lineup_stats(game.away_lineup_ids)
    return {
        "home_sp_fi_ra9": h_fi["fi_ra9"], "home_sp_fi_bb_rate": h_fi["fi_bb_rate"],
        "away_sp_fi_ra9": a_fi["fi_ra9"], "away_sp_fi_bb_rate": a_fi["fi_bb_rate"],
        "home_top3_obp": h_top["top3_obp"], "home_top3_iso": h_top["top3_iso"],
        "away_top3_obp": a_top["top3_obp"], "away_top3_iso": a_top["top3_iso"],
    }


# ==========================================================================
# STEP 5 -- Model Training & Inference
# ==========================================================================
def _synthetic_history(n: int = 4000) -> pd.DataFrame:
    """Generate plausible synthetic training rows so the pipeline runs even
    without mlb_history_2024_2025.csv. NOT for real betting use."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "home_sp_k9": rng.normal(8.6, 1.6, n), "home_sp_bb9": rng.normal(3.2, 1.0, n),
        "home_sp_xfip": rng.normal(4.1, 0.8, n), "home_sp_gb_pct": rng.normal(0.435, 0.06, n),
        "away_sp_k9": rng.normal(8.6, 1.6, n), "away_sp_bb9": rng.normal(3.2, 1.0, n),
        "away_sp_xfip": rng.normal(4.1, 0.8, n), "away_sp_gb_pct": rng.normal(0.435, 0.06, n),
        "home_team_wrc_plus": rng.normal(100, 10, n), "home_team_obp": rng.normal(0.315, 0.012, n),
        "away_team_wrc_plus": rng.normal(100, 10, n), "away_team_obp": rng.normal(0.315, 0.012, n),
        "home_bullpen_era14": rng.normal(4.2, 0.9, n), "away_bullpen_era14": rng.normal(4.2, 0.9, n),
        "park_factor": rng.normal(1.0, 0.04, n),
        "home_sp_fi_ra9": rng.normal(4.4, 1.8, n), "home_sp_fi_bb_rate": rng.normal(0.085, 0.03, n),
        "away_sp_fi_ra9": rng.normal(4.4, 1.8, n), "away_sp_fi_bb_rate": rng.normal(0.085, 0.03, n),
        "home_top3_obp": rng.normal(0.335, 0.02, n), "home_top3_iso": rng.normal(0.165, 0.03, n),
        "away_top3_obp": rng.normal(0.335, 0.02, n), "away_top3_iso": rng.normal(0.165, 0.03, n),
    })
    df["home_sp_xra9"] = df["home_sp_xfip"]
    df["away_sp_xra9"] = df["away_sp_xfip"]
    df["combined_sp_xra9"] = df["home_sp_xra9"] + df["away_sp_xra9"]

    # Latent run expectations drive correlated, realistic targets,
    # calibrated so team runs average ~4.4 (league total ~8.8).
    home_exp = (0.55 * df["away_sp_xra9"] + 0.30 * df["away_bullpen_era14"]) \
        * (df["home_team_wrc_plus"] / 100) * df["park_factor"] * 1.25 + 0.15  # home edge
    away_exp = (0.55 * df["home_sp_xra9"] + 0.30 * df["home_bullpen_era14"]) \
        * (df["away_team_wrc_plus"] / 100) * df["park_factor"] * 1.25
    df[TARGET_HOME_RUNS] = rng.poisson(np.clip(home_exp, 1.5, 9))
    df[TARGET_AWAY_RUNS] = rng.poisson(np.clip(away_exp, 1.5, 9))
    df[TARGET_TOTAL_RUNS] = df[TARGET_HOME_RUNS] + df[TARGET_AWAY_RUNS]
    ties = df[TARGET_HOME_RUNS] == df[TARGET_AWAY_RUNS]
    df.loc[ties, TARGET_HOME_RUNS] += rng.integers(0, 2, ties.sum()) * 2 - 1
    df[TARGET_HOME_RUNS] = df[TARGET_HOME_RUNS].clip(lower=0)
    df[TARGET_ML] = (df[TARGET_HOME_RUNS] > df[TARGET_AWAY_RUNS]).astype(int)

    fi_lambda = (df["home_sp_fi_ra9"] + df["away_sp_fi_ra9"]) / 18.0 \
        * ((df["home_top3_obp"] + df["away_top3_obp"]) / (2 * 0.335))
    df[TARGET_NRFI] = (rng.random(n) < 1 - np.exp(-np.clip(fi_lambda, 0.05, 2))).astype(int)
    return df


def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_CSV):
        print(f"\n[Step 5] Training on {HISTORY_CSV} ...")
        df = pd.read_csv(HISTORY_CSV)
        needed = set(ML_FEATURES + TOTALS_FEATURES + NRFI_FEATURES +
                     [TARGET_ML, TARGET_HOME_RUNS, TARGET_AWAY_RUNS,
                      TARGET_TOTAL_RUNS, TARGET_NRFI])
        missing = needed - set(df.columns)
        if missing:
            sys.exit(f"  [error] {HISTORY_CSV} is missing columns: {sorted(missing)}")
        return df
    print(f"\n[Step 5] WARNING: {HISTORY_CSV} not found -- training on "
          f"SYNTHETIC data. Predictions are for pipeline testing ONLY.")
    return _synthetic_history()


@dataclass
class ModelBundle:
    moneyline: object
    nrfi: object
    home_runs: object
    away_runs: object
    total_runs: object


def _make_classifier():
    if _HAVE_XGB:
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42,
        )
    return XGBClassifier(n_estimators=400, max_depth=8, random_state=42)


def _make_regressor():
    if _HAVE_XGB:
        return XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    return XGBRegressor(n_estimators=400, max_depth=10, random_state=42)


def train_models(hist: pd.DataFrame) -> ModelBundle:
    """Train the three market-specific models with a holdout report."""
    engine = "XGBoost" if _HAVE_XGB else "RandomForest (xgboost not installed)"
    print(f"  Engine: {engine}")

    # --- Moneyline classifier -------------------------------------------
    X, y = hist[ML_FEATURES], hist[TARGET_ML]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y)
    ml_model = _make_classifier().fit(Xtr, ytr)
    print(f"  Moneyline  holdout accuracy: "
          f"{accuracy_score(yte, ml_model.predict(Xte)):.3f}")

    # --- Totals regressors (home / away / combined) ---------------------
    regs = {}
    for tag, target in (("home", TARGET_HOME_RUNS), ("away", TARGET_AWAY_RUNS),
                        ("total", TARGET_TOTAL_RUNS)):
        X, y = hist[TOTALS_FEATURES], hist[target]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        regs[tag] = _make_regressor().fit(Xtr, ytr)
        print(f"  Totals[{tag:<5}] holdout MAE: "
              f"{mean_absolute_error(yte, regs[tag].predict(Xte)):.2f} runs")

    # --- NRFI classifier -------------------------------------------------
    X, y = hist[NRFI_FEATURES], hist[TARGET_NRFI]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y)
    nrfi_model = _make_classifier().fit(Xtr, ytr)
    print(f"  NRFI/YRFI  holdout accuracy: "
          f"{accuracy_score(yte, nrfi_model.predict(Xte)):.3f}")

    return ModelBundle(moneyline=ml_model, nrfi=nrfi_model,
                       home_runs=regs["home"], away_runs=regs["away"],
                       total_runs=regs["total"])


@dataclass
class Prediction:
    game: GameInfo
    winner: str
    winner_conf: float          # strict percentage from .predict_proba()
    away_runs: float
    home_runs: float
    total_runs: float
    first_inning: str           # "NRFI" or "YRFI"
    first_inning_conf: float


def predict_game(models: ModelBundle, game: GameInfo,
                 ml_row: dict, tot_row: dict, nrfi_row: dict) -> Prediction:
    Xml = pd.DataFrame([ml_row])[ML_FEATURES]
    Xtot = pd.DataFrame([tot_row])[TOTALS_FEATURES]
    Xfi = pd.DataFrame([nrfi_row])[NRFI_FEATURES]

    # Moneyline: proba of class 1 == home win
    p_home = float(models.moneyline.predict_proba(Xml)[0][1])
    winner = game.home_team if p_home >= 0.5 else game.away_team
    winner_conf = max(p_home, 1 - p_home) * 100

    home_r = max(0.0, float(models.home_runs.predict(Xtot)[0]))
    away_r = max(0.0, float(models.away_runs.predict(Xtot)[0]))
    # Blend the dedicated combined-total model with the sum of team models
    # so the three numbers stay mutually coherent.
    total_direct = max(0.0, float(models.total_runs.predict(Xtot)[0]))
    total_r = 0.5 * total_direct + 0.5 * (home_r + away_r)

    # NRFI: class 1 == a run scored in the 1st (YRFI)
    p_yrfi = float(models.nrfi.predict_proba(Xfi)[0][1])
    call = "YRFI" if p_yrfi >= 0.5 else "NRFI"
    fi_conf = max(p_yrfi, 1 - p_yrfi) * 100

    return Prediction(game=game, winner=winner, winner_conf=winner_conf,
                      away_runs=away_r, home_runs=home_r, total_runs=total_r,
                      first_inning=call, first_inning_conf=fi_conf)


# ==========================================================================
# STEP 6 -- The Final Output Formatter
# ==========================================================================
def print_predictions(preds: list[Prediction]) -> None:
    print("\n" + "=" * 62)
    print(" TODAY'S MLB PREDICTIONS")
    print("=" * 62)
    for p in preds:
        g = p.game
        print(f"\nGame: {g.away_team} @ {g.home_team}")
        print(f"\nMoneyline: {p.winner} (Confidence: {p.winner_conf:.1f}%)")
        print(f"\nProjected Score: {p.away_runs:.1f} - {p.home_runs:.1f} "
              f"(Total: {p.total_runs:.1f})")
        print(f"\n1st Inning: {p.first_inning} "
              f"(Confidence: {p.first_inning_conf:.1f}%)")
        print("-" * 62)
    if not preds:
        print("\nNo games scheduled today.")


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Daily MLB market predictor")
    parser.add_argument("--date", default=dt.date.today().strftime("%Y-%m-%d"),
                        help="Slate date, YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    asof = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
    season = asof.year

    # Step 1 -- schedule / pitchers / lineups
    games = load_todays_games(args.date)
    if not games:
        print("No games found; exiting.")
        return

    # Step 5 (training happens once, before per-game feature pulls)
    models = train_models(load_history())

    # Steps 2-4 -- per-game feature engineering, then inference
    print("\n[Steps 2-4] Building feature matrices "
          "(Statcast pulls may take a minute per pitcher on a cold cache)...")
    preds: list[Prediction] = []
    for game in games:
        print(f"  Features: {game.away_team} @ {game.home_team}")
        ml_row = build_moneyline_features(game, asof, season)     # Step 2
        tot_row = build_totals_features(game, ml_row, asof)       # Step 3
        nrfi_row = build_nrfi_features(game, asof)                # Step 4
        preds.append(predict_game(models, game, ml_row, tot_row, nrfi_row))

    # Step 6 -- formatted console output
    print_predictions(preds)


if __name__ == "__main__":
    main()
