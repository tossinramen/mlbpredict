from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


import statsapi 

import pybaseball
from pybaseball import statcast

from mlb_common import BB_EVENTS, EVENT_OUTS, HIT_EVENTS, NON_AB_EVENTS, fip

pybaseball.cache.enable()

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAVE_XGB = True
except ImportError:  
    from sklearn.ensemble import (
        RandomForestClassifier as XGBClassifier,
        RandomForestRegressor as XGBRegressor,
    )
    _HAVE_XGB = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error


HISTORY_CSV = "mlb_history_2024_2025.csv"


LEAGUE_AVG = {
    "k9": 8.6, "bb9": 3.2, "xfip": 4.10, "gb_pct": 0.435,
    "wrc_plus": 100.0, "obp": 0.315,
    "bullpen_era14": 4.20, "park_factor": 1.00,
    "fi_ra9": 4.40, "fi_bb_rate": 0.085,  
    "top3_obp": 0.335, "top3_iso": 0.165,  
}


TEAM_TO_SC = {
    "Arizona Diamondbacks": "AZ", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "ATH",
    "Oakland Athletics": "ATH", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


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

TARGET_ML = "home_win"           
TARGET_HOME_RUNS = "home_runs"
TARGET_AWAY_RUNS = "away_runs"
TARGET_TOTAL_RUNS = "total_runs"
TARGET_NRFI = "first_inning_run"  



@dataclass
class GameInfo:
    game_pk: int
    away_team: str
    home_team: str
    away_pitcher_name: str
    home_pitcher_name: str
    away_pitcher_id: Optional[int] = None
    home_pitcher_id: Optional[int] = None
    away_lineup_ids: list = field(default_factory=list)  
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
        if g.get("game_type") not in ("R", "F", "D", "L", "W"): 
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



_BULK_DF: Optional[pd.DataFrame] = None
_STARTERS: Optional[dict] = None

_BULK_COLS = [
    "game_pk", "game_date", "game_type", "home_team", "away_team",
    "pitcher", "batter", "p_throws", "stand",
    "events", "bb_type", "inning", "inning_topbot",
    "at_bat_number", "pitch_number",
    "bat_score", "post_bat_score", "woba_value", "woba_denom",
]


def bulk_statcast(asof: dt.date) -> pd.DataFrame:

    global _BULK_DF
    if _BULK_DF is not None:
        return _BULK_DF
    start = asof - dt.timedelta(days=90)
    print(f"  [bulk] one-time league Statcast download {start} .. {asof} "
          f"(cached by pybaseball) ...")
    try:
        df = statcast(start.strftime("%Y-%m-%d"), asof.strftime("%Y-%m-%d"),
                      verbose=False)
    except Exception as exc:
        print(f"    [warn] bulk statcast failed: {exc}")
        df = None
    if df is None or not len(df):
        df = pd.DataFrame(columns=_BULK_COLS)
    df = df[[c for c in _BULK_COLS if c in df.columns]].copy()
    if "game_type" in df.columns and len(df):
        df = df[df["game_type"] == "R"]
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number",
                         "pitch_number"], ignore_index=True)
    df["bat_team"] = np.where(df["inning_topbot"] == "Top",
                              df["away_team"], df["home_team"])
    df["pitch_team"] = np.where(df["inning_topbot"] == "Top",
                                df["home_team"], df["away_team"])
    df["runs_on_pitch"] = (df["post_bat_score"] - df["bat_score"]).clip(lower=0)
    _BULK_DF = df
    return _BULK_DF


def _game_starters(df: pd.DataFrame) -> dict:

    global _STARTERS
    if _STARTERS is None:
        first = (df[df["inning"] == 1]
                 .drop_duplicates(subset=["game_pk", "inning_topbot"],
                                  keep="first"))
        _STARTERS = {
            (r.game_pk, "home" if r.inning_topbot == "Top" else "away"):
                int(r.pitcher)
            for r in first.itertuples()
        }
    return _STARTERS


def _window(df: pd.DataFrame, asof: dt.date, days: int) -> pd.DataFrame:
    return df[df["game_date"] >= pd.Timestamp(asof - dt.timedelta(days=days))]


def pitcher_hand(pid: Optional[int], asof: dt.date) -> str:

    if pid is None:
        return "R"
    rows = bulk_statcast(asof)
    rows = rows[rows["pitcher"] == pid]
    return str(rows.iloc[0]["p_throws"]) if len(rows) else "R"


def pitcher_form_last30(pid: Optional[int], asof: dt.date) -> dict:

    out = {"k9": LEAGUE_AVG["k9"], "bb9": LEAGUE_AVG["bb9"],
           "gb_pct": LEAGUE_AVG["gb_pct"], "fip": LEAGUE_AVG["xfip"]}
    if pid is None:
        return out
    df = bulk_statcast(asof)
    mine = _window(df[df["pitcher"] == pid], asof, 30)
    ev = mine.dropna(subset=["events"])
    if ev.empty:
        return out
    outs = ev["events"].map(EVENT_OUTS).fillna(0).sum()
    ip = outs / 3.0
    if ip < 3:
        return out
    k = ev["events"].str.startswith("strikeout").sum()
    bb = ev["events"].isin(BB_EVENTS).sum()
    hbp = ev["events"].eq("hit_by_pitch").sum()
    hr = ev["events"].eq("home_run").sum()
    out["k9"] = round(9.0 * k / ip, 2)
    out["bb9"] = round(9.0 * bb / ip, 2)
    out["fip"] = round(fip(hr, bb, hbp, k, ip), 2)

    bip = ev.dropna(subset=["bb_type"])
    if len(bip) >= 10:
        out["gb_pct"] = round((bip["bb_type"] == "ground_ball").mean(), 3)
    return out


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


def team_offense(team_name: str, opp_hand: str, asof: dt.date,
                 season: int) -> dict:

    out = {"wrc_plus": LEAGUE_AVG["wrc_plus"], "obp": LEAGUE_AVG["obp"]}
    code = TEAM_TO_SC.get(team_name)
    ev = _window(bulk_statcast(asof).dropna(subset=["events"]), asof, 30)
    if code is not None and len(ev):
        for split in (ev[ev["p_throws"] == opp_hand], ev):
            team = split[split["bat_team"] == code]
            if len(team) < 120:  # platoon sample too thin, widen
                continue
            wden = pd.to_numeric(team["woba_denom"], errors="coerce").sum()
            lg_wden = pd.to_numeric(split["woba_denom"], errors="coerce").sum()
            if not wden or not lg_wden:
                continue
            woba = pd.to_numeric(team["woba_value"],
                                 errors="coerce").sum() / wden
            lg_woba = pd.to_numeric(split["woba_value"],
                                    errors="coerce").sum() / lg_wden
            e = team["events"]
            bb = e.isin(BB_EVENTS).sum()
            hbp = e.eq("hit_by_pitch").sum()
            sf = e.isin(["sac_fly", "sac_fly_double_play"]).sum()
            ob = e.isin(HIT_EVENTS).sum() + bb + hbp
            obp_den = (~e.isin(NON_AB_EVENTS)).sum() + bb + hbp + sf
            if lg_woba > 0:
                out["wrc_plus"] = round(100.0 * woba / lg_woba, 1)
            if obp_den > 0:
                out["obp"] = round(ob / obp_den, 3)
            return out

    mlb = _mlb_api_team_hitting(season)
    if team_name in mlb:
        out["obp"] = mlb[team_name]["obp"]
        league_ops = mlb.get("_league_ops", 0.715)
        out["wrc_plus"] = round(100.0 * mlb[team_name]["ops"] / league_ops, 1)
    return out


def bullpen_era_last14(team_name: str, asof: dt.date) -> float:
    code = TEAM_TO_SC.get(team_name)
    if code is None:
        return LEAGUE_AVG["bullpen_era14"]
    df = bulk_statcast(asof)
    win = _window(df[df["pitch_team"] == code], asof, 14)
    if win.empty:
        return LEAGUE_AVG["bullpen_era14"]
    starters = _game_starters(df)
    side = np.where(win["inning_topbot"] == "Top", "home", "away")
    starter_ids = np.array([starters.get(k, -1)
                            for k in zip(win["game_pk"], side)])
    rel = win[win["pitcher"].to_numpy() != starter_ids]
    ev = rel.dropna(subset=["events"])
    outs = ev["events"].map(EVENT_OUTS).fillna(0).sum()
    if outs < 30:  # < 10 IP of reliever work in window
        return LEAGUE_AVG["bullpen_era14"]
    return round(9.0 * rel["runs_on_pitch"].sum() / (outs / 3.0), 2)


def get_park_factor(home_team: str) -> float:
    return PARK_FACTORS.get(home_team, LEAGUE_AVG["park_factor"])



def build_moneyline_features(game: GameInfo, asof: dt.date, season: int) -> dict:
    h_form = pitcher_form_last30(game.home_pitcher_id, asof)
    a_form = pitcher_form_last30(game.away_pitcher_id, asof)
    h_off = team_offense(game.home_team,
                         pitcher_hand(game.away_pitcher_id, asof), asof, season)
    a_off = team_offense(game.away_team,
                         pitcher_hand(game.home_pitcher_id, asof), asof, season)
    return {
        "home_sp_k9": h_form["k9"], "home_sp_bb9": h_form["bb9"],
        "home_sp_xfip": h_form["fip"],
        "home_sp_gb_pct": h_form["gb_pct"],
        "away_sp_k9": a_form["k9"], "away_sp_bb9": a_form["bb9"],
        "away_sp_xfip": a_form["fip"],
        "away_sp_gb_pct": a_form["gb_pct"],
        "home_team_wrc_plus": h_off["wrc_plus"], "home_team_obp": h_off["obp"],
        "away_team_wrc_plus": a_off["wrc_plus"], "away_team_obp": a_off["obp"],
    }



def build_totals_features(game: GameInfo, ml_feats: dict, asof: dt.date) -> dict:
    """Starter expected runs allowed (calculated FIP as RA/9 proxy),
    14-day bullpen RA/9, park factor, plus each offense's wRC+ proxy."""
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



def pitcher_first_inning(pid: Optional[int], asof: dt.date) -> dict:
    """1st-inning RA/9 (run-average from Statcast score deltas -- for
    NRFI/YRFI an unearned run cashes the bet exactly like an earned one)
    and 1st-inning walk rate over the pitcher's last 90 days."""
    out = {"fi_ra9": LEAGUE_AVG["fi_ra9"], "fi_bb_rate": LEAGUE_AVG["fi_bb_rate"]}
    if pid is None:
        return out
    df = bulk_statcast(asof)
    mine = _window(df[df["pitcher"] == pid], asof, 90)
    fi = mine[mine["inning"] == 1]
    if fi.empty:
        return out

    starts = fi["game_pk"].nunique()
    if starts < 3:
        return out
    out["fi_ra9"] = round(9.0 * fi["runs_on_pitch"].sum() / starts, 2)

    ev = fi.dropna(subset=["events"])
    pa = len(ev)
    if pa >= 10:
        bb = ev["events"].isin(BB_EVENTS).sum()
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



def _synthetic_history(n: int = 4000) -> pd.DataFrame:
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


    X, y = hist[ML_FEATURES], hist[TARGET_ML]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y)
    ml_model = _make_classifier().fit(Xtr, ytr)
    print(f"  Moneyline  holdout accuracy: "
          f"{accuracy_score(yte, ml_model.predict(Xte)):.3f}")


    regs = {}
    for tag, target in (("home", TARGET_HOME_RUNS), ("away", TARGET_AWAY_RUNS),
                        ("total", TARGET_TOTAL_RUNS)):
        X, y = hist[TOTALS_FEATURES], hist[target]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        regs[tag] = _make_regressor().fit(Xtr, ytr)
        print(f"  Totals[{tag:<5}] holdout MAE: "
              f"{mean_absolute_error(yte, regs[tag].predict(Xte)):.2f} runs")

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
    winner_conf: float          
    away_runs: float
    home_runs: float
    total_runs: float
    first_inning: str          
    first_inning_conf: float


def predict_game(models: ModelBundle, game: GameInfo,
                 ml_row: dict, tot_row: dict, nrfi_row: dict) -> Prediction:
    Xml = pd.DataFrame([ml_row])[ML_FEATURES]
    Xtot = pd.DataFrame([tot_row])[TOTALS_FEATURES]
    Xfi = pd.DataFrame([nrfi_row])[NRFI_FEATURES]


    p_home = float(models.moneyline.predict_proba(Xml)[0][1])
    winner = game.home_team if p_home >= 0.5 else game.away_team
    winner_conf = max(p_home, 1 - p_home) * 100

    home_r = max(0.0, float(models.home_runs.predict(Xtot)[0]))
    away_r = max(0.0, float(models.away_runs.predict(Xtot)[0]))

    total_direct = max(0.0, float(models.total_runs.predict(Xtot)[0]))
    total_r = 0.5 * total_direct + 0.5 * (home_r + away_r)


    p_yrfi = float(models.nrfi.predict_proba(Xfi)[0][1])
    call = "YRFI" if p_yrfi >= 0.5 else "NRFI"
    fi_conf = max(p_yrfi, 1 - p_yrfi) * 100

    return Prediction(game=game, winner=winner, winner_conf=winner_conf,
                      away_runs=away_r, home_runs=home_r, total_runs=total_r,
                      first_inning=call, first_inning_conf=fi_conf)


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



def main() -> None:
    parser = argparse.ArgumentParser(description="Daily MLB market predictor")
    parser.add_argument("--date", default=dt.date.today().strftime("%Y-%m-%d"),
                        help="Slate date, YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    asof = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
    season = asof.year


    games = load_todays_games(args.date)
    if not games:
        print("No games found; exiting.")
        return


    models = train_models(load_history())


    print("\n[Steps 2-4] Building feature matrices "
          "(one bulk Statcast download, then in-memory slicing)...")
    preds: list[Prediction] = []
    for game in games:
        print(f"  Features: {game.away_team} @ {game.home_team}")
        ml_row = build_moneyline_features(game, asof, season)     
        tot_row = build_totals_features(game, ml_row, asof)       
        nrfi_row = build_nrfi_features(game, asof)             
        preds.append(predict_game(models, game, ml_row, tot_row, nrfi_row))

    
    print_predictions(preds)


if __name__ == "__main__":
    main()
