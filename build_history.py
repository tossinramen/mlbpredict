"""Build mlb_history_2024_2025.csv chronologically, with zero lookahead.

Inputs (produced by scrape_bref_schedules.py and download_statcast.py):
    data/schedule_backbone.csv      one row per game: date, teams, targets
    data/statcast_{2024,2025}.parquet   league-wide pitch-level Statcast

For a game played on date D, every feature is computed from Statcast data
in a window that ends on D-1 -- never from end-of-season totals:

    starter form        K/9, BB/9, GB%, FIP        last 30 days
    bullpen             reliever RA/9 (era14 col)  last 14 days
    team offense        wOBA-indexed "wRC+" & OBP  last 30 days,
                        split vs the opposing starter's handedness
                        (p_throws), falling back to all-hand window
    first inning        starter RA/9 & BB rate     last 90 days, inning 1
    top-3 lineup        OBP / ISO                  season-to-date
                        (batters 1-3 actually used in that game --
                        known pregame, so not lookahead)

FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + 3.15, straight from Statcast
events; it fills the *_sp_xfip columns so the existing model schema is
unchanged. first_inning_run (YRFI target) comes from Statcast inning-1
score deltas.

Usage:
    python build_history.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_common import (
    BB_EVENTS, EVENT_OUTS, FIP_CONSTANT, HIT_EVENTS, NON_AB_EVENTS, TB_MAP,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = Path(__file__).parent / "data"
BACKBONE_CSV = DATA_DIR / "schedule_backbone.csv"
OUT_CSV = Path(__file__).parent / "mlb_history_2024_2025.csv"
SEASONS = [2024, 2025]

LEAGUE_AVG = {
    "k9": 8.6, "bb9": 3.2, "fip": 4.10, "gb_pct": 0.435,
    "wrc_plus": 100.0, "obp": 0.315,
    "bullpen_era14": 4.20,
    "fi_ra9": 4.40, "fi_bb_rate": 0.085,
    "top3_obp": 0.335, "top3_iso": 0.165,
}

# B-Ref team code -> Statcast (Baseball Savant) team code.
# Savant re-coded the Athletics as ATH retroactively (even for Oakland 2024).
BR_TO_SC = {
    "ARI": "AZ", "CHW": "CWS", "KCR": "KC", "OAK": "ATH", "SDP": "SD",
    "SFG": "SF", "TBR": "TB", "WSN": "WSH",
}

# Park run factors keyed by B-Ref home-team code. ATH = Sutter Health Park
# (Sacramento, 2025+), a hitter-friendly minor-league park.
PARK_FACTORS = {
    "ARI": 1.03, "ATL": 1.02, "BAL": 0.98, "BOS": 1.06, "CHC": 0.99,
    "CHW": 1.01, "CIN": 1.07, "CLE": 0.97, "COL": 1.12, "DET": 0.96,
    "HOU": 1.00, "KCR": 1.02, "LAA": 0.99, "LAD": 1.01, "MIA": 0.97,
    "MIL": 1.00, "MIN": 0.99, "NYM": 0.96, "NYY": 1.02, "OAK": 0.98,
    "ATH": 1.05, "PHI": 1.02, "PIT": 0.97, "SDP": 0.96, "SEA": 0.93,
    "SFG": 0.95, "STL": 0.99, "TBR": 0.98, "TEX": 1.01, "TOR": 1.00,
    "WSN": 1.00,
}

EVENT_OUTS = {
    "strikeout": 1, "strikeout_double_play": 2, "field_out": 1,
    "force_out": 1, "grounded_into_double_play": 2, "double_play": 2,
    "sac_fly": 1, "sac_bunt": 1, "fielders_choice_out": 1,
    "sac_fly_double_play": 2, "sac_bunt_double_play": 2, "triple_play": 3,
    "other_out": 1, "caught_stealing_2b": 1, "caught_stealing_3b": 1,
    "caught_stealing_home": 1, "pickoff_1b": 1, "pickoff_2b": 1,
    "pickoff_3b": 1,
}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
BB_EVENTS = {"walk", "intent_walk", "intentional_walk"}
TB_MAP = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
NON_AB_EVENTS = BB_EVENTS | {
    "hit_by_pitch", "sac_fly", "sac_fly_double_play",
    "sac_bunt", "sac_bunt_double_play", "catcher_interf", "truncated_pa",
}


class RollingLookup:
    """Per-key cumulative sums over dates -> O(log n) window queries."""

    def __init__(self, df: pd.DataFrame, key_cols, value_cols):
        self.value_cols = value_cols
        self.data = {}
        for key, g in df.groupby(key_cols, sort=False):
            g = g.sort_values("game_date")
            dates = g["game_date"].to_numpy(dtype="datetime64[D]")
            cums = {c: np.concatenate([[0.0], np.cumsum(g[c].to_numpy(float))])
                    for c in value_cols}
            self.data[key] = (dates, cums)

    def window(self, key, start: np.datetime64, end: np.datetime64):
        """Sums over [start, end] inclusive, or None if no data."""
        entry = self.data.get(key)
        if entry is None:
            return None
        dates, cums = entry
        i0 = np.searchsorted(dates, start, side="left")
        i1 = np.searchsorted(dates, end, side="right")
        if i1 <= i0:
            return None
        return {c: arr[i1] - arr[i0] for c, arr in cums.items()}


def load_statcast() -> pd.DataFrame:
    frames = []
    for y in SEASONS:
        f = DATA_DIR / f"statcast_{y}.parquet"
        if not f.exists():
            sys.exit(f"[error] {f} missing -- run download_statcast.py first")
        frames.append(pd.read_parquet(f))
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number",
                         "pitch_number"], ignore_index=True)
    df["season"] = df["game_date"].dt.year
    df["bat_team"] = np.where(df["inning_topbot"] == "Top",
                              df["away_team"], df["home_team"])
    df["pitch_team"] = np.where(df["inning_topbot"] == "Top",
                                df["home_team"], df["away_team"])
    df["runs_on_pitch"] = (df["post_bat_score"] - df["bat_score"]).clip(lower=0)
    return df


def annotate_events(df: pd.DataFrame) -> pd.DataFrame:
    ev = df[df["events"].notna() & (df["events"] != "")].copy()
    e = ev["events"]
    ev["outs"] = e.map(EVENT_OUTS).fillna(0)
    ev["is_k"] = e.str.startswith("strikeout").astype(int)
    ev["is_bb"] = e.isin(BB_EVENTS).astype(int)
    ev["is_hbp"] = (e == "hit_by_pitch").astype(int)
    ev["is_hr"] = (e == "home_run").astype(int)
    ev["is_hit"] = e.isin(HIT_EVENTS).astype(int)
    ev["is_sf"] = e.isin(["sac_fly", "sac_fly_double_play"]).astype(int)
    ev["is_ab"] = (~e.isin(NON_AB_EVENTS)).astype(int)
    ev["tb"] = e.map(TB_MAP).fillna(0)
    ev["pa"] = 1
    ev["is_gb"] = (ev["bb_type"] == "ground_ball").astype(int)
    ev["is_bip"] = ev["bb_type"].notna().astype(int)
    ev["woba_value"] = pd.to_numeric(ev["woba_value"], errors="coerce").fillna(0)
    ev["woba_denom"] = pd.to_numeric(ev["woba_denom"], errors="coerce").fillna(0)
    return ev


def find_starters(df: pd.DataFrame) -> dict:
    """{(game_pk, 'home'|'away'): pitcher_id} from the first pitch of
    each half of the 1st inning (home pitches the Top)."""
    first = (df[df["inning"] == 1]
             .drop_duplicates(subset=["game_pk", "inning_topbot"], keep="first"))
    starters = {}
    for _, r in first.iterrows():
        side = "home" if r["inning_topbot"] == "Top" else "away"
        starters[(r["game_pk"], side)] = int(r["pitcher"])
    return starters


def pitcher_hands(df: pd.DataFrame) -> dict:
    return (df.drop_duplicates("pitcher").set_index("pitcher")["p_throws"]
            .to_dict())


def game_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Per game_pk: date, teams, final score, YRFI flag."""
    last = df.drop_duplicates("game_pk", keep="last")
    gs = last[["game_pk", "game_date", "home_team", "away_team",
               "post_home_score", "post_away_score"]].copy()
    gs = gs.rename(columns={"post_home_score": "sc_home_runs",
                            "post_away_score": "sc_away_runs"})
    fi = df[df["inning"] == 1].groupby("game_pk")["runs_on_pitch"].sum()
    gs["first_inning_run"] = (gs["game_pk"].map(fi).fillna(0) > 0).astype(int)
    return gs


def match_backbone(backbone: pd.DataFrame, gs: pd.DataFrame) -> pd.DataFrame:
    """Attach Statcast game_pk to each B-Ref game (score-based for DHs)."""
    backbone = backbone.copy()
    backbone["home_sc"] = backbone["home_team_br"].map(
        lambda t: BR_TO_SC.get(t, t))
    backbone["away_sc"] = backbone["away_team_br"].map(
        lambda t: BR_TO_SC.get(t, t))
    idx = defaultdict(list)
    for _, r in gs.iterrows():
        key = (r["game_date"].strftime("%Y-%m-%d"),
               r["home_team"], r["away_team"])
        idx[key].append(r)

    pks, fi_flags, mismatch = [], [], 0
    for _, b in backbone.iterrows():
        cands = idx.get((b["date"], b["home_sc"], b["away_sc"]), [])
        pick = None
        if len(cands) == 1:
            pick = cands[0]
        elif len(cands) > 1:
            exact = [c for c in cands
                     if c["sc_home_runs"] == b["home_runs"]
                     and c["sc_away_runs"] == b["away_runs"]]
            if len(exact) == 1:
                pick = exact[0]
            else:  # identical-score doubleheader: order by game_pk
                ordered = sorted(cands, key=lambda c: c["game_pk"])
                pick = ordered[min(b["dh_game"] - 1, len(ordered) - 1)]
        if pick is None:
            pks.append(np.nan)
            fi_flags.append(np.nan)
            continue
        if (pick["sc_home_runs"] != b["home_runs"]
                or pick["sc_away_runs"] != b["away_runs"]):
            mismatch += 1
        pks.append(pick["game_pk"])
        fi_flags.append(pick["first_inning_run"])

    backbone["game_pk"] = pks
    backbone["first_inning_run"] = fi_flags
    unmatched = backbone["game_pk"].isna().sum()
    if unmatched:
        print(f"[warn] {unmatched} B-Ref games had no Statcast match -- dropped:")
        print(backbone[backbone["game_pk"].isna()]
              [["date", "home_team_br", "away_team_br"]].to_string(index=False))
        backbone = backbone.dropna(subset=["game_pk"])
    if mismatch:
        print(f"[warn] {mismatch} games matched with B-Ref/Statcast score "
              f"disagreement (kept; B-Ref scores are the targets)")
    backbone["game_pk"] = backbone["game_pk"].astype(int)
    backbone["first_inning_run"] = backbone["first_inning_run"].astype(int)
    return backbone


def build_lookups(df: pd.DataFrame, ev: pd.DataFrame, starters: dict) -> dict:
    print("  aggregating pitcher daily lines ...")
    pit = (ev.groupby(["pitcher", "game_date"], sort=False)
           .agg(outs=("outs", "sum"), k=("is_k", "sum"), bb=("is_bb", "sum"),
                hbp=("is_hbp", "sum"), hr=("is_hr", "sum"),
                gb=("is_gb", "sum"), bip=("is_bip", "sum"))
           .reset_index())
    pit_lu = RollingLookup(pit, "pitcher",
                           ["outs", "k", "bb", "hbp", "hr", "gb", "bip"])

    print("  aggregating first-inning starter lines ...")
    fi_pitch = df[df["inning"] == 1]
    fi_runs = (fi_pitch.groupby(["pitcher", "game_date"], sort=False)
               .agg(fi_runs=("runs_on_pitch", "sum"),
                    fi_games=("game_pk", "nunique"))
               .reset_index())
    fi_ev = ev[ev["inning"] == 1]
    fi_rates = (fi_ev.groupby(["pitcher", "game_date"], sort=False)
                .agg(fi_bb=("is_bb", "sum"), fi_pa=("pa", "sum"))
                .reset_index())
    fi_daily = fi_runs.merge(fi_rates, on=["pitcher", "game_date"], how="left")
    fi_daily[["fi_bb", "fi_pa"]] = fi_daily[["fi_bb", "fi_pa"]].fillna(0)
    fi_lu = RollingLookup(fi_daily, "pitcher",
                          ["fi_runs", "fi_games", "fi_bb", "fi_pa"])

    print("  aggregating bullpen daily lines ...")
    side = np.where(df["inning_topbot"] == "Top", "home", "away")
    keys = list(zip(df["game_pk"], side))
    starter_of_row = np.array([starters.get(k, -1) for k in keys])
    is_rel = df["pitcher"].to_numpy() != starter_of_row
    rel = df[is_rel]
    rel_runs = (rel.groupby(["pitch_team", "game_date"], sort=False)
                .agg(runs=("runs_on_pitch", "sum")).reset_index())
    # ev is a subset of df rows; recompute the reliever mask on ev directly
    ev_keys = list(zip(ev["game_pk"], np.where(ev["inning_topbot"] == "Top",
                                               "home", "away")))
    ev_starter = np.array([starters.get(k, -1) for k in ev_keys])
    rel_ev = ev[ev["pitcher"].to_numpy() != ev_starter]
    rel_outs = (rel_ev.groupby(["pitch_team", "game_date"], sort=False)
                .agg(outs=("outs", "sum")).reset_index())
    pen = rel_runs.merge(rel_outs, on=["pitch_team", "game_date"], how="outer")
    pen[["runs", "outs"]] = pen[["runs", "outs"]].fillna(0)
    pen_lu = RollingLookup(pen, "pitch_team", ["runs", "outs"])

    print("  aggregating team offense daily lines (by pitcher hand) ...")
    off_cols = dict(pa=("pa", "sum"), ob=("is_ob", "sum"),
                    obp_den=("obp_den", "sum"),
                    wnum=("woba_value", "sum"), wden=("woba_denom", "sum"))
    ev = ev.copy()
    ev["is_ob"] = ev["is_hit"] + ev["is_bb"] + ev["is_hbp"]
    ev["obp_den"] = ev["is_ab"] + ev["is_bb"] + ev["is_hbp"] + ev["is_sf"]
    off_split = (ev.groupby(["bat_team", "p_throws", "game_date"], sort=False)
                 .agg(**off_cols).reset_index())
    off_all = (ev.groupby(["bat_team", "game_date"], sort=False)
               .agg(**off_cols).reset_index())
    lg_split = (ev.groupby(["p_throws", "game_date"], sort=False)
                .agg(**off_cols).reset_index())
    lg_all = ev.groupby("game_date", sort=False).agg(**off_cols).reset_index()
    lg_all["all"] = "ALL"
    vals = ["pa", "ob", "obp_den", "wnum", "wden"]
    off_split_lu = RollingLookup(off_split, ["bat_team", "p_throws"], vals)
    off_all_lu = RollingLookup(off_all, "bat_team", vals)
    lg_split_lu = RollingLookup(lg_split, "p_throws", vals)
    lg_all_lu = RollingLookup(lg_all, "all", vals)

    print("  aggregating batter season-to-date lines ...")
    bat = (ev.groupby(["season", "batter", "game_date"], sort=False)
           .agg(pa=("pa", "sum"), ob=("is_ob", "sum"),
                obp_den=("obp_den", "sum"), ab=("is_ab", "sum"),
                hit=("is_hit", "sum"), tb=("tb", "sum"))
           .reset_index())
    bat_lu = RollingLookup(bat, ["season", "batter"],
                           ["pa", "ob", "obp_den", "ab", "hit", "tb"])

    print("  extracting top-3 lineup slots per game ...")
    ab_first = df.drop_duplicates(subset=["game_pk", "inning_topbot",
                                          "at_bat_number"], keep="first")
    top3 = {}
    for (pk, tb_), g in ab_first.groupby(["game_pk", "inning_topbot"],
                                         sort=False):
        side = "away" if tb_ == "Top" else "home"
        batters = g.sort_values("at_bat_number")["batter"].drop_duplicates()
        top3[(pk, side)] = [int(b) for b in batters.head(3)]

    return {"pit": pit_lu, "fi": fi_lu, "pen": pen_lu,
            "off_split": off_split_lu, "off_all": off_all_lu,
            "lg_split": lg_split_lu, "lg_all": lg_all_lu,
            "bat": bat_lu, "top3": top3}


def starter_form(lu, pid, d0, d1) -> dict:
    out = {"k9": LEAGUE_AVG["k9"], "bb9": LEAGUE_AVG["bb9"],
           "fip": LEAGUE_AVG["fip"], "gb_pct": LEAGUE_AVG["gb_pct"]}
    w = lu.window(pid, d0, d1)
    if w is None or w["outs"] < 9:  # < 3 IP in window
        return out
    ip = w["outs"] / 3.0
    out["k9"] = round(9.0 * w["k"] / ip, 2)
    out["bb9"] = round(9.0 * w["bb"] / ip, 2)
    out["fip"] = round((13 * w["hr"] + 3 * (w["bb"] + w["hbp"])
                        - 2 * w["k"]) / ip + FIP_CONSTANT, 2)
    if w["bip"] >= 10:
        out["gb_pct"] = round(w["gb"] / w["bip"], 3)
    return out


def bullpen_ra9(lu, team_sc, d0, d1) -> float:
    w = lu.window(team_sc, d0, d1)
    if w is None or w["outs"] < 30:  # < 10 IP in window
        return LEAGUE_AVG["bullpen_era14"]
    return round(9.0 * w["runs"] / (w["outs"] / 3.0), 2)


def offense(lus, team_sc, hand, d0, d1) -> dict:
    """30-day wOBA-indexed wRC+ proxy & OBP vs `hand`, with fallbacks."""
    out = {"wrc_plus": LEAGUE_AVG["wrc_plus"], "obp": LEAGUE_AVG["obp"]}
    for team_lu, team_key, lg_lu, lg_key, min_pa in (
            (lus["off_split"], (team_sc, hand), lus["lg_split"], hand, 120),
            (lus["off_all"], team_sc, lus["lg_all"], "ALL", 120)):
        w = team_lu.window(team_key, d0, d1)
        lg = lg_lu.window(lg_key, d0, d1)
        if w is None or lg is None or w["pa"] < min_pa or w["wden"] < 1:
            continue
        team_woba = w["wnum"] / w["wden"]
        lg_woba = lg["wnum"] / lg["wden"]
        if lg_woba > 0:
            out["wrc_plus"] = round(100.0 * team_woba / lg_woba, 1)
        if w["obp_den"] > 0:
            out["obp"] = round(w["ob"] / w["obp_den"], 3)
        return out
    return out


def first_inning_form(lu, pid, d0, d1) -> dict:
    out = {"fi_ra9": LEAGUE_AVG["fi_ra9"],
           "fi_bb_rate": LEAGUE_AVG["fi_bb_rate"]}
    w = lu.window(pid, d0, d1)
    if w is None or w["fi_games"] < 3:
        return out
    out["fi_ra9"] = round(9.0 * w["fi_runs"] / w["fi_games"], 2)
    if w["fi_pa"] >= 10:
        out["fi_bb_rate"] = round(w["fi_bb"] / w["fi_pa"], 3)
    return out


def top3_stats(lus, pk, side, season, d1) -> dict:
    out = {"obp": LEAGUE_AVG["top3_obp"], "iso": LEAGUE_AVG["top3_iso"]}
    season_start = np.datetime64(f"{season}-01-01")
    obps, isos = [], []
    for pid in lus["top3"].get((pk, side), []):
        w = lus["bat"].window((season, pid), season_start, d1)
        if w is None or w["ab"] < 30 or w["obp_den"] < 1:
            continue
        obps.append(w["ob"] / w["obp_den"])
        isos.append((w["tb"] - w["hit"]) / w["ab"])
    if obps:
        out["obp"] = round(float(np.mean(obps)), 3)
        out["iso"] = round(float(np.mean(isos)), 3)
    return out


def main() -> None:
    if not BACKBONE_CSV.exists():
        sys.exit(f"[error] {BACKBONE_CSV} missing -- run "
                 f"scrape_bref_schedules.py first")
    backbone = pd.read_csv(BACKBONE_CSV)
    print(f"Backbone: {len(backbone)} games")

    print("Loading Statcast ...")
    df = load_statcast()
    print(f"  {len(df):,} pitches, {df['game_pk'].nunique():,} games")

    ev = annotate_events(df)
    starters = find_starters(df)
    hands = pitcher_hands(df)
    gs = game_summaries(df)

    backbone = match_backbone(backbone, gs)
    print(f"Matched: {len(backbone)} games")

    print("Building rolling lookups ...")
    lus = build_lookups(df, ev, starters)

    print("Computing as-of-date features per game ...")
    rows = []
    no_starter = 0
    for _, b in backbone.sort_values("date").iterrows():
        pk = b["game_pk"]
        d = np.datetime64(b["date"])
        d1 = d - np.timedelta64(1, "D")           # window always ends day before
        d30 = d - np.timedelta64(30, "D")
        d14 = d - np.timedelta64(14, "D")
        d90 = d - np.timedelta64(90, "D")
        season = int(b["season"])

        h_sp = starters.get((pk, "home"))
        a_sp = starters.get((pk, "away"))
        if h_sp is None or a_sp is None:
            no_starter += 1
            continue

        h_form = starter_form(lus["pit"], h_sp, d30, d1)
        a_form = starter_form(lus["pit"], a_sp, d30, d1)
        # Offense faces the OPPOSING starter's throwing hand.
        h_off = offense(lus, b["home_sc"], hands.get(a_sp, "R"), d30, d1)
        a_off = offense(lus, b["away_sc"], hands.get(h_sp, "R"), d30, d1)
        h_fi = first_inning_form(lus["fi"], h_sp, d90, d1)
        a_fi = first_inning_form(lus["fi"], a_sp, d90, d1)
        h_top = top3_stats(lus, pk, "home", season, d1)
        a_top = top3_stats(lus, pk, "away", season, d1)

        rows.append({
            "date": b["date"], "season": season, "game_pk": pk,
            "home_team": b["home_team_br"], "away_team": b["away_team_br"],
            "home_sp_id": h_sp, "away_sp_id": a_sp,
            # -- moneyline features
            "home_sp_k9": h_form["k9"], "home_sp_bb9": h_form["bb9"],
            "home_sp_xfip": h_form["fip"], "home_sp_gb_pct": h_form["gb_pct"],
            "away_sp_k9": a_form["k9"], "away_sp_bb9": a_form["bb9"],
            "away_sp_xfip": a_form["fip"], "away_sp_gb_pct": a_form["gb_pct"],
            "home_team_wrc_plus": h_off["wrc_plus"],
            "home_team_obp": h_off["obp"],
            "away_team_wrc_plus": a_off["wrc_plus"],
            "away_team_obp": a_off["obp"],
            # -- totals features
            "home_sp_xra9": h_form["fip"], "away_sp_xra9": a_form["fip"],
            "combined_sp_xra9": round(h_form["fip"] + a_form["fip"], 2),
            "home_bullpen_era14": bullpen_ra9(lus["pen"], b["home_sc"], d14, d1),
            "away_bullpen_era14": bullpen_ra9(lus["pen"], b["away_sc"], d14, d1),
            "park_factor": PARK_FACTORS.get(b["home_team_br"], 1.0),
            # -- NRFI features
            "home_sp_fi_ra9": h_fi["fi_ra9"],
            "home_sp_fi_bb_rate": h_fi["fi_bb_rate"],
            "away_sp_fi_ra9": a_fi["fi_ra9"],
            "away_sp_fi_bb_rate": a_fi["fi_bb_rate"],
            "home_top3_obp": h_top["obp"], "home_top3_iso": h_top["iso"],
            "away_top3_obp": a_top["obp"], "away_top3_iso": a_top["iso"],
            # -- targets
            "home_win": int(b["home_win"]),
            "home_runs": int(b["home_runs"]),
            "away_runs": int(b["away_runs"]),
            "total_runs": int(b["total_runs"]),
            "first_inning_run": int(b["first_inning_run"]),
        })

    if no_starter:
        print(f"[warn] {no_starter} games skipped (no starter found in Statcast)")
    hist = pd.DataFrame(rows)
    hist.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}: {len(hist)} rows")
    print(f"  home_win rate       : {hist['home_win'].mean():.3f}")
    print(f"  avg total runs      : {hist['total_runs'].mean():.2f}")
    print(f"  YRFI rate           : {hist['first_inning_run'].mean():.3f}")
    print(f"  mean starter FIP    : {hist['home_sp_xfip'].mean():.2f}")
    print(f"  mean wRC+ proxy     : {hist['home_team_wrc_plus'].mean():.1f}")


if __name__ == "__main__":
    main()
