"""Bulk-download league-wide Statcast pitch data for whole seasons.

One macro-download per season instead of statcast_pitcher() in a loop:
slicing the resulting dataframe for any pitcher/team is then instant.
Only the ~20 columns the feature builder needs are kept, so each season
parquet stays small (~40 MB) despite ~750k pitches.

Usage:
    python download_statcast.py                  # 2024 + 2025
    python download_statcast.py --years 2025

Output:
    data/statcast_{year}.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

import pybaseball
from pybaseball import statcast

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pybaseball.cache.enable()

DATA_DIR = Path(__file__).parent / "data"

# Regular-season spans (incl. Seoul 2024 / Tokyo 2025 openers and the
# 2024-09-30 makeup doubleheader).
SEASON_SPAN = {
    2024: ("2024-03-20", "2024-09-30"),
    2025: ("2025-03-18", "2025-09-28"),
}

KEEP_COLS = [
    "game_pk", "game_date", "game_type", "home_team", "away_team",
    "pitcher", "batter", "p_throws", "stand",
    "events", "bb_type", "inning", "inning_topbot",
    "at_bat_number", "pitch_number",
    "bat_score", "post_bat_score", "post_home_score", "post_away_score",
    "woba_value", "woba_denom",
]

CHUNK_DAYS = 10


def download_season(year: int) -> None:
    out = DATA_DIR / f"statcast_{year}.parquet"
    if out.exists():
        print(f"{out} already exists, skipping (delete it to re-download).")
        return
    start = dt.date.fromisoformat(SEASON_SPAN[year][0])
    end = dt.date.fromisoformat(SEASON_SPAN[year][1])

    chunks = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=CHUNK_DAYS - 1), end)
        print(f"  {year}: {cur} .. {chunk_end}")
        df = statcast(cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"),
                      verbose=False)
        if df is not None and len(df):
            cols = [c for c in KEEP_COLS if c in df.columns]
            chunks.append(df[cols])
        cur = chunk_end + dt.timedelta(days=1)

    season = pd.concat(chunks, ignore_index=True)
    # Regular season only (Seoul/Tokyo openers are game_type 'R' too).
    if "game_type" in season.columns:
        season = season[season["game_type"] == "R"]
    season["game_date"] = pd.to_datetime(season["game_date"])
    season = season.sort_values(["game_date", "game_pk", "at_bat_number",
                                 "pitch_number"], ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    season.to_parquet(out, index=False)
    print(f"  -> {out}: {len(season):,} pitches, "
          f"{season['game_pk'].nunique():,} games")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="*", type=int, default=[2024, 2025])
    args = ap.parse_args()
    for y in args.years:
        print(f"Season {y} ...")
        download_season(y)


if __name__ == "__main__":
    main()
