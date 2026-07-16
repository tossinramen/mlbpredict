"""Scrape Baseball-Reference team schedule-scores pages with pydoll.

Builds the historical "schedule backbone" for mlb_history_2024_2025.csv:
one row per team-game with date, opponent, home/away, runs scored/allowed,
and W/L -- i.e. the targets, tied to a calendar date. Feature values are
computed separately (build_history.py) from as-of-date data only, so no
end-of-season totals ever leak into training rows.

Usage:
    python scrape_bref_schedules.py                # all 30 teams x 2024+2025
    python scrape_bref_schedules.py --teams ARI    # quick single-team test
    python scrape_bref_schedules.py --years 2024

Output:
    data/bref/{TEAM}_{YEAR}.csv      raw per-team schedule rows
    data/schedule_backbone.csv       one row per GAME (home-team rows only)

"""

from __future__ import annotations

import argparse
import asyncio
import io
import re
import sys
from pathlib import Path

import pandas as pd

from pydoll.browser.chromium import Chrome
from pydoll.browser.options import ChromiumOptions

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "bref"
BACKBONE_CSV = DATA_DIR / "schedule_backbone.csv"

REQUEST_GAP_S = 3.5  

TEAMS_2024 = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

TEAMS_2025 = [t if t != "OAK" else "ATH" for t in TEAMS_2024]

TEAMS_BY_YEAR = {2024: TEAMS_2024, 2025: TEAMS_2025}

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def parse_bref_date(raw: str, year: int) -> tuple[str, int]:
    """'Thursday, Mar 28' / 'Friday, Jul 4 (2)' -> ('YYYY-MM-DD', dh_game)."""
    dh = 1
    m = re.search(r"\((\d)\)", raw)
    if m:
        dh = int(m.group(1))
        raw = raw[: m.start()].strip()
    m = re.search(r"([A-Z][a-z]{2})\s+(\d{1,2})", raw)
    if not m:
        raise ValueError(f"unparseable date: {raw!r}")
    return f"{year:04d}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}", dh


def extract_schedule_table(html: str) -> pd.DataFrame:
    """Find the team_schedule table (live or inside an HTML comment)."""
    candidates = [html]
    
    candidates.append(re.sub(r"<!--|-->", "", html))
    for blob in candidates:
        try:
            tables = pd.read_html(io.StringIO(blob))
        except ValueError:
            continue
        for t in tables:
            cols = [str(c) for c in t.columns]
            if "Gm#" in cols and "Opp" in cols and "R" in cols:
                return t
    raise ValueError("team_schedule table not found in page")


def clean_team_schedule(df: pd.DataFrame, team: str, year: int) -> pd.DataFrame:
    """Raw B-Ref table -> tidy per-team-game rows."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
   
    at_col = df.columns[list(df.columns).index("Opp") - 1]

    df = df[df["Gm#"].astype(str).str.fullmatch(r"\d+")]        
    df = df[df["W/L"].notna() & (df["W/L"].astype(str).str.len() > 0)] 

    rows = []
    for _, r in df.iterrows():
        date_iso, dh = parse_bref_date(str(r["Date"]), year)
        wl = str(r["W/L"]).strip()         
        rows.append({
            "season": year,
            "date": date_iso,
            "dh_game": dh,
            "team": team,
            "opp": str(r["Opp"]).strip(),
            "is_home": 0 if str(r[at_col]).strip() == "@" else 1,
            "win": 1 if wl.startswith("W") else 0,
            "runs_scored": int(float(r["R"])),
            "runs_allowed": int(float(r["RA"])),
            "innings": (int(float(r["Inn"])) if pd.notna(r.get("Inn")) and
                        str(r.get("Inn")).strip() not in ("", "nan") else 9),
            "winning_pitcher": str(r.get("Win", "")),
            "losing_pitcher": str(r.get("Loss", "")),
        })
    return pd.DataFrame(rows)


async def fetch_page(tab, url: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        await tab.go_to(url)
        await asyncio.sleep(2.0)
        html = await tab.page_source
        if "Gm#" in html or "team_schedule" in html:
            return html
        
        print(f"    table not present yet (attempt {attempt}), waiting ...")
        await asyncio.sleep(6.0 * attempt)
        html = await tab.page_source
        if "Gm#" in html or "team_schedule" in html:
            return html
    raise RuntimeError(f"could not load schedule table: {url}")


async def scrape(years: list[int], teams_filter: list[str] | None) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(y, t) for y in years for t in TEAMS_BY_YEAR[y]
            if not teams_filter or t in teams_filter]
   
    pending = [(y, t) for y, t in jobs if not (RAW_DIR / f"{t}_{y}.csv").exists()]
    print(f"{len(jobs)} team-seasons requested, {len(pending)} to fetch "
          f"({len(jobs) - len(pending)} cached in {RAW_DIR}).")

    if pending:
        options = ChromiumOptions()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1280,900")
        async with Chrome(options=options) as browser:
            tab = await browser.start()
            for i, (year, team) in enumerate(pending, 1):
                url = (f"https://www.baseball-reference.com/teams/"
                       f"{team}/{year}-schedule-scores.shtml")
                print(f"[{i}/{len(pending)}] {team} {year} ...")
                try:
                    html = await fetch_page(tab, url)
                    tidy = clean_team_schedule(extract_schedule_table(html), team, year)
                    if len(tidy) < 150:
                        print(f"    [warn] only {len(tidy)} completed games parsed")
                    tidy.to_csv(RAW_DIR / f"{team}_{year}.csv", index=False)
                    print(f"    saved {len(tidy)} games")
                except Exception as exc:
                    print(f"    [error] {team} {year}: {exc}")
                if i < len(pending):
                    await asyncio.sleep(REQUEST_GAP_S)

    build_backbone(years)


def build_backbone(years: list[int]) -> None:
    """Combine per-team CSVs into one row per game (home-team perspective)."""
    frames = []
    missing = []
    for y in years:
        for t in TEAMS_BY_YEAR[y]:
            f = RAW_DIR / f"{t}_{y}.csv"
            if f.exists():
                frames.append(pd.read_csv(f))
            else:
                missing.append(f"{t}_{y}")
    if missing:
        print(f"[warn] backbone missing team-seasons: {missing}")
    if not frames:
        print("[error] nothing scraped; backbone not written")
        return

    allg = pd.concat(frames, ignore_index=True)
    home = allg[allg["is_home"] == 1].copy()
    home = home.rename(columns={
        "team": "home_team_br", "opp": "away_team_br",
        "runs_scored": "home_runs", "runs_allowed": "away_runs",
        "win": "home_win",
    })
    home["total_runs"] = home["home_runs"] + home["away_runs"]
    backbone = home[["season", "date", "dh_game", "home_team_br", "away_team_br",
                     "home_win", "home_runs", "away_runs", "total_runs",
                     "innings", "winning_pitcher", "losing_pitcher"]]
    backbone = backbone.sort_values(["date", "home_team_br", "dh_game"])
    backbone.to_csv(BACKBONE_CSV, index=False)


    n_away = (allg["is_home"] == 0).sum()
    print(f"\nBackbone: {len(backbone)} games -> {BACKBONE_CSV}")
    print(f"  home rows {len(backbone)} vs away rows {n_away} "
          f"({'OK' if len(backbone) == n_away else 'MISMATCH - check logs'})")
    for y in years:
        n = (backbone["season"] == y).sum()
        print(f"  {y}: {n} games (expect ~2430)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="*", type=int, default=[2024, 2025])
    ap.add_argument("--teams", nargs="*", default=None,
                    help="subset of B-Ref team codes, e.g. --teams ARI NYY")
    args = ap.parse_args()
    asyncio.run(scrape(args.years, args.teams))


if __name__ == "__main__":
    main()
