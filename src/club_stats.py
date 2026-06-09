"""Step 3 data: recent club-season stats per player, from FBref via soccerdata.

We pull 2025-26 club-season counting stats (minutes, goals, assists, shots,
shots-on-target, cards) for a configurable set of leagues, then write a tidy
table keyed by (player, nation, team, league). The `nation` column is the
second join key we use later to match these rows back to the FIFA fantasy
player universe (`players.csv`) without relying on names alone.

soccerdata 1.9.0 season stats expose only counting stats (no xG/xA/tackles);
those would need Understat (Big-5 only) and are deferred. See project plan.

Usage:
    python src/club_stats.py                # scrape the default league set
    python src/club_stats.py --season 2025-2026
"""

import argparse
import json
import os

import pandas as pd
# NOTE: soccerdata reads its league config at import time, so we import it
# lazily inside main() *after* register_leagues() has written the config.

# Canonical league id -> FBref display name. "Big 5 European Leagues Combined"
# is a single fast endpoint covering ENG/ESP/ITA/GER/FRA. Everything else is a
# per-league scrape we register into soccerdata's config below.
# FBref display names must match the /en/comps/ index EXACTLY or the league
# silently resolves to nothing ("No objects to concatenate").
EXTRA_LEAGUES = {
    "NED-Eredivisie":      {"FBref": "Eredivisie",            "season_start": "Aug", "season_end": "May"},
    "POR-Primeira Liga":   {"FBref": "Primeira Liga",         "season_start": "Aug", "season_end": "May"},
    "ENG-Championship":    {"FBref": "EFL Championship",      "season_start": "Aug", "season_end": "May"},
    "USA-MLS":             {"FBref": "Major League Soccer",   "season_start": "Feb", "season_end": "Dec", "season_code": "single-year"},
    "MEX-Liga MX":         {"FBref": "Liga MX",               "season_start": "Jul", "season_end": "May"},
    "SAU-Pro League":      {"FBref": "Saudi Pro League",      "season_start": "Aug", "season_end": "May"},
    "TUR-Super Lig":       {"FBref": "Süper Lig",             "season_start": "Aug", "season_end": "May"},
    "BEL-Pro League":      {"FBref": "Belgian Pro League",    "season_start": "Aug", "season_end": "May"},
    "BRA-Serie A":         {"FBref": "Campeonato Brasileiro Série A",        "season_start": "Apr", "season_end": "Dec", "season_code": "single-year"},
    "ARG-Primera":         {"FBref": "Liga Profesional de Fútbol Argentina", "season_start": "Jan", "season_end": "Dec", "season_code": "single-year"},
}

# Calendar-year leagues need a single-year season string; everyone else uses
# the Aug-May code passed via --season.
SEASON_OVERRIDE = {
    "BRA-Serie A": "2025",
    "ARG-Primera": "2025",
}

# Leagues to actually scrape (start broad; expand via coverage gap analysis).
DEFAULT_LEAGUES = ["Big 5 European Leagues Combined"] + list(EXTRA_LEAGUES)

RAW_OUT = "data/raw/club_stats_raw.pkl"
CSV_OUT = "data/processed/club_player_stats.csv"


def register_leagues():
    """Merge our extra leagues into soccerdata's custom league_dict.json."""
    cfg_dir = os.path.expanduser("~/soccerdata/config")
    os.makedirs(cfg_dir, exist_ok=True)
    path = os.path.join(cfg_dir, "league_dict.json")
    existing = {}
    if os.path.isfile(path):
        existing = json.load(open(path))
    json.dump({**existing, **EXTRA_LEAGUES}, open(path, "w"), indent=2)


def _col(df, *candidates):
    """Return the first matching column (multiindex tuple) as a Series, else NaN."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(pd.NA, index=df.index)


def pull_league(league, season):
    """Pull standard + shooting season stats for one league, return a flat frame."""
    import soccerdata as sd
    fb = sd.FBref(leagues=league, seasons=season)
    std = fb.read_player_season_stats(stat_type="standard")
    sht = fb.read_player_season_stats(stat_type="shooting")

    out = pd.DataFrame(index=std.index)
    out["nation"] = _col(std, ("nation", ""))
    out["fbref_pos"] = _col(std, ("pos", ""))
    out["born"] = _col(std, ("born", ""))
    out["mp"] = _col(std, ("Playing Time", "MP"))
    out["starts"] = _col(std, ("Playing Time", "Starts"))
    out["minutes"] = _col(std, ("Playing Time", "Min"))
    out["nineties"] = _col(std, ("Playing Time", "90s"))
    out["goals"] = _col(std, ("Performance", "Gls"))
    out["assists"] = _col(std, ("Performance", "Ast"))
    out["pk"] = _col(std, ("Performance", "PK"))
    out["yellow"] = _col(std, ("Performance", "CrdY"))
    out["red"] = _col(std, ("Performance", "CrdR"))
    # shooting shares the (league, season, team, player) index
    out["shots"] = _col(sht, ("Standard", "Sh"))
    out["sot"] = _col(sht, ("Standard", "SoT"))

    out = out.reset_index()  # league, season, team, player -> columns
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-2026")
    ap.add_argument("--leagues", nargs="*", default=DEFAULT_LEAGUES)
    args = ap.parse_args()

    register_leagues()  # must happen before soccerdata is imported (in pull_league)

    frames = []
    for lg in args.leagues:
        season = SEASON_OVERRIDE.get(lg, args.season)
        try:
            df = pull_league(lg, season)
            print(f"  {lg:34} {len(df):4d} players")
            frames.append(df)
        except Exception as e:  # one bad league shouldn't sink the whole run
            print(f"  {lg:34} FAILED: {type(e).__name__}: {e}")

    allp = pd.concat(frames, ignore_index=True)
    # numeric coercion (FBref cells are strings)
    num = ["mp", "starts", "minutes", "nineties", "goals", "assists",
           "pk", "yellow", "red", "shots", "sot", "born"]
    for c in num:
        allp[c] = pd.to_numeric(allp[c], errors="coerce")

    # soccerdata's Big-5 combined parser fails to map the German sub-league
    # ("Fußball-Bundesliga"), leaving it NaN. Every other league is explicitly
    # labelled (separate pulls), so any remaining NaN is Bundesliga.
    allp["league"] = allp["league"].fillna("GER-Bundesliga")

    allp.to_pickle(RAW_OUT)
    allp.to_csv(CSV_OUT, index=False)
    print(f"\n{len(allp)} player-rows across {allp['league'].nunique()} leagues "
          f"-> {CSV_OUT}")


if __name__ == "__main__":
    main()
