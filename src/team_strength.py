"""
team_strength.py
================
Step 2 (part 1) of the WC26 fantasy optimizer: attach a strength rating to each
of the 48 teams, so later steps can estimate match outcomes and -- crucially --
how deep each team is likely to go (you score no fantasy points once your team
is knocked out).

We use the public World Football Elo ratings from eloratings.net:

    data/raw/elo_world.tsv     current rating per national team (one row each)
    data/raw/elo_teams.tsv     eloratings' own 2-letter-code -> country-name map

and join them onto our team list to produce:

    data/processed/team_strength.csv   one row per WC26 team, with its Elo rating

eloratings.net uses its own (mostly ISO-ish) 2-letter codes and English names.
41 of our 48 teams match their names exactly; the 7 that differ are aliased in
NAME_ALIASES below. Higher Elo = stronger.

Run it directly:   python src/team_strength.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ELO_BASE = "https://www.eloratings.net"
HEADERS = {"User-Agent": "Mozilla/5.0 (wc26-fantasy-optimizer)"}

# Our country name (as it appears in players.csv) -> eloratings.net's name, for
# the 7 teams whose names don't match verbatim. The other 41 join directly.
NAME_ALIASES = {
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USA": "United States",
}


# --- fetch -----------------------------------------------------------------
def _fetch(name: str, url: str, force: bool) -> str:
    """Download (and cache) one raw text file under data/raw/."""
    path = RAW / name
    if force or not path.exists():
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", "replace")
        path.write_text(text, encoding="utf-8")
        print(f"  fetched {name}  ({len(text.encode('utf-8')):,} bytes)")
    else:
        print(f"  cached  {name}")
    return path.read_text(encoding="utf-8")


def fetch_elo(force: bool = False) -> tuple[dict[str, int], dict[str, str]]:
    """Return (code -> rating) and (code -> country-name) from eloratings.net."""
    RAW.mkdir(parents=True, exist_ok=True)
    world = _fetch("elo_world.tsv", f"{ELO_BASE}/World.tsv", force)
    teams = _fetch("elo_teams.tsv", f"{ELO_BASE}/en.teams.tsv", force)

    # World.tsv columns: rank, _, code, rating, ...  (we only need code + rating)
    ratings: dict[str, int] = {}
    for line in world.splitlines():
        cols = line.split("\t")
        if len(cols) > 3 and cols[3].lstrip("-").isdigit():
            ratings[cols[2]] = int(cols[3])

    # en.teams.tsv columns: code, country-name
    code_to_name = {
        c[0]: c[1] for c in (l.split("\t") for l in teams.splitlines()) if len(c) >= 2
    }
    return ratings, code_to_name


# --- build -----------------------------------------------------------------
def build_team_strength(force: bool = False) -> pd.DataFrame:
    ratings, code_to_name = fetch_elo(force=force)
    name_to_code = {name: code for code, name in code_to_name.items()}

    players = pd.read_csv(PROCESSED / "players.csv")
    teams = (
        players[["country", "group", "abbr"]]
        .drop_duplicates()
        .sort_values(["group", "country"])
        .reset_index(drop=True)
    )

    def lookup(country: str) -> tuple[str | None, int | None]:
        elo_name = NAME_ALIASES.get(country, country)
        code = name_to_code.get(elo_name)
        return code, ratings.get(code) if code else None

    teams[["elo_code", "elo"]] = teams["country"].apply(
        lambda c: pd.Series(lookup(c))
    )

    # Every team must resolve -- a missing Elo silently zeroes out a country's
    # advancement odds later, so fail loud instead.
    missing = teams[teams["elo"].isna()]["country"].tolist()
    if missing:
        raise ValueError(f"No Elo rating for: {missing}")

    teams["elo"] = teams["elo"].astype(int)
    return teams.sort_values("elo", ascending=False).reset_index(drop=True)


# --- main ------------------------------------------------------------------
def main(force: bool = False) -> None:
    print("Building team strength (Elo)...")
    teams = build_team_strength(force=force)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "team_strength.csv"
    teams.to_csv(out, index=False)

    print(f"\nteam_strength.csv : {len(teams)} teams -> {out}")
    print(f"  strongest: {teams.iloc[0]['country']} ({teams.iloc[0]['elo']})")
    print(f"  weakest:   {teams.iloc[-1]['country']} ({teams.iloc[-1]['elo']})")
    print("\nTop 10:")
    print(teams.head(10)[["country", "group", "elo"]].to_string(index=False))


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
