"""Match FIFA fantasy players (players.csv) to FBref club stats.

Join key = normalized name + country. Country is the disambiguator that makes
name matching safe: two "Luis Garcia"s only collide if they're the same
nationality. We aggregate a player's club rows (handles mid-season transfers)
into season totals, then attach to the fantasy universe.

Outputs:
    data/processed/players_with_stats.csv  - fantasy players + club stats (NaN if unmatched)
    prints a coverage report (matched %, and unmatched counts by country)

Usage:
    python src/match_players.py
"""

import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd
from unidecode import unidecode

FIFA_IN = "data/processed/players.csv"
STATS_IN = "data/processed/club_player_stats.csv"
OUT = "data/processed/players_with_stats.csv"

# FBref nation codes that differ from FIFA `abbr`. Extend as gaps surface.
NATION_FIX = {
    # fbref -> fifa (only where they disagree)
}

FUZZY_THRESHOLD = 0.86  # difflib ratio for a same-country fuzzy accept


def norm(name):
    """Accent-strip, lowercase, drop punctuation, collapse spaces."""
    if not isinstance(name, str):
        return ""
    s = unidecode(unicodedata.normalize("NFKD", name)).lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def aggregate_stats(stats):
    """Sum a player's club rows into season totals keyed by (norm_name, nation)."""
    stats = stats.copy()
    stats["norm"] = stats["player"].map(norm)
    stats["nation"] = stats["nation"].map(lambda x: NATION_FIX.get(x, x))
    sum_cols = ["minutes", "starts", "mp", "goals", "assists", "pk",
                "yellow", "red", "shots", "sot", "nineties"]

    # primary league = where the player logged the most minutes (for transfers)
    def primary_league(g):
        return g.loc[g["minutes"].idxmax(), "league"]
    league = (stats.groupby(["norm", "nation"]).apply(primary_league, include_groups=False)
                   .rename("league").reset_index())

    agg = (stats.groupby(["norm", "nation"], as_index=False)
                .agg({**{c: "sum" for c in sum_cols},
                      "team": lambda s: " / ".join(sorted(set(s))),
                      "fbref_pos": "first"}))
    return agg.merge(league, on=["norm", "nation"], how="left")


def main():
    fifa = pd.read_csv(FIFA_IN)
    stats = pd.read_csv(STATS_IN)
    agg = aggregate_stats(stats)

    fifa["norm"] = fifa["display_name"].map(norm)
    # index stats by country for fast same-country lookup
    by_nat = {nat: grp for nat, grp in agg.groupby("nation")}

    matched_rows = []
    match_kind = []
    for _, p in fifa.iterrows():
        nat = p["abbr"]
        cand = by_nat.get(nat)
        row, kind = None, "none"
        if cand is not None:
            exact = cand[cand["norm"] == p["norm"]]
            if len(exact):
                row, kind = exact.iloc[0], "exact"
            else:
                # fuzzy within same nationality
                best, best_r = None, 0.0
                for _, c in cand.iterrows():
                    r = SequenceMatcher(None, p["norm"], c["norm"]).ratio()
                    if r > best_r:
                        best, best_r = c, r
                if best is not None and best_r >= FUZZY_THRESHOLD:
                    row, kind = best, f"fuzzy"
        matched_rows.append(row)
        match_kind.append(kind)

    stat_cols = ["minutes", "starts", "goals", "assists", "pk", "yellow",
                 "red", "shots", "sot", "nineties", "team", "fbref_pos", "league"]
    for c in stat_cols:
        fifa[c] = [r[c] if r is not None else pd.NA for r in matched_rows]
    fifa["match"] = match_kind
    fifa = fifa.drop(columns=["norm"])
    fifa.to_csv(OUT, index=False)

    # --- coverage report ---
    n = len(fifa)
    mk = pd.Series(match_kind)
    print(f"\nMatched {(mk!='none').sum()}/{n} ({(mk!='none').mean()*100:.1f}%)  "
          f"[exact {(mk=='exact').sum()}, fuzzy {(mk=='fuzzy').sum()}]")
    print(f"-> {OUT}\n")

    unm = fifa[fifa["match"] == "none"]
    print("Unmatched players by country (top 20):")
    print(unm["country"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
