"""
fifa_data.py
============
Pulls the public FIFA World Cup 2026 Fantasy feeds and turns them into two
tidy tables that the rest of the project joins onto:

    data/processed/players.csv    one row per fantasy-eligible player
    data/processed/fixtures.csv   one row per scheduled match (all 8 rounds)

The feeds are public JSON (no auth) served from play.fifa.com. We cache the raw
responses under data/raw/ so we can re-build the tables offline and so you can
eyeball exactly what FIFA gave us.

Run it directly:   python src/fifa_data.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

BASE_URL = "https://play.fifa.com/json/fantasy"
FEEDS = ["players", "squads", "rounds"]  # the three feeds we actually use
HEADERS = {"User-Agent": "Mozilla/5.0 (wc26-fantasy-optimizer)"}

# Map FIFA's group-stage round ids -> a human label. Rounds 1-3 are the three
# group matchdays; 4-8 are the knockout bracket.
ROUND_LABELS = {
    1: "Group MD1", 2: "Group MD2", 3: "Group MD3",
    4: "Round of 32", 5: "Round of 16", 6: "Quarter-final",
    7: "Semi-final", 8: "Final",
}


# --- fetch -----------------------------------------------------------------
def fetch_feeds(force: bool = False) -> dict[str, object]:
    """Download (and cache) the raw FIFA feeds. Returns parsed JSON per feed."""
    RAW.mkdir(parents=True, exist_ok=True)
    out: dict[str, object] = {}
    for name in FEEDS:
        path = RAW / f"{name}.json"
        if force or not path.exists():
            url = f"{BASE_URL}/{name}.json"
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8")
            path.write_text(text, encoding="utf-8")
            print(f"  fetched {name}.json  ({len(text.encode('utf-8')):,} bytes)")
        else:
            print(f"  cached  {name}.json")
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


# --- build: squads ---------------------------------------------------------
def _squads_frame(squads: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(squads)[["id", "name", "group", "abbr", "isEliminated"]]
    df = df.rename(columns={"id": "squad_id", "name": "country"})
    df["group"] = df["group"].str.upper()
    return df


# --- build: players --------------------------------------------------------
def build_players(feeds: dict) -> pd.DataFrame:
    players = pd.DataFrame(feeds["players"])
    squads = _squads_frame(feeds["squads"])

    # A readable display name: prefer the "known as" name, else first + last.
    # NB: missing knownName comes through pandas as a float NaN, and bool(nan)
    # is True -- so guard on it being a real, non-empty string.
    def display(row):
        known = row.get("knownName")
        if isinstance(known, str) and known.strip():
            return known
        return f"{row.get('firstName', '')} {row.get('lastName', '')}".strip()

    players["display_name"] = players.apply(display, axis=1)

    keep = {
        "id": "player_id",
        "display_name": "display_name",
        "squadId": "squad_id",
        "position": "position",
        "price": "price",
        "status": "status",
        "percentSelected": "percent_selected",
    }
    df = players[list(keep)].rename(columns=keep)

    # Attach country / group / abbreviation from the squads feed.
    df = df.merge(squads, on="squad_id", how="left")

    # Sensible ordering for human reading.
    df = df.sort_values(["position", "price"], ascending=[True, False]).reset_index(drop=True)
    return df


# --- build: fixtures -------------------------------------------------------
def build_fixtures(feeds: dict) -> pd.DataFrame:
    rows = []
    for rnd in feeds["rounds"]:
        rid = rnd["id"]
        for m in rnd.get("tournaments", []):
            rows.append({
                "round_id": rid,
                "round_label": ROUND_LABELS.get(rid, f"Round {rid}"),
                "match_id": m.get("id"),
                "date": m.get("date"),
                "venue_city": m.get("venueCity"),
                "home_squad_id": m.get("homeSquadId"),
                "away_squad_id": m.get("awaySquadId"),
                "home_country": m.get("homeSquadName"),
                "away_country": m.get("awaySquadName"),
                "home_score": m.get("homeScore"),
                "away_score": m.get("awayScore"),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df.sort_values(["round_id", "date"]).reset_index(drop=True)


# --- main ------------------------------------------------------------------
def main(force: bool = False) -> None:
    print("Fetching FIFA Fantasy feeds...")
    feeds = fetch_feeds(force=force)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    players = build_players(feeds)
    fixtures = build_fixtures(feeds)

    players.to_csv(PROCESSED / "players.csv", index=False)
    fixtures.to_csv(PROCESSED / "fixtures.csv", index=False)

    print(f"\nplayers.csv : {len(players):>4} rows  -> {PROCESSED/'players.csv'}")
    print(players["position"].value_counts().to_string())
    print(f"\nfixtures.csv: {len(fixtures):>4} rows  -> {PROCESSED/'fixtures.csv'}")
    print(fixtures["round_label"].value_counts().reindex(ROUND_LABELS.values()).to_string())


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
