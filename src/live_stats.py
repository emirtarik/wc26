"""Extract actual WC26 tournament returns from the cached FIFA feeds.

Once matches start, the pipeline conditions its projections on what has actually
happened. This module pulls the *player-level* live data:

  - goals & assists per player   (from rounds.json goal events)
  - fantasy points & form        (from players.json stats)

-> data/processed/player_live_stats.csv
   player_id, wc_games, wc_goals, wc_assists, wc_points, form

Completed match *scores* are already written to fixtures.csv by fifa_data; the
modules that re-predict standings/scorelines (advancement, predict_scores) read
those directly. This module only covers the per-player numbers that feed the xP
projection.

Run it directly:   python src/live_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "player_live_stats.csv"


def goal_events(rounds: list[dict]) -> dict[int, list[int]]:
    """player_id -> [goals, assists] aggregated over completed group matches."""
    tally: dict[int, list[int]] = {}
    for r in rounds:
        if r["id"] > 3:                       # group stage only (knockouts undrawn)
            continue
        for m in r.get("tournaments", []):
            if m.get("status") != "complete":
                continue
            for side in ("homeGoalScorersAssists", "awayGoalScorersAssists"):
                for ev in (m.get(side) or []):
                    pid, aid = ev.get("playerId"), ev.get("assistId")
                    if pid:
                        tally.setdefault(pid, [0, 0])[0] += 1
                    if aid:
                        tally.setdefault(aid, [0, 0])[1] += 1
    return tally


def main() -> None:
    rounds = json.loads((RAW / "rounds.json").read_text(encoding="utf-8"))
    players = json.loads((RAW / "players.json").read_text(encoding="utf-8"))
    goals = goal_events(rounds)

    rows = []
    for p in players:
        st = p.get("stats", {}) or {}
        rp = st.get("roundPoints") or {}
        games = len(rp) if isinstance(rp, dict) else 0   # a played round logs points
        g, a = goals.get(p["id"], [0, 0])
        pts = st.get("totalPoints", 0) or 0
        if games == 0 and pts == 0 and g == 0 and a == 0:
            continue                                     # hasn't featured yet
        rows.append({
            "player_id": p["id"],
            "wc_games": games,
            "wc_goals": g,
            "wc_assists": a,
            "wc_points": pts,
            "form": st.get("form", 0) or 0,
        })

    out = pd.DataFrame(rows, columns=[
        "player_id", "wc_games", "wc_goals", "wc_assists", "wc_points", "form"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"player_live_stats.csv : {len(out)} players with WC returns -> {OUT}")
    if len(out):
        top = out.sort_values("wc_points", ascending=False).head(8)
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
