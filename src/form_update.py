"""Update team Elo from actual World Cup results (the 'form' signal).

Base Elo (team_strength.csv) is a pre-tournament rating. Once matches are played
we nudge each team's rating with a World-Football-Elo update so that over- and
under-performance flows into the *future* projections: a side that beats a
higher-rated opponent gains Elo, lifting its later-matchday expected goals.

  E_home = 1 / (1 + 10^(((elo_away+hb) - (elo_home+hb))/400))   # host bump hb
  S_home = 1 win / 0.5 draw / 0 loss
  G      = goal-difference multiplier (1, 1.5, then (11+GD)/8)
  elo_home += K*G*(S_home - E_home);  elo_away -= same           # K=60 (WC)

`updated_elo(..., max_round=r)` applies only results from rounds <= r, which is
how predict_scores gets the "as-of matchday N" rating (results from rounds < N).

-> data/processed/team_strength_live.csv  (base table with a live `elo` column)

Run it directly:   python src/form_update.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from advancement import HOSTS, HOST_ELO_BUMP

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "team_strength_live.csv"

K_WC = 60.0   # World Cup K-factor (high: each game is informative)


def _gd_multiplier(gd: int) -> float:
    """Goal-difference weight, World Football Elo style."""
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def updated_elo(base: dict[str, float], results: pd.DataFrame,
                max_round: int | None = None, k: float = K_WC) -> dict[str, float]:
    """Base Elo nudged by every completed group result (optionally up to a round).

    `results` is the fixtures table (needs round_id, date, home/away_country,
    home/away_score). Matches are applied in chronological order; unplayed rows
    and rows past `max_round` are skipped.
    """
    elo = dict(base)
    df = results.sort_values(["round_id", "date"])
    for _, m in df.iterrows():
        if max_round is not None and m["round_id"] > max_round:
            continue
        if pd.isna(m["home_score"]) or pd.isna(m["away_score"]):
            continue
        h, a = m["home_country"], m["away_country"]
        if h not in elo or a not in elo:
            continue
        gh, ga = int(m["home_score"]), int(m["away_score"])
        hb = HOST_ELO_BUMP if h in HOSTS else 0.0
        ab = HOST_ELO_BUMP if a in HOSTS else 0.0
        e_home = 1.0 / (1.0 + 10 ** (((elo[a] + ab) - (elo[h] + hb)) / 400.0))
        s_home = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        delta = k * _gd_multiplier(abs(gh - ga)) * (s_home - e_home)
        elo[h] += delta
        elo[a] -= delta
    return elo


def main() -> None:
    teams = pd.read_csv(PROCESSED / "team_strength.csv")
    fixtures = pd.read_csv(PROCESSED / "fixtures.csv", parse_dates=["date"])
    base = dict(zip(teams["country"], teams["elo"]))

    live = updated_elo(base, fixtures[fixtures["round_id"] <= 3])
    out = teams.copy()
    out["elo_base"] = out["elo"]
    out["elo"] = out["country"].map(live)
    out["elo_delta"] = out["elo"] - out["elo_base"]
    out.to_csv(OUT, index=False)

    moved = out[out["elo_delta"].round(1) != 0].sort_values("elo_delta", ascending=False)
    print(f"team_strength_live.csv : {len(out)} teams, {len(moved)} moved by results -> {OUT}")
    if len(moved):
        cols = ["country", "elo_base", "elo", "elo_delta"]
        fmt = lambda x: f"{x:.1f}"
        print(moved[cols].to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
