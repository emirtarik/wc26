"""Side-project: group-stage scoreline predictions for a score-prediction game.

Reuses the Step-2 match engine (advancement.expected_goals: Elo gap -> each
side's expected goals, with a host bump) and Monte-Carlo simulates every group
fixture N times as two independent Poissons. For each match we report:

  - the average (mean) score        -> e.g. 2.1 - 0.8
  - the rounded prediction          -> 2 - 1   (round the means)
  - the single most likely exact scoreline + its probability
  - win / draw / loss probabilities

The mean of a Poisson is just its lambda, so the *average* score is fixed by the
model; the value of simulating is the scoreline distribution (mode, outcome
odds) that a prediction game actually scores you on.

Output: data/processed/group_score_predictions.csv  (+ printed per-round table)

Usage:
    python src/predict_scores.py            # default 50,000 sims/match
    python src/predict_scores.py 200000     # more sims, tighter estimates
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from advancement import HOSTS, expected_goals

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "group_score_predictions.csv"


def simulate_match(la: float, lb: float, n: int, rng) -> dict:
    """Monte-Carlo one fixture; return mean score, modal score, W/D/L probs."""
    gh = rng.poisson(la, n)
    ga = rng.poisson(lb, n)

    # most likely exact scoreline: mode of the joint (home, away) distribution
    pair = gh * 100 + ga                       # encode (h,a) -> single int
    vals, counts = np.unique(pair, return_counts=True)
    top = vals[np.argsort(-counts)][:3]        # 3 most common scorelines
    top_scores = [(int(v // 100), int(v % 100)) for v in top]
    top_probs = [counts[vals == v][0] / n for v in top]

    return {
        "exp_home": gh.mean(), "exp_away": ga.mean(),
        "mode_home": top_scores[0][0], "mode_away": top_scores[0][1],
        "mode_prob": top_probs[0],
        "p_home_win": float((gh > ga).mean()),
        "p_draw": float((gh == ga).mean()),
        "p_away_win": float((gh < ga).mean()),
        "top3": "  ".join(f"{h}-{a} ({p:.0%})"
                          for (h, a), p in zip(top_scores, top_probs)),
    }


def main(n_sims: int = 50_000) -> None:
    teams = pd.read_csv(PROCESSED / "team_strength.csv")
    fixtures = pd.read_csv(PROCESSED / "fixtures.csv")
    elo = dict(zip(teams["country"], teams["elo"]))
    grp = dict(zip(teams["country"], teams["group"]))

    rng = np.random.default_rng(42)
    group_fx = fixtures[fixtures["round_id"] <= 3]

    rows = []
    for _, m in group_fx.iterrows():
        h, a = m["home_country"], m["away_country"]
        if pd.isna(h) or pd.isna(a) or h not in elo or a not in elo:
            continue
        la, lb = expected_goals(elo[h], elo[a], h in HOSTS, a in HOSTS)
        sim = simulate_match(la, lb, n_sims, rng)
        rows.append({
            "match_id": m["match_id"], "round_label": m["round_label"],
            "date": m["date"], "group": grp.get(h, "?"),
            "home": h, "away": a,
            "pred_home": int(round(sim["exp_home"])),
            "pred_away": int(round(sim["exp_away"])),
            **sim,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    # --- printed table, grouped by matchday ---
    print(f"Group-stage score predictions ({n_sims:,} sims/match)\n")
    for rnd in ["Group MD1", "Group MD2", "Group MD3"]:
        sub = out[out["round_label"] == rnd].sort_values(["group", "home"])
        if sub.empty:
            continue
        print(f"=== {rnd} ===")
        for _, r in sub.iterrows():
            line = (f"  [{r.group}] {r.home[:14]:14} {r.exp_home:.1f}-{r.exp_away:.1f} "
                    f"{r.away[:14]:14}  | most likely {r.mode_home}-{r.mode_away} "
                    f"({r.mode_prob:.0%})  | W/D/L {r.p_home_win:.0%}/{r.p_draw:.0%}/{r.p_away_win:.0%}")
            print(line)
        print()
    print(f"-> {OUT}")


if __name__ == "__main__":
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 50_000)
    main(n_sims=n)
