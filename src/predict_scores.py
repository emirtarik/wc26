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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from advancement import HOSTS, expected_goals
from form_update import updated_elo

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "group_score_predictions.csv"
LOCK = PROCESSED / "score_predictions_locked.csv"   # frozen per-matchday snapshots

# Columns that make up a projection (everything except actual-result / status).
PROJ_COLS = ["match_id", "round_label", "matchday", "date", "group", "home", "away",
             "pred_home", "pred_away", "exp_home", "exp_away",
             "mode_home", "mode_away", "mode_prob",
             "p_home_win", "p_draw", "p_away_win", "top3"]


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


def project_fixtures(group_fx, grp, elo_by_md, n_sims, rng):
    """Fresh per-match projection. Matchday N uses the as-of-MD-N rating
    (elo_by_md[N] = base Elo updated with results from earlier matchdays only),
    so already-played matches keep the projection we *would have* made, never the
    actual score."""
    rows = []
    for _, m in group_fx.iterrows():
        h, a = m["home_country"], m["away_country"]
        md = int(m["round_id"])
        elo = elo_by_md[md]
        if pd.isna(h) or pd.isna(a) or h not in elo or a not in elo:
            continue
        la, lb = expected_goals(elo[h], elo[a], h in HOSTS, a in HOSTS)
        sim = simulate_match(la, lb, n_sims, rng)
        rows.append({
            "match_id": m["match_id"], "round_label": m["round_label"],
            "matchday": md, "date": m["date"], "group": grp.get(h, "?"),
            "home": h, "away": a,
            "pred_home": int(round(sim["exp_home"])),
            "pred_away": int(round(sim["exp_away"])),
            **sim,
        })
    return pd.DataFrame(rows)[PROJ_COLS]


def main(n_sims: int = 50_000) -> None:
    teams = pd.read_csv(PROCESSED / "team_strength.csv")
    fixtures = pd.read_csv(PROCESSED / "fixtures.csv", parse_dates=["date"])
    base = dict(zip(teams["country"], teams["elo"]))
    grp = dict(zip(teams["country"], teams["group"]))
    group_fx = fixtures[fixtures["round_id"] <= 3]
    results = group_fx

    # Rating to project each matchday with: results from *earlier* matchdays only.
    elo_by_md = {md: updated_elo(base, results, max_round=md - 1) for md in (1, 2, 3)}

    rng = np.random.default_rng(42)
    proj = project_fixtures(group_fx, grp, elo_by_md, n_sims, rng)

    # --- freeze each matchday at its first kickoff -------------------------
    now = datetime.now(timezone.utc)
    md_start = group_fx.groupby("round_id")["date"].min()
    started = {md: now >= md_start[md] for md in (1, 2, 3)}

    lock = pd.read_csv(LOCK) if LOCK.exists() else pd.DataFrame(columns=PROJ_COLS + ["locked_at"])
    locked_ids = set(lock["match_id"]) if len(lock) else set()
    fresh_to_lock = proj[proj["matchday"].map(started) & ~proj["match_id"].isin(locked_ids)].copy()
    if len(fresh_to_lock):
        fresh_to_lock["locked_at"] = now.isoformat()
        lock = pd.concat([lock, fresh_to_lock], ignore_index=True)
        lock.to_csv(LOCK, index=False)
        locked_ids = set(lock["match_id"])
        print(f"  froze {len(fresh_to_lock)} projections for matchdays "
              f"{sorted(set(fresh_to_lock['matchday']))} -> {LOCK.name}")

    # Final projection = frozen snapshot where a matchday has started, fresh else.
    locked_rows = lock[lock["match_id"].isin(proj["match_id"])][PROJ_COLS]
    out = pd.concat([locked_rows,
                     proj[~proj["match_id"].isin(locked_ids)]], ignore_index=True)
    out["locked"] = out["match_id"].isin(locked_ids)

    # --- attach actual results for projection-vs-reality comparison --------
    res = group_fx[["match_id", "home_score", "away_score"]].rename(
        columns={"home_score": "actual_home", "away_score": "actual_away"})
    out = out.merge(res, on="match_id", how="left")
    played = out["actual_home"].notna()
    out["exact_hit"] = played & (out["pred_home"] == out["actual_home"]) \
                              & (out["pred_away"] == out["actual_away"])
    out["result_hit"] = played & (np.sign(out["pred_home"] - out["pred_away"])
                                  == np.sign(out["actual_home"] - out["actual_away"]))
    out = out.sort_values(["matchday", "group", "home"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    # --- printed table, grouped by matchday ---
    print(f"\nGroup-stage score predictions ({n_sims:,} sims/match)\n")
    for md, rnd in [(1, "Group MD1"), (2, "Group MD2"), (3, "Group MD3")]:
        sub = out[out["matchday"] == md]
        if sub.empty:
            continue
        tag = "FROZEN" if started[md] else "live"
        print(f"=== {rnd} [{tag}] ===")
        for _, r in sub.iterrows():
            line = (f"  [{r.group}] {r.home[:14]:14} {r.pred_home}-{r.pred_away} "
                    f"{r.away[:14]:14}  (most likely {r.mode_home}-{r.mode_away} "
                    f"{r.mode_prob:.0%})")
            if pd.notna(r.actual_home):
                mark = "✓ exact" if r.exact_hit else "✓ result" if r.result_hit else "✗ miss"
                line += f"  | ACTUAL {int(r.actual_home)}-{int(r.actual_away)} {mark}"
            print(line)
        print()
    graded = out[out["actual_home"].notna()]
    if len(graded):
        print(f"Accuracy so far: {graded['exact_hit'].sum()}/{len(graded)} exact, "
              f"{graded['result_hit'].sum()}/{len(graded)} correct result")
    print(f"-> {OUT}")


if __name__ == "__main__":
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 50_000)
    main(n_sims=n)
