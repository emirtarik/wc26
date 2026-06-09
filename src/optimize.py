"""Step 4: pick the optimal $100m fantasy squad with a MILP (PuLP).

We choose three nested 0/1 decisions per player:
    in_squad  - one of the 15 picked
    in_xi     - one of the 11 that start (only these score)
    is_captain- the single captain (scores double)

and maximise total expected points of the starting XI plus the captain's
points again, subject to the official constraints:
    squad = 2 GK / 5 DEF / 5 MID / 3 FWD, <= $100m, <= 3 players per country,
    XI = 11 with a valid formation (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD).

The 4 bench players still cost money and fill squad/country quotas but don't
score, so the optimiser naturally parks cheap "enablers" there.

Input : data/processed/players_with_xp.csv (+ status from players.csv)
Output: prints the squad; writes data/processed/optimal_squad.csv

Usage:
    python src/optimize.py
    python src/optimize.py --include-transferred   # don't filter on status
    python src/optimize.py --budget 100 --max-per-country 3
"""

import argparse

import pandas as pd
import pulp

XP_IN = "data/processed/players_with_xp.csv"
STATUS_IN = "data/processed/players.csv"
OUT = "data/processed/optimal_squad.csv"

SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}      # full 15
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}     # starting 11 formation
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}

# Deterministic tie-breaker. Many squads tie on xP (esp. the cheap bench
# "enabler" slots), and a raw solver returns an arbitrary one — sometimes an
# obscure player who may never feature. We add a tiny secondary objective that
# prefers higher-owned players (the crowd's proxy for "will actually play"), so
# exact ties always resolve the same, sensible way. EPS is small enough to never
# override a real xP gap (max swing across 15 picks ~0.015 xP).
TIEBREAK_EPS = 1e-3


def load(include_transferred):
    df = pd.read_csv(XP_IN)
    status = pd.read_csv(STATUS_IN)[["player_id", "status"]]
    df = df.merge(status, on="player_id", how="left")
    if not include_transferred:
        df = df[df["status"] == "playing"].copy()
    return df.reset_index(drop=True)


def solve(df, budget, max_per_country, bench_weight=0.0,
          bench_cover=0, bench_cover_max=None, min_bench_xp=15.0):
    P = list(df.index)
    pos = df["position"].to_dict()
    price = df["price"].to_dict()
    xp = df["xp"].to_dict()
    country = df["country"].to_dict()
    own = df["percent_selected"].fillna(0).to_dict()   # tie-breaker signal

    prob = pulp.LpProblem("fantasy_squad", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", P, cat="Binary")
    xi = pulp.LpVariable.dicts("xi", P, cat="Binary")
    cap = pulp.LpVariable.dicts("cap", P, cat="Binary")

    # objective: XI points + captain's points again + (down-weighted) bench
    # points. bench_weight=0 -> bench is free to be cheap dummies (stars &
    # scrubs); higher -> solver spends XI budget to buy real auto-sub cover.
    bench = {p: squad[p] - xi[p] for p in P}   # 1 iff picked but not starting
    prob += (pulp.lpSum(xp[p] * xi[p] + xp[p] * cap[p]
                        + bench_weight * xp[p] * bench[p] for p in P)
             # tiny tie-breaker: among equal-xP squads, prefer higher ownership
             + TIEBREAK_EPS * pulp.lpSum((own[p] / 100.0) * squad[p] for p in P))

    # squad size & composition
    prob += pulp.lpSum(squad[p] for p in P) == 15
    for k, n in SQUAD.items():
        prob += pulp.lpSum(squad[p] for p in P if pos[p] == k) == n

    # budget & country cap
    prob += pulp.lpSum(price[p] * squad[p] for p in P) <= budget
    for c in set(country.values()):
        prob += pulp.lpSum(squad[p] for p in P if country[p] == c) <= max_per_country

    # starting XI: subset of squad, size 11, valid formation
    prob += pulp.lpSum(xi[p] for p in P) == 11
    for p in P:
        prob += xi[p] <= squad[p]
    for k in SQUAD:
        prob += pulp.lpSum(xi[p] for p in P if pos[p] == k) >= XI_MIN[k]
        prob += pulp.lpSum(xi[p] for p in P if pos[p] == k) <= XI_MAX[k]

    # exactly one captain, who must start
    prob += pulp.lpSum(cap[p] for p in P) == 1
    for p in P:
        prob += cap[p] <= xi[p]

    # bench cover: at least `bench_cover` benched players must clear an xP floor
    # (= real auto-sub options, not cheap enablers). The rest stay scrubs.
    if bench_cover > 0 or bench_cover_max is not None:
        real = [p for p in P if xp[p] >= min_bench_xp]
        if bench_cover > 0:
            prob += pulp.lpSum(bench[p] for p in real) >= bench_cover
        if bench_cover_max is not None:   # force the rest to be scrubs
            prob += pulp.lpSum(bench[p] for p in real) <= bench_cover_max

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    df = df.copy()
    df["in_squad"] = [int(squad[p].value()) for p in P]
    df["in_xi"] = [int(xi[p].value()) for p in P]
    df["is_captain"] = [int(cap[p].value()) for p in P]
    return df[df["in_squad"] == 1].copy(), status, pulp.value(prob.objective)


def report(squad, status, obj):
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad = squad.sort_values(
        by=["in_xi", "position", "xp"],
        key=lambda s: s.map(order) if s.name == "position" else s,
        ascending=[False, True, False])

    print(f"\nSolver: {status}   |   objective (XI + captain) = {obj:.1f} xP")
    print(f"Cost: ${squad.price.sum():.1f}m / 100   |   "
          f"raw squad xP (all 15): {squad.xp.sum():.1f}\n")

    def line(r):
        c = " (C)" if r.is_captain else ""
        flag = "*" if r.imputed else " "
        return (f"  {r.position:3} {r.display_name[:22]:22} {r.country[:13]:13} "
                f"${r.price:4.1f}  {r.xp:5.1f}xP{flag}{c}")

    for label, mask in [("STARTING XI", squad.in_xi == 1), ("BENCH", squad.in_xi == 0)]:
        print(label)
        for _, r in squad[mask].iterrows():
            print(line(r))
        print()
    cap = squad[squad.is_captain == 1].iloc[0]
    print(f"Captain: {cap.display_name} ({cap.country}) — {cap.xp:.1f} xP doubled")
    print("(* = projection imputed from price/position, no club-stat match)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--max-per-country", type=int, default=3)
    ap.add_argument("--include-transferred", action="store_true")
    ap.add_argument("--bench-weight", type=float, default=0.0,
                    help="0 = stars & scrubs; higher buys better bench cover")
    ap.add_argument("--bench-cover", type=int, default=0,
                    help="require this many benched players above --min-bench-xp")
    ap.add_argument("--min-bench-xp", type=float, default=15.0,
                    help="xP floor defining a 'real' sub for --bench-cover")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    df = load(args.include_transferred)
    squad, status, obj = solve(df, args.budget, args.max_per_country,
                               args.bench_weight, args.bench_cover, args.min_bench_xp)
    squad.to_csv(args.out, index=False)
    report(squad, status, obj)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
