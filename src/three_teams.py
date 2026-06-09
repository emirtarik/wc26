"""Build the three candidate squads along the XI-ceiling vs bench-cover axis.

Same MILP as optimize.py, run three ways:
  1. Stars & scrubs  - max XI, bench = cheap dummies            (bench_cover 0)
  2. All-round       - 4 real auto-sub options on the bench     (bench_cover 4)
  3. In-between      - 2 real subs + 2 scrubs                   (bench_cover 2)

Writes optimal_squad_{1,2,3}.csv and prints a side-by-side summary.
A small bench_weight breaks ties toward better (not just cheaper) subs.

Usage: python src/three_teams.py
"""

import optimize as opt

CONFIGS = [
    ("1. Stars & scrubs", dict(bench_weight=0.0, bench_cover=0)),
    ("2. All-round",      dict(bench_weight=0.05, bench_cover=4, min_bench_xp=15)),
    ("3. In-between",     dict(bench_weight=0.05, bench_cover=2, bench_cover_max=2, min_bench_xp=15)),
]


def main():
    df = opt.load(include_transferred=False)
    results = []
    for i, (name, kw) in enumerate(CONFIGS, 1):
        squad, status, obj = opt.solve(df, budget=100.0, max_per_country=3, **kw)
        squad.to_csv(f"data/processed/optimal_squad_{i}.csv", index=False)
        results.append((name, squad))
        print("=" * 60)
        print(name)
        opt.report(squad, status, obj)
        print()

    print("=" * 60)
    print("SUMMARY")
    print(f"{'team':<20} {'XI xP':>7} {'benchXP':>8} {'cost':>7} {'capt':>16}")
    for name, sq in results:
        xi, bn = sq[sq.in_xi == 1], sq[sq.in_xi == 0]
        cap = sq[sq.is_captain == 1].iloc[0]
        print(f"{name:<20} {xi.xp.sum():>7.1f} {bn.xp.sum():>8.1f} "
              f"${sq.price.sum():>5.1f}m {cap.display_name[:15]:>16}")


if __name__ == "__main__":
    main()
