"""
advancement.py
==============
Step 2 (part 2) of the WC26 fantasy optimizer: turn Elo ratings into

  (a) per-fixture outcome probabilities  -> data/processed/fixture_probs.csv
  (b) per-team advancement probabilities -> data/processed/team_advancement.csv

WHY THIS MATTERS FOR FANTASY
  (a) is the workhorse for the *group-stage* squad: clean-sheet probability and
      expected goals per fixture drive almost every points projection (a clean
      sheet is +5 for GK/DEF, and only the strongest sides facing weak opponents
      get them cheaply).
  (b) is the longer-horizon weight: the squad you build now rolls into the
      knockouts with only 2 transfers/round, so players on teams likely to be
      eliminated after the group stage are dead weight. P(advance) and expected
      games played capture that.

THE MODEL  (intentionally simple + editable -- every knob is a CONSTANT below)
  Match engine: from the two Elo ratings we derive each side's expected goals,
  then treat goals as independent Poisson. That one engine gives us win/draw/
  loss splits, clean-sheet probabilities (the opponent scoring 0), AND realistic
  scorelines to rank groups by points -> goal difference -> goals for.

  - Expected goals: a neutral baseline BASE_GOALS per side, scaled up/down by the
    Elo gap. Two evenly-matched teams average ~2*BASE total; a gap shifts goals
    to the favourite while keeping the total roughly stable.
  - Host bump: USA / Mexico / Canada get HOST_ELO_BUMP in their group games.

  The group stage is simulated on the REAL fixtures. The knockout is modelled as
  a random re-draw among survivors each round (we do not hardcode the official
  R32 bracket / best-third lookup -- that is a "later" refinement); Elo still
  decides each tie, so deep-run odds are driven by strength, which is what the
  expected-games weight needs.

Run it directly:   python src/advancement.py            (default 20,000 sims)
                   python src/advancement.py 5000        (fewer sims, faster)
"""

from __future__ import annotations

from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# --- model constants (tweak me) --------------------------------------------
BASE_GOALS = 1.35          # neutral expected goals per side (WC avg total ~2.7)
ELO_GOAL_SCALE = 700.0     # larger -> Elo gap moves goals less; 700 is moderate
MAX_LAMBDA = 5.0           # cap a side's expected goals (avoid blowout nonsense)
HOST_ELO_BUMP = 80.0       # home-ground bump for the three host nations
HOSTS = {"USA", "Mexico", "Canada"}

N_BEST_THIRDS = 8          # top 2 per group + 8 best thirds advance (WC26 format)
KO_ROUNDS = ["reach_r16", "reach_qf", "reach_sf", "reach_final", "win_cup"]


# --- match engine ----------------------------------------------------------
def expected_goals(elo_a: float, elo_b: float,
                   host_a: bool = False, host_b: bool = False) -> tuple[float, float]:
    """Expected goals for A and B from their Elo ratings (A's perspective)."""
    dr = (elo_a + (HOST_ELO_BUMP if host_a else 0.0)) \
        - (elo_b + (HOST_ELO_BUMP if host_b else 0.0))
    la = BASE_GOALS * 10 ** (dr / (2 * ELO_GOAL_SCALE))
    lb = BASE_GOALS * 10 ** (-dr / (2 * ELO_GOAL_SCALE))
    return min(la, MAX_LAMBDA), min(lb, MAX_LAMBDA)


def _poisson_pmf(lam: float, kmax: int) -> np.ndarray:
    k = np.arange(kmax + 1)
    return np.exp(-lam) * lam ** k / np.array([factorial(i) for i in k], dtype=float)


def fixture_probs(la: float, lb: float, kmax: int = 12) -> dict[str, float]:
    """Analytic W/D/L, clean-sheet and expected-goal numbers from two Poissons."""
    pa, pb = _poisson_pmf(la, kmax), _poisson_pmf(lb, kmax)
    joint = np.outer(pa, pb)                       # joint[i, j] = P(home=i, away=j)
    joint /= joint.sum()                           # renormalise the truncated tail
    pa, pb = pa / pa.sum(), pb / pb.sum()          # so clean-sheet probs match too
    return {
        "exp_goals_home": la,
        "exp_goals_away": lb,
        "p_home_win": float(np.tril(joint, -1).sum()),   # home > away
        "p_draw": float(np.trace(joint)),                # home == away
        "p_away_win": float(np.triu(joint, 1).sum()),    # away > home
        "p_cs_home": float(pb[0]),                       # away scored 0
        "p_cs_away": float(pa[0]),                       # home scored 0
    }


# --- (a) per-fixture probabilities -----------------------------------------
def build_fixture_probs(fixtures: pd.DataFrame, elo: dict[str, float]) -> pd.DataFrame:
    rows = []
    for _, m in fixtures.iterrows():
        h, a = m["home_country"], m["away_country"]
        if pd.isna(h) or pd.isna(a) or h not in elo or a not in elo:
            continue  # knockout rows have no teams yet
        la, lb = expected_goals(elo[h], elo[a], h in HOSTS, a in HOSTS)
        rows.append({
            "match_id": m["match_id"], "round_id": m["round_id"],
            "round_label": m["round_label"], "date": m["date"],
            "home_country": h, "away_country": a,
            **fixture_probs(la, lb),
        })
    return pd.DataFrame(rows)


# --- (b) group + knockout simulation ---------------------------------------
def _knockout_round(survivors: list[int], elo_arr: np.ndarray, rng) -> list[int]:
    """Random pairing of survivors; Elo win-probability decides each tie."""
    s = list(survivors)
    rng.shuffle(s)
    winners = []
    for i in range(0, len(s) - 1, 2):
        a, b = s[i], s[i + 1]
        p_a = 1.0 / (1.0 + 10 ** ((elo_arr[b] - elo_arr[a]) / 400.0))
        winners.append(a if rng.random() < p_a else b)
    if len(s) % 2:                       # carry an unpaired team (shouldn't happen)
        winners.append(s[-1])
    return winners


def simulate(fixtures: pd.DataFrame, teams: pd.DataFrame,
             n_sims: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    countries = list(teams["country"])
    idx = {c: i for i, c in enumerate(countries)}
    n = len(countries)
    elo = dict(zip(teams["country"], teams["elo"]))
    elo_arr = np.array([elo[c] for c in countries], dtype=float)

    # Group membership (list of team indices per group).
    groups: dict[str, list[int]] = {}
    for c in countries:
        groups.setdefault(teams.set_index("country").loc[c, "group"], []).append(idx[c])

    # Pre-compute each real group match as (home_idx, away_idx, lambda_home, lambda_away).
    gmatches = []
    for _, m in fixtures[fixtures["round_id"] <= 3].iterrows():
        h, a = m["home_country"], m["away_country"]
        la, lb = expected_goals(elo[h], elo[a], h in HOSTS, a in HOSTS)
        gmatches.append((idx[h], idx[a], la, lb))

    first = np.zeros(n); second = np.zeros(n); third = np.zeros(n); adv = np.zeros(n)
    ko = {r: np.zeros(n) for r in KO_ROUNDS}

    for _ in range(n_sims):
        pts = np.zeros(n); gf = np.zeros(n); ga = np.zeros(n)
        for hi, ai, la, lb in gmatches:
            gh, gA = rng.poisson(la), rng.poisson(lb)
            gf[hi] += gh; ga[hi] += gA; gf[ai] += gA; ga[ai] += gh
            if gh > gA:
                pts[hi] += 3
            elif gh < gA:
                pts[ai] += 3
            else:
                pts[hi] += 1; pts[ai] += 1
        gd = gf - ga

        # Rank each group by points -> GD -> GF -> tiny Elo tiebreak (deterministic).
        survivors: list[int] = []
        thirds: list[tuple[float, int]] = []
        for members in groups.values():
            m = np.array(members)
            key = pts[m] * 1e9 + gd[m] * 1e5 + gf[m] * 10 + elo_arr[m] / 1e4
            order = m[np.argsort(-key)]
            first[order[0]] += 1; second[order[1]] += 1; third[order[2]] += 1
            adv[order[0]] += 1; adv[order[1]] += 1
            survivors.extend([order[0], order[1]])
            third_key = (pts[order[2]] * 1e9 + gd[order[2]] * 1e5
                         + gf[order[2]] * 10 + elo_arr[order[2]] / 1e4)
            thirds.append((third_key, int(order[2])))

        best_thirds = [t for _, t in sorted(thirds, reverse=True)[:N_BEST_THIRDS]]
        for t in best_thirds:
            adv[t] += 1
        survivors.extend(best_thirds)

        # Knockout: 32 survivors enter the R32. Play each round, then credit the
        # WINNERS with having reached the next round (so win_cup = 1 team, not 2).
        for r in KO_ROUNDS:
            survivors = _knockout_round(survivors, elo_arr, rng)
            for t in survivors:
                ko[r][t] += 1

    order_idx = [idx[c] for c in teams["country"]]
    out = teams.copy()
    out["p_first"] = first[order_idx] / n_sims
    out["p_second"] = second[order_idx] / n_sims
    out["p_third"] = third[order_idx] / n_sims
    out["p_advance"] = adv[order_idx] / n_sims
    for r in KO_ROUNDS:
        out[r] = ko[r][order_idx] / n_sims
    # Expected games = 3 group games + one knockout game per round *participated*
    # in. A team plays the R32 if it advanced, the R16 if it reached it, etc.
    # (winning the final adds no game, so win_cup is excluded here.)
    out["exp_games"] = 3.0 + out["p_advance"] + sum(
        out[r] for r in ["reach_r16", "reach_qf", "reach_sf", "reach_final"]
    )
    return out.sort_values("p_advance", ascending=False).reset_index(drop=True)


# --- main ------------------------------------------------------------------
def main(n_sims: int = 20_000) -> None:
    teams = pd.read_csv(PROCESSED / "team_strength.csv")
    fixtures = pd.read_csv(PROCESSED / "fixtures.csv")
    elo = dict(zip(teams["country"], teams["elo"]))

    print("Building per-fixture probabilities...")
    fp = build_fixture_probs(fixtures, elo)
    fp.to_csv(PROCESSED / "fixture_probs.csv", index=False)
    print(f"  fixture_probs.csv : {len(fp)} matches")

    print(f"Simulating tournament ({n_sims:,} sims)...")
    adv = simulate(fixtures, teams, n_sims=n_sims)
    adv.to_csv(PROCESSED / "team_advancement.csv", index=False)
    print(f"  team_advancement.csv : {len(adv)} teams")

    fmt = lambda x: f"{x:.3f}"
    print("\nMost likely to advance from group:")
    print(adv.head(8)[["country", "group", "elo", "p_advance", "exp_games"]]
          .to_string(index=False, float_format=fmt))
    print("\nLeast likely to advance:")
    print(adv.tail(5)[["country", "group", "elo", "p_advance", "exp_games"]]
          .to_string(index=False, float_format=fmt))
    print("\nTitle favourites:")
    print(adv.sort_values("win_cup", ascending=False).head(6)
          [["country", "elo", "p_advance", "reach_sf", "win_cup"]]
          .to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    import sys
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 20_000)
    main(n_sims=n)
