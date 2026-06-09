"""
odds_blend.py
=============
Step 2 (part 3) of the WC26 fantasy optimizer: nudge the pure-simulation
advancement probabilities (from advancement.py) toward the bookmaker market.

    data/processed/team_advancement.csv   (sim)   +   data/odds/wc26_odds.csv
        -> data/processed/team_advancement_blended.csv

WHY BLEND
  The Elo simulation is internally consistent but blind to news the market prices
  in (injuries, form, motivation, tactical mismatches). Bookmaker odds aggregate
  all of that but carry a margin ("vig") and only cover teams people bet on. We
  de-vig the odds, then take a weighted GEOMETRIC opinion pool of the two views
  -- the standard way to merge probabilistic forecasts -- and finally re-impose
  the structural truths the market ignores: exactly ONE team wins the cup, and
  exactly 32 of 48 advance from the groups.

HOW (each step is a constant you can tweak)
  - Outright (win-cup) market: implied prob = 1/odds, de-vigged by dividing out
    the overround (sum of implied probs). Teams with no listed odds keep their
    (tiny) simulated win prob. Blend, then renormalise so the total is 1.
  - Qualify market (optional): implied prob = 1/odds per team. Blend with the
    simulated p_advance, then rescale every team so the total advancers = 32.
  - Blend: pooled = sim^(1-W) * market^W, with W = BLEND_WEIGHT (0 = trust the
    sim entirely, 1 = trust the market entirely).

Run it directly:   python src/odds_blend.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ODDS_CSV = ROOT / "data" / "odds" / "wc26_odds.csv"

# --- constants (tweak me) --------------------------------------------------
BLEND_WEIGHT = 0.5         # weight on the MARKET (0 = pure sim, 1 = pure market)
N_QUALIFIERS = 32          # exactly 32 of 48 advance (WC26)
EPS = 1e-9


# --- helpers ---------------------------------------------------------------
def load_odds() -> pd.DataFrame:
    """Read the editable odds CSV, ignoring comment (#) lines and blanks."""
    df = pd.read_csv(ODDS_CSV, comment="#")
    for col in ("outright_odds", "qualify_odds"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    return df


def geometric_pool(sim: np.ndarray, market: np.ndarray, w: float) -> np.ndarray:
    """Weighted geometric mean of two probability views (un-normalised)."""
    return sim ** (1 - w) * market ** w


def blend_win_cup(df: pd.DataFrame, w: float) -> np.ndarray:
    """Blend simulated win_cup with de-vigged outright odds; renormalise to 1."""
    sim = df["sim_win_cup"].to_numpy(dtype=float)
    odds = df["outright_odds"].to_numpy(dtype=float)
    have = np.isfinite(odds) & (odds > 1.0)

    market = np.full(len(df), np.nan)
    implied = 1.0 / odds[have]
    market[have] = implied / implied.sum()          # de-vig (remove overround)

    blended = sim.copy()
    blended[have] = geometric_pool(np.clip(sim[have], EPS, 1),
                                   np.clip(market[have], EPS, 1), w)
    return blended / blended.sum()                   # exactly one winner


def blend_advance(df: pd.DataFrame, w: float) -> np.ndarray:
    """Blend simulated p_advance with qualify odds; rescale total to 32."""
    sim = df["sim_p_advance"].to_numpy(dtype=float)
    odds = df["qualify_odds"].to_numpy(dtype=float)
    have = np.isfinite(odds) & (odds > 1.0)

    blended = sim.copy()
    if have.any():
        market = np.clip(1.0 / odds[have], EPS, 1.0)  # implied P(advance)
        blended[have] = geometric_pool(np.clip(sim[have], EPS, 1), market, w)

    # Rescale so exactly N_QUALIFIERS advance, keeping probabilities in [0, 1].
    blended = np.clip(blended, 0.0, 1.0)
    for _ in range(50):                               # iterative proportional fit
        total = blended.sum()
        if abs(total - N_QUALIFIERS) < 1e-6:
            break
        room = blended if total > N_QUALIFIERS else (1.0 - blended)
        scale = (N_QUALIFIERS - (0 if total > N_QUALIFIERS else total)) / \
                (total if total > N_QUALIFIERS else room.sum())
        blended = blended * scale if total > N_QUALIFIERS else blended + room * scale
        blended = np.clip(blended, 0.0, 1.0)
    return blended


def rebuild_ladder(df: pd.DataFrame, win_cup_target: np.ndarray) -> np.ndarray:
    """Re-derive the knockout reach-rounds so they end at the blended win_cup.

    p_advance is held at the simulated value (the outright market says nothing
    about group survival, and we have no qualify odds). The five per-round
    knockout win probabilities (R32, R16, QF, SF, Final) are each multiplied by
    a COMMON factor so that their product moves from the simulated win_cup to
    the blended target -- a uniform 'strength bump'. This preserves the
    simulation's round-by-round shape, keeps the ladder monotone, and lets the
    market's deep-run view reach exp_games (which is built from these rungs).

    Returns an (n, 6) array of ladder values:
        [p_advance, reach_r16, reach_qf, reach_sf, reach_final, win_cup].
    """
    pa = df["sim_p_advance"].to_numpy(dtype=float)            # held fixed
    mids = df[["reach_r16", "reach_qf", "reach_sf", "reach_final"]].to_numpy(float)
    wc_sim = df["sim_win_cup"].to_numpy(dtype=float)

    # Full ladder incl. endpoints; q_i = ladder[i+1] / ladder[i] is the
    # conditional prob of winning the i-th knockout game (5 of them).
    ladder = np.column_stack([pa, mids, wc_sim])
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(ladder[:, :-1] > EPS, ladder[:, 1:] / ladder[:, :-1], 0.0)

    # Common per-round factor f so prod(f * q_i) hits the target product.
    with np.errstate(divide="ignore", invalid="ignore"):
        p_sim = np.where(pa > EPS, wc_sim / pa, 0.0)
        p_tgt = np.where(pa > EPS, win_cup_target / pa, 0.0)
        ratio = np.where(p_sim > EPS, p_tgt / p_sim, 1.0)
    f = ratio ** (1.0 / 5)
    cond_new = np.clip(cond * f[:, None], 0.0, 1.0)

    # Walk the ladder forward from the fixed p_advance.
    out = np.empty_like(ladder)
    out[:, 0] = pa
    for i in range(5):
        out[:, i + 1] = out[:, i] * cond_new[:, i]
    return out


# --- main ------------------------------------------------------------------
def main(w: float = BLEND_WEIGHT) -> None:
    adv = pd.read_csv(PROCESSED / "team_advancement.csv")
    odds = load_odds()

    df = adv.merge(odds, on="country", how="left")
    df = df.rename(columns={"p_advance": "sim_p_advance", "win_cup": "sim_win_cup"})
    df["sim_exp_games"] = df["exp_games"]                  # keep for comparison

    df["win_cup"] = blend_win_cup(df, w)                  # target top of ladder
    df["p_advance"] = blend_advance(df, w)                # = sim (no qualify odds)

    # Push the deep-run signal down the knockout ladder, then rebuild exp_games.
    ladder = rebuild_ladder(df, df["win_cup"].to_numpy(dtype=float))
    df["reach_r16"], df["reach_qf"] = ladder[:, 1], ladder[:, 2]
    df["reach_sf"], df["reach_final"] = ladder[:, 3], ladder[:, 4]
    df["win_cup"] = ladder[:, 5]                          # realized (clip-safe)
    df["exp_games"] = (3.0 + df["p_advance"]
                       + df[["reach_r16", "reach_qf", "reach_sf",
                             "reach_final"]].sum(axis=1))

    cols = ["country", "group", "elo", "sim_p_advance", "p_advance",
            "sim_win_cup", "win_cup", "outright_odds", "qualify_odds",
            "p_first", "p_second", "p_third",
            "reach_r16", "reach_qf", "reach_sf", "reach_final",
            "sim_exp_games", "exp_games"]
    out = df[cols].sort_values("p_advance", ascending=False).reset_index(drop=True)
    out.to_csv(PROCESSED / "team_advancement_blended.csv", index=False)

    n_odds = int(df["outright_odds"].notna().sum())
    print(f"Blended {n_odds} outright-odds teams (BLEND_WEIGHT={w}) "
          f"-> team_advancement_blended.csv")
    print(f"  invariants: sum win_cup={out.win_cup.sum():.4f} (=1), "
          f"sum p_advance={out.p_advance.sum():.2f} (=32)")
    fmt = lambda x: f"{x:.3f}"
    print("\nBiggest exp_games move from the market blend:")
    moved = out.reindex((out.exp_games - out.sim_exp_games).abs()
                        .sort_values(ascending=False).index)
    print(moved.head(6)[["country", "sim_win_cup", "win_cup",
                         "sim_exp_games", "exp_games", "outright_odds"]]
          .to_string(index=False, float_format=fmt))
    print("\nFinal advancement (top 8):")
    print(out.head(8)[["country", "group", "sim_p_advance", "p_advance",
                       "exp_games"]].to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    import sys
    weight = next((float(a) for a in sys.argv[1:]
                   if a.replace(".", "", 1).isdigit()), BLEND_WEIGHT)
    main(w=weight)
