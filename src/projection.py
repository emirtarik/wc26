"""Step 3 (part 3): per-player expected fantasy points (xP).

Method: club-rate, opponent-adjusted. For each player we estimate
  - minutes security  (will they play, and 60+ min?) from club minutes
  - attacking rates    (goals/assists/shots-on-target per 90) from club stats,
                        shrunk toward a position prior so low-minute samples
                        don't blow up
  - clean-sheet points from the Step-2 per-fixture clean-sheet probability
and combine them on the scoring table (user-confirmed "Table A"). Attacking
output in a given World Cup match is anchored to that fixture's team expected
goals (Step 2), so a striker is worth more vs Cabo Verde than vs France.

Total xP = (sum over the team's 3 real group fixtures of per-match xP)
         + (expected knockout games) x (per-match xP vs a generic KO opponent)

Players with no matched club stats fall back to a position+price prior and are
flagged `imputed=True`.

Inputs : players_with_stats.csv, fixture_probs.csv, team_advancement_blended.csv
Output : data/processed/players_with_xp.csv  (+ a printed sanity check)
"""

from pathlib import Path

import numpy as np
import pandas as pd

PLAYERS = "data/processed/players_with_stats.csv"
FIXTURES = "data/processed/fixture_probs.csv"
TEAMS = "data/processed/team_advancement_blended.csv"
LIVE = "data/processed/player_live_stats.csv"   # actual WC returns (may not exist pre-KO)
OUT = "data/processed/players_with_xp.csv"

# --- scoring table (user-confirmed Table A) --------------------------------
GOAL_PTS = {"GK": 9, "DEF": 7, "MID": 6, "FWD": 5}
CS_PTS = {"GK": 5, "DEF": 5, "MID": 1, "FWD": 0}
ASSIST_PTS = 3
SOT_PTS = 0.5          # +1 per 2 shots on target (FWD only)
APP_PTS, MIN60_PTS = 1, 1  # +1 for playing, +1 more for 60+ (=> 2 total)

# --- league strength: Opta Power Ratings, avg team quality 0-100 ------------
# Source: Opta Power Rankings, June 2025 (theanalyst.com). Values marked (est)
# weren't published numerically and are interpolated from the league's ranked
# position; tune freely. Used to put scoring rates on a common difficulty
# scale so a goal in MLS/Saudi isn't worth the same as one in the Premier League.
LEAGUE_RATING = {
    "ENG-Premier League": 92.6, "ESP-La Liga": 87.0, "ITA-Serie A": 87.0,
    "GER-Bundesliga": 86.3, "FRA-Ligue 1": 85.5,
    "ENG-Championship": 82.0,   # (est) ranks ~6th
    "POR-Primeira Liga": 79.8, "BRA-Serie A": 79.4, "NED-Eredivisie": 78.8,
    "ARG-Primera": 78.6, "USA-MLS": 78.5, "MEX-Liga MX": 78.5,
    "BEL-Pro League": 77.5,     # (est)
    "TUR-Super Lig": 78.5,      # (est) ranks ~15th
    "SAU-Pro League": 75.0,     # (est) low average, strong top clubs
}
TOP_RATING = 92.6               # Premier League = coefficient 1.0
LEAGUE_EXPONENT = 2.0           # coef = (rating/TOP)^exp; higher = harsher discount

# --- model knobs (edit these) ----------------------------------------------
TYPICAL_CLUB_GAMES = 32   # denominator turning season minutes -> minutes/game
SHRINK_90S = 5.0          # empirical-Bayes weight (in 90s) toward position prior
BASELINE_TG = 1.35        # league-average team goals/match; the opp-adjust pivot
KO_DAMP = 0.88            # KO opponents are tougher than a team's group average
SOT_OPP_SENSITIVITY = 0.5 # how much SoT (vs goals) reacts to opponent strength

# --- live (actual WC) blending --------------------------------------------
# How much one already-played World Cup match counts, in club-90 units, when we
# fold a player's real WC goals/assists into their projected rates. >1 because
# WC games are recent, top-opposition, and national-team evidence; keep modest
# so a single hot/quiet game doesn't dominate a full club season. Tune freely.
LIVE_WEIGHT = 2.0
LIVE_MIN_SEC = 0.80       # a player who has already featured is confirmed to play


def minutes_security(minutes):
    """0..1: ~1 for an ever-present, scaling down with fewer minutes/game."""
    mpg = np.clip(minutes / TYPICAL_CLUB_GAMES, 0, 90)
    return mpg / 90.0


def per_match_xp(pos, g90, a90, sot90, sec, team_xg, p_cs):
    """Expected fantasy points for one match given fixture context."""
    p_play = min(sec * 1.15, 0.98)     # rotation players still often feature
    p_60 = sec                          # P(play 60+ minutes)
    exp90 = sec                         # expected fraction of a full match

    opp = team_xg / BASELINE_TG         # >1 vs weak opponents, <1 vs strong
    sot_opp = 1 + SOT_OPP_SENSITIVITY * (opp - 1)

    pts = APP_PTS * p_play + MIN60_PTS * p_60
    pts += g90 * exp90 * opp * GOAL_PTS[pos]
    pts += a90 * exp90 * opp * ASSIST_PTS
    pts += p_cs * p_60 * CS_PTS[pos]
    if pos == "FWD":
        pts += sot90 * exp90 * sot_opp * SOT_PTS
    return pts


def team_fixtures(fx):
    """country -> list of (team_xg, p_cs) for its *remaining* group matches.

    Already-played fixtures are dropped: their points are banked, not future xP,
    so the squad you pick now should be scored only on the games still to come.
    """
    if "played" in fx.columns:
        fx = fx[~fx["played"].astype(bool)]
    rows = {}
    for _, m in fx.iterrows():
        rows.setdefault(m.home_country, []).append((m.exp_goals_home, m.p_cs_home))
        rows.setdefault(m.away_country, []).append((m.exp_goals_away, m.p_cs_away))
    return rows


def blend_live(df):
    """Fold actual WC goals/assists into the per-90 rates and confirm minutes for
    players who have already featured. The established (club) rate carries `eff`
    90s of weight; each WC game adds LIVE_WEIGHT 90s at the observed WC rate. WC
    opposition is ~EPL difficulty, so the WC tallies need no league rescale."""
    df = df.copy()
    games = df["wc_games"].fillna(0)
    eff = df["nineties"].fillna(0).clip(lower=SHRINK_90S)
    for col, rate in [("goals", "g90"), ("assists", "a90")]:
        wc = df[f"wc_{col}"].fillna(0)
        df[rate] = (df[rate] * eff + LIVE_WEIGHT * wc) / (eff + LIVE_WEIGHT * games)
    confirmed = games > 0
    df.loc[confirmed, "sec"] = np.maximum(df.loc[confirmed, "sec"], LIVE_MIN_SEC)
    return df


def build_rates(df):
    """Add shrunk per-90 rates, using per-position priors for the shrink target."""
    df = df.copy()
    nineties = df["nineties"].fillna(0)
    for col, rate in [("goals", "g90"), ("assists", "a90"), ("sot", "sot90")]:
        raw = df[col].fillna(0)
        # position prior = pooled rate among players with real minutes
        played = nineties > SHRINK_90S
        prior = (df.loc[played].groupby("position")
                   .apply(lambda g: g[col].sum() / g["nineties"].sum(), include_groups=False))
        prior_vec = df["position"].map(prior).fillna(0)
        df[rate] = (raw + prior_vec * SHRINK_90S) / (nineties + SHRINK_90S)
    return df


def league_coef(league):
    """Map a league to a 0..1 difficulty coefficient (Premier League = 1.0)."""
    rating = LEAGUE_RATING.get(league)
    if rating is None:
        return np.nan
    return (rating / TOP_RATING) ** LEAGUE_EXPONENT


def apply_league_strength(df):
    """Scale matched players' rates to an EPL-equivalent difficulty scale."""
    df = df.copy()
    coef = df["league"].map(league_coef)
    matched = df["match"] != "none"
    df["lg_coef"] = np.where(matched, coef, np.nan)
    for r in ["g90", "a90", "sot90"]:
        df.loc[matched, r] = df.loc[matched, r] * df.loc[matched, "lg_coef"]
    return df


def impute_unmatched(df):
    """Fill rates + minutes for players with no club match, from price+position."""
    df = df.copy()
    df["imputed"] = df["match"] == "none"
    # price percentile within position -> a 0..1 'quality' proxy
    df["price_pct"] = df.groupby("position")["price"].rank(pct=True)
    for pos in df["position"].unique():
        m = (df["position"] == pos)
        matched = m & ~df["imputed"]
        med = {r: df.loc[matched, r].median() for r in ["g90", "a90", "sot90"]}
        need = m & df["imputed"]
        # scale median rate by price standing (stars score more than fringe)
        scale = 0.4 + df.loc[need, "price_pct"]
        for r in ["g90", "a90", "sot90"]:
            df.loc[need, r] = (med[r] if not np.isnan(med[r]) else 0) * scale
        # minutes security for unmatched: lean on price standing
        df.loc[need, "sec"] = np.clip(0.25 + 0.7 * df.loc[need, "price_pct"], 0, 0.95)
    return df


def main():
    df = pd.read_csv(PLAYERS)
    fx = pd.read_csv(FIXTURES)
    teams = pd.read_csv(TEAMS)[["country", "exp_games"]]
    df = df.merge(teams, on="country", how="left")

    # actual WC returns so far (file is absent until the tournament starts)
    live_cols = ["wc_games", "wc_goals", "wc_assists", "wc_points", "form"]
    if Path(LIVE).exists():
        live = pd.read_csv(LIVE)
        df = df.merge(live, on="player_id", how="left")
    for c in live_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    df = build_rates(df)
    df = apply_league_strength(df)   # rates -> EPL-equivalent before imputing
    df["sec"] = minutes_security(df["minutes"].fillna(0))
    df = impute_unmatched(df)   # fills sec/rates for unmatched, sets `imputed`
    df = blend_live(df)         # fold in actual WC goals/assists + confirmed minutes

    tf = team_fixtures(fx)

    xps = []
    for _, p in df.iterrows():
        games = tf.get(p["country"], [])
        if not games:                       # team somehow absent from fixtures
            xps.append(np.nan)
            continue
        # real group matches
        group_xp = sum(per_match_xp(p.position, p.g90, p.a90, p.sot90, p.sec,
                                    txg, pcs) for txg, pcs in games)
        # generic knockout match: team's group means, dampened for tougher draw
        mean_xg = np.mean([g[0] for g in games]) * KO_DAMP
        mean_cs = np.mean([g[1] for g in games]) * KO_DAMP
        ko_games = max(p.exp_games - 3, 0)
        ko_xp = ko_games * per_match_xp(p.position, p.g90, p.a90, p.sot90,
                                        p.sec, mean_xg, mean_cs)
        xps.append(group_xp + ko_xp)

    df["xp"] = xps
    df["value"] = df["xp"] / df["price"]
    out = df[["player_id", "display_name", "position", "country", "price",
              "percent_selected", "exp_games", "sec", "league", "lg_coef",
              "g90", "a90", "sot90", "wc_games", "wc_goals", "wc_assists",
              "xp", "value", "imputed"]].sort_values("xp", ascending=False)
    out.to_csv(OUT, index=False)

    print(f"{len(out)} players projected -> {OUT}")
    print(f"  imputed (no club stats): {out.imputed.sum()}\n")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        print(f"=== top 8 {pos} by xP ===")
        cols = ["display_name", "country", "price", "exp_games", "sec", "xp", "value", "imputed"]
        print(out[out.position == pos].head(8)[cols].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
