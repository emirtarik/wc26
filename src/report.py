"""Step 6 (reporting): render the pipeline outputs into a clean Markdown report.

Reads the processed CSVs (squad projection, score predictions, advancement) and
writes report.md: the three candidate squads + the group-stage score
predictions, in human-readable tables. Run standalone to rebuild just the report
from existing data, or via run_all.py after refreshing everything.

Usage: python src/report.py
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

import optimize as opt
from three_teams import CONFIGS

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT = ROOT / "report.md"

POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def formation(xi):
    c = xi["position"].value_counts()
    return f"{c.get('DEF',0)}-{c.get('MID',0)}-{c.get('FWD',0)}"


def squad_tables(squad):
    """Return (xi_md, bench_md) markdown tables for one squad."""
    squad = squad.sort_values(["position", "xp"],
                              key=lambda s: s.map(POS_ORDER) if s.name == "position" else s,
                              ascending=[True, False])
    out = {}
    for label, mask in [("xi", squad.in_xi == 1), ("bench", squad.in_xi == 0)]:
        rows = ["| Pos | Player | Team | Price | xP |",
                "|:---|:---|:---|---:|---:|"]
        for _, r in squad[mask].iterrows():
            cap = " **(C)**" if r.is_captain else ""
            star = " ⚠️" if r.imputed else ""
            rows.append(f"| {r.position} | {r.display_name}{cap}{star} | {r.country} "
                        f"| ${r.price:.1f} | {r.xp:.1f} |")
        out[label] = "\n".join(rows)
    return out["xi"], out["bench"]


def build_squads():
    df = opt.load(include_transferred=False)
    squads = []
    for name, kw in CONFIGS:
        sq, _, obj = opt.solve(df, budget=100.0, max_per_country=3, **kw)
        squads.append((name, sq, obj))
    return squads


def squads_section(squads):
    md = ["## 🏆 Optimised Squads\n"]

    # comparison table first
    md.append("| Option | XI xP | Bench xP | Cost | Captain |")
    md.append("|:---|---:|---:|---:|:---|")
    for name, sq, obj in squads:
        xi, bn = sq[sq.in_xi == 1], sq[sq.in_xi == 0]
        cap = sq[sq.is_captain == 1].iloc[0]
        md.append(f"| {name} | {xi.xp.sum():.1f} | {bn.xp.sum():.1f} "
                  f"| ${sq.price.sum():.1f}m | {cap.display_name} |")
    md.append("\n> **Recommended: All-round** — the XI is near-identical across "
              "options, and real bench cover costs ~1 xP because cheap clean-sheet "
              "defenders/keepers price the same as dummies. ⚠️ = projection imputed "
              "(no club-stat match).\n")

    # full detail per squad
    for name, sq, obj in squads:
        xi = sq[sq.in_xi == 1]
        cap = sq[sq.is_captain == 1].iloc[0]
        xi_md, bench_md = squad_tables(sq)
        md.append(f"<details>\n<summary><b>{name}</b> — "
                  f"{formation(xi)}, ${sq.price.sum():.1f}m, "
                  f"captain {cap.display_name} (XI {xi.xp.sum():.1f} xP)</summary>\n")
        md.append("**Starting XI**\n")
        md.append(xi_md + "\n")
        md.append("**Bench**\n")
        md.append(bench_md + "\n")
        md.append("</details>\n")
    return "\n".join(md)


def predictions_section():
    preds = pd.read_csv(PROCESSED / "group_score_predictions.csv")
    md = ["## 📊 Group-Stage Score Predictions\n",
          "_Each matchday's prediction is **frozen at its first kickoff** "
          "(🔒) and graded against the actual result. Not-yet-started matchdays "
          "stay live and are refreshed from results so far._\n",
          "_**Mean goals (λ)** are each side's expected goals; **Prediction** "
          "rounds them; **Most likely** is the single most common exact scoreline "
          "in the simulation (the mode). They differ because goals are "
          "Poisson-distributed and right-skewed — the long tail of high-scoring "
          "games pulls the mean above the modal (most probable) score._\n"]
    for mday, rnd in [(1, "Group MD1"), (2, "Group MD2"), (3, "Group MD3")]:
        sub = preds[preds["matchday"] == mday].sort_values(["group", "home"])
        if sub.empty:
            continue
        frozen = bool(sub["locked"].iloc[0])
        graded = sub[sub["actual_home"].notna()]
        head = f"### {rnd}" + (" 🔒" if frozen else " · live")
        if len(graded):
            head += (f"  —  {int(graded['exact_hit'].sum())}/{len(graded)} exact, "
                     f"{int(graded['result_hit'].sum())}/{len(graded)} result")
        md.append(head + "\n")
        md.append("| Grp | Match | Mean goals (λ) | Prediction (round λ) "
                  "| Most likely (mode) | Actual |")
        md.append("|:---:|:---|:---:|:---:|:---:|:---:|")
        for _, r in sub.iterrows():
            if pd.notna(r.actual_home):
                mark = "✅" if r.exact_hit else "🟨" if r.result_hit else "❌"
                actual = f"**{int(r.actual_home)}–{int(r.actual_away)}** {mark}"
            else:
                actual = "—"
            md.append(f"| {r.group} | {r.home} vs {r.away} "
                      f"| {r.exp_home:.2f}–{r.exp_away:.2f} "
                      f"| {r.pred_home}–{r.pred_away} "
                      f"| {r.mode_home}–{r.mode_away} ({r.mode_prob:.0%}) "
                      f"| {actual} |")
        md.append("")
    return "\n".join(md)


def group_tables_section():
    """Predicted final group standings, ranked by expected finishing position."""
    adv = pd.read_csv(PROCESSED / "team_advancement_blended.csv").copy()
    adv["p_fourth"] = (1 - adv[["p_first", "p_second", "p_third"]].sum(axis=1)).clip(lower=0)
    adv["exp_pos"] = (adv["p_first"] + 2 * adv["p_second"]
                      + 3 * adv["p_third"] + 4 * adv["p_fourth"])

    md = ["## 🗂️ Predicted Group Tables\n",
          "_Final group standings by expected finishing position. Top 2 advance, "
          "plus the 8 best 3rd-placed teams._\n"]
    for g in sorted(adv["group"].unique()):
        sub = adv[adv["group"] == g].sort_values("exp_pos")
        md.append(f"### Group {g}\n")
        md.append("| Pos | Team | P(1st) | P(2nd) | P(3rd) | P(advance) |")
        md.append("|:---:|:---|---:|---:|---:|---:|")
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            md.append(f"| {i} | {r.country} | {r.p_first:.0%} | {r.p_second:.0%} "
                      f"| {r.p_third:.0%} | **{r.p_advance:.0%}** |")
        md.append("")
    return "\n".join(md)


def knockout_section(top_n=16):
    """Knockout run probabilities (reach each round), sorted by title odds."""
    adv = pd.read_csv(PROCESSED / "team_advancement_blended.csv")
    top = adv.sort_values("win_cup", ascending=False).head(top_n)
    md = ["## 🏟️ Knockout Predictions\n",
          "_Probability of reaching each round (blended Elo + betting market). "
          "Knockout pairings are modelled as a random re-draw among survivors, so "
          "these are run-depth odds, not a fixed bracket._\n",
          "| Team | Grp | Reach R16 | Reach QF | Reach SF | Final | 🏆 Champion |",
          "|:---|:---:|---:|---:|---:|---:|---:|"]
    for _, r in top.iterrows():
        md.append(f"| {r.country} | {r.group} | {r.reach_r16:.0%} | {r.reach_qf:.0%} "
                  f"| {r.reach_sf:.0%} | {r.reach_final:.0%} | **{r.win_cup:.1%}** |")
    return "\n".join(md) + "\n"


def build_report():
    squads = build_squads()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        f"# WC26 Fantasy & Predictions Report",
        f"_Generated {stamp}_\n",
        group_tables_section(),
        knockout_section(),
        squads_section(squads),
        predictions_section(),
    ]
    REPORT.write_text("\n".join(parts), encoding="utf-8")
    print(f"-> {REPORT}")


if __name__ == "__main__":
    build_report()
