"""Step 6: one-command pipeline. Refresh data -> rebuild everything -> report.md

Run this after each matchday to regenerate the optimised squads and score
predictions against the latest FIFA feed (prices, availability, eliminations).

Stages:
  1. FIFA feeds + live stats            - re-fetched every run (cheap): prices,
                                          availability, played scores, and per-
                                          player goals/assists/fantasy form
  2. Elo -> team strength               - cached unless --refresh-elo
  3. advancement sim + odds blend       - re-run every time
  4. club stats (FBref scrape)          - SKIPPED by default: the 2025-26 club
                                          season is static. Use --refresh-club to
                                          re-scrape (slow, selenium).
  5. player matching + xP projection    - re-run every time
  6. group-stage score predictions      - re-run every time
  7. report.md                          - rendered last

Usage:
    python src/run_all.py                  # standard matchday refresh
    python src/run_all.py --sims 50000     # more advancement sims
    python src/run_all.py --refresh-club   # also re-scrape club stats
    python src/run_all.py --refresh-elo    # also re-download Elo ratings
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                     # modules use paths relative to the repo root
sys.path.insert(0, str(ROOT / "src"))

import advancement
import fifa_data
import form_update
import live_stats
import match_players
import odds_blend
import projection
import predict_scores
import report
import team_strength


def step(msg):
    print(f"\n{'='*60}\n▶ {msg}\n{'='*60}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000, help="advancement sims")
    ap.add_argument("--pred-sims", type=int, default=50000, help="score-prediction sims")
    ap.add_argument("--refresh-club", action="store_true", help="re-scrape FBref club stats")
    ap.add_argument("--refresh-elo", action="store_true", help="re-download Elo ratings")
    ap.add_argument("--no-refresh", action="store_true",
                    help="reuse cached FIFA feed (fully reproducible run)")
    args = ap.parse_args()

    step("1/7  FIFA feeds (prices, availability, eliminations, live scores)")
    fifa_data.main(force=not args.no_refresh)
    # actual goals/assists + fantasy form for matches played so far
    live_stats.main()

    step("2/7  Elo -> team strength (+ live form update from results)")
    team_strength.main(force=args.refresh_elo)
    form_update.main()   # nudge Elo by actual results -> team_strength_live.csv

    step(f"3/7  Advancement simulation ({args.sims:,} sims) + odds blend")
    advancement.main(n_sims=args.sims)
    odds_blend.main()

    step("4/7  Club stats")
    club_csv = ROOT / "data" / "processed" / "club_player_stats.csv"
    if args.refresh_club or not club_csv.exists():
        # heavy selenium scrape -> isolate in a subprocess
        subprocess.run([sys.executable, "src/club_stats.py"], check=True)
    else:
        print("  using cached club_player_stats.csv (pass --refresh-club to re-scrape)")

    step("5/7  Player matching + xP projection")
    match_players.main()
    projection.main()

    step(f"6/7  Group-stage score predictions ({args.pred_sims:,} sims)")
    predict_scores.main(n_sims=args.pred_sims)

    step("7/7  Rendering report.md")
    report.build_report()

    print("\n✅ Done. Open report.md")


if __name__ == "__main__":
    main()
