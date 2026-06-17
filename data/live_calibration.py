"""
data/live_calibration.py
------------------------
Calibrates team stats using real 2026 World Cup results.

Two calibrations applied on top of the historical base:

  1. Elo update — for each completed WC26 match, update both teams' Elo
     ratings using the standard formula with a World Cup K-factor (60)
     and neutral-venue assumption.

  2. Goal calibration — blend real WC26 goals into each team's
     avg_goals_scored / avg_goals_conceded, with opponent-strength
     adjustment so a 7-1 win over Curacao doesn't inflate Germany's
     rating the same as 7-1 against Argentina.
     Weight grows with games played: 1 game → 8%, 2 → 16%, 3 → 24%, max 30%.

Usage (called automatically by models/simulator.py):
    from data.live_calibration import calibrate_team_stats
    team_stats = calibrate_team_stats(team_stats, real_results)

Or run standalone to see calibration details:
    python3 data/live_calibration.py
"""

import os
import sys
import json
import copy

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, "data", "processed")
LIVE_DIR  = os.path.join(BASE, "data", "live")

sys.path.insert(0, BASE)

# ── Constants ────────────────────────────────────────────────────────────────
WC_K_FACTOR      = 60      # FIFA recommends 60 for major tournaments
MAX_GOAL_WEIGHT  = 0.30    # cap WC goal evidence at 30% of final average
GOAL_WEIGHT_PER  = 0.08    # 8% per WC game played (1→8%, 2→16%, 3→24%, 4+→30%)
LEAGUE_AVG_GOALS = 1.35    # global historical average goals per team per game


# ── Helpers ──────────────────────────────────────────────────────────────────

def elo_expected(elo_h: float, elo_a: float) -> float:
    """Expected score for home team (neutral venue — no home advantage)."""
    return 1.0 / (1.0 + 10.0 ** ((elo_a - elo_h) / 400.0))


def elo_update(elo_h: float, elo_a: float, actual_h: float) -> tuple[float, float]:
    """
    Update both teams' Elo after one match.
    actual_h: 1.0 = home win, 0.5 = draw, 0.0 = away win
    Returns (new_elo_h, new_elo_a).
    """
    e_h   = elo_expected(elo_h, elo_a)
    delta = WC_K_FACTOR * (actual_h - e_h)
    return round(elo_h + delta, 2), round(elo_a - delta, 2)


def result_actual(home_goals: int, away_goals: int) -> float:
    """Convert scoreline to Elo actual score (home perspective)."""
    if home_goals > away_goals:
        return 1.0
    elif home_goals == away_goals:
        return 0.5
    return 0.0


# ── Main calibration ─────────────────────────────────────────────────────────

def calibrate_team_stats(
    base_stats: dict,
    real_results: list,
    verbose: bool = False,
) -> dict:
    """
    Apply WC26 results on top of base_stats (loaded from team_stats.json).

    Parameters
    ----------
    base_stats   : team_stats dict (will NOT be mutated — a copy is returned)
    real_results : list of match dicts from real_results.json
    verbose      : print calibration details per team

    Returns
    -------
    Calibrated copy of team_stats with updated 'elo', 'avg_goals_scored',
    'avg_goals_conceded' for every team that played in WC26.
    """
    stats = copy.deepcopy(base_stats)

    # ── 1. Elo updates ────────────────────────────────────────────────────────
    # Process in chronological order (sort by date if available)
    sorted_results = sorted(
        real_results,
        key=lambda r: r.get("date", "9999-99-99"),
    )

    # Deduplicate by match key (same match may appear twice in older files)
    seen = set()
    unique_results = []
    for r in sorted_results:
        h, a = r.get("home", ""), r.get("away", "")
        key  = f"{min(h,a)}||{max(h,a)}"
        if key not in seen and h and a:
            seen.add(key)
            unique_results.append(r)

    for r in unique_results:
        home = r.get("home", "")
        away = r.get("away", "")
        hg   = r.get("home_goals")
        ag   = r.get("away_goals")

        if hg is None or ag is None:
            continue
        if home not in stats or away not in stats:
            continue  # team not in our system — skip

        hg, ag = int(hg), int(ag)
        actual = result_actual(hg, ag)

        old_h = stats[home]["elo"]
        old_a = stats[away]["elo"]
        new_h, new_a = elo_update(old_h, old_a, actual)
        stats[home]["elo"] = new_h
        stats[away]["elo"] = new_a

        if verbose:
            outcome = "W" if actual == 1.0 else ("D" if actual == 0.5 else "L")
            print(
                f"  Elo  {home:22} {outcome}  "
                f"{old_h:.0f}→{new_h:.0f}   "
                f"{away:22} {old_a:.0f}→{new_a:.0f}"
            )

    # ── 2. Goal calibration ───────────────────────────────────────────────────
    # Per team, accumulate opponent-adjusted WC goals scored / conceded.
    wc_scored    = {}   # {team: [adj_goal, adj_goal, ...]}
    wc_conceded  = {}

    for r in unique_results:
        home = r.get("home", "")
        away = r.get("away", "")
        hg   = r.get("home_goals")
        ag   = r.get("away_goals")

        if hg is None or ag is None:
            continue
        if home not in stats or away not in stats:
            continue

        hg, ag = int(hg), int(ag)

        # Opponent-strength adjustment:
        # If Curacao concedes 7 to Germany, that's inflated because Curacao
        # have very weak defence. We scale the raw goals down by the ratio
        # of opponent's average conceded vs the league average.
        #   adj_hg = hg * (LEAGUE_AVG / opp_avg_conceded)
        # A strong defensive opponent (avg < league_avg) gives a HIGHER adj,
        # meaning scoring against good teams is rewarded more.
        opp_a_conc = stats[away].get("avg_goals_conceded", LEAGUE_AVG_GOALS)
        opp_h_conc = stats[home].get("avg_goals_conceded", LEAGUE_AVG_GOALS)

        # Clamp to avoid division by near-zero
        opp_a_conc = max(0.3, opp_a_conc)
        opp_h_conc = max(0.3, opp_h_conc)

        adj_hg = hg * (LEAGUE_AVG_GOALS / opp_a_conc)
        adj_ag = ag * (LEAGUE_AVG_GOALS / opp_h_conc)

        wc_scored.setdefault(home, []).append(adj_hg)
        wc_scored.setdefault(away, []).append(adj_ag)
        wc_conceded.setdefault(home, []).append(ag)   # raw conceded (no adj needed)
        wc_conceded.setdefault(away, []).append(hg)

    # Blend: posterior = (1-α) * historical + α * wc_avg
    for team, adj_goals in wc_scored.items():
        if team not in stats:
            continue
        n     = len(adj_goals)
        alpha = min(MAX_GOAL_WEIGHT, n * GOAL_WEIGHT_PER)

        hist_scored   = stats[team].get("avg_goals_scored",   LEAGUE_AVG_GOALS)
        hist_conceded = stats[team].get("avg_goals_conceded", LEAGUE_AVG_GOALS)

        wc_avg_scored   = sum(adj_goals) / n
        wc_avg_conceded = sum(wc_conceded.get(team, [hist_conceded])) / n

        new_scored   = round((1 - alpha) * hist_scored   + alpha * wc_avg_scored,   3)
        new_conceded = round((1 - alpha) * hist_conceded + alpha * wc_avg_conceded, 3)

        if verbose:
            print(
                f"  Goals {team:22}  scored {hist_scored:.2f}→{new_scored:.2f} "
                f"  conceded {hist_conceded:.2f}→{new_conceded:.2f}  "
                f"(α={alpha:.0%}, n={n})"
            )

        stats[team]["avg_goals_scored"]   = new_scored
        stats[team]["avg_goals_conceded"] = new_conceded

    return stats


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results_path = os.path.join(LIVE_DIR, "real_results.json")
    stats_path   = os.path.join(PROCESSED, "team_stats.json")

    if not os.path.exists(results_path):
        print("❌ No real_results.json found. Run simulator.py first or feed results manually.")
        sys.exit(1)
    if not os.path.exists(stats_path):
        print("❌ No team_stats.json found. Run: python3 features/engineering.py")
        sys.exit(1)

    with open(results_path) as f:
        real_results = json.load(f)
    with open(stats_path) as f:
        base_stats = json.load(f)

    print(f"\n🔬 Live calibration — {len(real_results)} WC26 matches loaded\n")
    print("  Elo updates:")
    calibrated = calibrate_team_stats(base_stats, real_results, verbose=True)

    # Show top movers
    elo_deltas = []
    for team in calibrated:
        if team in base_stats:
            delta = calibrated[team]["elo"] - base_stats[team]["elo"]
            elo_deltas.append((team, base_stats[team]["elo"], calibrated[team]["elo"], delta))

    elo_deltas.sort(key=lambda x: -abs(x[3]))
    print("\n  Biggest Elo movers (WC26 results):\n")
    for team, old, new, delta in elo_deltas[:10]:
        sign = "▲" if delta > 0 else "▼"
        print(f"    {team:25} {old:.0f} → {new:.0f}  {sign}{abs(delta):.0f}")

    print("\n✅ Calibration complete (stats not saved — applied in-memory by simulator).\n")
