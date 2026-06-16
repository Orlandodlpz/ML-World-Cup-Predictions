"""
models/simulator.py
--------------------
Monte Carlo tournament simulator for the 2026 FIFA World Cup.

Runs the full tournament 10,000 times and outputs:
  - Win probability per team
  - Probability of reaching each stage (R32, R16, QF, SF, Final, Champion)
  - Most likely match predictions for upcoming games

Run:  python models/simulator.py
"""

import os
import sys
import json
import random
import numpy as np
from collections import defaultdict
from typing import Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.flags import flag, team_str

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(__file__))
PROCESSED = os.path.join(BASE, "data", "processed")
RAW       = os.path.join(BASE, "data", "raw")
OUTPUTS   = os.path.join(BASE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: get win probability for a match
# ══════════════════════════════════════════════════════════════════════════════

def match_win_prob(
    home: str,
    away: str,
    team_stats: dict,
    predictor=None,
    neutral: bool = True,
) -> dict:
    """
    Returns {"home_win": p, "draw": p, "away_win": p}.
    Uses XGBoost predictor if available, falls back to Elo-based estimate.
    """
    if predictor is not None:
        try:
            return predictor.predict_from_stats(home, away, team_stats, neutral)
        except Exception:
            pass

    # Elo-based fallback
    h_elo = team_stats.get(home, {}).get("elo", 1500)
    a_elo = team_stats.get(away, {}).get("elo", 1500)
    diff  = h_elo - a_elo
    # Elo expected score for home team
    e_h = 1 / (1 + 10 ** (-diff / 400))
    # Map to win/draw/loss: draws are ~25-28% of matches on average
    draw_base = 0.26
    home_win  = e_h * (1 - draw_base)
    away_win  = (1 - e_h) * (1 - draw_base)
    draw      = draw_base
    return {
        "home_win": round(home_win, 4),
        "draw":     round(draw,     4),
        "away_win": round(away_win, 4),
    }


def simulate_match(
    home: str,
    away: str,
    team_stats: dict,
    predictor=None,
    neutral: bool = True,
    allow_draw: bool = True,
) -> str:
    """
    Simulate one match. Returns the winner (or None on draw if allowed).
    Uses weighted random draw based on match probabilities.
    """
    probs = match_win_prob(home, away, team_stats, predictor, neutral)

    if allow_draw:
        outcomes  = [home, "draw", away]
        weights   = [probs["home_win"], probs["draw"], probs["away_win"]]
    else:
        # Knockout: redistribute draw probability proportionally
        p_h = probs["home_win"]
        p_a = probs["away_win"]
        total = p_h + p_a
        outcomes = [home, away]
        weights  = [p_h / total, p_a / total]

    return random.choices(outcomes, weights=weights, k=1)[0]


# ══════════════════════════════════════════════════════════════════════════════
# GROUP STAGE
# ══════════════════════════════════════════════════════════════════════════════

def simulate_group_stage(
    groups: dict,
    group_matches: list,
    team_stats: dict,
    predictor=None,
    known_results: Optional[list] = None,
) -> tuple[dict, list]:
    """
    Simulate all group stage matches for the real 2026 WC format:
      12 groups of 4 → top 2 per group (24) + best 8 third-place = 32 advance.

    Returns:
      qualifiers   : dict of group → [1st, 2nd]
      third_place  : list of dicts {team, group, points, gd, gf} for all 12 third-place teams

    known_results: list of [home, away, home_score, away_score, date]
                   where scores are not None (already played).
    """
    points = defaultdict(int)
    gd     = defaultdict(int)
    gf     = defaultdict(int)

    # Index known results by (home, away) pair
    played = {}
    if known_results:
        for match in known_results:
            h, a, hs, as_, *_ = match
            if hs is not None and as_ is not None:
                played[(h, a)] = (int(hs), int(as_))

    for match in group_matches:
        home, away, *rest = match

        if (home, away) in played:
            home_goals, away_goals = played[(home, away)]
        else:
            winner = simulate_match(home, away, team_stats, predictor, neutral=False)
            if winner == home:
                home_goals, away_goals = np.random.randint(1, 4), np.random.randint(0, 2)
                while home_goals <= away_goals:
                    home_goals, away_goals = np.random.randint(1, 4), np.random.randint(0, 2)
            elif winner == away:
                away_goals, home_goals = np.random.randint(1, 4), np.random.randint(0, 2)
                while away_goals <= home_goals:
                    away_goals, home_goals = np.random.randint(1, 4), np.random.randint(0, 2)
            else:
                g = np.random.randint(0, 3)
                home_goals, away_goals = g, g

        gf[home] += home_goals
        gf[away] += away_goals
        gd[home] += home_goals - away_goals
        gd[away] += away_goals - home_goals

        if home_goals > away_goals:
            points[home] += 3
        elif home_goals == away_goals:
            points[home] += 1
            points[away] += 1
        else:
            points[away] += 3

    qualifiers  = {}
    third_place = []

    for grp, teams in groups.items():
        ranked = sorted(
            teams,
            key=lambda t: (points[t], gd[t], gf[t]),
            reverse=True,
        )
        qualifiers[grp] = ranked[:2]
        third = ranked[2]
        third_place.append({
            "team":   third,
            "group":  grp,
            "points": points[third],
            "gd":     gd[third],
            "gf":     gf[third],
        })

    return qualifiers, third_place


def select_best_thirds(third_place: list, n: int = 8) -> list:
    """
    Select the best N third-place teams from all groups.
    Tiebreaker: points → goal difference → goals for.
    Returns a list of team names (strings).
    """
    ranked = sorted(
        third_place,
        key=lambda x: (x["points"], x["gd"], x["gf"]),
        reverse=True,
    )
    return [t["team"] for t in ranked[:n]]


# ══════════════════════════════════════════════════════════════════════════════
# KNOCKOUT STAGE
# ══════════════════════════════════════════════════════════════════════════════

def simulate_knockout_stage(
    qualifiers: dict,
    best_thirds: list,
    team_stats: dict,
    predictor=None,
) -> dict:
    """
    Simulate the full knockout stage from Round of 32 onward.
    Returns dict of stage → [teams that reached that stage].

    2026 WC format: 12 groups A–L.
      24 top-2 qualifiers + 8 best third-place = 32 teams.

    R32 bracket design:
      • Groups A–H (8 groups, 16 teams): cross-pair within adjacent-group duos
          A1 vs B2, B1 vs A2
          C1 vs D2, D1 vs C2
          E1 vs F2, F1 vs E2
          G1 vs H2, H1 vs G2
          → 8 R32 matches, 16 top-2 teams used
      • Groups I–L (4 groups, 8 teams): each group winner and runner-up face
        one of the 8 best third-place teams
          I1 vs 3rd[0], I2 vs 3rd[4]
          J1 vs 3rd[1], J2 vs 3rd[5]
          K1 vs 3rd[2], K2 vs 3rd[6]
          L1 vs 3rd[3], L2 vs 3rd[7]
          → 8 R32 matches, 8 top-2 teams + all 8 third-place teams used

    Total: 16 R32 matches, 32 unique teams. ✓
    """
    grp_keys = sorted(qualifiers.keys())   # ['A', 'B', ..., 'L']
    thirds   = list(best_thirds)           # exactly 8 team names

    r32_matches = []

    # ── Groups A–H: winner vs runner-up cross-pairs ───────────────────────────
    for i in range(0, 8, 2):
        g1, g2 = grp_keys[i], grp_keys[i + 1]
        r32_matches.append((qualifiers[g1][0], qualifiers[g2][1]))   # e.g. A1 vs B2
        r32_matches.append((qualifiers[g2][0], qualifiers[g1][1]))   # e.g. B1 vs A2

    # ── Groups I–L: each slot faces a third-place qualifier ───────────────────
    for j, i in enumerate(range(8, 12)):
        g = grp_keys[i]
        r32_matches.append((qualifiers[g][0], thirds[j]))            # e.g. I1 vs 3rd
        r32_matches.append((qualifiers[g][1], thirds[j + 4]))        # e.g. I2 vs 3rd

    # ── Simulate R32 → R16 → QF → SF → Final ────────────────────────────────
    stage_results = {}
    current_round = r32_matches
    stage_names   = ["Round of 32", "Round of 16", "Quarter-Finals",
                     "Semi-Finals", "Final"]

    for stage in stage_names:
        winners = []
        for home, away in current_round:
            winner = simulate_match(
                home, away, team_stats, predictor,
                neutral=True, allow_draw=False,
            )
            winners.append(winner)
        stage_results[stage] = winners

        if stage == "Final":
            stage_results["Champion"] = [winners[0]]
            break

        current_round = [
            (winners[i], winners[i + 1])
            for i in range(0, len(winners), 2)
        ]

    return stage_results


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    fixtures: dict,
    team_stats: dict,
    predictor=None,
    n_simulations: int = 10_000,
) -> dict:
    """
    Run the tournament N times. Returns win probabilities per team per stage.
    """
    groups        = fixtures["groups"]
    group_matches = fixtures["group_matches"]
    known         = [m for m in group_matches if m[2] is not None]

    # Accumulators
    stage_counts = defaultdict(lambda: defaultdict(int))

    print(f"\n🎲 Running {n_simulations:,} tournament simulations...\n")

    for _ in tqdm(range(n_simulations), ncols=70):
        # Group stage → top 2 per group + all 12 third-place records
        qualifiers, third_place = simulate_group_stage(
            groups, group_matches, team_stats, predictor, known_results=known
        )

        # Best 8 third-place teams advance to R32
        best_thirds = select_best_thirds(third_place, n=8)

        # Track group-stage advancement (top 2 + best 8 third-place)
        for grp, top2 in qualifiers.items():
            for team in top2:
                stage_counts["Group Stage"][team] += 1
        for team in best_thirds:
            stage_counts["Group Stage"][team] += 1

        # Knockout stage
        ko_results = simulate_knockout_stage(qualifiers, best_thirds, team_stats, predictor)
        for stage, teams in ko_results.items():
            for team in teams:
                stage_counts[stage][team] += 1

    # Convert counts to probabilities
    results = {}
    stages  = ["Group Stage", "Round of 32", "Round of 16",
               "Quarter-Finals", "Semi-Finals", "Final", "Champion"]

    for stage in stages:
        counts = stage_counts[stage]
        results[stage] = {
            team: round(count / n_simulations, 4)
            for team, count in sorted(counts.items(), key=lambda x: -x[1])
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Load fixtures
    fixtures_path = os.path.join(RAW, "wc2026_fixtures.json")
    if not os.path.exists(fixtures_path):
        print("❌ wc2026_fixtures.json not found. Run: python data/download_data.py")
        return

    with open(fixtures_path) as f:
        fixtures = json.load(f)

    # Load team stats (or use defaults)
    stats_path = os.path.join(PROCESSED, "team_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            team_stats = json.load(f)
        print("✓ Loaded real team stats from feature engineering")
    else:
        print("⚠️  No team_stats.json found — using Elo-only fallback")
        team_stats = _default_team_stats(fixtures["groups"])

    # ── Blend player strength into team stats ────────────────────────────────
    # If team_player_strengths.json exists, adjust each team's expected goals
    # based on their squad quality (attack/defense strength scores).
    # This makes predictions player-aware without changing the core Elo logic.
    player_path = os.path.join(PROCESSED, "team_player_strengths.json")
    if os.path.exists(player_path):
        with open(player_path) as f:
            player_strengths = json.load(f)
        AVERAGE_STRENGTH = 55.0   # baseline overall strength (0-100 scale)
        for team, stats in team_stats.items():
            ps = player_strengths.get(team)
            if ps:
                atk = ps.get("attack_strength",  AVERAGE_STRENGTH)
                dfd = ps.get("defense_strength",  AVERAGE_STRENGTH)
                # Scale factor: each 10pts above avg = +5% goals scored / -5% conceded
                atk_mult = 1.0 + (atk - AVERAGE_STRENGTH) / 200.0
                dfd_mult = 1.0 - (dfd - AVERAGE_STRENGTH) / 200.0   # stronger def → fewer conceded
                stats["avg_goals_scored"]   = round(stats.get("avg_goals_scored",   1.2) * atk_mult, 3)
                stats["avg_goals_conceded"] = round(stats.get("avg_goals_conceded", 1.2) * max(0.5, dfd_mult), 3)
        print(f"✓ Player strength data blended into team stats ({len(player_strengths)} teams)")
    else:
        print("ℹ️  No player strength data found — run: python features/player_features.py")

    # Try to load XGBoost predictor
    predictor = None
    try:
        from models.match_predictor import MatchPredictor
        predictor = MatchPredictor()
        print("✓ XGBoost predictor loaded")
    except Exception:
        print("⚠️  XGBoost model not found — using Elo fallback for predictions")

    # Run simulation
    results = run_simulation(fixtures, team_stats, predictor, n_simulations=10_000)

    # Save results
    out_path = os.path.join(OUTPUTS, "simulation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved → {out_path}")

    # Print top 15 World Cup winner probabilities
    print("\n🏆 World Cup Winner Probabilities (Top 15):\n")
    champ = results.get("Champion", {})
    for i, (team, prob) in enumerate(list(champ.items())[:15], 1):
        bar = "█" * int(prob * 200)
        print(f"  {i:2}. {team_str(team, 24)} {prob:.1%}  {bar}")

    print("\n📊 Teams most likely to reach the Final:\n")
    final = results.get("Final", {})
    for team, prob in list(final.items())[:10]:
        print(f"  {team_str(team, 24)} {prob:.1%}")

    print("\n✅ Simulation complete! Next: python analysis/explainability.py")


def _default_team_stats(groups: dict) -> dict:
    """
    Rough Elo and stats for all 48 WC teams when real data isn't available.
    Based on FIFA rankings as of early 2026.
    """
    # Tier 1 = elite, Tier 2 = strong, Tier 3 = competitive, Tier 4 = underdogs
    tiers = {
        "Brazil": (2100, 2.1, 0.9, 0.80),
        "Argentina": (2090, 2.0, 0.85, 0.82),
        "France": (2050, 1.9, 1.0, 0.74),
        "Spain": (2040, 1.85, 0.95, 0.72),
        "England": (2000, 1.75, 1.0, 0.70),
        "Germany": (1980, 1.80, 1.1, 0.68),
        "Portugal": (1970, 1.80, 1.05, 0.70),
        "Netherlands": (1960, 1.75, 1.05, 0.68),
        "Belgium": (1940, 1.70, 1.1, 0.66),
        "Italy": (1930, 1.65, 1.0, 0.65),
        "Croatia": (1920, 1.60, 1.05, 0.64),
        "Uruguay": (1910, 1.65, 1.1, 0.65),
        "Colombia": (1900, 1.65, 1.15, 0.63),
        "Mexico": (1890, 1.55, 1.15, 0.60),
        "USA": (1880, 1.50, 1.2, 0.58),
        "Japan": (1870, 1.50, 1.1, 0.60),
        "Senegal": (1860, 1.55, 1.2, 0.60),
        "Morocco": (1850, 1.45, 1.1, 0.60),
        "South Korea": (1840, 1.45, 1.15, 0.58),
        "Denmark": (1840, 1.45, 1.05, 0.60),
        "Switzerland": (1830, 1.40, 1.1, 0.58),
        "Turkey": (1810, 1.45, 1.2, 0.57),
        "Poland": (1800, 1.40, 1.2, 0.56),
        "Austria": (1800, 1.40, 1.2, 0.56),
        "Serbia": (1790, 1.40, 1.2, 0.55),
        "Ukraine": (1790, 1.38, 1.2, 0.55),
        "Ecuador": (1780, 1.40, 1.25, 0.55),
        "Chile": (1770, 1.35, 1.25, 0.54),
        "Paraguay": (1750, 1.30, 1.3, 0.52),
        "IR Iran": (1750, 1.25, 1.2, 0.52),
        "Canada": (1750, 1.30, 1.2, 0.54),
        "Australia": (1740, 1.25, 1.25, 0.52),
        "Nigeria": (1730, 1.30, 1.3, 0.52),
        "Czechia": (1730, 1.25, 1.25, 0.51),
        "Costa Rica": (1700, 1.20, 1.3, 0.49),
        "Saudi Arabia": (1690, 1.20, 1.35, 0.48),
        "Egypt": (1680, 1.20, 1.3, 0.48),
        "Cameroon": (1670, 1.20, 1.35, 0.47),
        "Peru": (1660, 1.15, 1.3, 0.47),
        "Panama": (1640, 1.10, 1.4, 0.45),
        "Honduras": (1630, 1.10, 1.4, 0.44),
        "Venezuela": (1620, 1.10, 1.4, 0.44),
        "Albania": (1610, 1.05, 1.4, 0.43),
        "New Zealand": (1580, 1.00, 1.5, 0.41),
        "Bolivia": (1560, 0.95, 1.5, 0.40),
        "Jamaica": (1550, 0.95, 1.5, 0.39),
        "Qatar": (1540, 0.90, 1.6, 0.37),
        "Guatemala": (1520, 0.85, 1.6, 0.35),
    }

    stats = {}
    for team, (elo, gs, gc, form) in tiers.items():
        stats[team] = {
            "elo": elo,
            "avg_goals_scored":   gs,
            "avg_goals_conceded": gc,
            "form": form,
        }

    # Any team not in our list gets average values
    for grp_teams in groups.values():
        for team in grp_teams:
            if team not in stats:
                stats[team] = {
                    "elo": 1650,
                    "avg_goals_scored": 1.2,
                    "avg_goals_conceded": 1.3,
                    "form": 0.50,
                }
    return stats


if __name__ == "__main__":
    main()
