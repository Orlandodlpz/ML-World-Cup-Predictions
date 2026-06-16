"""
models/lineup_simulator.py
---------------------------
Lineup-aware scenario simulator. Answers questions like:

  "What happens to France's World Cup chances if Mbappé is injured?"
  "How much does Brazil improve with Vinicius Jr. in the starting 11?"
  "What if Argentina play a 5-3-2 instead of 4-3-3?"
  "Compare England's best XI vs their B-team."

How it works:
  1. Start from base team stats (Elo + form + goal averages)
  2. Load player strengths for each lineup
  3. Adjust expected goals using player strength delta vs baseline
  4. Re-run 10,000 Monte Carlo simulations with adjusted numbers
  5. Compare champion probabilities: baseline vs scenario

Usage:
  python models/lineup_simulator.py --injury "France" "Kylian Mbappé"
  python models/lineup_simulator.py --injury "Brazil" "Vinicius Jr." "Rodrygo"
  python models/lineup_simulator.py --formation "Germany" "3-4-3"
  python models/lineup_simulator.py --compare "England" "Jude Bellingham" "Phil Foden"
  python models/lineup_simulator.py --top5       ← show top 5 teams & their key-player impact
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)

PLAYERS_DIR = os.path.join(BASE, "data", "players")
PROCESSED   = os.path.join(BASE, "data", "processed")
RAW         = os.path.join(BASE, "data", "raw")
OUTPUTS     = os.path.join(BASE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_squads() -> dict:
    path = os.path.join(PLAYERS_DIR, "wc2026_squads.json")
    if not os.path.exists(path):
        # Generate embedded squads on the fly
        print("  Generating embedded squad data...")
        from data.download_players import _write_squads
        _write_squads()
    with open(path) as f:
        return json.load(f)


def load_fixtures() -> dict:
    path = os.path.join(RAW, "wc2026_fixtures.json")
    if not os.path.exists(path):
        from data.download_data import _write_wc2026_fixtures
        _write_wc2026_fixtures()
    with open(path) as f:
        return json.load(f)


def load_team_stats() -> dict:
    """Load Elo + goal averages. Fall back to defaults if not computed yet."""
    path = os.path.join(PROCESSED, "team_stats.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Use simulator defaults
    from models.simulator import _default_team_stats
    fixtures = load_fixtures()
    return _default_team_stats(fixtures["groups"])


# ══════════════════════════════════════════════════════════════════════════════
# LINEUP-ADJUSTED MATCH PROBABILITY
# ══════════════════════════════════════════════════════════════════════════════

def lineup_adjusted_probs(
    home: str,
    away: str,
    team_stats: dict,
    engine,                          # PlayerStrengthEngine
    home_absent: list[str] | None = None,
    away_absent: list[str] | None = None,
    home_formation: str | None = None,
    away_formation: str | None = None,
    neutral: bool = True,
) -> dict:
    """
    Returns win/draw/loss probabilities adjusted for lineup strength.
    """
    from features.player_features import adjust_xg_for_lineup

    # Base expected goals from Elo/form
    h = team_stats.get(home, {})
    a = team_stats.get(away, {})
    base_h_xg = h.get("avg_goals_scored",   1.2) * a.get("avg_goals_conceded", 1.2) / 1.2
    base_a_xg = a.get("avg_goals_scored",   1.2) * h.get("avg_goals_conceded", 1.2) / 1.2

    if not neutral:
        base_h_xg *= 1.10   # home advantage

    # Player strength for both lineups
    h_strength = engine.compute_team_strength(home, home_absent, home_formation)
    a_strength = engine.compute_team_strength(away, away_absent, away_formation)

    # Adjust xG
    adj_h_xg = adjust_xg_for_lineup(base_h_xg, h_strength, a_strength, "home")
    adj_a_xg = adjust_xg_for_lineup(base_a_xg, a_strength, h_strength, "away")

    # Poisson probabilities
    from scipy.stats import poisson
    MAX_G = 8
    h_probs = np.array([poisson.pmf(g, adj_h_xg) for g in range(MAX_G + 1)])
    a_probs = np.array([poisson.pmf(g, adj_a_xg) for g in range(MAX_G + 1)])
    mat     = np.outer(h_probs, a_probs)

    home_win = float(np.sum(np.tril(mat, -1)))
    draw     = float(np.sum(np.diag(mat)))
    away_win = float(np.sum(np.triu(mat, 1)))

    return {
        "home_win": round(home_win, 4),
        "draw":     round(draw,     4),
        "away_win": round(away_win, 4),
        "home_xg":  round(adj_h_xg, 3),
        "away_xg":  round(adj_a_xg, 3),
        "home_strength": h_strength["overall_strength"],
        "away_strength": a_strength["overall_strength"],
    }


def simulate_match_lineup(
    home: str,
    away: str,
    team_stats: dict,
    engine,
    home_absent=None,
    away_absent=None,
    home_formation=None,
    away_formation=None,
    neutral: bool = True,
    allow_draw: bool = True,
) -> str:
    probs = lineup_adjusted_probs(
        home, away, team_stats, engine,
        home_absent, away_absent,
        home_formation, away_formation, neutral,
    )
    if allow_draw:
        outcomes = [home, "draw", away]
        weights  = [probs["home_win"], probs["draw"], probs["away_win"]]
    else:
        p_h = probs["home_win"]
        p_a = probs["away_win"]
        t   = p_h + p_a
        outcomes = [home, away]
        weights  = [p_h / t, p_a / t]
    return random.choices(outcomes, weights=weights, k=1)[0]


# ══════════════════════════════════════════════════════════════════════════════
# FULL TOURNAMENT SIMULATION WITH LINEUPS
# ══════════════════════════════════════════════════════════════════════════════

def run_lineup_simulation(
    fixtures: dict,
    team_stats: dict,
    engine,
    absent_map: dict[str, list[str]] | None = None,   # team → [absent players]
    formation_map: dict[str, str] | None = None,       # team → formation override
    n_simulations: int = 10_000,
    silent: bool = False,
) -> dict[str, float]:
    """
    Runs full tournament simulation with lineup adjustments.
    Returns {team: champion_probability} dict.
    """
    absent_map    = absent_map    or {}
    formation_map = formation_map or {}

    groups        = fixtures["groups"]
    group_matches = fixtures["group_matches"]
    known         = [m for m in group_matches if m[2] is not None]
    played_map    = {(m[0], m[1]): (int(m[2]), int(m[3])) for m in known}

    champion_counts = defaultdict(int)

    iterator = range(n_simulations)
    if not silent:
        iterator = tqdm(iterator, ncols=70)

    for _ in iterator:
        # ── Group Stage ───────────────────────────────────────────────────────
        points = defaultdict(int)
        gd     = defaultdict(int)
        gf     = defaultdict(int)

        for match in group_matches:
            home, away = match[0], match[1]
            if (home, away) in played_map:
                hg, ag = played_map[(home, away)]
            else:
                winner = simulate_match_lineup(
                    home, away, team_stats, engine,
                    absent_map.get(home), absent_map.get(away),
                    formation_map.get(home), formation_map.get(away),
                    neutral=False, allow_draw=True,
                )
                if winner == home:
                    hg, ag = np.random.randint(1, 4), np.random.randint(0, 2)
                    while hg <= ag: hg, ag = np.random.randint(1,4), np.random.randint(0,2)
                elif winner == away:
                    ag, hg = np.random.randint(1, 4), np.random.randint(0, 2)
                    while ag <= hg: ag, hg = np.random.randint(1,4), np.random.randint(0,2)
                else:
                    g = np.random.randint(0, 3); hg = ag = g

            gf[home] += hg; gf[away] += ag
            gd[home] += hg - ag; gd[away] += ag - hg
            if hg > ag:   points[home] += 3
            elif hg == ag: points[home] += 1; points[away] += 1
            else:          points[away] += 3

        # ── Group advancement: top 2 + track 3rd-place ───────────────────────
        qualifiers  = {}
        third_place = []
        for grp, teams in groups.items():
            ranked = sorted(teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True)
            qualifiers[grp] = ranked[:2]
            third = ranked[2]
            third_place.append({
                "team":   third,
                "points": points[third],
                "gd":     gd[third],
                "gf":     gf[third],
            })

        # Best 8 third-place teams also advance
        best_thirds = [
            t["team"] for t in sorted(
                third_place,
                key=lambda x: (x["points"], x["gd"], x["gf"]),
                reverse=True,
            )[:8]
        ]

        # ── Knockout Stage: same bracket as simulator.py ──────────────────────
        #   Groups A–H: cross-paired (winner vs runner-up)
        #   Groups I–L: each slot faces a third-place qualifier
        grp_keys = sorted(qualifiers.keys())   # ['A'..'L']
        thirds   = best_thirds                  # 8 team names

        current = []
        for i in range(0, 8, 2):
            g1, g2 = grp_keys[i], grp_keys[i + 1]
            current.append((qualifiers[g1][0], qualifiers[g2][1]))
            current.append((qualifiers[g2][0], qualifiers[g1][1]))
        for j, i in enumerate(range(8, 12)):
            g = grp_keys[i]
            current.append((qualifiers[g][0], thirds[j]))
            current.append((qualifiers[g][1], thirds[j + 4]))

        # Run knockout rounds until champion
        while len(current) > 0:
            next_round = []
            for home, away in current:
                winner = simulate_match_lineup(
                    home, away, team_stats, engine,
                    absent_map.get(home), absent_map.get(away),
                    formation_map.get(home), formation_map.get(away),
                    neutral=True, allow_draw=False,
                )
                next_round.append(winner)
            if len(next_round) == 1:
                champion_counts[next_round[0]] += 1
                break
            current = [
                (next_round[i], next_round[i + 1])
                for i in range(0, len(next_round), 2)
            ]

    return {
        team: round(count / n_simulations, 4)
        for team, count in sorted(champion_counts.items(), key=lambda x: -x[1])
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(
    scenario_name: str,
    affected_team: str,
    absent_players: list[str],
    formation_override: str | None,
    fixtures: dict,
    team_stats: dict,
    engine,
    n_sims: int = 5_000,
):
    """
    Run baseline vs scenario and print the comparison.
    """
    print(f"\n{'═'*60}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'═'*60}")

    if absent_players:
        print(f"\n  Players absent for {affected_team}:")
        for p in absent_players:
            strength_with    = engine.compute_team_strength(affected_team)
            strength_without = engine.compute_team_strength(affected_team, [p])
            diff = strength_without["overall_strength"] - strength_with["overall_strength"]
            print(f"    ✗ {p:<25}  team strength: {strength_with['overall_strength']:.1f} → {strength_without['overall_strength']:.1f}  ({diff:+.1f})")

    if formation_override:
        print(f"\n  Formation change for {affected_team}: → {formation_override}")

    print(f"\n  Running {n_sims:,} simulations each...\n")

    # Baseline
    print("  [1/2] Baseline (full squad)...")
    baseline = run_lineup_simulation(
        fixtures, team_stats, engine,
        absent_map={}, formation_map={},
        n_simulations=n_sims, silent=False,
    )

    # Scenario
    print(f"\n  [2/2] Scenario ({scenario_name})...")
    absent_map    = {affected_team: absent_players} if absent_players else {}
    formation_map = {affected_team: formation_override} if formation_override else {}
    scenario = run_lineup_simulation(
        fixtures, team_stats, engine,
        absent_map=absent_map,
        formation_map=formation_map,
        n_simulations=n_sims, silent=False,
    )

    # Print comparison
    all_teams = sorted(set(list(baseline.keys()) + list(scenario.keys())),
                       key=lambda t: -baseline.get(t, 0))

    print(f"\n\n  {'Team':<25} {'Baseline':>10} {'Scenario':>10} {'Change':>8}")
    print(f"  {'─'*55}")
    for team in all_teams[:15]:
        b = baseline.get(team, 0)
        s = scenario.get(team, 0)
        d = s - b
        marker = " ◄" if team == affected_team else ""
        arrow  = "▲" if d > 0.005 else ("▼" if d < -0.005 else "─")
        print(f"  {team:<25} {b:>9.1%} {s:>10.1%} {arrow}{abs(d):>6.1%}{marker}")

    # Save
    out = {"baseline": baseline, "scenario": scenario, "scenario_name": scenario_name}
    save_path = os.path.join(OUTPUTS, f"scenario_{scenario_name.replace(' ','_')}.json")
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  💾 Saved → {save_path}")


def show_top5_key_players(team_stats: dict, engine, fixtures: dict, n_sims: int = 3_000):
    """
    For the top 5 favourites, show how much each key player matters.
    """
    # Quick baseline
    print("\n🔬 Key Player Impact Analysis (top 5 teams)\n")
    baseline = run_lineup_simulation(
        fixtures, team_stats, engine,
        n_simulations=n_sims, silent=True,
    )
    top5 = list(baseline.items())[:5]

    for team, base_prob in top5:
        keys = engine.get_key_players(team, top_n=3)
        print(f"\n  {team}  (baseline: {base_prob:.1%} to win WC)")
        for player in keys:
            absent = run_lineup_simulation(
                fixtures, team_stats, engine,
                absent_map={team: [player["name"]]},
                n_simulations=n_sims, silent=True,
            )
            new_prob = absent.get(team, 0)
            impact   = new_prob - base_prob
            print(f"    Without {player['name']:<22}  → {new_prob:.1%}  ({impact:+.1%})")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Lineup Scenario Simulator")
    parser.add_argument("--injury",    nargs="+", metavar="ARG",
        help="TEAM [PLAYER1 PLAYER2 ...]  — simulate without those players")
    parser.add_argument("--formation", nargs=2,   metavar=("TEAM", "FORMATION"),
        help="Simulate with a different formation, e.g. 'Germany' '3-4-3'")
    parser.add_argument("--compare",   nargs="+", metavar="ARG",
        help="TEAM PLAYER1 [PLAYER2 ...]  — compare with vs without players")
    parser.add_argument("--top5",      action="store_true",
        help="Show key player impact for top 5 WC favourites")
    parser.add_argument("--sims",      type=int, default=5_000,
        help="Number of simulations (default 5000)")
    args = parser.parse_args()

    # Load data
    squads     = load_squads()
    fixtures   = load_fixtures()
    team_stats = load_team_stats()

    from features.player_features import PlayerStrengthEngine
    engine = PlayerStrengthEngine(squads)

    if args.top5:
        show_top5_key_players(team_stats, engine, fixtures, n_sims=args.sims)

    elif args.injury:
        team    = args.injury[0]
        players = args.injury[1:] if len(args.injury) > 1 else []
        label   = f"No {', '.join(players) or 'changes'} for {team}"
        run_scenario(label, team, players, None, fixtures, team_stats, engine, args.sims)

    elif args.formation:
        team, formation = args.formation
        label = f"{team} in {formation}"
        run_scenario(label, team, [], formation, fixtures, team_stats, engine, args.sims)

    elif args.compare:
        team    = args.compare[0]
        players = args.compare[1:]
        label   = f"{team} without {', '.join(players)}"
        run_scenario(label, team, players, None, fixtures, team_stats, engine, args.sims)

    else:
        parser.print_help()
        print("\nExamples:")
        print('  python models/lineup_simulator.py --injury "France" "Kylian Mbappé"')
        print('  python models/lineup_simulator.py --injury "Brazil" "Vinicius Jr." "Rodrygo"')
        print('  python models/lineup_simulator.py --formation "Germany" "3-4-3"')
        print('  python models/lineup_simulator.py --top5')
        print('  python models/lineup_simulator.py --top5 --sims 10000')


if __name__ == "__main__":
    main()
