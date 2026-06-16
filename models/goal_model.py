"""
models/goal_model.py
---------------------
Poisson-based goal model. Given two teams, simulates realistic scorelines.

Why Poisson?
  Goals in football are rare, random events that fit the Poisson distribution
  almost perfectly. If Brazil averages 1.8 goals/game, Poisson tells us
  the exact probability of scoring 0, 1, 2, 3... goals in any single match.

Outputs:
  - Scoreline probabilities (e.g. P(Brazil 2-1 France) = 8.3%)
  - Expected goals for each team
  - Win / Draw / Loss probabilities (from summing scoreline probs)
"""

import os
import json
import numpy as np
from scipy.stats import poisson
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(__file__))
PROCESSED = os.path.join(BASE, "data", "processed")

MAX_GOALS = 10   # we compute up to 10 goals per team (covers 99.99% of cases)


class PoissonGoalModel:
    """
    Simulates match scorelines using the Poisson distribution.

    Expected goals (λ) calculation (Dixon-Coles inspired):
      λ_home = home_attack × away_defence × home_advantage
      λ_away = away_attack × home_defence

    Where:
      - attack  = team's avg goals scored  / league avg goals
      - defence = team's avg goals conceded / league avg goals
    """

    HOME_ADVANTAGE = 1.1   # home teams score ~10% more than on neutral ground
    AVG_GOALS      = 1.35  # global average goals scored per team per game

    def __init__(self, team_stats: Optional[dict] = None):
        if team_stats is None:
            stats_path = os.path.join(PROCESSED, "team_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path) as f:
                    team_stats = json.load(f)
            else:
                team_stats = {}
        self.stats = team_stats

    def _lambda(
        self,
        attack_avg: float,
        defence_avg: float,
        home: bool = False,
    ) -> float:
        """
        Expected goals for one team in one match.
        """
        lam = (attack_avg / self.AVG_GOALS) * defence_avg
        if home:
            lam *= self.HOME_ADVANTAGE
        return max(lam, 0.1)   # at least 0.1 to avoid division issues

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Returns (expected_home_goals, expected_away_goals)."""
        h = self.stats.get(home_team, {})
        a = self.stats.get(away_team, {})

        h_scored   = h.get("avg_goals_scored",   self.AVG_GOALS)
        h_conceded = h.get("avg_goals_conceded",  self.AVG_GOALS)
        a_scored   = a.get("avg_goals_scored",   self.AVG_GOALS)
        a_conceded = a.get("avg_goals_conceded",  self.AVG_GOALS)

        lam_h = self._lambda(h_scored,   a_conceded, home=not neutral)
        lam_a = self._lambda(a_scored,   h_conceded, home=False)
        return round(lam_h, 3), round(lam_a, 3)

    def scoreline_matrix(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
    ) -> np.ndarray:
        """
        Returns a (MAX_GOALS+1 × MAX_GOALS+1) matrix where
        matrix[i][j] = P(home scores i, away scores j).
        """
        lam_h, lam_a = self.expected_goals(home_team, away_team, neutral)
        home_probs = np.array([poisson.pmf(g, lam_h) for g in range(MAX_GOALS + 1)])
        away_probs = np.array([poisson.pmf(g, lam_a) for g in range(MAX_GOALS + 1)])
        return np.outer(home_probs, away_probs)

    def outcome_probs(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
    ) -> dict:
        """
        Returns win/draw/loss probabilities derived from the scoreline matrix.
        """
        mat = self.scoreline_matrix(home_team, away_team, neutral)
        home_win = float(np.sum(np.tril(mat, -1)))   # home scores more
        draw      = float(np.sum(np.diag(mat)))       # equal scores
        away_win  = float(np.sum(np.triu(mat, 1)))    # away scores more
        lam_h, lam_a = self.expected_goals(home_team, away_team, neutral)
        return {
            "home_win":   round(home_win, 4),
            "draw":       round(draw, 4),
            "away_win":   round(away_win, 4),
            "home_xg":    lam_h,
            "away_xg":    lam_a,
        }

    def simulate_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        allow_draw: bool = True,
    ) -> tuple[int, int]:
        """
        Simulate a single match. Returns (home_goals, away_goals).
        If allow_draw=False (knockout stage), simulate extra time + penalties.
        """
        lam_h, lam_a = self.expected_goals(home_team, away_team, neutral)
        hg = np.random.poisson(lam_h)
        ag = np.random.poisson(lam_a)

        if not allow_draw and hg == ag:
            # Extra time: add ~30% of a half's worth of goals
            et_lam_h = lam_h * 0.30
            et_lam_a = lam_a * 0.30
            hg += np.random.poisson(et_lam_h)
            ag += np.random.poisson(et_lam_a)

            if hg == ag:
                # Penalties: 50/50 (simplified)
                if np.random.random() > 0.5:
                    hg += 1
                else:
                    ag += 1

        return int(hg), int(ag)

    def top_scorelines(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False,
        top_n: int = 10,
    ) -> list[dict]:
        """Returns the top N most likely scorelines with probabilities."""
        mat = self.scoreline_matrix(home_team, away_team, neutral)
        results = []
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                results.append({
                    "score": f"{h}-{a}",
                    "home_goals": h,
                    "away_goals": a,
                    "prob": round(float(mat[h, a]), 4),
                })
        return sorted(results, key=lambda x: -x["prob"])[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("⚽ Poisson Goal Model Demo\n")

    # Try to load real stats, fall back to demo values
    stats_path = os.path.join(PROCESSED, "team_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            team_stats = json.load(f)
        print("   Loaded real team stats ✓")
    else:
        print("   No team_stats.json found — using demo values")
        team_stats = {
            "Brazil":    {"avg_goals_scored": 2.1, "avg_goals_conceded": 0.9, "elo": 2100, "form": 0.78},
            "France":    {"avg_goals_scored": 1.9, "avg_goals_conceded": 1.0, "elo": 2050, "form": 0.74},
            "Argentina": {"avg_goals_scored": 2.0, "avg_goals_conceded": 0.85,"elo": 2090, "form": 0.80},
            "Germany":   {"avg_goals_scored": 1.8, "avg_goals_conceded": 1.1, "elo": 1980, "form": 0.68},
        }

    model = PoissonGoalModel(team_stats)

    matchups = [
        ("Brazil", "France", True),
        ("Argentina", "Germany", True),
    ]

    for home, away, neutral in matchups:
        print(f"\n{'='*50}")
        print(f"  {home} vs {away}  (neutral={'yes' if neutral else 'no'})")
        print(f"{'='*50}")

        probs = model.outcome_probs(home, away, neutral)
        print(f"  Expected goals: {home} {probs['home_xg']}  |  {away} {probs['away_xg']}")
        print(f"  {home} win: {probs['home_win']:.1%}")
        print(f"  Draw:        {probs['draw']:.1%}")
        print(f"  {away} win: {probs['away_win']:.1%}")

        print(f"\n  Top 5 most likely scorelines:")
        for s in model.top_scorelines(home, away, neutral, top_n=5):
            bar = "█" * int(s["prob"] * 200)
            print(f"    {s['score']:<6}  {s['prob']:.1%}  {bar}")

    print("\n✅ Goal model ready. Next: python models/simulator.py")
