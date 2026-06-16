"""
features/engineering.py
-----------------------
Turns raw historical match data into model-ready features.

Produces:
  - data/processed/match_features.csv   ← one row per historical match
  - data/processed/team_elos.json       ← latest Elo rating per team
  - data/processed/team_stats.json      ← goal averages, recent form per team

Run: python features/engineering.py
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(BASE, "data", "raw")
PROCESSED = os.path.join(BASE, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ELO RATING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class EloRating:
    """
    Calculates Elo ratings for every national team from historical results.

    How it works:
      - Every team starts at 1500
      - Win vs stronger team  → bigger gain
      - Loss vs weaker team   → bigger drop
      - K-factor varies by match importance (World Cup > friendlies)
    """

    BASE_ELO = 1500
    K_FACTORS = {
        "FIFA World Cup": 60,
        "Confederations Cup": 50,
        "UEFA Euro": 50,
        "Copa America": 50,
        "AFC Asian Cup": 40,
        "Africa Cup of Nations": 40,
        "UEFA Nations League": 40,
        "Friendly": 20,
    }
    DEFAULT_K = 35

    def __init__(self):
        self.ratings: dict[str, float] = {}

    def _k(self, tournament: str) -> float:
        for name, k in self.K_FACTORS.items():
            if name.lower() in str(tournament).lower():
                return k
        return self.DEFAULT_K

    def _expected(self, r_a: float, r_b: float) -> float:
        """Expected score for team A given ratings r_a and r_b."""
        return 1 / (1 + 10 ** ((r_b - r_a) / 400))

    def _get(self, team: str) -> float:
        return self.ratings.get(team, self.BASE_ELO)

    def update(self, home: str, away: str, home_score: int, away_score: int,
               tournament: str) -> tuple[float, float]:
        """Update Elo after a match. Returns (new_home_elo, new_away_elo)."""
        r_h = self._get(home)
        r_a = self._get(away)
        e_h = self._expected(r_h, r_a)
        e_a = 1 - e_h

        if home_score > away_score:
            s_h, s_a = 1.0, 0.0
        elif home_score == away_score:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        # Goal difference multiplier (big wins count more)
        diff = abs(home_score - away_score)
        gd_mult = 1 + 0.1 * min(diff, 5)

        k = self._k(tournament) * gd_mult
        self.ratings[home] = r_h + k * (s_h - e_h)
        self.ratings[away] = r_a + k * (s_a - e_a)
        return self.ratings[home], self.ratings[away]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class FeatureBuilder:
    """
    Iterates through all historical matches chronologically and builds
    a rich feature row for each one.
    """

    FORM_WINDOW = 10   # last N matches for form calculation
    H2H_WINDOW  = 10   # last N head-to-head matches

    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df.sort_values("date").reset_index(drop=True)
        self.elo = EloRating()
        # team → list of recent match results (1=win, 0.5=draw, 0=loss)
        self._form:  dict[str, list] = {}
        # (team_a, team_b) → list of results from team_a's perspective
        self._h2h:   dict[tuple, list] = {}
        # team → list of goals scored
        self._goals_scored: dict[str, list] = {}
        self._goals_conceded: dict[str, list] = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _record_form(self, team: str, result: float):
        self._form.setdefault(team, []).append(result)

    def _record_goals(self, team: str, scored: int, conceded: int):
        self._goals_scored.setdefault(team, []).append(scored)
        self._goals_conceded.setdefault(team, []).append(conceded)

    def _record_h2h(self, team_a: str, team_b: str, result: float):
        key = tuple(sorted([team_a, team_b]))
        entry = result if team_a == key[0] else 1 - result
        self._h2h.setdefault(key, []).append(entry)

    def _avg_form(self, team: str) -> float:
        hist = self._form.get(team, [])
        if not hist:
            return 0.5
        recent = hist[-self.FORM_WINDOW:]
        return np.mean(recent)

    def _avg_goals(self, team: str) -> tuple[float, float]:
        scored   = self._goals_scored.get(team, [])
        conceded = self._goals_conceded.get(team, [])
        s = np.mean(scored[-self.FORM_WINDOW:])   if scored   else 1.2
        c = np.mean(conceded[-self.FORM_WINDOW:]) if conceded else 1.2
        return float(s), float(c)

    def _h2h_win_rate(self, home: str, away: str) -> float:
        key = tuple(sorted([home, away]))
        hist = self._h2h.get(key, [])
        if not hist:
            return 0.5
        recent = hist[-self.H2H_WINDOW:]
        # perspective of key[0]
        rate = np.mean(recent)
        return rate if home == key[0] else 1 - rate

    # ── main loop ──────────────────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        rows = []

        for _, row in self.df.iterrows():
            home       = row["home_team"]
            away       = row["away_team"]
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])
            tournament = row.get("tournament", "")
            neutral    = bool(row.get("neutral", False))

            # ── snapshot features BEFORE updating state ────────────────────
            h_elo = self.elo._get(home)
            a_elo = self.elo._get(away)
            h_form = self._avg_form(home)
            a_form = self._avg_form(away)
            h_gs, h_gc = self._avg_goals(home)
            a_gs, a_gc = self._avg_goals(away)
            h2h = self._h2h_win_rate(home, away)

            # ── outcome label ──────────────────────────────────────────────
            if home_score > away_score:
                outcome = 2   # home win
            elif home_score == away_score:
                outcome = 1   # draw
            else:
                outcome = 0   # away win

            feature_row = {
                "date":           row["date"],
                "home_team":      home,
                "away_team":      away,
                "tournament":     tournament,
                "neutral":        int(neutral),
                # Elo features
                "home_elo":       round(h_elo, 1),
                "away_elo":       round(a_elo, 1),
                "elo_diff":       round(h_elo - a_elo, 1),
                # Form features
                "home_form":      round(h_form, 3),
                "away_form":      round(a_form, 3),
                "form_diff":      round(h_form - a_form, 3),
                # Goal rate features
                "home_avg_scored":   round(h_gs, 2),
                "home_avg_conceded": round(h_gc, 2),
                "away_avg_scored":   round(a_gs, 2),
                "away_avg_conceded": round(a_gc, 2),
                # Expected goals (Dixon-Coles style attack/defence)
                "home_xg": round(h_gs * a_gc / 1.2, 2),
                "away_xg": round(a_gs * h_gc / 1.2, 2),
                # Head-to-head
                "h2h_home_win_rate": round(h2h, 3),
                # Actual result (target variable)
                "home_score":     home_score,
                "away_score":     away_score,
                "outcome":        outcome,       # 2=home win, 1=draw, 0=away win
            }
            rows.append(feature_row)

            # ── update state AFTER recording features ──────────────────────
            self.elo.update(home, away, home_score, away_score, tournament)

            h_res = 1.0 if home_score > away_score else (0.5 if home_score == away_score else 0.0)
            a_res = 1 - h_res
            self._record_form(home, h_res)
            self._record_form(away, a_res)
            self._record_goals(home, home_score, away_score)
            self._record_goals(away, away_score, home_score)
            self._record_h2h(home, away, h_res)

        return pd.DataFrame(rows)

    def get_team_elos(self) -> dict:
        return dict(sorted(self.elo.ratings.items(), key=lambda x: -x[1]))

    def get_team_stats(self) -> dict:
        """Returns latest goal averages and form per team."""
        stats = {}
        for team in set(list(self._form.keys())):
            gs, gc = self._avg_goals(team)
            stats[team] = {
                "elo":     round(self.elo._get(team), 1),
                "form":    round(self._avg_form(team), 3),
                "avg_goals_scored":   round(gs, 2),
                "avg_goals_conceded": round(gc, 2),
            }
        return stats


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    results_path = os.path.join(RAW, "results.csv")
    if not os.path.exists(results_path):
        print("❌ data/raw/results.csv not found.")
        print("   Run: python data/download_data.py first.")
        return

    print("📊 Loading historical match data...")
    df = pd.read_csv(results_path, parse_dates=["date"])

    # Filter to matches from 1990 onward (older data less relevant)
    df = df[df["date"] >= "1990-01-01"].copy()
    # Drop rows with missing scores
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    print(f"   {len(df):,} matches loaded (1990–2026)")

    print("⚙️  Building features...")
    builder = FeatureBuilder(df)
    features_df = builder.build()

    # Save
    feat_path = os.path.join(PROCESSED, "match_features.csv")
    features_df.to_csv(feat_path, index=False)
    print(f"   ✓ match_features.csv saved ({len(features_df):,} rows)")

    elos = builder.get_team_elos()
    elo_path = os.path.join(PROCESSED, "team_elos.json")
    with open(elo_path, "w") as f:
        json.dump(elos, f, indent=2)
    print(f"   ✓ team_elos.json saved ({len(elos)} teams)")

    stats = builder.get_team_stats()
    stats_path = os.path.join(PROCESSED, "team_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"   ✓ team_stats.json saved")

    # Print top 10 teams by Elo
    print("\n🏆 Top 10 teams by Elo:\n")
    for i, (team, elo) in enumerate(list(elos.items())[:10], 1):
        bar = "█" * int(elo / 100)
        print(f"  {i:2}. {team:<25} {elo:>7.1f}  {bar}")

    print("\n✅ Done! Next: python models/match_predictor.py")


if __name__ == "__main__":
    main()
