"""
features/player_features.py
----------------------------
Computes team-level strength scores from individual player ratings.

Each team gets four scores (0-100 scale):
  - attack_strength    ← weighted avg of ATT players' FIFA ratings + market value
  - midfield_strength  ← weighted avg of MID players
  - defense_strength   ← weighted avg of DEF + GK players
  - squad_depth        ← how much quality drops from starters to bench

These scores AUGMENT the Elo/form features, making the model lineup-aware.

Run: python features/player_features.py
"""

import os
import json
import numpy as np

BASE       = os.path.dirname(os.path.dirname(__file__))
PLAYERS    = os.path.join(BASE, "data", "players")
PROCESSED  = os.path.join(BASE, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)


# ── Starter selection per position ────────────────────────────────────────────
STARTERS = {"GK": 1, "DEF": 4, "MID": 4, "ATT": 3}   # classic 4-4-3 / 4-3-3


class PlayerStrengthEngine:
    """
    Converts a squad (list of players) into numerical team strength scores.

    Scoring philosophy:
      - FIFA rating carries 60% weight (quality)
      - Market value carries 40% weight (recent form / peak)
      - We normalize both to 0-100 before combining
      - Starters (top N per position) weight more than bench
    """

    # Normalisation anchors (approx real-world range)
    FIFA_MIN,  FIFA_MAX  = 65,  95
    VALUE_MIN, VALUE_MAX = 0,   200   # €M

    def __init__(self, squads: dict):
        self.squads = squads

    def _norm_fifa(self, v: float) -> float:
        return (v - self.FIFA_MIN) / (self.FIFA_MAX - self.FIFA_MIN)

    def _norm_value(self, v: float) -> float:
        return min(1.0, v / self.VALUE_MAX)

    def _player_score(self, p: dict) -> float:
        """Single composite score for one player (0-1 scale)."""
        nf = self._norm_fifa(p.get("fifa", 70))
        nv = self._norm_value(p.get("value_m", 5))
        return 0.60 * nf + 0.40 * nv

    def compute_team_strength(
        self,
        team: str,
        absent_players: list[str] | None = None,
        formation: str | None = None,
    ) -> dict:
        """
        Returns strength scores for the given team, optionally with players
        marked as absent (injured / suspended).

        absent_players: list of player names to exclude
        formation: override the team's default formation
        """
        squad_data = self.squads.get(team)
        if squad_data is None:
            return self._default_strength()

        players = squad_data["players"]
        absent  = set(absent_players or [])

        # Filter out absent players
        available = [p for p in players if p["name"] not in absent]

        # Determine starters per position based on formation
        starters_config = _formation_to_slots(formation or squad_data.get("formation", "4-3-3"))

        # Split by position and pick best available starters
        by_pos = {"GK": [], "DEF": [], "MID": [], "ATT": []}
        for p in available:
            pos = p.get("pos", "MID")
            if pos in by_pos:
                by_pos[pos].append(p)

        # Sort each group by composite score descending
        for pos in by_pos:
            by_pos[pos].sort(key=lambda p: self._player_score(p), reverse=True)

        strengths = {}
        depth_scores = {}

        for pos, n_starters in starters_config.items():
            group = by_pos.get(pos, [])
            starters = group[:n_starters]
            bench    = group[n_starters:]

            if not starters:
                strengths[pos]    = 0.5
                depth_scores[pos] = 0.0
                continue

            s_scores = [self._player_score(p) for p in starters]
            b_scores = [self._player_score(p) for p in bench] if bench else s_scores

            # Starters score (weighted: first-choice player gets 1.5× weight)
            weights = np.linspace(1.5, 1.0, len(s_scores))
            strengths[pos] = float(np.average(s_scores, weights=weights))

            # Depth = how much quality drops from starters to bench
            depth_scores[pos] = max(0.0, float(np.mean(s_scores) - np.mean(b_scores)))

        # Aggregate: attack / midfield / defense / overall
        atk = strengths.get("ATT", 0.5)
        mid = strengths.get("MID", 0.5)
        dfd = (strengths.get("DEF", 0.5) * 0.75 + strengths.get("GK", 0.5) * 0.25)

        # Key player bonus: if top player is absent, how much does attack drop?
        top_player = max(available, key=self._player_score, default=None)
        key_player_impact = self._player_score(top_player) if top_player else 0.5

        return {
            "attack_strength":   round(atk * 100, 1),
            "midfield_strength": round(mid * 100, 1),
            "defense_strength":  round(dfd * 100, 1),
            "overall_strength":  round((atk * 0.35 + mid * 0.30 + dfd * 0.35) * 100, 1),
            "squad_depth":       round(np.mean(list(depth_scores.values())) * 100, 1),
            "top_player_score":  round(key_player_impact * 100, 1),
            "n_available":       len(available),
            "n_absent":          len(absent),
        }

    def _default_strength(self) -> dict:
        """Fallback for teams with no squad data."""
        return {
            "attack_strength":   55.0,
            "midfield_strength": 55.0,
            "defense_strength":  55.0,
            "overall_strength":  55.0,
            "squad_depth":       10.0,
            "top_player_score":  60.0,
            "n_available":       26,
            "n_absent":          0,
        }

    def all_team_strengths(self) -> dict[str, dict]:
        return {team: self.compute_team_strength(team) for team in self.squads}

    def get_key_players(self, team: str, top_n: int = 5) -> list[dict]:
        """Returns the N most impactful players for a team."""
        squad_data = self.squads.get(team, {})
        players    = squad_data.get("players", [])
        ranked     = sorted(players, key=self._player_score, reverse=True)
        return [
            {
                "name":  p["name"],
                "pos":   p["pos"],
                "fifa":  p.get("fifa", 70),
                "value_m": p.get("value_m", 5),
                "score": round(self._player_score(p) * 100, 1),
            }
            for p in ranked[:top_n]
        ]


def _formation_to_slots(formation: str) -> dict[str, int]:
    """Parse '4-3-3' into {'GK':1, 'DEF':4, 'MID':3, 'ATT':3}."""
    parts = formation.strip().split("-")
    try:
        if len(parts) == 3:
            return {"GK": 1, "DEF": int(parts[0]), "MID": int(parts[1]), "ATT": int(parts[2])}
        elif len(parts) == 4:
            # e.g. 4-2-3-1: DEF=4, MID1=2, MID2=3 (all mid), ATT=1
            return {"GK": 1, "DEF": int(parts[0]), "MID": int(parts[1]) + int(parts[2]), "ATT": int(parts[3])}
    except (ValueError, IndexError):
        pass
    return STARTERS   # default


# ══════════════════════════════════════════════════════════════════════════════
# HOW PLAYER STRENGTH ADJUSTS MATCH PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def adjust_xg_for_lineup(
    base_xg: float,
    lineup_strength: dict,
    opponent_strength: dict,
    role: str = "home",
) -> float:
    """
    Adjusts the base expected goals (from Elo/form) using lineup strength scores.

    Logic:
      - Your attack strength vs their defense strength → multiplier
      - A 10-point attack advantage → ~8% more xG
      - A 10-point defense disadvantage against you → ~6% more xG for you

    This keeps Elo as the anchor and uses player data as a modifier.
    """
    atk = lineup_strength.get("attack_strength",   55.0)
    mid = lineup_strength.get("midfield_strength",  55.0)
    dfd = opponent_strength.get("defense_strength", 55.0)

    # Normalized differences (each ±10 points = ±1 tier)
    atk_edge = (atk - dfd) / 20.0          # attack vs opponent defence
    mid_edge = (mid - 55.0) / 30.0         # midfield above/below average

    # Multiplier centered at 1.0
    mult = 1.0 + 0.08 * atk_edge + 0.04 * mid_edge
    mult = np.clip(mult, 0.70, 1.40)       # cap at ±40% adjustment

    return round(float(base_xg * mult), 3)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    squads_path = os.path.join(PLAYERS, "wc2026_squads.json")
    if not os.path.exists(squads_path):
        print("❌ wc2026_squads.json not found.")
        print("   Run: python data/download_players.py")
        return

    with open(squads_path) as f:
        squads = json.load(f)

    print(f"⚽ Computing player-based strength scores for {len(squads)} teams...\n")
    engine   = PlayerStrengthEngine(squads)
    strengths = engine.all_team_strengths()

    # Save
    out_path = os.path.join(PROCESSED, "team_player_strengths.json")
    with open(out_path, "w") as f:
        json.dump(strengths, f, indent=2)
    print(f"✓ Saved → {out_path}\n")

    # Print top teams
    ranked = sorted(strengths.items(), key=lambda x: -x[1]["overall_strength"])
    print(f"{'Team':<25} {'Overall':>8} {'Attack':>8} {'Midfield':>9} {'Defense':>8} {'Depth':>7}")
    print("─" * 70)
    for team, s in ranked[:15]:
        print(
            f"{team:<25} {s['overall_strength']:>7.1f}"
            f"  {s['attack_strength']:>7.1f}"
            f"  {s['midfield_strength']:>8.1f}"
            f"  {s['defense_strength']:>7.1f}"
            f"  {s['squad_depth']:>6.1f}"
        )

    # Show key players for top teams
    print("\n🌟 Key players (top 3 per team):\n")
    for team, _ in ranked[:5]:
        keys = engine.get_key_players(team, top_n=3)
        names = ", ".join(f"{p['name']} ({p['pos']}, {p['score']:.0f})" for p in keys)
        print(f"  {team:<20} → {names}")

    print("\n✅ Player features ready! Next: python models/lineup_simulator.py")


if __name__ == "__main__":
    main()
