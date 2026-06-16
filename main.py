"""
main.py
-------
One-command runner for the full World Cup ML pipeline.

Usage:
  python main.py            ← run everything end-to-end
  python main.py --sim      ← skip training, just re-simulate
  python main.py --predict "Brazil" "Germany"  ← single match prediction
"""

import os
import sys
import argparse

BASE = os.path.dirname(__file__)
sys.path.insert(0, BASE)


def banner(msg: str):
    print(f"\n{'═'*60}")
    print(f"  {msg}")
    print(f"{'═'*60}")


def run_pipeline(skip_training: bool = False):
    banner("🌍 2026 FIFA World Cup — ML Prediction Pipeline")

    # Step 1: Download data (if not already done)
    from data.download_data import main as download
    banner("Step 1/5 — Data Download")
    download()

    # Step 2: Feature Engineering
    banner("Step 2/5 — Feature Engineering")
    from features.engineering import main as engineer
    engineer()

    # Step 3: Train model (unless skipped)
    if not skip_training:
        banner("Step 3/5 — Training XGBoost Model")
        from models.match_predictor import train
        train()
    else:
        print("\nSkipping model training (--sim flag)")

    # Step 4: Simulate tournament
    banner("Step 4/5 — Monte Carlo Tournament Simulation")
    from models.simulator import main as simulate
    simulate()

    banner("✅ Pipeline Complete!")
    print("  → outputs/simulation_results.json   (win probabilities)")
    print("  → Run: python analysis/explainability.py --explain 'Brazil' 'France'")
    print("  → Run: python analysis/explainability.py --report")


def predict_match(home: str, away: str):
    banner(f"🔮 Predicting: {home} vs {away}")
    import json

    stats_path = os.path.join(BASE, "data", "processed", "team_stats.json")
    if not os.path.exists(stats_path):
        # Use defaults
        from models.simulator import _default_team_stats
        from data.download_data import _write_wc2026_fixtures
        import tempfile, json as _json
        _write_wc2026_fixtures()
        fixtures_path = os.path.join(BASE, "data", "raw", "wc2026_fixtures.json")
        with open(fixtures_path) as f:
            fixtures = _json.load(f)
        stats = _default_team_stats(fixtures["groups"])
    else:
        with open(stats_path) as f:
            stats = json.load(f)

    # Try XGBoost, fall back to Elo
    try:
        from models.match_predictor import MatchPredictor
        predictor = MatchPredictor()
        result = predictor.predict_from_stats(home, away, stats, neutral=True)
        method = "XGBoost"
    except Exception:
        from models.simulator import match_win_prob
        result = match_win_prob(home, away, stats, neutral=True)
        method = "Elo-based"

    print(f"\n  Method: {method}")
    print(f"  {home} win : {result['home_win']:.1%}")
    print(f"  Draw        : {result['draw']:.1%}")
    print(f"  {away} win : {result['away_win']:.1%}")

    # Also show SHAP if possible
    try:
        from analysis.explainability import explain_match
        explain_match(home, away, neutral=True)
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026 WC ML Predictor")
    parser.add_argument("--sim",     action="store_true", help="Skip training, just simulate")
    parser.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"), help="Predict one match")
    args = parser.parse_args()

    if args.predict:
        predict_match(args.predict[0], args.predict[1])
    else:
        run_pipeline(skip_training=args.sim)
