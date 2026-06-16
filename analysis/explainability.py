"""
analysis/explainability.py
---------------------------
Two jobs:

1. SHAP explainability
   → After any prediction, show WHY the model picked that team.
   → Feature importance plots.

2. Post-match learning
   → After a real result comes in, compare to what we predicted.
   → If we were wrong, explain WHY using SHAP and suggest what feature
     we should weight more next time.

Usage:
  # See why the model predicted something:
  python analysis/explainability.py --explain "Brazil" "Germany"

  # Update with a real result and learn from any mistake:
  python analysis/explainability.py --update "Brazil" "Germany" 2 1

  # Full post-match report:
  python analysis/explainability.py --report
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(__file__))
PROCESSED = os.path.join(BASE, "data", "processed")
LIVE      = os.path.join(BASE, "data", "live")
OUTPUTS   = os.path.join(BASE, "outputs")
os.makedirs(LIVE, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

sys.path.insert(0, BASE)

FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "home_form", "away_form", "form_diff",
    "home_avg_scored", "home_avg_conceded",
    "away_avg_scored", "away_avg_conceded",
    "home_xg", "away_xg",
    "h2h_home_win_rate",
    "neutral",
]

FEATURE_LABELS = {
    "home_elo":            "Home team Elo rating",
    "away_elo":            "Away team Elo rating",
    "elo_diff":            "Elo gap (home − away)",
    "home_form":           "Home recent form (0–1)",
    "away_form":           "Away recent form (0–1)",
    "form_diff":           "Form gap (home − away)",
    "home_avg_scored":     "Home avg goals scored",
    "home_avg_conceded":   "Home avg goals conceded",
    "away_avg_scored":     "Away avg goals scored",
    "away_avg_conceded":   "Away avg goals conceded",
    "home_xg":             "Home expected goals",
    "away_xg":             "Away expected goals",
    "h2h_home_win_rate":   "Head-to-head win rate",
    "neutral":             "Neutral venue",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_stats() -> dict:
    stats_path = os.path.join(PROCESSED, "team_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    # Fallback
    return {}


def build_feature_vector(home: str, away: str, stats: dict, neutral: bool = True) -> dict:
    h = stats.get(home, {})
    a = stats.get(away, {})
    h_elo  = h.get("elo",  1500)
    a_elo  = a.get("elo",  1500)
    h_form = h.get("form", 0.5)
    a_form = a.get("form", 0.5)
    h_gs   = h.get("avg_goals_scored",   1.2)
    h_gc   = h.get("avg_goals_conceded", 1.2)
    a_gs   = a.get("avg_goals_scored",   1.2)
    a_gc   = a.get("avg_goals_conceded", 1.2)
    return {
        "home_elo":            h_elo,
        "away_elo":            a_elo,
        "elo_diff":            h_elo - a_elo,
        "home_form":           h_form,
        "away_form":           a_form,
        "form_diff":           h_form - a_form,
        "home_avg_scored":     h_gs,
        "home_avg_conceded":   h_gc,
        "away_avg_scored":     a_gs,
        "away_avg_conceded":   a_gc,
        "home_xg":             h_gs * a_gc / 1.2,
        "away_xg":             a_gs * h_gc / 1.2,
        "h2h_home_win_rate":   0.5,
        "neutral":             int(neutral),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def explain_match(home: str, away: str, neutral: bool = True):
    """
    Show why the model predicts what it does for a given match.
    Produces a SHAP waterfall chart saved to outputs/.
    """
    try:
        import shap
        import joblib
        from models.match_predictor import MatchPredictor, MODEL_PATH
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return

    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Run: python models/match_predictor.py")
        return

    stats  = load_stats()
    fvec   = build_feature_vector(home, away, stats, neutral)
    X      = np.array([[fvec[f] for f in FEATURES]])

    model = joblib.load(MODEL_PATH)

    # Get probabilities
    proba = model.predict_proba(X)[0]
    labels = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    print(f"\n📊 Prediction: {home} vs {away}")
    print(f"   {home} Win : {proba[2]:.1%}")
    print(f"   Draw        : {proba[1]:.1%}")
    print(f"   {away} Win : {proba[0]:.1%}")

    # SHAP explanation for the most likely outcome
    predicted_class = int(np.argmax(proba))
    outcome_name    = labels[predicted_class]
    print(f"\n🔍 SHAP explanation for predicted outcome: {outcome_name}\n")

    explainer    = shap.TreeExplainer(model)
    shap_values  = explainer.shap_values(X)

    # shap_values shape: (n_classes, n_samples, n_features)
    sv = shap_values[predicted_class][0]

    # Print text-based SHAP explanation
    contributions = sorted(
        zip(FEATURES, sv),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    print(f"  {'Feature':<30} {'Impact':>8}  Direction")
    print(f"  {'-'*55}")
    for feat, val in contributions:
        label  = FEATURE_LABELS.get(feat, feat)
        arrow  = "▲ pushes toward this outcome" if val > 0 else "▼ pushes against"
        actual = fvec[feat]
        print(f"  {label:<30} {val:>+8.4f}  ({actual:.2f}) {arrow}")

    # Save SHAP bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    feats  = [FEATURE_LABELS.get(f, f) for f, _ in contributions]
    values = [v for _, v in contributions]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
    ax.barh(feats[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"SHAP — Why model predicts: {outcome_name}\n{home} vs {away}", fontsize=13)
    ax.set_xlabel("SHAP value (impact on prediction)")
    plt.tight_layout()
    chart_path = os.path.join(OUTPUTS, f"shap_{home}_vs_{away}.png".replace(" ", "_"))
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n  📈 SHAP chart saved → {chart_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. POST-MATCH LEARNING
# ══════════════════════════════════════════════════════════════════════════════

LIVE_RESULTS_PATH = os.path.join(LIVE, "real_results.json")


def load_live_results() -> list:
    if os.path.exists(LIVE_RESULTS_PATH):
        with open(LIVE_RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_live_results(results: list):
    with open(LIVE_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def update_with_result(home: str, away: str, home_goals: int, away_goals: int):
    """
    Record a real result, compare to prediction, and explain any mistake.
    """
    stats = load_stats()
    fvec  = build_feature_vector(home, away, stats, neutral=True)

    # What did we predict?
    predicted_proba = None
    try:
        import joblib
        from models.match_predictor import MODEL_PATH
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            X = np.array([[fvec[f] for f in FEATURES]])
            predicted_proba = model.predict_proba(X)[0]
    except Exception:
        pass

    # Actual outcome
    if home_goals > away_goals:
        actual = 2; actual_label = f"{home} Win"
    elif home_goals == away_goals:
        actual = 1; actual_label = "Draw"
    else:
        actual = 0; actual_label = f"{away} Win"

    # Save to live results log
    live = load_live_results()
    entry = {
        "home": home, "away": away,
        "home_goals": home_goals, "away_goals": away_goals,
        "actual_outcome": actual,
        "actual_label": actual_label,
        "predicted_proba": predicted_proba.tolist() if predicted_proba is not None else None,
    }
    live.append(entry)
    save_live_results(live)

    print(f"\n✅ Result recorded: {home} {home_goals}–{away_goals} {away}  ({actual_label})")

    if predicted_proba is not None:
        predicted_class = int(np.argmax(predicted_proba))
        labels = {0: f"{away} Win", 1: "Draw", 2: f"{home} Win"}
        pred_label = labels[predicted_class]
        pred_conf  = predicted_proba[predicted_class]

        if predicted_class == actual:
            print(f"   ✓ Correct! We predicted {pred_label} ({pred_conf:.1%} confidence)")
        else:
            print(f"   ✗ Wrong!   We predicted {pred_label} ({pred_conf:.1%} confidence)")
            print(f"              Actual was {actual_label}")
            print(f"\n   🔍 Analyzing why we got it wrong...\n")
            _analyze_mistake(home, away, fvec, predicted_class, actual, predicted_proba)


def _analyze_mistake(
    home: str,
    away: str,
    fvec: dict,
    predicted: int,
    actual: int,
    proba: np.ndarray,
):
    """
    When the model gets a match wrong, diagnose the likely cause.
    """
    labels = {0: f"{away} Win", 1: "Draw", 2: f"{home} Win"}
    elo_diff  = fvec["elo_diff"]
    form_diff = fvec["form_diff"]
    xg_diff   = fvec["home_xg"] - fvec["away_xg"]

    print(f"   Key features going into the prediction:")
    print(f"     Elo gap (home − away)  : {elo_diff:+.0f}")
    print(f"     Form gap               : {form_diff:+.3f}")
    print(f"     xG gap                 : {xg_diff:+.2f}")

    # Heuristic lessons
    lessons = []
    if predicted == 2 and actual != 2 and elo_diff > 150:
        lessons.append(
            "⚠️  High Elo gap suggested a clear home favorite, but they didn't win.\n"
            "     Lesson: Elo alone doesn't capture match-day factors (injuries,\n"
            "             motivation, key player absences, tournament pressure)."
        )
    if predicted != 1 and actual == 1 and abs(elo_diff) < 100:
        lessons.append(
            "⚠️  Evenly-matched teams often draw, but we predicted a winner.\n"
            "     Lesson: When Elo gap < 100, draw probability should be higher."
        )
    if abs(form_diff) < 0.05 and predicted != actual:
        lessons.append(
            "⚠️  Both teams had similar recent form — these are the hardest matches\n"
            "     to predict. Consider adding squad depth / injuries as features."
        )
    if not lessons:
        lessons.append(
            "ℹ️  This looks like a genuine upset — even perfect models can't predict\n"
            "    these. Football has irreducible randomness (~40% of outcomes are upsets)."
        )

    for lesson in lessons:
        print(f"\n   {lesson}")

    print(f"\n   Model's actual confidence for the real outcome ({labels[actual]}):")
    print(f"     {proba[actual]:.1%} — ", end="")
    if proba[actual] < 0.20:
        print("very low. This was a genuine surprise.")
    elif proba[actual] < 0.35:
        print("low. The model sensed some chance but didn't expect it.")
    else:
        print("moderate. The model was on the fence.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FULL POST-MATCH REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report():
    """Print a summary of all predictions vs. real results so far."""
    live = load_live_results()
    if not live:
        print("No live results recorded yet. Use --update to add real results.")
        return

    correct  = 0
    total    = 0
    upsets   = []

    print(f"\n{'='*60}")
    print(f"  POST-MATCH REPORT — {len(live)} matches recorded")
    print(f"{'='*60}\n")

    for r in live:
        home  = r["home"]
        away  = r["away"]
        hg    = r["home_goals"]
        ag    = r["away_goals"]
        act   = r["actual_outcome"]
        proba = r.get("predicted_proba")

        if proba:
            pred = int(np.argmax(proba))
            conf = proba[pred]
            hit  = pred == act
            correct += int(hit)
            total   += 1
            status   = "✓" if hit else "✗"
            if not hit and proba[act] < 0.25:
                upsets.append(f"  {home} {hg}–{ag} {away}")
            print(f"  {status} {home} {hg}–{ag} {away:<20} pred_conf={conf:.0%}")

    if total > 0:
        acc = correct / total
        print(f"\n  Overall accuracy: {correct}/{total}  ({acc:.1%})")
        if upsets:
            print(f"\n  Biggest upsets (model gave <25% to actual outcome):")
            for u in upsets:
                print(u)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup ML — Explainability & Learning")
    parser.add_argument("--explain", nargs=2, metavar=("HOME", "AWAY"),
                        help="Explain prediction for HOME vs AWAY")
    parser.add_argument("--update", nargs=4, metavar=("HOME", "AWAY", "HOME_GOALS", "AWAY_GOALS"),
                        help="Record a real result and learn from it")
    parser.add_argument("--report", action="store_true",
                        help="Print full post-match report")
    args = parser.parse_args()

    if args.explain:
        explain_match(args.explain[0], args.explain[1])
    elif args.update:
        home, away, hg, ag = args.update
        update_with_result(home, away, int(hg), int(ag))
    elif args.report:
        print_report()
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python analysis/explainability.py --explain "Brazil" "Germany"')
        print('  python analysis/explainability.py --update "Brazil" "Germany" 2 1')
        print('  python analysis/explainability.py --report')
