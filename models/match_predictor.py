"""
models/match_predictor.py
--------------------------
XGBoost model that predicts the outcome of a football match.

Output per match:
  - P(Home Win)   e.g. 0.52
  - P(Draw)       e.g. 0.24
  - P(Away Win)   e.g. 0.24

Run:  python models/match_predictor.py
      → trains, evaluates, and saves the model to models/xgb_model.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(__file__))
PROCESSED = os.path.join(BASE, "data", "processed")
MODELS    = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

MODEL_PATH = os.path.join(MODELS, "xgb_model.joblib")
META_PATH  = os.path.join(MODELS, "model_meta.json")

# Features used by the model
FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "home_form", "away_form", "form_diff",
    "home_avg_scored", "home_avg_conceded",
    "away_avg_scored", "away_avg_conceded",
    "home_xg", "away_xg",
    "h2h_home_win_rate",
    "neutral",
]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train():
    feat_path = os.path.join(PROCESSED, "match_features.csv")
    if not os.path.exists(feat_path):
        print("❌ match_features.csv not found. Run: python features/engineering.py")
        return None

    print("📂 Loading features...")
    df = pd.read_csv(feat_path)

    # Drop any rows with NaN in features
    df = df.dropna(subset=FEATURES + ["outcome"])

    X = df[FEATURES].values
    y = df["outcome"].values   # 0=away win, 1=draw, 2=home win

    # Train/test split — use last 20% of matches as test (time-aware)
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"   Train: {len(X_train):,} matches | Test: {len(X_test):,} matches")

    # ── XGBoost model ─────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    print("🚀 Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    preds      = model.predict(X_test)
    proba      = model.predict_proba(X_test)
    acc        = accuracy_score(y_test, preds)
    ll         = log_loss(y_test, proba)

    print(f"\n📈 Results on held-out test set:")
    print(f"   Accuracy : {acc:.3f}  ({acc*100:.1f}%)")
    print(f"   Log-Loss : {ll:.4f}  (lower = better calibrated probabilities)")

    # Class distribution in test set
    for label, name in [(2, "Home Win"), (1, "Draw"), (0, "Away Win")]:
        actual = (y_test == label).mean()
        predicted = (preds == label).mean()
        print(f"   {name:<12}  actual={actual:.2%}  predicted={predicted:.2%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    meta = {
        "features": FEATURES,
        "accuracy": round(acc, 4),
        "log_loss": round(ll, 4),
        "n_train":  int(len(X_train)),
        "n_test":   int(len(X_test)),
        "labels":   {0: "away_win", 1: "draw", 2: "home_win"},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Model saved → {MODEL_PATH}")
    print(f"   Metadata  → {META_PATH}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION API
# ══════════════════════════════════════════════════════════════════════════════

class MatchPredictor:
    """
    Simple API for predicting a single match.

    Usage:
        predictor = MatchPredictor()
        result = predictor.predict_match(home_features, away_features)
        # returns {"home_win": 0.52, "draw": 0.24, "away_win": 0.24}
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model not found. Run: python models/match_predictor.py"
            )
        self.model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            self.meta = json.load(f)

    def predict(self, feature_dict: dict) -> dict:
        """
        feature_dict keys must match FEATURES list.
        Returns dict with home_win / draw / away_win probabilities.
        """
        row = np.array([[feature_dict[f] for f in FEATURES]])
        proba = self.model.predict_proba(row)[0]
        return {
            "away_win":  round(float(proba[0]), 4),
            "draw":      round(float(proba[1]), 4),
            "home_win":  round(float(proba[2]), 4),
        }

    def predict_from_stats(
        self,
        home_team: str,
        away_team: str,
        team_stats: dict,
        neutral: bool = False,
    ) -> dict:
        """
        High-level helper: just pass team names and the stats dict.
        """
        h = team_stats.get(home_team, {})
        a = team_stats.get(away_team, {})

        def safe(d, k, default):
            return d.get(k, default)

        h_elo  = safe(h, "elo",  1500)
        a_elo  = safe(a, "elo",  1500)
        h_form = safe(h, "form", 0.5)
        a_form = safe(a, "form", 0.5)
        h_gs   = safe(h, "avg_goals_scored",   1.2)
        h_gc   = safe(h, "avg_goals_conceded", 1.2)
        a_gs   = safe(a, "avg_goals_scored",   1.2)
        a_gc   = safe(a, "avg_goals_conceded", 1.2)

        features = {
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
            "h2h_home_win_rate":   0.5,   # default; overridden by simulator
            "neutral":             int(neutral),
        }
        return self.predict(features)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model = train()

    if model is not None:
        print("\n--- Quick sanity check ---")
        print("Loading predictor and testing Brazil vs. Germany...")
        try:
            stats_path = os.path.join(PROCESSED, "team_stats.json")
            with open(stats_path) as f:
                stats = json.load(f)

            predictor = MatchPredictor()
            result = predictor.predict_from_stats("Brazil", "Germany", stats, neutral=True)
            print(f"Brazil vs Germany (neutral): {result}")
        except Exception as e:
            print(f"Skipped sanity check: {e}")

        print("\nNext step: python models/goal_model.py")
