# CLAUDE.md — ML World Cup Predictions

> This file is the project brain. Updated every 5 prompts or on topic change.
> Last updated: 2026-06-15 | Phase: Full System — Ready to Run ✅

---

## What This Project Does

A machine learning system that predicts the entire 2026 FIFA World Cup:

1. **Predicts every match** — Win / Draw / Loss probabilities using XGBoost trained on 150+ years of international football history
2. **Simulates the full tournament 10,000 times** — Monte Carlo simulation gives each team a realistic championship probability
3. **Uses real player data** — Player strength (FIFA ratings + market value) adjusts expected goals, making predictions squad-aware
4. **Simulates "what if?" scenarios** — Remove injured players or change formations and instantly see the probability impact
5. **Learns from real results** — After each 2026 WC match, update the model and see why predictions were right or wrong (SHAP)
6. **Generates an HTML dashboard** — Visual predictions report with country flags, probability bars, group standings, and upcoming match previews

---

## Project Structure

```
ML World Cup Predictions/
│
├── CLAUDE.md                        ← You are here (project brain)
├── learning.md                      ← Plain-English explanations of every concept
├── requirements.txt                 ← All Python packages needed
│
├── data/
│   ├── download_data.py             ← Downloads results.csv + writes real 2026 WC fixtures
│   ├── download_players.py          ← Embeds 26-man squads for all 48 teams
│   ├── raw/
│   │   ├── results.csv              ← 50,000+ international matches (1872–2026)
│   │   ├── wc2026_fixtures.json     ← Real 2026 WC groups A–L, 72 matches, known scores
│   │   └── fifa_rankings.csv        ← Optional FIFA rankings (may 404, not required)
│   ├── processed/
│   │   ├── match_features.csv       ← Feature-engineered training data
│   │   ├── team_stats.json          ← Per-team Elo, form, goals (used by simulator)
│   │   ├── team_elos.json           ← Final Elo ratings for all teams
│   │   └── team_player_strengths.json ← Player-based strength scores (attack/mid/def)
│   ├── live/
│   │   └── real_results.json        ← 2026 WC real match results fed in by user
│   └── players/
│       ├── wc2026_squads.json       ← 26-man squads for all 48 teams
│       └── merged_players.csv       ← Optional extended player data
│
├── features/
│   ├── engineering.py               ← Builds Elo ratings + all match features from results.csv
│   └── player_features.py           ← Converts squad data → attack/midfield/defense scores
│
├── models/
│   ├── match_predictor.py           ← XGBoost: predicts Win/Draw/Loss per match
│   ├── goal_model.py                ← Poisson: simulates realistic scorelines
│   ├── simulator.py                 ← Monte Carlo: runs tournament 10,000 times
│   └── lineup_simulator.py          ← Scenario simulator: injuries, formations, key players
│
├── analysis/
│   └── explainability.py            ← SHAP values + post-match wrong-prediction analysis
│
├── utils/
│   └── flags.py                     ← Emoji flags for all 48 WC teams 🇧🇷🇦🇷🇫🇷
│
├── outputs/
│   ├── report_generator.py          ← Generates HTML predictions dashboard
│   ├── wc2026_predictions.html      ← The actual predictions dashboard (open in browser)
│   └── simulation_results.json      ← Raw probability data from last simulation run
│
└── main.py                          ← Pipeline runner (train + simulate in one command)
```

---

## How to Run — Full Pipeline (First Time)

Run these commands in order. Each step depends on the one before it.

### Step 0 — Install dependencies
```bash
pip install -r requirements.txt
brew install libomp          # Mac only — required for XGBoost to load
```

### Step 1 — Download data
```bash
python3 data/download_data.py
```
Downloads `results.csv` (50,000+ historical matches) and writes `wc2026_fixtures.json` with the real 2026 WC groups and all 72 match slots. Known scores are already filled in.

### Step 2 — Download player squads
```bash
python3 data/download_players.py
```
Writes `data/players/wc2026_squads.json` with 26-man squads (name, position, FIFA rating, market value) for all 48 teams.

### Step 3 — Build features
```bash
python3 features/engineering.py
```
Reads `results.csv`, computes Elo ratings from scratch for every team (going match by match from 1872 to today), then builds features for every historical match: Elo diff, form, head-to-head, goals, xG proxy. Saves `match_features.csv`, `team_stats.json`, `team_elos.json`.

### Step 4 — Build player strength scores
```bash
python3 features/player_features.py
```
Reads the squad JSON and computes attack / midfield / defense / overall scores (0–100) for each team based on FIFA ratings (60%) and market values (40%). Saves `team_player_strengths.json`. Prints a ranked table of all 48 teams.

### Step 5 — Train the prediction model
```bash
python3 models/match_predictor.py
```
Trains an XGBoost classifier on `match_features.csv` to predict Win / Draw / Loss for any match. Uses an 80/20 time-based train/test split (trains on old matches, tests on recent ones). Saves `models/xgb_model.joblib`.

### Step 6 — Run tournament simulation
```bash
python3 models/simulator.py
```
Runs 10,000 Monte Carlo simulations of the full 2026 WC. Each run simulates every match using the XGBoost model (or Elo fallback). Applies player strength adjustments from Step 4. Saves `outputs/simulation_results.json` and prints winner probabilities with emoji flags.

### Step 7 — Generate HTML report
```bash
python3 outputs/report_generator.py
```
Reads `simulation_results.json` and generates `outputs/wc2026_predictions.html` — a visual dashboard with country flags, champion probability bars, group standings, and upcoming match predictions. **Open the HTML file in any browser.**

---

## How to Run — Daily During the Tournament

Just run the simulator — it auto-fetches new results from ESPN, updates Elo ratings and goal averages, and prints what changed since last time:

```bash
python3 models/simulator.py
python3 outputs/report_generator.py
```

On startup the simulator prints a report like:
```
🆕 3 match(es) added since your last simulation:
   🇫🇷 France 3–1 Senegal 🇸🇳  [2026-06-16]
   🇳🇴 Norway 4–1 Iraq 🇮🇶     [2026-06-16]
   🇮🇷 Iran 2–2 New Zealand 🇳🇿 [2026-06-16]
```

To fetch results only (without simulating):
```bash
python3 data/fetch_results.py
```

To see how each WC result moved team Elo and goal averages:
```bash
python3 data/live_calibration.py
```

---

## How to Run — Scenario Analysis

```bash
# What happens if Mbappé gets injured?
python3 models/lineup_simulator.py --injury "France" "Kylian Mbappé"

# What if Germany plays a 3-4-3 instead?
python3 models/lineup_simulator.py --formation "Germany" "3-4-3"

# How much is each key player worth for the top 5 teams?
python3 models/lineup_simulator.py --top5

# Remove multiple players at once
python3 models/lineup_simulator.py --injury "Brazil" "Vinicius Jr." "Rodrygo"

# Compare England with and without Bellingham
python3 models/lineup_simulator.py --compare "England" "Jude Bellingham"
```

---

## How to Run — Single Match Prediction

```bash
# Get Win/Draw/Loss % for any matchup
python3 main.py --predict "Argentina" "France"

# Skip retraining and just simulate
python3 main.py --sim

# Why did we predict the wrong winner?
python3 analysis/explainability.py --explain "Brazil" "Morocco"
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| XGBoost for match prediction | Best accuracy on tabular data, fast, interpretable via SHAP |
| Poisson for goal simulation | Goals in football are Poisson-distributed (proven by sports science) |
| Monte Carlo with 10,000 runs | Enough to get stable probabilities without being slow (~1 sec) |
| Elo computed from scratch | Avoids broken/outdated download URLs; gives us full control |
| Player strength as xG modifier | Keeps Elo as the anchor, player data fine-tunes it (±40% max) |
| Flags via Unicode emoji | Works in terminal + HTML without any external dependencies |

---

## Data Sources

| Dataset | Source | What it gives us |
|---|---|---|
| Historical results (1872–2026) | github.com/martj42/international_results | Training data for XGBoost + Elo calculation |
| FIFA Rankings | FIFA.com / Kaggle (optional) | Secondary signal — we use computed Elo instead |
| Player squad data | Embedded in `download_players.py` | FIFA ratings + market values for all 48 squads |
| 2026 WC fixtures | Embedded in `download_data.py` | Real groups A–L, 72 match slots, 16 known results |

---

## 2026 WC Format (what the simulator models)

| Stat | Value |
|---|---|
| Teams | 48 |
| Groups | 12 × 4 teams (A through L) |
| Group matches | 72 |
| Advancing per group | Top 2 = 24 teams |
| Wild-card | Best 8 third-place teams |
| Round of 32 field | 32 teams total |
| Knockout rounds | R32 → R16 → QF → SF → Final |
| R32 bracket | Groups A–H cross-paired; Groups I–L vs 3rd-place teams |

---

## Known Results (as of 2026-06-15)

| Match | Score |
|---|---|
| 🇲🇽 Mexico vs 🇿🇦 South Africa | 2–0 |
| 🇰🇷 South Korea vs 🇨🇿 Czechia | 2–1 |
| 🇨🇦 Canada vs 🇧🇦 Bosnia & Herz. | 1–1 |
| 🇶🇦 Qatar vs 🇨🇭 Switzerland | 1–1 |
| 🇧🇷 Brazil vs 🇲🇦 Morocco | 1–1 |
| 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland vs 🇭🇹 Haiti | 1–0 (Haiti home) |
| 🇺🇸 USA vs 🇵🇾 Paraguay | 4–1 |
| 🇦🇺 Australia vs 🇹🇷 Turkey | 2–0 |
| 🇩🇪 Germany vs 🇨🇼 Curacao | 7–1 |
| 🇨🇮 Ivory Coast vs 🇪🇨 Ecuador | 1–0 |
| 🇳🇱 Netherlands vs 🇯🇵 Japan | 2–2 |
| 🇸🇪 Sweden vs 🇹🇳 Tunisia | 5–1 |
| 🇧🇪 Belgium vs 🇪🇬 Egypt | 1–1 |
| 🇮🇷 Iran vs 🇳🇿 New Zealand | 2–2 |
| 🇪🇸 Spain vs 🇨🇻 Cape Verde | 0–0 |
| 🇸🇦 Saudi Arabia vs 🇺🇾 Uruguay | 1–1 |

---

## Current Status

**All code complete and tested. Ready to run on your machine.**

Completed components: Data pipeline · Feature engineering · XGBoost match predictor · Poisson goal model · Monte Carlo tournament simulator · SHAP explainability + post-match learning · Player squad data · Player strength engine · Lineup scenario simulator · Flag emoji support · HTML predictions dashboard · Real 2026 WC fixtures (12 groups × 4 teams) · Correct R32 bracket logic (32 teams, best-8-third-place)