# 🌍 Learning Guide — ML World Cup Predictions

> Everything we built, explained clearly. No jargon walls.
> The goal: you understand every line of this project and why it exists.
> Last updated: 2026-06-16

---

## Table of Contents

1. [The Big Picture — How It All Fits Together](#1-the-big-picture--how-it-all-fits-together)
2. [The Data — Where It All Starts](#2-the-data--where-it-all-starts)
3. [Feature Engineering — Turning Matches Into Math](#3-feature-engineering--turning-matches-into-math)
4. [Elo Ratings — The Heart of Team Strength](#4-elo-ratings--the-heart-of-team-strength)
5. [XGBoost — The Match Prediction Model](#5-xgboost--the-match-prediction-model)
6. [The Poisson Goal Model — Simulating Scorelines](#6-the-poisson-goal-model--simulating-scorelines)
7. [Monte Carlo Simulation — Running the Tournament 10,000 Times](#7-monte-carlo-simulation--running-the-tournament-10000-times)
8. [Player Stats — Making It Squad-Aware](#8-player-stats--making-it-squad-aware)
9. [Lineup Scenario Simulator — The "What If?" Engine](#9-lineup-scenario-simulator--the-what-if-engine)
10. [SHAP — Why Did the Model Predict That?](#10-shap--why-did-the-model-predict-that)
11. [The HTML Report — Seeing It All](#11-the-html-report--seeing-it-all)
12. [The 2026 World Cup Format — What the Simulator Models](#12-the-2026-world-cup-format--what-the-simulator-models)
13. [Live Results — The ESPN API Pipeline](#13-live-results--the-espn-api-pipeline)
14. [Live Calibration — Updating the Model as the Tournament Plays Out](#14-live-calibration--updating-the-model-as-the-tournament-plays-out)
15. [How the Models Talk to Each Other](#15-how-the-models-talk-to-each-other)
16. [Why Any of This Works](#16-why-any-of-this-works)

---

## 1. The Big Picture — How It All Fits Together

Before diving into the details, here's the whole system in one diagram:

```
RAW DATA
  results.csv (50,000 historical matches)
  wc2026_fixtures.json (real 2026 bracket)
  wc2026_squads.json (48 × 26-man squads)
       │
       ▼
FEATURE ENGINEERING  [features/engineering.py]
  Elo ratings · Recent form · Head-to-head · xG proxy
       │
       ├──────────────────────────────────────────────────────┐
       ▼                                                      ▼
XGBOOST MODEL                                    PLAYER STRENGTH ENGINE
[models/match_predictor.py]                      [features/player_features.py]
Trained on 150 years of football.                FIFA ratings + market values →
Predicts Win/Draw/Loss % for any match.          attack / midfield / defense scores
       │                                                      │
       └──────────────────┬───────────────────────────────────┘
                          │
       ┌──────────────────▼──────────────────┐
       │   LIVE CALIBRATION LAYER            │  ← NEW
       │   fetch_results.py (ESPN API)       │
       │   live_calibration.py              │
       │   Updates Elo + goals from real     │
       │   WC26 matches, in-memory           │
       └──────────────────┬──────────────────┘
                          ▼
              MONTE CARLO SIMULATOR
              [models/simulator.py]
              Run the full tournament 10,000×
              Count how often each team wins
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
      SHAP EXPLAINABILITY      HTML REPORT
      [analysis/explainability] [outputs/report_generator.py]
      Why did we predict that?  Visual dashboard with flags,
      Post-match learning.      bars, standings, predictions
```

Every piece has a job. Remove any one and the system gets worse. They work together like a chain.

---

## 2. The Data — Where It All Starts

**File: `data/download_data.py`**

We need two kinds of data:

### Historical match results (`results.csv`)
Every international football match since 1872. Over 50,000 rows. Each row looks like:
```
date,       home_team,  away_team,  home_score, away_score, tournament,    neutral
1872-11-30, Scotland,   England,    0,          0,          Friendly,      False
2014-07-08, Brazil,     Germany,    1,          7,          World Cup,     False
```

This is the training data for our XGBoost model. It also lets us compute Elo ratings for every team going all the way back to the 19th century.

### 2026 WC fixtures (`wc2026_fixtures.json`)
The actual draw: 12 groups, 48 teams, 72 matches. We embed this directly in code because FIFA's website needs JavaScript to load, which our script can't run. Known results are pre-filled; future matches have `null` scores so the simulator fills them in.

### Player squads (`wc2026_squads.json`)
26-man squads for all 48 teams. Each player has: name, position (GK/DEF/MID/ATT), FIFA 25 rating, market value in €M, and club. This lets us model lineup changes and injuries.

**Why download from 1872?** Because Elo is cumulative — Brazil's strength in 2026 reflects a century of results, not just last year. The older the data, the more accurate the Elo.

---

## 3. Feature Engineering — Turning Matches Into Math

**File: `features/engineering.py`**

You can't feed "Brazil vs France" into an algorithm — you need numbers. Feature engineering converts every match into a row of meaningful numbers called **features**.

For each historical match, we compute these features **before the match happened** (critical — no cheating by looking at future data):

| Feature | What it means | Example |
|---|---|---|
| `home_elo` | Home team's Elo strength | 2050 |
| `away_elo` | Away team's Elo strength | 1890 |
| `elo_diff` | Strength gap | +160 (home stronger) |
| `home_form` | Win rate over last 10 matches | 0.72 (72%) |
| `away_form` | Same for away team | 0.51 |
| `form_diff` | Form gap | +0.21 |
| `home_avg_scored` | Goals scored per game (recent) | 1.9 |
| `home_avg_conceded` | Goals conceded per game | 0.8 |
| `away_avg_scored` | Same for away | 1.4 |
| `away_avg_conceded` | Same for away | 1.1 |
| `home_xg` | Expected goals proxy | 1.85 |
| `away_xg` | Same for away | 1.20 |
| `h2h_home_win_rate` | Head-to-head historical rate | 0.62 |
| `neutral` | Is it on neutral ground? | 0 or 1 |
| `outcome` | **The label**: 2=home win, 1=draw, 0=away win | 2 |

The last column (`outcome`) is what XGBoost tries to predict. Everything else is its input.

The most important rule: **compute features using only data that existed before the match.** If we use the final score to compute form, we're cheating — the model would be predicting things it "already knew."

---

## 4. Elo Ratings — The Heart of Team Strength

**File: `features/engineering.py` → `EloRating` class**

Elo was invented by a Hungarian-American physics professor named Arpad Elo in the 1960s for chess rankings. It answers:

> "How strong is this team, given *who* they beat — not just how many they beat?"

**The math:**
```
Expected score for Home = 1 / (1 + 10^(-(Elo_home - Elo_away) / 400))

New Elo = Old Elo + K × (Actual result - Expected score)

Where:
  Actual result = 1 (win), 0.5 (draw), 0 (loss)
  K = how much this match changes the rating
      K is higher for World Cup matches, lower for friendlies
```

**Why K varies by tournament type:** A World Cup match matters more than a friendly. Beating Germany in a World Cup final vs. a June friendly are very different things. Our K-factor table:
- World Cup: K = 60
- Continental championships (Euros, Copa América): K = 50
- Qualifying matches: K = 40
- Friendlies: K = 20

**An example:** Brazil (Elo 2050) beats Curacao (Elo 1450). The expected result was basically certain, so Brazil only gains ~2 points. If Curacao had somehow won, they'd gain ~58 points and Brazil would lose ~58. The bigger the upset, the bigger the rating swing.

**Why we compute Elo ourselves instead of downloading:** Rating systems like FIFA rankings are updated monthly and reset. We want ratings that go back to 1872 and are fully transparent about how they were calculated.

---

## 5. XGBoost — The Match Prediction Model

**File: `models/match_predictor.py`**

XGBoost stands for **eXtreme Gradient Boosting**. It's the most-used algorithm for tabular data (spreadsheet-style). It won more Kaggle competitions than any other algorithm in history.

**The core idea — boosting:**

1. Build a simple decision tree. Example: "If Elo diff > 150, predict home win."
2. See where it got things wrong.
3. Build a *second* tree specifically to fix those mistakes.
4. Repeat 500–1000 times.
5. The final prediction is a weighted vote of all the trees.

Each new tree "boosts" the accuracy by correcting errors from the previous ones. This is why it's called boosting.

**What ours predicts:**
- Input: the 14 features from Step 3
- Output: probabilities for Home Win / Draw / Away Win (3 classes)

For example: `Argentina vs France` →
```
Argentina win: 44%
Draw:          26%
France win:    30%
```

**How we train it:**
- We use an 80/20 time-based split: train on older matches, test on recent ones
- This mimics the real world — you can't train on future results
- The model never sees 2026 WC matches during training

**Why XGBoost over other models?**
- Handles the 14 mixed features naturally
- Resistant to overfitting (via regularization)
- Fast to train (~seconds on a laptop)
- Works with SHAP for explainability

---

## 6. The Poisson Goal Model — Simulating Scorelines

**File: `models/goal_model.py`**

The XGBoost model tells us *who wins*, but not *by how much*. For knockout rounds, we need to simulate extra time and penalties — which requires actual goal counts, not just win/draw/loss.

The **Poisson distribution** is perfect for this. It models how many times a rare event happens in a fixed time window. Goals in football are rare (average ~1.4 per team per 90 minutes), so they follow a Poisson distribution almost exactly — this has been confirmed by sports science studies since the 1980s.

**How it works:**

We compute an expected goals rate (λ, "lambda") for each team:
```
λ_home = base_goals × (home_attack / avg_attack) × (away_defense / avg_defense) × home_advantage
λ_away = similar calculation for the away team
```

Then we ask: given Brazil scores at a rate of λ=1.8 goals/game, what's the probability of exactly 0, 1, 2, 3... goals?

```
P(0 goals) = e^(-1.8) × 1.8^0 / 0! = 16.5%
P(1 goal)  = e^(-1.8) × 1.8^1 / 1! = 29.7%
P(2 goals) = e^(-1.8) × 1.8^2 / 2! = 26.7%
P(3 goals) = e^(-1.8) × 1.8^3 / 3! = 16.0%
...
```

We compute this for both teams independently, producing an 11×11 probability matrix of all possible scorelines (0-0 through 10-10+). Then we draw randomly from this matrix to get a realistic scoreline.

**For knockout rounds:** If the Poisson draw results in a draw, we simulate 30 minutes of extra time (with reduced scoring rates), then if still level, run a penalty shootout (50/50 per penalty, first to miss loses).

---

## 7. Monte Carlo Simulation — Running the Tournament 10,000 Times

**File: `models/simulator.py`**

Named after the famous casino in Monaco. The idea: **run something random thousands of times and count the outcomes to get reliable probabilities.**

**Why 10,000?** Think about flipping a coin. After 10 flips you might get 7 heads. After 1,000 you'll be close to 50%. After 10,000 you'll be at 50.0% ± 0.5%. Same idea here — 10,000 tournament simulations gives stable win probabilities (error margin < 1%).

**Each simulation does this:**

```
1. GROUP STAGE (72 matches):
   For each match:
     → If it's already been played, use the real score
     → If not, predict using XGBoost + player strength → draw random result
   Track points, goal difference, goals for each team

2. DETERMINE WHO ADVANCES:
   → Top 2 from each of 12 groups = 24 teams
   → Rank all 12 third-place teams by points → GD → GF
   → Best 8 third-place teams also advance
   → Total: 32 teams

3. ROUND OF 32 (16 matches):
   → Groups A-H: cross-paired (A1 vs B2, B1 vs A2, etc.)
   → Groups I-L: each winner and runner-up vs a third-place team
   → No draws — if level, run Poisson extra time + penalties

4. ROUND OF 16 → QUARTER-FINALS → SEMI-FINALS → FINAL:
   → Winners of each match advance
   → Bracket is fixed (no re-seeding)

5. Record the champion of this simulation
```

After 10,000 simulations:
- Brazil won 2,280 times → **22.8% champion probability**
- Argentina won 1,440 times → **14.4%**
- etc.

**Player strength is baked in at the team stats level:** Before simulating, we adjust each team's expected goals using their squad strength scores. A team with a 70/100 attack score will have their xG boosted; a team with weak defense has their xG conceded boosted for opponents.

---

## 8. Player Stats — Making It Squad-Aware

**File: `features/player_features.py`**

Without player data, our model treats Brazil the same whether Vinicius Jr. is playing or on the bench with a torn hamstring. Player stats fix that.

**Where the data comes from:**
- FIFA 25 ratings (1–99, overall skill)
- Market values from Transfermarkt (€M, a great real-time proxy for form and quality)

**How we score each player:**
```
Player score = 0.60 × normalized_FIFA_rating
             + 0.40 × normalized_market_value
```

The FIFA rating captures skill ceiling; market value captures current form and prime years.

**How we aggregate to team level:**
```
attack_strength  = weighted average of ATT players (starters weighted 1.5×)
midfield_strength = weighted average of MID players
defense_strength = 75% × DEF average + 25% × GK average
overall_strength = 35% attack + 30% midfield + 35% defense
squad_depth      = how much quality drops from starters to bench
```

**How it connects to the simulator:**

Player strength doesn't replace Elo — it adjusts the expected goals up or down. The formula:
```
xG multiplier = 1.0 + 0.08 × attack_edge + 0.04 × midfield_edge

where:
  attack_edge = (your attack - their defense) / 20
  midfield_edge = (your midfield - average midfield) / 30

multiplier is capped at [0.70, 1.40] — max ±40% adjustment
```

So if France's attack (80/100) faces a weak defense (45/100), their xG might be boosted by ~14%. This keeps Elo as the dominant signal but uses squad quality to fine-tune every match.

---

## 9. Lineup Scenario Simulator — The "What If?" Engine

**File: `models/lineup_simulator.py`**

This is the most fun part. It runs full tournament simulations under hypothetical conditions:

**Injury scenario:**
```
python3 models/lineup_simulator.py --injury "France" "Kylian Mbappé"
```
1. Remove Mbappé from France's squad
2. Recalculate France's attack strength (big drop — he's their best player)
3. Run 5,000 full tournament simulations without him
4. Compare: France had 12.3% WC probability → drops to 7.1% without him
5. Show how this affects every other team too (Brazil gains ~2% because France got weaker)

**Formation change:**
```
python3 models/lineup_simulator.py --formation "Germany" "3-4-3"
```
Changes Germany's formation from their default (4-2-3-1) to 3-4-3. This shifts players between attack/midfield/defense, changing all three strength scores, which changes their xG in every match.

**Formation parsing:**
```
"4-3-3"   → {GK:1, DEF:4, MID:3, ATT:3}
"4-2-3-1" → {GK:1, DEF:4, MID:5, ATT:1}   (2+3 counted as mid)
"3-5-2"   → {GK:1, DEF:3, MID:5, ATT:2}
```

The model picks the best available players for each slot, recomputes team strengths, then reruns the tournament.

**Key player impact:**
```
python3 models/lineup_simulator.py --top5
```
For the top 5 WC favorites, removes each key player one at a time to show exactly how many percentage points of WC probability each player is worth. This answers "who is the single most irreplaceable player in the tournament?"

---

## 10. SHAP — Why Did the Model Predict That?

**File: `analysis/explainability.py`**

SHAP stands for **SHapley Additive exPlanations** (from game theory — Shapley values). It answers:

> "The model predicted Argentina wins with 55% probability — *why* exactly?"

Without SHAP, the XGBoost model is a black box. With SHAP, every prediction decomposes into contributions from each feature:

```
Argentina win probability: 55%

+12%  Elo advantage (Argentina rated 160 points higher)
 +8%  Recent form (Argentina won 8 of last 10)
 +5%  Head-to-head record (Argentina has 60% H2H win rate)
 -4%  Away match (slight disadvantage)
 -3%  Opponent's strong defense (France concedes only 0.9 goals/game)
 -3%  Other features
─────
= 55% net probability
```

**Post-match learning:** After a real match is played, if our prediction was wrong, SHAP shows us *why* we got it wrong. Was it because:
- We underestimated the opponent's form?
- The Elo gap wasn't as decisive as we thought?
- An injury we didn't account for?

This is the "learning" part of the system. It doesn't automatically retrain the model, but it tells you what signal you're missing so you can add it as a future feature.

**How to use it:**
```bash
# Explain why the model predicted a specific result
python3 analysis/explainability.py --explain "Brazil" "Morocco"

# Feed in a real match result after it happens
python3 analysis/explainability.py --update "Brazil" "Morocco" 1 1

# See a full report of all past predictions vs reality
python3 analysis/explainability.py --report
```

---

## 11. The HTML Report — Seeing It All

**File: `outputs/report_generator.py`**

Generates a self-contained HTML dashboard (`outputs/wc2026_predictions.html`) that you open in any browser. It pulls from:
- `simulation_results.json` → championship probabilities and stage breakdown
- `wc2026_fixtures.json` → group standings and upcoming matches
- `team_stats.json` → Elo-based win/draw/loss estimates for upcoming games
- `utils/flags.py` → emoji flags for every team name

The report includes:
- 🏆 **Championship odds** — all 24 teams shown with probability bars
- 📊 **Group standings** — live standings with green/gold shading for qualification zones
- 📅 **Upcoming matches** — each with a colored probability bar (blue=home win, gray=draw, red=away win)

To regenerate after new results:
```bash
python3 models/simulator.py          # run fresh simulations
python3 outputs/report_generator.py  # rebuild the HTML
```

---

## 12. The 2026 World Cup Format — What the Simulator Models

The 2026 World Cup is the first ever with **48 teams** instead of 32. This required a complete rewrite of our simulator.

**Group Stage:**
- 12 groups (A through L), 4 teams each
- Every team plays 3 matches within their group (round-robin)
- Points: Win=3, Draw=1, Loss=0
- Tiebreaker: points → goal difference → goals for

**Who Advances:**
- Top 2 from each group = **24 teams** (guaranteed)
- All 12 third-place teams are ranked against each other
- Best 8 third-place teams = **8 more teams** (wild card)
- Total advancing: **32 teams**

This "best third-place" rule is interesting to think about. A team could finish third in their group (out of 4 teams) but still advance if they outperformed the third-place finishers in other groups. So every point matters even if you're already eliminated from top-2.

**Round of 32 bracket:**
- Groups A–H: winner of one group plays runner-up of an adjacent group
  - A1 vs B2, B1 vs A2, C1 vs D2, D1 vs C2, E1 vs F2, F1 vs E2, G1 vs H2, H1 vs G2
- Groups I–L: each group's winner and runner-up face one of the 8 third-place qualifiers
- Result: 32 teams, 16 R32 matches, no team gets a bye

**Then it's standard knockout:** R32 → R16 → QF → SF → Final. 5 rounds, 31 matches total.

---

## 13. Live Results — The ESPN API Pipeline

**File: `data/fetch_results.py`**

Every time you run the simulator, it automatically checks for new match results so you never have to feed them in manually. Here's how that works.

### What is an API?

API stands for **Application Programming Interface**. It's a way for two programs to talk to each other over the internet. Instead of scraping a webpage (which breaks every time the design changes), you ask a server directly: "give me the data in a structured format."

ESPN runs a public, undocumented API that their own apps use. We discovered that if you hit a specific URL, it returns clean JSON data for any sport — including the 2026 World Cup. No login, no API key required.

### The URL pattern

```
https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard
    ?dates=20260611-20260616
    &limit=200
```

Breaking that down:
- `site.api.espn.com` — ESPN's internal data server
- `/sports/soccer/fifa.world/scoreboard` — the World Cup scoreboard endpoint (`fifa.world` is ESPN's internal league ID for the FIFA World Cup)
- `?dates=20260611-20260616` — date range filter (WC start to today)
- `&limit=200` — return up to 200 events

### What it returns

The API responds with a large JSON blob containing a list of `events`. Each event looks like this (simplified):

```json
{
  "date": "2026-06-16T19:00Z",
  "competitions": [{
    "status": { "type": { "completed": true } },
    "competitors": [
      { "homeAway": "home", "team": { "displayName": "France" }, "score": "3" },
      { "homeAway": "away", "team": { "displayName": "Senegal" }, "score": "1" }
    ]
  }]
}
```

We check `completed: true` before recording a match — ongoing games are skipped.

### Team name normalization

ESPN uses different names than our internal system. For example:

| ESPN name | Our canonical name |
|---|---|
| `"United States"` | `"USA"` |
| `"Bosnia and Herzegovina"` | `"Bosnia & Herzegovina"` |
| `"Türkiye"` | `"Turkey"` |
| `"Czech Republic"` | `"Czechia"` |
| `"Côte d'Ivoire"` | `"Ivory Coast"` |
| `"Korea Republic"` | `"South Korea"` |
| `"Curaçao"` | `"Curacao"` |

A lookup dictionary (`ESPN_NAME_MAP`) handles all of these before any match is saved.

### Deduplication

Every match gets a stable **key** based on both team names sorted alphabetically:

```python
key = f"{min(home, away)}||{max(home, away)}"
# France vs Senegal → "France||Senegal"
```

Using sorted names means it doesn't matter if we see France listed as "home" or "away" — the key is always the same. Before adding any match, we check if its key already exists in `real_results.json`. If yes, skip it. If no, append it.

### The "since last run" report

After each successful simulation, `simulator.py` saves a snapshot — the complete set of match keys it processed. The *next* time you run the simulator, it compares the current `real_results.json` against that snapshot and reports only the truly new matches:

```
🆕 3 match(es) added since your last simulation:
   🇫🇷 France 3–1 Senegal 🇸🇳  [2026-06-16]
   🇳🇴 Norway 4–1 Iraq 🇮🇶     [2026-06-16]
   🇮🇷 Iran 2–2 New Zealand 🇳🇿 [2026-06-16]
```

The snapshot lives in `data/live/last_run_snapshot.json`.

---

## 14. Live Calibration — Updating the Model as the Tournament Plays Out

**File: `data/live_calibration.py`**

Now that we have real WC26 results, we can use them to make predictions sharper. The calibration does two things: updates Elo and adjusts expected goals. Both happen **in-memory** on top of the historical base — the original `team_stats.json` is never overwritten.

### Elo updates from real results

The standard Elo update formula (described in Section 4) applies here too, but with the World Cup K-factor of 60 and no home advantage (all WC matches are on neutral ground):

```
E_home = 1 / (1 + 10^(-(Elo_home - Elo_away) / 400))   # expected score
actual = 1.0 (win) | 0.5 (draw) | 0.0 (loss)

new_Elo_home = old_Elo_home + 60 × (actual - E_home)
new_Elo_away = old_Elo_away + 60 × (E_home - actual)    # zero-sum
```

Results are processed in chronological order so Elo flows correctly — Spain's Elo going into game 2 already reflects what happened in game 1.

**After the first round of group games, here's how Elo moved:**

| Team | Before | After | Δ |
|---|---|---|---|
| Spain | 2147 | 2120 | ▼27 — drew 0-0 with Cape Verde, huge underperformance |
| Australia | 1942 | 1970 | ▲28 — beat Turkey 2-0, strong overperformance |
| Ivory Coast | 1844 | 1882 | ▲38 — beat Ecuador 1-0 as underdogs |
| Norway | 1940 | 1957 | ▲17 — beat Iraq 4-1 |
| France | 2055 | 2070 | ▲15 — beat Senegal 3-1 |
| Belgium | 1892 | 1883 | ▼9 — drew 1-1 with Egypt |
| Uruguay | 1903 | 1888 | ▼15 — drew 1-1 with Saudi Arabia |

These shifts are small but meaningful over the course of a full simulation. Spain's drop changes their probability of topping Group H; Australia's rise makes them harder to knock out.

### Opponent-adjusted goal calibration

Raw goals alone are misleading. Germany scoring 7 against Curacao tells us very little about how Germany will do against a proper defence. So before blending WC goals into the model, we adjust them for opponent quality:

```
adjusted_goals = raw_goals × (league_avg_conceded / opponent_avg_conceded)
```

If Curacao's `avg_goals_conceded` is 3.5 and the league average is 1.35:
```
Germany's 7 goals → 7 × (1.35 / 3.5) = 2.7 adjusted goals
```

So Germany's WC "evidence" is treated as scoring 2.7 goals, not 7 — which is actually close to their historical average of 3.5 anyway. This is why Germany's goal stats barely moved after the Curacao game, which is exactly correct behaviour.

A true blowout against a good team (say, 4-0 vs France) would push the attacker's goals up significantly because the divisor would be small (France concede very little).

### The Bayesian blend

We don't throw away 150 years of data after one WC game. Instead we blend with a small weight that grows as more games are played:

```
α (alpha) = min(0.30, games_played × 0.08)

1 game  → α = 8%   (8% WC data, 92% historical)
2 games → α = 16%
3 games → α = 24%  ← end of group stage
max     → α = 30%  (after ~4 games)

new_avg = (1 - α) × historical_avg + α × wc_adjusted_avg
```

The cap at 30% keeps the model stable even if a team has an unusually good or bad group stage. By the knockout rounds, we'll have 3 WC games per team and the WC signal will carry meaningful weight.

---

## 15. How the Models Talk to Each Other

This is the most important thing to understand. Each model is independent but feeds into the next:

```
Step 1: FEATURE ENGINEERING produces team_stats.json + match_features.csv
         │
         ├─► Step 2: XGBOOST is trained on match_features.csv
         │           XGBoost can now predict any match
         │
         ├─► Step 3: PLAYER ENGINE reads squads, produces team_player_strengths.json
         │
         └─► Step 4: SIMULATOR (main) at startup:
                       [A] fetch_results.py → ESPN API → real_results.json
                       [B] live_calibration.py → updates Elo + goals in-memory
                     Then: team_stats.json + player strength + live calibration
                     For each of 10,000 simulated matches:
                       → Calls XGBoost.predict(features) to get win/draw/loss odds
                       → Calls Poisson.simulate() for scorelines in knockouts
                     Saves simulation_results.json + last_run_snapshot.json
                               │
                               └─► Step 5: REPORT GENERATOR reads simulation_results.json
                                           Builds wc2026_predictions.html
```

**The key handoff:** `team_stats.json` is the universal connector. Feature engineering writes it; the simulator reads it (then calibrates on top in-memory); the lineup simulator reads it too. Every team's stats (Elo, form, goals, xG) live there and flow through the entire system.

**Why calibration stays in-memory:** We never overwrite `team_stats.json` with calibrated data. If we did, the second run would re-apply calibration on top of already-calibrated data — doubling the effect. By always starting fresh from the historical base and computing calibration from scratch each run, the results are always correct no matter how many times you run it.

---

## 16. Why Any of This Works

This is the question worth sitting with. Why does 150 years of football results tell us anything useful about who wins in 2026?

**Three reasons:**

**1. Football has stable patterns.** The team with the higher Elo wins more often than not — not always, but reliably enough to learn from. Over 50,000+ matches, patterns emerge: home advantage is real (~60% home win rate historically), form streaks matter, head-to-head history matters. These patterns haven't changed fundamentally since the 1870s.

**2. Elo is self-correcting.** Because Elo updates after every match, a team that's been on a recent run of strong form will have a higher Elo than their 5-year average suggests. The model is never purely historical — it reflects recent results.

**3. Monte Carlo handles uncertainty honestly.** Instead of pretending we know who will win (we don't), we quantify *uncertainty*. "Brazil has a 22% chance to win" is honest — it means that in a world where we ran this tournament 100 times, Brazil would win about 22 of them. Some of those 22 times involve lucky draws. Some involve dominant performances. The probability reflects all of that.

**What the model can't do:**
- It doesn't know about injuries that happened this morning (unless you run `--injury`)
- It doesn't know about team chemistry, morale, or fatigue
- It can't predict true upsets (by definition, upsets happen less often than favorites win)
- Match-fixing, weather, referee decisions — all invisible to the model

**What it's actually good at:**
- Ranking teams from most to least likely to win with calibrated confidence
- Showing how a single player absence ripples through an entire tournament
- Giving you a quantitative baseline to argue against when your gut says something different

The fun isn't in treating the predictions as certain — it's in understanding *why* the model thinks what it thinks, and deciding whether you agree.

---

## 🏆 Conditioning a Simulation on a Live Bracket (added July 7)

### The problem, in plain English

Before the knockout stage, our Monte Carlo simulator answered the question: "if the whole tournament were played 10,000 times, who wins most often?" But once real matches are played, that question changes. Brazil and Germany are already out. Simulating worlds where Germany wins the final is not just wasted compute, it is *wrong*: it dilutes the probabilities of the teams that are actually still alive.

### The fix: condition on what already happened

"Conditioning" is the probability word for locking in known facts and only rolling dice for the unknown. Think of it like this:

- **Before the tournament:** every match is a dice roll. 10,000 simulated tournaments.
- **Now (quarter-finals):** the group tables are facts. The Round of 32 and Round of 16 winners are facts. So we start every one of the 10,000 simulations from the REAL bracket with the REAL 8 quarter-finalists, and only roll dice for the 7 matches that remain (4 QFs, 2 SFs, 1 final).

Same simulator, same XGBoost and Poisson models per match. The only change is the starting state.

### Why probabilities jump around at this stage

Spain went from a modest pre-tournament favorite to ~32%. Two reasons:

1. **Survivorship**: every eliminated rival's probability mass gets redistributed to the survivors. 8 teams share 100% instead of 48.
2. **Bracket luck**: it matters *who* is on your side of the bracket. A favorite whose half contains the other favorites has a harder road than one whose half opened up. Conditioning captures this automatically because we simulate the real pairings, not hypothetical ones.

### One subtlety: knockout matches cannot end in draws

Our match model predicts Win / Draw / Loss. In the knockouts, a "draw" after 90 minutes goes to extra time and maybe penalties. The simulator resolves simulated draws with a near-coin-flip weighted by team strength (penalties are close to random, which real-world data supports). For REAL matches that went to penalties, ESPN tells us who advanced, and we lock in the advancer rather than re-deciding it.

### The takeaway

A forecast is only as good as the information you refuse to ignore. Updating the simulation to start from the real bracket is the same idea as Bayesian updating: new evidence (results) narrows the space of possible futures, and the model's job is to only explore futures that are still possible.

---

## ⏱️ Predicting HOW a Match Ends: Extra Time and Penalties (added July 8)

### The question

"Who advances?" is one number. But a knockout match can end three ways: decided in 90 minutes, decided in extra time, or decided on penalties. Each path has its own probability, and we can compute all of them exactly, without simulating anything.

### Step 1: P(extra time) is already sitting in the scoreline matrix

The Poisson model gives us a matrix where cell (i, j) = P(home scores i, away scores j). Sum the diagonal (0-0, 1-1, 2-2...) and you have the probability of a draw after 90 minutes. That IS the probability of extra time. For an even quarter-final like France vs Morocco it is about 22%. For a mismatch it drops, because mismatches produce fewer draws.

### Step 2: extra time is just a tiny football match

Extra time is 30 minutes of tired football. We model it as a mini Poisson match where each team scores at 30% of its full-match rate. Build the same matrix for that mini match:

- Off-diagonal cells: someone scored more, the match ends in extra time
- Diagonal cells: still level, we are going to penalties

Conditional probabilities falling out of a smaller version of the exact same math we already had. No new theory needed.

### Step 3: penalties are (almost) a coin flip

Studies of real shootouts show outcomes close to 50/50 regardless of team quality: it is a high-pressure skill lottery. So we give the stronger team only a small Elo-based edge, capped at ±5%. Spain might get 55% against Switzerland, never 80%.

### Why this changed the champion odds

The old simulator resolved knockout draws "proportionally": if Spain had a 4-to-1 edge in 90 minutes, it kept a 4-to-1 edge in the draw resolution too. That silently assumed favorites dominate shootouts, which is false. Chaining the honest three-stage math (90 minutes, then ET, then near-random pens) gives underdogs their real upset path: about 12% of Spain vs Belgium ends in a shootout, and in that world Belgium is nearly even. That is why Belgium and Switzerland each gained roughly 2 points of final-appearance probability when we shipped this.

### The takeaway

Decompose an outcome into its real-world paths and probability multiplication does the rest: P(advance) = P(win in 90) + P(draw) × P(win ET) + P(draw) × P(level ET) × P(win pens). Each factor comes from a model matched to how that phase of football actually behaves. The lesson generalizes: when one number hides several different mechanisms, model the mechanisms.

---

*Last updated: 2026-07-08 | Phase: Knockout Stage + Per-Match Predictions ✅*
