"""
download_data.py
----------------
Run this script ONCE on your local machine to download all datasets.
It will save everything to data/raw/ automatically.

Usage:
    python data/download_data.py
"""

import os
import urllib.request
import zipfile
import json

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def download(url: str, filename: str, description: str, optional: bool = False):
    dest = os.path.join(RAW_DIR, filename)
    if os.path.exists(dest):
        print(f"  ✓ Already have {filename} — skipping")
        return dest
    print(f"  ↓ Downloading {description}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"    Saved → {dest}")
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)   # remove partial file
        if optional:
            print(f"  ⚠️  Skipped {filename} ({e}) — not required, continuing...")
            return None
        raise
    return dest


def main():
    print("\n🌍 Downloading World Cup ML datasets...\n")

    # ── 1. Historical International Results (1872–2026) ──────────────────────
    # Source: github.com/martj42/international_results
    download(
        url="https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        filename="results.csv",
        description="Historical international results (1872–2026)",
    )

    # ── 2. FIFA World Rankings ────────────────────────────────────────────────
    # NOTE: We compute Elo ratings ourselves from results.csv, so this is
    # supplementary only. We use the fifa-ranking dataset on Kaggle-mirrors.
    download(
        url="https://raw.githubusercontent.com/datasets/fifa-world-ranking/main/data/fifa-world-ranking-2024.csv",
        filename="fifa_rankings.csv",
        description="FIFA World Rankings",
        optional=True,
    )

    # ── 3. 2026 World Cup Groups & Fixtures ──────────────────────────────────
    # We embed this directly since the official source requires JS rendering
    print("  ✓ Writing 2026 WC fixtures (embedded)...")
    _write_wc2026_fixtures()

    print("\n✅ All data downloaded to data/raw/\n")
    print("Next step: python features/engineering.py")


def _write_wc2026_fixtures():
    """
    2026 FIFA World Cup — REAL draw (Dec 5, 2025):
      48 teams · 12 groups of 4 · 72 group matches
      Top 2 per group (24) + best 8 third-place teams = 32 advance
      Knockout: Round of 32 → 16 → QF → SF → Final

    Scores filled where played (as of 2026-06-15).
    Note: UEFA/FIFA play-off winners are named by their resolved team.
      Play-off D → Czechia  |  Play-off A → Bosnia & Herzegovina
      Play-off C → Turkey   |  Play-off B → Sweden
      FIFA Play-off 2 → Iraq  |  FIFA Play-off 1 → DR Congo (TBC)
    """
    fixtures = {
        "tournament": "2026 FIFA World Cup",
        "hosts": ["USA", "Canada", "Mexico"],
        "start_date": "2026-06-11",
        "final_date": "2026-07-19",
        "format": {
            "groups": 12,
            "teams_per_group": 4,
            "advance_per_group": 2,
            "best_third_place": 8,
            "total_advancing": 32,
        },
        "groups": {
            "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
            "B": ["Canada", "Switzerland", "Qatar", "Bosnia & Herzegovina"],
            "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
            "D": ["USA", "Australia", "Paraguay", "Turkey"],
            "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
            "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
            "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
            "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
            "I": ["France", "Senegal", "Norway", "Iraq"],
            "J": ["Argentina", "Austria", "Algeria", "Jordan"],
            "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
            "L": ["England", "Croatia", "Panama", "Ghana"],
        },
        "group_matches": [
            # Format: [home, away, home_score, away_score, date]
            # null score = not yet played

            # ── Group A ──────────────────────────────────────────────────────
            ["Mexico",       "South Africa", 2, 0,    "2026-06-11"],  # ✓
            ["South Korea",  "Czechia",      2, 1,    "2026-06-11"],  # ✓
            ["Mexico",       "South Korea",  None, None, "2026-06-18"],
            ["South Africa", "Czechia",      None, None, "2026-06-18"],
            ["South Africa", "South Korea",  None, None, "2026-06-24"],
            ["Czechia",      "Mexico",       None, None, "2026-06-24"],

            # ── Group B ──────────────────────────────────────────────────────
            ["Canada",              "Bosnia & Herzegovina", 1, 1, "2026-06-12"],  # ✓
            ["Qatar",               "Switzerland",          1, 1, "2026-06-13"],  # ✓
            ["Switzerland",         "Bosnia & Herzegovina", None, None, "2026-06-18"],
            ["Canada",              "Qatar",                None, None, "2026-06-18"],
            ["Switzerland",         "Canada",               None, None, "2026-06-24"],
            ["Bosnia & Herzegovina","Qatar",                None, None, "2026-06-24"],

            # ── Group C ──────────────────────────────────────────────────────
            ["Brazil",   "Morocco",  1, 1, "2026-06-13"],  # ✓
            ["Haiti",    "Scotland", 0, 1, "2026-06-13"],  # ✓
            ["Scotland", "Morocco",  None, None, "2026-06-19"],
            ["Brazil",   "Haiti",    None, None, "2026-06-19"],
            ["Scotland", "Brazil",   None, None, "2026-06-24"],
            ["Morocco",  "Haiti",    None, None, "2026-06-24"],

            # ── Group D ──────────────────────────────────────────────────────
            ["USA",       "Paraguay",  4, 1, "2026-06-12"],  # ✓
            ["Australia", "Turkey",    2, 0, "2026-06-13"],  # ✓
            ["USA",       "Australia", None, None, "2026-06-19"],
            ["Turkey",    "Paraguay",  None, None, "2026-06-19"],
            ["Turkey",    "USA",       None, None, "2026-06-25"],
            ["Paraguay",  "Australia", None, None, "2026-06-25"],

            # ── Group E ──────────────────────────────────────────────────────
            ["Germany",      "Curacao",      7, 1, "2026-06-14"],  # ✓
            ["Ivory Coast",  "Ecuador",      1, 0, "2026-06-14"],  # ✓
            ["Germany",      "Ivory Coast",  None, None, "2026-06-20"],
            ["Ecuador",      "Curacao",      None, None, "2026-06-20"],
            ["Ecuador",      "Germany",      None, None, "2026-06-25"],
            ["Curacao",      "Ivory Coast",  None, None, "2026-06-25"],

            # ── Group F ──────────────────────────────────────────────────────
            ["Netherlands",  "Japan",        2, 2, "2026-06-14"],  # ✓
            ["Sweden",       "Tunisia",      5, 1, "2026-06-14"],  # ✓
            ["Netherlands",  "Sweden",       None, None, "2026-06-20"],
            ["Tunisia",      "Japan",        None, None, "2026-06-21"],
            ["Japan",        "Sweden",       None, None, "2026-06-25"],
            ["Tunisia",      "Netherlands",  None, None, "2026-06-25"],

            # ── Group G ──────────────────────────────────────────────────────
            ["Belgium",     "Egypt",        1, 1, "2026-06-15"],  # ✓
            ["Iran",        "New Zealand",  2, 2, "2026-06-15"],  # ✓
            ["Belgium",     "Iran",         None, None, "2026-06-21"],
            ["New Zealand", "Egypt",        None, None, "2026-06-21"],
            ["New Zealand", "Belgium",      None, None, "2026-06-26"],
            ["Egypt",       "Iran",         None, None, "2026-06-26"],

            # ── Group H ──────────────────────────────────────────────────────
            ["Spain",        "Cape Verde",   0, 0, "2026-06-15"],  # ✓
            ["Saudi Arabia", "Uruguay",      1, 1, "2026-06-15"],  # ✓
            ["Spain",        "Saudi Arabia", None, None, "2026-06-21"],
            ["Uruguay",      "Cape Verde",   None, None, "2026-06-21"],
            ["Uruguay",      "Spain",        None, None, "2026-06-26"],
            ["Cape Verde",   "Saudi Arabia", None, None, "2026-06-26"],

            # ── Group I ──────────────────────────────────────────────────────
            ["France",   "Senegal",  None, None, "2026-06-16"],
            ["Iraq",     "Norway",   None, None, "2026-06-16"],
            ["France",   "Iraq",     None, None, "2026-06-22"],
            ["Norway",   "Senegal",  None, None, "2026-06-22"],
            ["Norway",   "France",   None, None, "2026-06-26"],
            ["Senegal",  "Iraq",     None, None, "2026-06-26"],

            # ── Group J ──────────────────────────────────────────────────────
            ["Argentina", "Algeria", None, None, "2026-06-16"],
            ["Austria",   "Jordan",  None, None, "2026-06-17"],
            ["Argentina", "Austria", None, None, "2026-06-22"],
            ["Jordan",    "Algeria", None, None, "2026-06-22"],
            ["Jordan",    "Argentina", None, None, "2026-06-27"],
            ["Algeria",   "Austria",   None, None, "2026-06-27"],

            # ── Group K ──────────────────────────────────────────────────────
            ["Portugal",  "DR Congo",   None, None, "2026-06-17"],
            ["Uzbekistan","Colombia",   None, None, "2026-06-17"],
            ["Portugal",  "Uzbekistan", None, None, "2026-06-23"],
            ["Colombia",  "DR Congo",   None, None, "2026-06-23"],
            ["Colombia",  "Portugal",   None, None, "2026-06-27"],
            ["DR Congo",  "Uzbekistan", None, None, "2026-06-27"],

            # ── Group L ──────────────────────────────────────────────────────
            ["England",  "Croatia", None, None, "2026-06-17"],
            ["Ghana",    "Panama",  None, None, "2026-06-17"],
            ["England",  "Ghana",   None, None, "2026-06-23"],
            ["Panama",   "Croatia", None, None, "2026-06-23"],
            ["Panama",   "England", None, None, "2026-06-27"],
            ["Croatia",  "Ghana",   None, None, "2026-06-27"],
        ],
    }

    dest = os.path.join(RAW_DIR, "wc2026_fixtures.json")
    with open(dest, "w") as f:
        json.dump(fixtures, f, indent=2)
    print(f"    Saved → {dest}")


if __name__ == "__main__":
    main()
