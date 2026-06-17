"""
data/fetch_results.py
---------------------
Fetches completed 2026 World Cup match results from ESPN's public API,
deduplicates against data/live/real_results.json, and appends new ones.

Called automatically by models/simulator.py at startup, or run directly:
    python3 data/fetch_results.py

Returns (and prints) the list of matches newly added since the last run.
"""

import os
import sys
import json
import urllib.request
from datetime import date

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DIR  = os.path.join(BASE, "data", "live")
SNAP_PATH = os.path.join(LIVE_DIR, "last_run_snapshot.json")   # tracks last-run match keys
RESULTS_PATH = os.path.join(LIVE_DIR, "real_results.json")

os.makedirs(LIVE_DIR, exist_ok=True)

# Tournament start date
WC_START = "20260611"

# ── Name normalizer: ESPN displayName → our canonical team names ─────────────
# Canonical names are defined by data/raw/wc2026_fixtures.json (Groups A–L).
ESPN_NAME_MAP = {
    # North America / CONCACAF
    "United States":                   "USA",
    "Trinidad and Tobago":             "Trinidad & Tobago",
    "Curaçao":                         "Curacao",
    # Europe
    "Türkiye":                         "Turkey",
    "Turkiye":                         "Turkey",
    "Czech Republic":                  "Czechia",
    "Bosnia and Herzegovina":          "Bosnia & Herzegovina",
    # Africa
    "Côte d'Ivoire":                   "Ivory Coast",
    "Cote d'Ivoire":                   "Ivory Coast",
    "DR Congo":                        "DR Congo",
    "Congo DR":                        "DR Congo",
    "Democratic Republic of Congo":    "DR Congo",
    "Congo, DR":                       "DR Congo",
    "Cape Verde Islands":              "Cape Verde",
    "Cabo Verde":                      "Cape Verde",
    # Asia
    "Korea Republic":                  "South Korea",
    "Republic of Korea":               "South Korea",
    "IR Iran":                         "Iran",
    "New Zealand":                     "New Zealand",
}


def normalize(name: str) -> str:
    """Map ESPN team name to our internal canonical name."""
    return ESPN_NAME_MAP.get(name, name)


def match_key(home: str, away: str) -> str:
    """Order-independent key for deduplication."""
    return f"{min(home, away)}||{max(home, away)}"


# ── Load / save ──────────────────────────────────────────────────────────────

def load_existing_results() -> list:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results: list) -> None:
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def load_last_snapshot() -> set:
    """Keys seen during the PREVIOUS simulator run."""
    if os.path.exists(SNAP_PATH):
        with open(SNAP_PATH) as f:
            return set(json.load(f))
    return set()


def save_snapshot(keys: set) -> None:
    with open(SNAP_PATH, "w") as f:
        json.dump(sorted(keys), f, indent=2)


# ── ESPN fetch ───────────────────────────────────────────────────────────────

def fetch_espn_results() -> list:
    """
    Pulls all completed WC2026 matches from ESPN's public scoreboard API.
    Returns a list of dicts: {home, away, home_goals, away_goals, date}.
    Raises on network error so caller can fall back gracefully.
    """
    today = date.today().strftime("%Y%m%d")
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
        f"scoreboard?dates={WC_START}-{today}&limit=200"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)

    matches = []
    for event in data.get("events", []):
        comp = (event.get("competitions") or [{}])[0]

        # Only process completed matches
        status = comp.get("status", {}).get("type", {})
        if not status.get("completed", False):
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        # Identify home / away
        home_c = next(
            (c for c in competitors if c.get("homeAway") == "home"),
            competitors[0],
        )
        away_c = next(
            (c for c in competitors if c.get("homeAway") == "away"),
            competitors[1],
        )

        home = normalize(home_c["team"]["displayName"])
        away = normalize(away_c["team"]["displayName"])

        try:
            home_goals = int(home_c.get("score", 0))
            away_goals = int(away_c.get("score", 0))
        except (ValueError, TypeError):
            continue

        match_date = (event.get("date") or "")[:10]
        matches.append({
            "home": home, "away": away,
            "home_goals": home_goals, "away_goals": away_goals,
            "date": match_date,
            "source": "espn_auto",
        })

    return matches


# ── Main entry ───────────────────────────────────────────────────────────────

def update_results(verbose: bool = True) -> dict:
    """
    Fetch latest ESPN results, deduplicate, persist, and return a report dict:
      {
        "newly_added":   [list of new match dicts],
        "already_known": [list of dicts that were already in the file],
        "total":         int,
      }
    """
    existing   = load_existing_results()
    last_snap  = load_last_snapshot()

    # Build a set of keys already on disk (normalize names for safety)
    existing_keys = set()
    for r in existing:
        h = normalize(r.get("home", ""))
        a = normalize(r.get("away", ""))
        existing_keys.add(match_key(h, a))

    # ── Fetch from ESPN ───────────────────────────────────────────────────────
    try:
        fetched = fetch_espn_results()
    except Exception as exc:
        if verbose:
            print(f"  ⚠️  ESPN fetch failed: {exc}")
            print("     Using existing real_results.json without updates.")
        fetched = []

    newly_added   = []
    already_known = []

    for m in fetched:
        key = match_key(m["home"], m["away"])
        if key in existing_keys:
            already_known.append(m)
        else:
            newly_added.append(m)
            existing.append(m)
            existing_keys.add(key)

    # Remove duplicate entries in the file (can happen if script was run twice)
    seen_keys = set()
    deduped = []
    for r in existing:
        h = normalize(r.get("home", ""))
        a = normalize(r.get("away", ""))
        k = match_key(h, a)
        if k not in seen_keys:
            deduped.append(r)
            seen_keys.add(k)
    existing = deduped

    # Persist updated file
    save_results(existing)

    # Compute "new since last run" = in existing_keys but not in last_snap
    new_since_last = [
        r for r in existing
        if match_key(normalize(r.get("home","")), normalize(r.get("away","")))
        not in last_snap
    ]

    return {
        "newly_added":      newly_added,
        "new_since_last":   new_since_last,
        "already_known":    already_known,
        "total":            len(existing),
        "all_results":      existing,
    }


def save_run_snapshot(results: list) -> None:
    """Call this AFTER a simulation run to mark the current set as 'seen'."""
    keys = {
        match_key(normalize(r.get("home","")), normalize(r.get("away","")))
        for r in results
    }
    save_snapshot(keys)


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔄 Fetching latest WC2026 results from ESPN...\n")
    report = update_results(verbose=True)

    if report["newly_added"]:
        print(f"  ✅ {len(report['newly_added'])} new match(es) added:\n")
        for m in report["newly_added"]:
            hg, ag = m["home_goals"], m["away_goals"]
            print(f"     {m['home']} {hg}–{ag} {m['away']}  ({m.get('date','')})")
    else:
        print("  ✓ No new matches — already up to date.")

    print(f"\n  📋 Total matches on file: {report['total']}")
