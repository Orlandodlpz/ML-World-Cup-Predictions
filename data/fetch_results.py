"""
data/fetch_results.py
---------------------
Fetches 2026 World Cup match data from ESPN's public API for the FULL
tournament (group stage + all knockout rounds, June 11 to July 19):

  1. Completed matches  -> merged into data/live/real_results.json
       Each record now carries: round, status (FT / AET / PEN),
       penalty shootout scores, and the advancing team ("winner").
  2. Scheduled matches  -> written to data/live/upcoming_fixtures.json
       Used by the simulator to build the real knockout bracket
       (e.g. actual Quarter-Final pairings once the R16 is decided).

Knockout care: a match that ends level after 120 minutes is decided on
penalties. ESPN reports the shootout separately (shootoutScore), so we
record BOTH the match score and the shootout, plus the team that
actually advanced. Elo/goal calibration keeps treating a PEN match as a
draw on the scoreboard, but the bracket advances the real winner.

Called automatically by models/simulator.py at startup, or run directly:
    python3 data/fetch_results.py
"""

import os
import sys
import json
import urllib.request
from datetime import date

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DIR  = os.path.join(BASE, "data", "live")
SNAP_PATH = os.path.join(LIVE_DIR, "last_run_snapshot.json")   # tracks last-run match keys
RESULTS_PATH  = os.path.join(LIVE_DIR, "real_results.json")
UPCOMING_PATH = os.path.join(LIVE_DIR, "upcoming_fixtures.json")

os.makedirs(LIVE_DIR, exist_ok=True)

# Tournament window (full tournament, not just the group stage)
WC_START = "20260611"
WC_END   = "20260719"   # date of the final

# ESPN season.slug -> our canonical round names
ROUND_NAMES = {
    "group-stage":     "Group Stage",
    "round-of-32":     "Round of 32",
    "round-of-16":     "Round of 16",
    "quarterfinals":   "Quarter-Finals",
    "semifinals":      "Semi-Finals",
    "3rd-place-match": "Third Place",
    "final":           "Final",
}

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
    "Bosnia-Herzegovina":              "Bosnia & Herzegovina",
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

# ESPN uses placeholder "teams" for undecided knockout slots
PLACEHOLDER_HINTS = ("Winner", "Loser", "TBD", "To Be Determined")


def normalize(name: str) -> str:
    """Map ESPN team name to our internal canonical name."""
    return ESPN_NAME_MAP.get(name, name)


def is_placeholder(name: str) -> bool:
    return any(h in name for h in PLACEHOLDER_HINTS)


def match_key(home: str, away: str, match_date: str = "") -> str:
    """
    Order-independent key for deduplication.
    The date is part of the key because two teams CAN meet twice in one
    World Cup (group stage + a knockout round).
    """
    return f"{min(home, away)}||{max(home, away)}||{match_date or ''}"


def pair_key(home: str, away: str) -> str:
    """Legacy order-independent pair key (no date)."""
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


def save_upcoming(fixtures: list) -> None:
    with open(UPCOMING_PATH, "w") as f:
        json.dump(fixtures, f, indent=2)


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

def fetch_espn_events() -> tuple:
    """
    Pulls ALL WC2026 events (completed + scheduled) from ESPN's public
    scoreboard API for the full tournament window.

    Returns (completed, scheduled):
      completed: [{home, away, home_goals, away_goals, date, round,
                   status, home_pens?, away_pens?, winner?}, ...]
      scheduled: [{home, away, date, round, placeholder}, ...]
    Raises on network error so caller can fall back gracefully.
    """
    end = max(WC_END, date.today().strftime("%Y%m%d"))
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
        f"scoreboard?dates={WC_START}-{end}&limit=350"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)

    completed, scheduled = [], []

    for event in data.get("events", []):
        comp = (event.get("competitions") or [{}])[0]

        slug       = (event.get("season") or {}).get("slug", "")
        round_name = ROUND_NAMES.get(slug)
        match_date = (event.get("date") or "")[:10]

        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        home_c = next((c for c in competitors if c.get("homeAway") == "home"),
                      competitors[0])
        away_c = next((c for c in competitors if c.get("homeAway") == "away"),
                      competitors[1])

        home_raw = home_c["team"]["displayName"]
        away_raw = away_c["team"]["displayName"]
        home = normalize(home_raw)
        away = normalize(away_raw)

        status_type = comp.get("status", {}).get("type", {})

        if not status_type.get("completed", False):
            # Scheduled (or in-progress) match — keep for bracket building
            scheduled.append({
                "home": home, "away": away,
                "date": match_date,
                "round": round_name,
                "placeholder": is_placeholder(home_raw) or is_placeholder(away_raw),
            })
            continue

        try:
            home_goals = int(home_c.get("score", 0))
            away_goals = int(away_c.get("score", 0))
        except (ValueError, TypeError):
            continue

        # Status: FT (90'), AET (extra time), PEN (penalty shootout)
        status_name = status_type.get("name", "")
        if status_name == "STATUS_FINAL_PEN":
            status = "PEN"
        elif status_name == "STATUS_FINAL_AET":
            status = "AET"
        else:
            status = "FT"

        record = {
            "home": home, "away": away,
            "home_goals": home_goals, "away_goals": away_goals,
            "date": match_date,
            "round": round_name,
            "status": status,
            "source": "espn_auto",
        }

        # Knockout matches cannot end level: record who actually advanced.
        if round_name and round_name != "Group Stage":
            if status == "PEN":
                try:
                    record["home_pens"] = int(home_c.get("shootoutScore"))
                    record["away_pens"] = int(away_c.get("shootoutScore"))
                except (ValueError, TypeError):
                    pass
            if home_c.get("winner"):
                record["winner"] = home
            elif away_c.get("winner"):
                record["winner"] = away
            elif "home_pens" in record and "away_pens" in record:
                record["winner"] = home if record["home_pens"] > record["away_pens"] else away
            elif home_goals != away_goals:
                record["winner"] = home if home_goals > away_goals else away

        completed.append(record)

    return completed, scheduled


# ── Main entry ───────────────────────────────────────────────────────────────

def update_results(verbose: bool = True) -> dict:
    """
    Fetch latest ESPN results, merge into real_results.json (enriching any
    older entries in place), write upcoming_fixtures.json, and return:
      {
        "newly_added":    [new match dicts],
        "new_since_last": [matches not seen by the last simulator run],
        "total":          int,
        "all_results":    [every match on file],
        "upcoming":       [scheduled fixtures],
      }
    """
    existing  = load_existing_results()
    last_snap = load_last_snapshot()

    # Normalize names on everything already on disk (fixes old aliases
    # like "Bosnia-Herzegovina" saved before the alias map was extended).
    for r in existing:
        r["home"] = normalize(r.get("home", ""))
        r["away"] = normalize(r.get("away", ""))

    # Index existing entries.
    # Group-stage pairings are unique, so group matches merge by pair alone.
    # Knockout matches merge by pair + date (two teams can meet twice).
    dated_index = {}   # (pair, date) -> record
    group_index = {}   # pair -> record (round is Group Stage or unknown/legacy)
    for r in existing:
        p = pair_key(r["home"], r["away"])
        if r.get("date"):
            dated_index[(p, r["date"])] = r
        if r.get("round") in (None, "Group Stage"):
            group_index.setdefault(p, r)

    # ── Fetch from ESPN ───────────────────────────────────────────────────────
    try:
        fetched, upcoming = fetch_espn_events()
    except Exception as exc:
        if verbose:
            print(f"  ⚠️  ESPN fetch failed: {exc}")
            print("     Using existing real_results.json without updates.")
        fetched, upcoming = [], None

    newly_added = []

    for m in fetched:
        p = pair_key(m["home"], m["away"])
        if m.get("round") == "Group Stage":
            target = group_index.get(p) or dated_index.get((p, m["date"]))
        else:
            target = dated_index.get((p, m["date"]))

        if target is not None:
            # Update / enrich existing record, keep any extra legacy fields
            target.update(m)
            dated_index[(p, m["date"])] = target
            group_index.setdefault(p, target)
        else:
            existing.append(m)
            dated_index[(p, m["date"])] = m
            if m.get("round") == "Group Stage":
                group_index[p] = m
            newly_added.append(m)

    # Final dedup pass: group matches by pair, knockout by pair + date
    seen, deduped = set(), []
    for r in existing:
        if r.get("round") in (None, "Group Stage"):
            k = pair_key(r["home"], r["away"])
        else:
            k = match_key(r["home"], r["away"], r.get("date", ""))
        if k not in seen:
            deduped.append(r)
            seen.add(k)
    existing = sorted(deduped, key=lambda r: (r.get("date") or "0000-00-00"))

    save_results(existing)
    if upcoming is not None:
        save_upcoming(upcoming)
    elif os.path.exists(UPCOMING_PATH):
        with open(UPCOMING_PATH) as f:
            upcoming = json.load(f)
    else:
        upcoming = []

    # "New since last run": lenient — accept legacy pair-only snapshot keys
    new_since_last = []
    for r in existing:
        k_dated = match_key(r["home"], r["away"], r.get("date", ""))
        k_pair  = pair_key(r["home"], r["away"])
        if k_dated not in last_snap and k_pair not in last_snap:
            new_since_last.append(r)

    return {
        "newly_added":    newly_added,
        "new_since_last": new_since_last,
        "total":          len(existing),
        "all_results":    existing,
        "upcoming":       upcoming,
    }


def save_run_snapshot(results: list) -> None:
    """Call this AFTER a simulation run to mark the current set as 'seen'."""
    keys = {
        match_key(normalize(r.get("home", "")), normalize(r.get("away", "")),
                  r.get("date", ""))
        for r in results
    }
    save_snapshot(keys)


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔄 Fetching latest WC2026 results from ESPN (full tournament)...\n")
    report = update_results(verbose=True)

    if report["newly_added"]:
        print(f"  ✅ {len(report['newly_added'])} new match(es) added:\n")
        for m in report["newly_added"]:
            hg, ag = m["home_goals"], m["away_goals"]
            extra = ""
            if m.get("status") == "PEN":
                extra = f"  ({m.get('winner','?')} win {m.get('home_pens','?')}-{m.get('away_pens','?')} on pens)"
            elif m.get("status") == "AET":
                extra = "  (AET)"
            rd = f" [{m['round']}]" if m.get("round") else ""
            print(f"     {m['home']} {hg}-{ag} {m['away']}{extra}  ({m.get('date','')}){rd}")
    else:
        print("  ✓ No new matches — already up to date.")

    print(f"\n  📋 Total matches on file: {report['total']}")

    real_upcoming = [u for u in report.get("upcoming", []) if not u.get("placeholder")]
    if real_upcoming:
        print(f"\n  📅 Next scheduled matches with known pairings:")
        for u in real_upcoming[:8]:
            print(f"     {u['home']} vs {u['away']}  ({u.get('date','')})  [{u.get('round','')}]")
