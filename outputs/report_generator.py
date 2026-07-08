"""
outputs/report_generator.py
----------------------------
Generates a self-contained HTML predictions report for the 2026 FIFA World Cup.

Includes:
  - Stage-by-stage breakdown (R32 → Final)
  - Group stage current standings (from known results)
  - Upcoming match predictions with clean probability pills

Run:
    python outputs/report_generator.py

Output:
    outputs/wc2026_predictions.html   ← open in any browser
"""

import os
import sys
import json
from collections import defaultdict
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)

RAW       = os.path.join(BASE, "data", "raw")
PROCESSED = os.path.join(BASE, "data", "processed")
LIVE_DIR  = os.path.join(BASE, "data", "live")
OUTPUTS   = os.path.join(BASE, "outputs")

from utils.flags import flag, TEAM_FLAGS


# ── Name normalization (mirrors fetch_results.py) ─────────────────────────────

_TEAM_ALIASES = {
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Bosnia-Herzegovina":     "Bosnia & Herzegovina",
    "Bosnia And Herzegovina": "Bosnia & Herzegovina",
    "United States":          "USA",
    "Türkiye":                "Turkey",
    "Turkiye":                "Turkey",
    "Czech Republic":         "Czechia",
    "Korea Republic":         "South Korea",
    "Republic of Korea":      "South Korea",
    "IR Iran":                "Iran",
    "Curaçao":                "Curacao",
    "Cape Verde Islands":     "Cape Verde",
    "Cabo Verde":             "Cape Verde",
    "Côte d'Ivoire":          "Ivory Coast",
    "Cote d'Ivoire":          "Ivory Coast",
    "Congo DR":               "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
}

def _norm(name: str) -> str:
    return _TEAM_ALIASES.get(name, name)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_fixtures():
    with open(os.path.join(RAW, "wc2026_fixtures.json")) as f:
        return json.load(f)


def load_results():
    path = os.path.join(OUTPUTS, "simulation_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_team_stats():
    path = os.path.join(PROCESSED, "team_stats.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def build_live_lookup():
    """
    Load real_results.json and return a dict keyed by frozenset({home, away})
    → (home_name, away_name, home_goals, away_goals).

    This fills in scores for matches that have been played but whose fixture
    entry still has null (e.g. ESPN-fetched results added after the fixture
    file was written).
    """
    path = os.path.join(LIVE_DIR, "real_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        live = json.load(f)

    lookup = {}
    seen   = set()
    for r in live:
        h  = _norm(r.get("home", ""))
        a  = _norm(r.get("away", ""))
        hg = r.get("home_goals")
        ag = r.get("away_goals")
        if not h or not a or hg is None or ag is None:
            continue
        key = frozenset([h, a])
        if key in seen:
            continue   # deduplicate
        seen.add(key)
        lookup[key] = (h, a, int(hg), int(ag))
    return lookup


def _resolve_score(home, away, fix_hs, fix_as, live_lookup):
    """
    Return (home_goals, away_goals) using the fixture score if known,
    otherwise falling back to the live lookup.  Returns (None, None) if
    neither source has a result yet.
    """
    if fix_hs is not None and fix_as is not None:
        return int(fix_hs), int(fix_as)
    key = frozenset([_norm(home), _norm(away)])
    if key in live_lookup:
        lh, la, lhg, lag = live_lookup[key]
        # Re-align goals to the fixture's home/away order
        if _norm(home) == lh:
            return lhg, lag
        else:
            return lag, lhg   # live result had teams reversed
    return None, None


# ── Group standings from known results ────────────────────────────────────────

def compute_standings(fixtures, live_lookup):
    groups        = fixtures["groups"]
    group_matches = fixtures["group_matches"]

    points = defaultdict(int)
    gd     = defaultdict(int)
    gf     = defaultdict(int)
    ga     = defaultdict(int)
    played = defaultdict(int)
    wins   = defaultdict(int)
    draws  = defaultdict(int)
    losses = defaultdict(int)

    for match in group_matches:
        home, away, fix_hs, fix_as = match[0], match[1], match[2], match[3]
        hs, as_ = _resolve_score(home, away, fix_hs, fix_as, live_lookup)
        if hs is None:
            continue
        played[home] += 1; played[away] += 1
        gf[home] += hs;    gf[away] += as_
        ga[home] += as_;   ga[away] += hs
        gd[home] += hs - as_; gd[away] += as_ - hs
        if hs > as_:
            points[home] += 3; wins[home] += 1; losses[away] += 1
        elif hs == as_:
            points[home] += 1; points[away] += 1
            draws[home]  += 1; draws[away]  += 1
        else:
            points[away] += 3; wins[away] += 1; losses[home] += 1

    standings = {}
    for grp, teams in groups.items():
        ranked = sorted(teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True)
        standings[grp] = [
            {
                "team": t, "flag": flag(t),
                "p": played[t], "w": wins[t], "d": draws[t], "l": losses[t],
                "gf": gf[t], "ga": ga[t], "gd": gd[t], "pts": points[t],
            }
            for t in ranked
        ]
    return standings


# ── Upcoming matches ───────────────────────────────────────────────────────────

def get_upcoming(fixtures, team_stats, live_lookup, limit=30):
    matches = []
    for m in fixtures["group_matches"]:
        home, away, fix_hs, fix_as = m[0], m[1], m[2], m[3]
        date = m[4] if len(m) > 4 else "TBD"
        hs, as_ = _resolve_score(home, away, fix_hs, fix_as, live_lookup)
        if hs is not None:
            continue   # already played — skip
        h_elo = (team_stats or {}).get(home, {}).get("elo", 1700)
        a_elo = (team_stats or {}).get(away, {}).get("elo", 1700)
        diff    = h_elo - a_elo
        e_h     = 1 / (1 + 10 ** (-diff / 400))
        # Draw probability shrinks as the Elo gap grows.
        # Equal teams (~gap 0): ~28%  |  Gap 200: ~22%  |  Gap 400: ~16%  |  Gap 600+: ~10%
        draw    = max(0.10, 0.28 - abs(diff) * 0.0003)
        hw      = round(e_h * (1 - draw), 3)
        aw      = round((1 - e_h) * (1 - draw), 3)
        draw    = round(draw, 3)
        matches.append({
            "home": home, "away": away, "date": date,
            "home_flag": flag(home), "away_flag": flag(away),
            "hw": hw, "draw": draw, "aw": aw,
        })
    return matches[:limit]


# ── Knockout bracket rendering ─────────────────────────────────────────────────

def _fmt_date(iso: str) -> str:
    """2026-07-09 → Jul 9 (falls back to raw string)."""
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%b %-d")
    except Exception:
        return iso or "TBD"


def _team_label(name, label, flag_first=True):
    """Team name with flag, or a muted placeholder like 'Winner QF1'."""
    if name:
        return f"{flag(name)} {name}" if flag_first else f"{name} {flag(name)}"
    return f'<span class="tbd">{label}</span>'


def _ko_match_html(m):
    """One knockout match card: real result, prediction, or TBD placeholder."""
    date = _fmt_date(m.get("date", ""))

    if m.get("played"):
        w = m.get("winner")
        h_cls = "winner" if w == m["home"] else "loser"
        a_cls = "winner" if w == m["away"] else "loser"
        note = ""
        if m.get("status") == "PEN":
            wp = max(m.get("home_pens") or 0, m.get("away_pens") or 0)
            lp = min(m.get("home_pens") or 0, m.get("away_pens") or 0)
            note = f"{w} advance {wp}-{lp} on penalties"
        elif m.get("status") == "AET":
            note = f"{w} win after extra time"
        note_html = f'<div class="ko-note">🕐 {note}</div>' if note else ""
        return f"""
        <div class="ko-match played">
          <div class="ko-line">
            <span class="ko-team {h_cls}">{_team_label(m["home"], m["home_label"])}</span>
            <span class="ko-score">{m["home_goals"]} - {m["away_goals"]}</span>
            <span class="ko-team right {a_cls}">{_team_label(m["away"], m["away_label"], False)}</span>
          </div>
          {note_html}
          <div class="ko-date">✅ {date}</div>
        </div>"""

    if m.get("home") and m.get("away") and m.get("p_home") is not None:
        ph = m["p_home"] * 100
        pa = 100 - ph
        return f"""
        <div class="ko-match upcoming">
          <div class="ko-line">
            <span class="ko-team">{_team_label(m["home"], m["home_label"])}</span>
            <span class="ko-score vs">vs</span>
            <span class="ko-team right">{_team_label(m["away"], m["away_label"], False)}</span>
          </div>
          <div class="prob-bar-stacked ko-bar">
            <div class="seg seg-home" style="width:{ph:.1f}%"></div>
            <div class="seg seg-away" style="width:{pa:.1f}%"></div>
          </div>
          <div class="ko-pcts"><span>{ph:.0f}%</span><span>{pa:.0f}%</span></div>
          <div class="ko-date">📅 {date}</div>
        </div>"""

    return f"""
    <div class="ko-match tbd-match">
      <div class="ko-line">
        <span class="ko-team">{_team_label(m.get("home"), m.get("home_label", "TBD"))}</span>
        <span class="ko-score vs">vs</span>
        <span class="ko-team right">{_team_label(m.get("away"), m.get("away_label", "TBD"))}</span>
      </div>
      <div class="ko-date">📅 {date}</div>
    </div>"""


def _pred_panel_html(pred):
    """
    Match prediction panel: expected goals, most likely scorelines, and how
    the match ends. The three path percentages (decided in 90 minutes,
    decided in extra time, penalties) always sum to 100.
    """
    if not pred:
        return ""
    home, away = pred["home"], pred["away"]
    p_reg  = pred["p_regulation"] * 100
    p_etd  = (pred["p_home_et"] + pred["p_away_et"]) * 100
    p_pens = pred["p_penalties"] * 100
    scores = " · ".join(
        f'<span class="score-chip">{s["score"]} <em>{s["prob"]*100:.0f}%</em></span>'
        for s in pred.get("top_scorelines", [])[:3]
    )
    return f"""
          <div class="pred-panel">
            <div class="pred-row xg-row">
              <span class="pred-label">⚽ Expected goals</span>
              <span>{flag(home)} <strong>{pred["home_xg"]:.1f}</strong>
                &nbsp;·&nbsp; {flag(away)} <strong>{pred["away_xg"]:.1f}</strong></span>
            </div>
            <div class="pred-row">
              <span class="pred-label">🎯 Likely scorelines</span>
              <span>{scores}</span>
            </div>
            <div class="path-pills">
              <div class="pill path-pill">
                <span class="pill-label">Decided in 90&prime;</span>
                <span class="pill-pct">{p_reg:.0f}%</span>
              </div>
              <div class="pill path-pill">
                <span class="pill-label">Extra time winner</span>
                <span class="pill-pct">{p_etd:.0f}%</span>
              </div>
              <div class="pill path-pill pens">
                <span class="pill-label">Penalties 🎲</span>
                <span class="pill-pct">{p_pens:.0f}%</span>
              </div>
            </div>
          </div>"""


def build_bracket_html(bracket):
    """Round-by-round knockout view with real results and live predictions."""
    icons = {"Round of 32": "3️⃣2️⃣", "Round of 16": "1️⃣6️⃣",
             "Quarter-Finals": "🎯", "Semi-Finals": "🔥", "Final": "🏆"}
    html = ""
    for rnd in bracket.get("rounds", []):
        name    = rnd["name"]
        matches = rnd["matches"]
        played  = sum(1 for m in matches if m.get("played"))
        status  = ("all played ✅" if played == len(matches)
                   else f"{played}/{len(matches)} played")
        html += (
            f'<div class="ko-round">'
            f'<h3>{icons.get(name, "⚽")} {name} '
            f'<span class="ko-status">{status}</span></h3>'
            f'<div class="ko-grid">'
        )
        for m in matches:
            html += _ko_match_html(m)
        html += "</div></div>"

    third = bracket.get("third_place")
    if third:
        html += (
            '<div class="ko-round"><h3>🥉 Third Place Match '
            '<span class="ko-status">display only</span></h3>'
            f'<div class="ko-grid">{_ko_match_html(third)}</div></div>'
        )
    return html


def get_remaining_knockout(bracket):
    """Unplayed knockout matches with known teams (for the upcoming section)."""
    remaining = []
    for rnd in bracket.get("rounds", []):
        for m in rnd["matches"]:
            if not m.get("played") and m.get("home") and m.get("away") \
                    and m.get("p_home") is not None:
                remaining.append({**m, "round": rnd["name"]})
    remaining.sort(key=lambda m: m.get("date", ""))
    return remaining


# ── HTML builder ───────────────────────────────────────────────────────────────

def build_html(fixtures, sim_results, standings, upcoming, team_stats):
    now = datetime.now().strftime("%B %d, %Y %H:%M")

    meta     = (sim_results or {}).get("meta", {})
    bracket  = (sim_results or {}).get("bracket")
    knockout = bool(bracket) and meta.get("phase") == "knockout"

    # Teams that actually advanced from the group stage (knockout phase)
    advanced32 = set()
    if knockout:
        advanced32 = {
            t for t, p in sim_results.get("Group Stage", {}).items() if p >= 0.999
        }

    # ── Champion probability rows ──────────────────────────────────────────────
    champ_rows = ""
    if sim_results:
        champ   = sim_results.get("Champion", {})
        final_p = sim_results.get("Final", {})
        sf_p    = sim_results.get("Semi-Finals", {})
        qf_p    = sim_results.get("Quarter-Finals", {})

        for i, (team, prob) in enumerate(list(champ.items())[:24], 1):
            pct   = prob * 100
            bar_w = min(100, pct * 4)
            f     = flag(team)
            medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"{i}."
            fp    = final_p.get(team, 0) * 100
            sfp   = sf_p.get(team, 0) * 100
            qfp   = qf_p.get(team, 0) * 100
            champ_rows += f"""
            <tr>
              <td class="rank">{medal}</td>
              <td class="team-name">{f} {team}</td>
              <td class="prob-cell">
                <div class="prob-bar-track">
                  <div class="prob-bar" style="width:{bar_w:.1f}%"></div>
                </div>
                <span class="prob-label">{pct:.1f}%</span>
              </td>
              <td class="stage-prob">{fp:.1f}%</td>
              <td class="stage-prob">{sfp:.1f}%</td>
              <td class="stage-prob">{qfp:.1f}%</td>
            </tr>"""
    else:
        champ_rows = (
            '<tr><td colspan="6" class="no-data">'
            "Run <code>python models/simulator.py</code> first to generate predictions."
            "</td></tr>"
        )

    # ── Group standings ────────────────────────────────────────────────────────
    # Each card wraps its table in overflow-x:auto so long team names never break layout
    group_html = ""
    for grp, rows in sorted(standings.items()):
        group_html += (
            f'<div class="group-card">'
            f'<h3>Group {grp}</h3>'
            f'<div class="table-scroll">'
            f'<table class="standings-table">'
            f'<thead><tr>'
            f'<th></th><th class="col-team">Team</th>'
            f'<th title="Played">P</th><th title="Won">W</th>'
            f'<th title="Drawn">D</th><th title="Lost">L</th>'
            f'<th title="Goals For">GF</th><th title="Goals Against">GA</th>'
            f'<th title="Goal Difference">GD</th><th title="Points">Pts</th>'
            f'</tr></thead><tbody>'
        )
        for idx, r in enumerate(rows):
            if knockout:
                # Group stage is over — highlight who ACTUALLY advanced
                cls = "qualify" if r["team"] in advanced32 else ""
            else:
                cls = "qualify" if idx < 2 else ("third" if idx == 2 else "")
            gd_str = f"+{r['gd']}" if r['gd'] > 0 else str(r['gd'])
            group_html += (
                f'<tr class="{cls}">'
                f'<td class="col-flag">{r["flag"]}</td>'
                f'<td class="col-team" title="{r["team"]}">{r["team"]}</td>'
                f'<td>{r["p"]}</td><td>{r["w"]}</td>'
                f'<td>{r["d"]}</td><td>{r["l"]}</td>'
                f'<td>{r["gf"]}</td><td>{r["ga"]}</td>'
                f'<td class="col-gd">{gd_str}</td>'
                f'<td class="col-pts"><strong>{r["pts"]}</strong></td>'
                f'</tr>'
            )
        group_html += "</tbody></table></div></div>"

    # ── Upcoming matches ───────────────────────────────────────────────────────
    # Bar segments are now purely visual — NO text inside them.
    # Percentages live in clean pill badges below each bar.
    # Knockout phase: predictions come from the live bracket (no draw pill,
    # knockout matches always produce a winner).
    if knockout:
        upcoming = []   # every group match has been played
        remaining = get_remaining_knockout(bracket)
        upcoming_html = ""
        current_date  = None
        for m in remaining:
            if m["date"] != current_date:
                if current_date is not None:
                    upcoming_html += "</div>"
                upcoming_html += (
                    f'<div class="date-section">'
                    f'<h4>📅 {m["date"]} · {m["round"]}</h4>'
                )
                current_date = m["date"]
            hw = int(round(m["p_home"] * 100))
            aw = 100 - hw
            upcoming_html += f"""
        <div class="match-card">
          <div class="match-teams">
            <span class="team-a">{flag(m["home"])} {m["home"]}</span>
            <span class="vs">vs</span>
            <span class="team-b">{m["away"]} {flag(m["away"])}</span>
          </div>
          <div class="prob-bar-stacked">
            <div class="seg seg-home" style="width:{hw}%"></div>
            <div class="seg seg-away" style="width:{aw}%"></div>
          </div>
          <div class="outcome-pills">
            <div class="pill pill-home">
              <span class="pill-label">{m["home"]} advance</span>
              <span class="pill-pct">{hw}%</span>
            </div>
            <div class="pill pill-away">
              <span class="pill-label">{m["away"]} advance</span>
              <span class="pill-pct">{aw}%</span>
            </div>
          </div>
          {_pred_panel_html(m.get("pred"))}
        </div>"""
        if current_date:
            upcoming_html += "</div>"
        if not remaining:
            upcoming_html = (
                '<p class="no-data">No remaining matches with known pairings — '
                'check the bracket above for upcoming slots.</p>'
            )
        return _finish_html(now, champ_rows, group_html, upcoming_html,
                            knockout=True, bracket=bracket, meta=meta)

    upcoming_html = ""
    current_date  = None
    for m in upcoming:
        if m["date"] != current_date:
            if current_date is not None:
                upcoming_html += "</div>"  # close previous date-section
            upcoming_html += f'<div class="date-section"><h4>📅 {m["date"]}</h4>'
            current_date = m["date"]

        hw_w = int(m["hw"] * 100)
        dr_w = int(m["draw"] * 100)
        aw_w = 100 - hw_w - dr_w   # ensure exactly 100%

        upcoming_html += f"""
        <div class="match-card">
          <div class="match-teams">
            <span class="team-a">{m["home_flag"]} {m["home"]}</span>
            <span class="vs">vs</span>
            <span class="team-b">{m["away"]} {m["away_flag"]}</span>
          </div>
          <div class="prob-bar-stacked">
            <div class="seg seg-home" style="width:{hw_w}%"></div>
            <div class="seg seg-draw" style="width:{dr_w}%"></div>
            <div class="seg seg-away" style="width:{aw_w}%"></div>
          </div>
          <div class="outcome-pills">
            <div class="pill pill-home">
              <span class="pill-label">Home win</span>
              <span class="pill-pct">{m['hw']:.0%}</span>
            </div>
            <div class="pill pill-draw">
              <span class="pill-label">Draw</span>
              <span class="pill-pct">{m['draw']:.0%}</span>
            </div>
            <div class="pill pill-away">
              <span class="pill-label">Away win</span>
              <span class="pill-pct">{m['aw']:.0%}</span>
            </div>
          </div>
        </div>"""

    if current_date:
        upcoming_html += "</div>"

    if not upcoming:
        upcoming_html = '<p class="no-data">No upcoming matches — all group stage matches have been played.</p>'

    return _finish_html(now, champ_rows, group_html, upcoming_html,
                        knockout=False, bracket=None, meta=meta)


def _finish_html(now, champ_rows, group_html, upcoming_html,
                 knockout, bracket, meta):
    """Assemble the final HTML page (shared by group and knockout phases)."""

    if knockout:
        phase_badge     = "Knockout Stage LIVE 🔴"
        bracket_section = f"""
  <!-- ── Knockout Bracket ────────────────────────────────────── -->
  <section>
    <h2>🗺️ Knockout Bracket (Live)</h2>
    <p class="phase-note">
      Real results fill the bracket as they happen. Remaining matches show
      the model's advance probabilities. Eliminated teams sit at 0% champion
      probability and no longer appear in the table above.
    </p>
    {build_bracket_html(bracket)}
  </section>
"""
        standings_title = "📊 Final Group Standings"
        standings_note  = (
            '<span><div class="dot dot-qualify"></div> '
            'Advanced to the Round of 32 (top 2 + best 8 third-place)</span>'
        )
        upcoming_title  = "📅 Remaining Match Predictions"
        upcoming_note   = (
            'Bar: <span style="color:#60a5fa">■ first team advances</span>'
            ' &nbsp;·&nbsp; <span style="color:#f87171">■ second team advances</span>'
            ' &nbsp;·&nbsp; Knockout matches cannot end in a draw:'
            ' level games go to extra time and penalties.'
        )
    else:
        phase_badge     = "Group Stage"
        bracket_section = ""
        standings_title = "📊 Group Stage Standings"
        standings_note  = (
            '<span><div class="dot dot-qualify"></div> Top 2 — qualify automatically</span>'
            '<span><div class="dot dot-third"></div> 3rd place — may qualify as best 8</span>'
        )
        upcoming_title  = "📅 Upcoming Match Predictions"
        upcoming_note   = (
            'Bar: <span style="color:#60a5fa">■ home win</span>'
            ' &nbsp;·&nbsp; <span style="color:#94a3b8">■ draw</span>'
            ' &nbsp;·&nbsp; <span style="color:#f87171">■ away win</span>'
            ' &nbsp;·&nbsp; Percentages shown in badges below each bar.'
        )

    # ── Full HTML ──────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🌍 2026 FIFA World Cup — ML Predictions</title>
<style>
  :root {{
    --bg:     #0a0f1e;
    --card:   #111827;
    --card2:  #1a2235;
    --accent: #3b82f6;
    --gold:   #f59e0b;
    --green:  #10b981;
    --red:    #ef4444;
    --text:   #e2e8f0;
    --muted:  #64748b;
    --border: #1e2d45;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh;
    line-height: 1.5;
  }}

  /* ── Header ── */
  header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    padding: 2.5rem 2rem;
    text-align: center;
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{ font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; }}
  header .subtitle {{ color: var(--gold); font-size: 1.4rem; font-weight: 700; margin-top: 0.3rem; }}
  header p  {{ color: var(--muted); margin-top: 0.5rem; font-size: 0.85rem; }}

  /* ── Layout ── */
  .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1rem; }}
  section {{ margin-bottom: 3rem; }}
  section > h2 {{
    font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 2px solid var(--accent);
  }}

  /* ── Champion probability table ── */
  .pred-table {{ width: 100%; border-collapse: collapse; }}
  .pred-table th {{
    padding: 0.55rem 0.75rem; text-align: left;
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--muted); border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  .pred-table td {{
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
    vertical-align: middle;
  }}
  .pred-table tr:hover td {{ background: var(--card2); }}
  .col-rank  {{ width: 2.5rem; font-size: 1.05rem; }}
  .col-tname {{ font-weight: 600; white-space: nowrap; }}
  .col-bar   {{ min-width: 200px; }}
  .col-stage {{ text-align: center; color: var(--muted); font-size: 0.8rem; white-space: nowrap; }}

  /* Champion prob bar — bar lives in a fixed-width track, label floats outside */
  .prob-cell {{ display: flex; align-items: center; gap: 0.6rem; }}
  .prob-bar-track {{
    width: 160px; flex-shrink: 0;
    height: 9px; border-radius: 5px;
    background: var(--card2); overflow: hidden;
  }}
  .prob-bar {{
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--gold));
    border-radius: 5px;
  }}
  .prob-label {{ font-weight: 700; color: var(--gold); font-size: 0.88rem; white-space: nowrap; flex-shrink: 0; }}

  /* ── Group standings ── */
  .legend {{
    display: flex; flex-wrap: wrap; gap: 1rem;
    margin-bottom: 0.75rem; font-size: 0.8rem; color: var(--muted);
  }}
  .legend span {{ display: flex; align-items: center; gap: 0.4rem; }}
  .dot {{ width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }}
  .dot-qualify {{ background: rgba(16,185,129,0.5); }}
  .dot-third   {{ background: rgba(245,158,11,0.4); }}

  .groups-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1rem;
  }}
  .group-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem;
    /* NOTE: no overflow:hidden here — it blocks child scroll */
  }}
  .group-card h3 {{ font-size: 0.95rem; font-weight: 700; margin-bottom: 0.6rem; color: var(--accent); }}

  /* scroll wrapper — table scrolls horizontally when wider than card */
  .table-scroll {{
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    border-radius: 6px;
  }}

  .standings-table {{
    min-width: 390px;      /* wider than the card → triggers horizontal scroll */
    border-collapse: collapse;
    font-size: 0.78rem;
    white-space: nowrap;
  }}
  .standings-table th {{
    color: var(--muted); padding: 0.2rem 0.35rem;
    text-align: center; border-bottom: 1px solid var(--border);
    font-weight: 600; font-size: 0.72rem;
  }}
  .standings-table .col-team {{ text-align: left; }}
  .standings-table td {{
    padding: 0.28rem 0.35rem; text-align: center;
  }}
  .standings-table td.col-team {{
    text-align: left; font-weight: 500;
    max-width: 130px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }}
  .standings-table td.col-flag  {{ font-size: 1rem; padding-right: 0.1rem; }}
  .standings-table td.col-gd    {{ font-weight: 500; }}
  .standings-table td.col-pts   {{ font-weight: 700; }}

  /* row colour bands */
  .standings-table tr.qualify {{ background: rgba(16,185,129,0.09); }}
  .standings-table tr.third   {{ background: rgba(245,158,11,0.07); }}
  .standings-table tbody tr:hover {{ background: var(--card2) !important; }}

  /* ── Upcoming match cards ── */
  .date-section > h4 {{
    color: var(--muted); font-size: 0.82rem;
    text-transform: uppercase; letter-spacing: 0.5px;
    margin: 1.2rem 0 0.5rem;
  }}
  .match-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.65rem;
  }}

  /* team names row */
  .match-teams {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.65rem; gap: 0.5rem;
  }}
  .team-a, .team-b {{ font-weight: 600; font-size: 0.92rem; white-space: nowrap; }}
  .vs {{ color: var(--muted); font-size: 0.78rem; flex-shrink: 0; }}

  /* purely visual stacked bar — NO text inside */
  .prob-bar-stacked {{
    display: flex; height: 14px; border-radius: 6px;
    overflow: hidden; margin-bottom: 0.55rem;
  }}
  .seg {{ height: 100%; flex-shrink: 0; }}
  .seg-home {{ background: var(--accent); }}
  .seg-draw {{ background: var(--muted); }}
  .seg-away {{ background: var(--red); }}

  /* three pill badges below the bar */
  .outcome-pills {{
    display: flex; gap: 0.4rem;
  }}
  .pill {{
    flex: 1; border-radius: 6px; padding: 0.3rem 0.4rem;
    text-align: center; font-size: 0.75rem; line-height: 1.4;
  }}
  .pill-label {{ display: block; color: var(--muted); font-size: 0.68rem; white-space: nowrap; }}
  .pill-pct   {{ display: block; font-weight: 700; font-size: 0.88rem; }}
  .pill-home  {{ background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.25); }}
  .pill-home .pill-pct {{ color: #60a5fa; }}
  .pill-draw  {{ background: rgba(100,116,139,0.12); border: 1px solid rgba(100,116,139,0.25); }}
  .pill-draw .pill-pct {{ color: #94a3b8; }}
  .pill-away  {{ background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.25); }}
  .pill-away .pill-pct {{ color: #f87171; }}

  /* ── Match prediction panel (xG, scorelines, how it ends) ── */
  .pred-panel {{
    margin-top: 0.6rem; padding-top: 0.6rem;
    border-top: 1px dashed var(--border);
  }}
  .pred-row {{
    display: flex; justify-content: space-between; align-items: center;
    gap: 0.6rem; font-size: 0.78rem; margin-bottom: 0.35rem;
    flex-wrap: wrap;
  }}
  .pred-label {{ color: var(--muted); font-size: 0.7rem; white-space: nowrap; }}
  .score-chip {{
    background: var(--card2); border: 1px solid var(--border);
    border-radius: 5px; padding: 0.1rem 0.4rem; font-weight: 700;
    font-size: 0.75rem; white-space: nowrap;
  }}
  .score-chip em {{ font-style: normal; font-weight: 500; color: var(--muted); font-size: 0.68rem; }}
  .path-pills {{ display: flex; gap: 0.4rem; margin-top: 0.45rem; }}
  .path-pill {{ background: rgba(100,116,139,0.10); border: 1px solid rgba(100,116,139,0.22); }}
  .path-pill .pill-pct {{ color: #cbd5e1; }}
  .path-pill.pens {{ background: rgba(234,179,8,0.10); border-color: rgba(234,179,8,0.25); }}
  .path-pill.pens .pill-pct {{ color: var(--gold); }}

  /* ── Knockout bracket ── */
  .phase-note {{ color: var(--muted); font-size: 0.82rem; margin-bottom: 1rem; }}
  .ko-round {{ margin-bottom: 1.6rem; }}
  .ko-round h3 {{
    font-size: 1rem; font-weight: 700; color: var(--accent);
    margin-bottom: 0.6rem;
  }}
  .ko-status {{
    color: var(--muted); font-size: 0.72rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.5px; margin-left: 0.5rem;
  }}
  .ko-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 0.6rem;
  }}
  .ko-match {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
  }}
  .ko-match.upcoming {{ border-color: rgba(59,130,246,0.35); }}
  .ko-match.tbd-match {{ opacity: 0.65; border-style: dashed; }}
  .ko-line {{
    display: flex; justify-content: space-between; align-items: center;
    gap: 0.5rem;
  }}
  .ko-team {{
    font-weight: 600; font-size: 0.88rem; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }}
  .ko-team.right {{ text-align: right; }}
  .ko-team.winner {{ color: var(--green); }}
  .ko-team.loser  {{ color: var(--muted); font-weight: 500; }}
  .ko-score {{
    font-weight: 800; font-size: 0.95rem; flex-shrink: 0;
    background: var(--card2); border-radius: 6px; padding: 0.15rem 0.5rem;
  }}
  .ko-score.vs {{ font-weight: 500; color: var(--muted); font-size: 0.75rem; }}
  .ko-note {{ color: var(--gold); font-size: 0.75rem; margin-top: 0.4rem; }}
  .ko-date {{ color: var(--muted); font-size: 0.7rem; margin-top: 0.35rem; }}
  .ko-bar {{ margin-top: 0.55rem; margin-bottom: 0.25rem; height: 10px; }}
  .ko-pcts {{
    display: flex; justify-content: space-between;
    font-size: 0.72rem; font-weight: 700;
  }}
  .ko-pcts span:first-child {{ color: #60a5fa; }}
  .ko-pcts span:last-child  {{ color: #f87171; }}
  .tbd {{ color: var(--muted); font-style: italic; font-weight: 500; }}

  /* ── Misc ── */
  .no-data {{ color: var(--muted); font-style: italic; padding: 1rem 0; }}
  code {{ background: var(--card2); padding: 0.1rem 0.35rem; border-radius: 3px; font-size: 0.85em; }}

  footer {{
    text-align: center; padding: 2rem;
    color: var(--muted); font-size: 0.78rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<header>
  <h1>🌍 2026 FIFA World Cup</h1>
  <div class="subtitle">ML Predictions Dashboard</div>
  <p>{phase_badge} &nbsp;·&nbsp; Generated {now} &nbsp;·&nbsp; Monte Carlo simulation (10,000 runs) &nbsp;·&nbsp; XGBoost + Elo + Player Stats</p>
</header>

<div class="container">

  <!-- ── Championship Probabilities ─────────────────────────── -->
  <section>
    <h2>🏆 Championship Probabilities</h2>
    <div style="overflow-x:auto;">
      <table class="pred-table">
        <thead>
          <tr>
            <th class="col-rank"></th>
            <th>Team</th>
            <th class="col-bar">Win WC</th>
            <th class="col-stage">Reach Final</th>
            <th class="col-stage">Reach SF</th>
            <th class="col-stage">Reach QF</th>
          </tr>
        </thead>
        <tbody>
          {champ_rows}
        </tbody>
      </table>
    </div>
  </section>

{bracket_section}
  <!-- ── Group Stage Standings ───────────────────────────────── -->
  <section>
    <h2>{standings_title}</h2>
    <div class="legend">
      {standings_note}
    </div>
    <div class="groups-grid">
      {group_html}
    </div>
  </section>

  <!-- ── Upcoming Matches ────────────────────────────────────── -->
  <section>
    <h2>{upcoming_title}</h2>
    <p style="color:var(--muted);font-size:0.8rem;margin-bottom:0.5rem;">
      {upcoming_note}
    </p>
    {upcoming_html}
  </section>

</div>

<footer>
  Built with XGBoost &middot; Poisson goal model &middot; Monte Carlo simulation &middot; SHAP explainability<br>
  Data: martj42/international_results &middot; Player stats (FIFA 25 ratings + Transfermarkt values)
</footer>

</body>
</html>"""

    return html


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("\n📊 Generating 2026 WC Predictions Report...\n")

    fixtures    = load_fixtures()
    sim_results = load_results()
    team_stats  = load_team_stats()
    live_lookup = build_live_lookup()
    standings   = compute_standings(fixtures, live_lookup)
    upcoming    = get_upcoming(fixtures, team_stats, live_lookup, limit=30)

    played_count = sum(
        1 for m in fixtures["group_matches"]
        if _resolve_score(m[0], m[1], m[2], m[3], live_lookup)[0] is not None
    )
    print(f"  ✅ {played_count} group matches played so far (standings updated)")

    if sim_results:
        champ = sim_results.get("Champion", {})
        print("  Top 3 predictions from simulation_results.json:")
        for team, prob in list(champ.items())[:3]:
            print(f"    {flag(team)} {team}: {prob:.1%} to win the World Cup")
    else:
        print("  ⚠️  No simulation results yet — run: python models/simulator.py")
        print("      Report will still show group standings + upcoming match predictions.\n")

    html     = build_html(fixtures, sim_results, standings, upcoming, team_stats)
    out_path = os.path.join(OUTPUTS, "wc2026_predictions.html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Report saved → {out_path}")
    print("   Open in your browser to see the full predictions dashboard!\n")


if __name__ == "__main__":
    main()
