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
OUTPUTS   = os.path.join(BASE, "outputs")

from utils.flags import flag, TEAM_FLAGS


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


# ── Group standings from known results ────────────────────────────────────────

def compute_standings(fixtures):
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
        home, away, hs, as_ = match[0], match[1], match[2], match[3]
        if hs is None or as_ is None:
            continue
        hs, as_ = int(hs), int(as_)
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

def get_upcoming(fixtures, team_stats, limit=30):
    matches = []
    for m in fixtures["group_matches"]:
        home, away, hs, as_, *rest = m
        date = rest[0] if rest else "TBD"
        if hs is not None:
            continue
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


# ── HTML builder ───────────────────────────────────────────────────────────────

def build_html(fixtures, sim_results, standings, upcoming, team_stats):
    now = datetime.now().strftime("%B %d, %Y %H:%M")

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
  <p>Generated {now} &nbsp;·&nbsp; Monte Carlo simulation (10,000 runs) &nbsp;·&nbsp; XGBoost + Elo + Player Stats</p>
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

  <!-- ── Group Stage Standings ───────────────────────────────── -->
  <section>
    <h2>📊 Group Stage Standings</h2>
    <div class="legend">
      <span><div class="dot dot-qualify"></div> Top 2 — qualify automatically</span>
      <span><div class="dot dot-third"></div> 3rd place — may qualify as best 8</span>
    </div>
    <div class="groups-grid">
      {group_html}
    </div>
  </section>

  <!-- ── Upcoming Matches ────────────────────────────────────── -->
  <section>
    <h2>📅 Upcoming Match Predictions</h2>
    <p style="color:var(--muted);font-size:0.8rem;margin-bottom:0.5rem;">
      Bar: <span style="color:#60a5fa">■ home win</span>
      &nbsp;·&nbsp; <span style="color:#94a3b8">■ draw</span>
      &nbsp;·&nbsp; <span style="color:#f87171">■ away win</span>
      &nbsp;·&nbsp; Percentages shown in badges below each bar.
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
    standings   = compute_standings(fixtures)
    upcoming    = get_upcoming(fixtures, team_stats, limit=30)

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
