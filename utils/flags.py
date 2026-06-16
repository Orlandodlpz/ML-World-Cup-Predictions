"""
utils/flags.py
--------------
Emoji flags for all 48 teams in the 2026 FIFA World Cup.

Usage:
    from utils.flags import flag, team_str

    flag("Brazil")           → "🇧🇷"
    team_str("Brazil")       → "🇧🇷 Brazil"
    team_str("Brazil", 25)   → "🇧🇷 Brazil         " (padded to 25 chars)
"""

# ── Flag lookup ────────────────────────────────────────────────────────────────
# Grouped by 2026 WC draw (Groups A–L)

TEAM_FLAGS: dict[str, str] = {
    # Group A
    "Mexico":               "🇲🇽",
    "South Korea":          "🇰🇷",
    "South Africa":         "🇿🇦",
    "Czechia":              "🇨🇿",

    # Group B
    "Canada":               "🇨🇦",
    "Switzerland":          "🇨🇭",
    "Qatar":                "🇶🇦",
    "Bosnia & Herzegovina": "🇧🇦",

    # Group C
    "Brazil":               "🇧🇷",
    "Morocco":              "🇲🇦",
    "Scotland":             "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Haiti":                "🇭🇹",

    # Group D
    "USA":                  "🇺🇸",
    "Australia":            "🇦🇺",
    "Paraguay":             "🇵🇾",
    "Turkey":               "🇹🇷",

    # Group E
    "Germany":              "🇩🇪",
    "Ecuador":              "🇪🇨",
    "Ivory Coast":          "🇨🇮",
    "Curacao":              "🇨🇼",

    # Group F
    "Netherlands":          "🇳🇱",
    "Japan":                "🇯🇵",
    "Tunisia":              "🇹🇳",
    "Sweden":               "🇸🇪",

    # Group G
    "Belgium":              "🇧🇪",
    "Iran":                 "🇮🇷",
    "Egypt":                "🇪🇬",
    "New Zealand":          "🇳🇿",

    # Group H
    "Spain":                "🇪🇸",
    "Uruguay":              "🇺🇾",
    "Saudi Arabia":         "🇸🇦",
    "Cape Verde":           "🇨🇻",

    # Group I
    "France":               "🇫🇷",
    "Senegal":              "🇸🇳",
    "Norway":               "🇳🇴",
    "Iraq":                 "🇮🇶",

    # Group J
    "Argentina":            "🇦🇷",
    "Austria":              "🇦🇹",
    "Algeria":              "🇩🇿",
    "Jordan":               "🇯🇴",

    # Group K
    "Portugal":             "🇵🇹",
    "Colombia":             "🇨🇴",
    "Uzbekistan":           "🇺🇿",
    "DR Congo":             "🇨🇩",

    # Group L
    "England":              "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Croatia":              "🇭🇷",
    "Panama":               "🇵🇦",
    "Ghana":                "🇬🇭",
}


def flag(team: str) -> str:
    """Return the emoji flag for a team, or 🏳 if unknown."""
    return TEAM_FLAGS.get(team, "🏳️")


def team_str(team: str, pad: int = 0) -> str:
    """
    Return '🇧🇷 Brazil' optionally padded to `pad` total visible characters.
    Emoji count as 2 chars wide in most terminals, so padding accounts for that.
    """
    f = flag(team)
    base = f"{f} {team}"
    if pad:
        # Emoji take 2 terminal columns; subtract 1 extra per emoji for alignment
        visible_len = len(team) + 2   # flag emoji (2 cols) + space (1) = effectively 3 but we just pad team name
        base = f"{f} {team:<{pad}}"
    return base
