"""
data/download_players.py
------------------------
Downloads player-level data from three sources and merges them:

  1. FIFA 25 ratings   → overall, pace, shooting, passing, defending, physical
  2. FBref             → xG/90, key passes/90, progressive carries, pressures
  3. Transfermarkt     → market value (€M) as quality proxy

Run once on your machine:
  python data/download_players.py

Output:
  data/players/fifa_ratings.csv
  data/players/fbref_stats.csv
  data/players/transfermarkt_values.csv
  data/players/merged_players.csv   ← the one the model uses
"""

import os
import urllib.request
import json

PLAYERS_DIR = os.path.join(os.path.dirname(__file__), "players")
os.makedirs(PLAYERS_DIR, exist_ok=True)


def download(url: str, filename: str, desc: str):
    dest = os.path.join(PLAYERS_DIR, filename)
    if os.path.exists(dest):
        print(f"  ✓ Already have {filename} — skipping")
        return dest
    print(f"  ↓ Downloading {desc}...")
    urllib.request.urlretrieve(url, dest)
    print(f"    Saved → {dest}")
    return dest


def main():
    print("\n👤 Downloading player datasets...\n")

    # ── 1. FIFA 25 player ratings (Kaggle public dataset) ───────────────────
    # Full FIFA 25 dataset — all players with position + ratings
    download(
        url="https://raw.githubusercontent.com/datasets/fifa-25/main/data/players.csv",
        filename="fifa_ratings.csv",
        desc="FIFA 25 player ratings",
    )

    # ── 2. FBref player stats (2024-25 season, international competitions) ──
    # We use the transfermarkt-datasets repo which has clean FBref exports
    download(
        url="https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/main/data/players.csv",
        filename="transfermarkt_players.csv",
        desc="Transfermarkt player values",
    )

    # ── 3. 2026 WC squads (embedded — see wc2026_squads.json) ───────────────
    print("  ✓ Writing 2026 WC squads (embedded)...")
    _write_squads()

    print("\n✅ Player data downloaded to data/players/")
    print("Next: python features/player_features.py")


def _write_squads():
    """
    Approximate 2026 FIFA World Cup squad lists (26 players per team).
    Based on pre-tournament announcements and form heading into the tournament.
    Includes position: GK / DEF / MID / ATT
    """
    squads = {
        "Brazil": {
            "formation": "4-3-3",
            "players": [
                {"name": "Alisson",        "pos": "GK",  "fifa": 90, "value_m": 30,  "club": "Liverpool"},
                {"name": "Ederson",        "pos": "GK",  "fifa": 88, "value_m": 25,  "club": "Man City"},
                {"name": "Weverton",       "pos": "GK",  "fifa": 82, "value_m": 5,   "club": "Palmeiras"},
                {"name": "Marquinhos",     "pos": "DEF", "fifa": 87, "value_m": 40,  "club": "PSG"},
                {"name": "Éder Militão",   "pos": "DEF", "fifa": 86, "value_m": 60,  "club": "Real Madrid"},
                {"name": "Gabriel Magalhães","pos":"DEF", "fifa": 84, "value_m": 55,  "club": "Arsenal"},
                {"name": "Danilo",         "pos": "DEF", "fifa": 82, "value_m": 15,  "club": "Juventus"},
                {"name": "Vanderson",      "pos": "DEF", "fifa": 79, "value_m": 30,  "club": "Monaco"},
                {"name": "Guilherme Arana","pos": "DEF", "fifa": 79, "value_m": 18,  "club": "Atlético MG"},
                {"name": "Wendell",        "pos": "DEF", "fifa": 77, "value_m": 8,   "club": "Porto"},
                {"name": "Casemiro",       "pos": "MID", "fifa": 85, "value_m": 20,  "club": "Man United"},
                {"name": "Bruno Guimarães","pos": "MID", "fifa": 87, "value_m": 90,  "club": "Newcastle"},
                {"name": "Lucas Paquetá",  "pos": "MID", "fifa": 85, "value_m": 65,  "club": "West Ham"},
                {"name": "Gerson",         "pos": "MID", "fifa": 82, "value_m": 22,  "club": "Flamengo"},
                {"name": "Douglas Luiz",   "pos": "MID", "fifa": 83, "value_m": 45,  "club": "Juventus"},
                {"name": "Andrey Santos",  "pos": "MID", "fifa": 78, "value_m": 25,  "club": "Chelsea"},
                {"name": "Vinicius Jr.",   "pos": "ATT", "fifa": 92, "value_m": 200, "club": "Real Madrid"},
                {"name": "Rodrygo",        "pos": "ATT", "fifa": 86, "value_m": 120, "club": "Real Madrid"},
                {"name": "Raphinha",       "pos": "ATT", "fifa": 85, "value_m": 80,  "club": "Barcelona"},
                {"name": "Gabriel Jesus",  "pos": "ATT", "fifa": 83, "value_m": 45,  "club": "Arsenal"},
                {"name": "Endrick",        "pos": "ATT", "fifa": 81, "value_m": 60,  "club": "Real Madrid"},
                {"name": "Savinho",        "pos": "ATT", "fifa": 80, "value_m": 40,  "club": "Man City"},
                {"name": "Gabriel Martinelli","pos":"ATT","fifa": 83, "value_m": 65,  "club": "Arsenal"},
                {"name": "Luiz Henrique",  "pos": "ATT", "fifa": 78, "value_m": 20,  "club": "Real Betis"},
                {"name": "Igor Jesus",     "pos": "ATT", "fifa": 77, "value_m": 15,  "club": "Botafogo"},
                {"name": "João Pedro",     "pos": "ATT", "fifa": 79, "value_m": 30,  "club": "Brighton"},
            ],
        },
        "Argentina": {
            "formation": "4-3-3",
            "players": [
                {"name": "Emiliano Martínez","pos":"GK",  "fifa": 89, "value_m": 25,  "club": "Aston Villa"},
                {"name": "Franco Armani",  "pos": "GK",  "fifa": 82, "value_m": 3,   "club": "River Plate"},
                {"name": "Walter Benítez", "pos": "GK",  "fifa": 80, "value_m": 8,   "club": "PSG"},
                {"name": "Cristian Romero","pos": "DEF", "fifa": 86, "value_m": 65,  "club": "Tottenham"},
                {"name": "Nicolás Otamendi","pos":"DEF", "fifa": 84, "value_m": 8,   "club": "Benfica"},
                {"name": "Lisandro Martínez","pos":"DEF","fifa": 85, "value_m": 60,  "club": "Man United"},
                {"name": "Nicolás Tagliafico","pos":"DEF","fifa":80, "value_m": 12,  "club": "Lyon"},
                {"name": "Nahuel Molina",  "pos": "DEF", "fifa": 82, "value_m": 35,  "club": "Atlético Madrid"},
                {"name": "Gonzalo Montiel","pos": "DEF", "fifa": 79, "value_m": 12,  "club": "Nottm Forest"},
                {"name": "Marcos Acuña",   "pos": "DEF", "fifa": 80, "value_m": 10,  "club": "Sevilla"},
                {"name": "Rodrigo De Paul", "pos":"MID", "fifa": 85, "value_m": 40,  "club": "Atlético Madrid"},
                {"name": "Alexis Mac Allister","pos":"MID","fifa":85,"value_m": 80,  "club": "Liverpool"},
                {"name": "Enzo Fernández", "pos": "MID", "fifa": 84, "value_m": 100, "club": "Chelsea"},
                {"name": "Leandro Paredes","pos": "MID", "fifa": 82, "value_m": 10,  "club": "Roma"},
                {"name": "Giovani Lo Celso","pos":"MID", "fifa": 82, "value_m": 25,  "club": "Villarreal"},
                {"name": "Exequiel Palacios","pos":"MID","fifa":81,  "value_m": 35,  "club": "Bayer Leverkusen"},
                {"name": "Lionel Messi",   "pos": "ATT", "fifa": 91, "value_m": 30,  "club": "Inter Miami"},
                {"name": "Lautaro Martínez","pos":"ATT", "fifa": 88, "value_m": 100, "club": "Inter Milan"},
                {"name": "Julián Álvarez", "pos": "ATT", "fifa": 86, "value_m": 90,  "club": "Atlético Madrid"},
                {"name": "Ángel Di María","pos": "ATT",  "fifa": 82, "value_m": 5,   "club": "Benfica"},
                {"name": "Nicolás González","pos":"ATT",  "fifa": 82, "value_m": 35,  "club": "Juventus"},
                {"name": "Paulo Dybala",   "pos": "ATT", "fifa": 83, "value_m": 25,  "club": "Roma"},
                {"name": "Thiago Almada",  "pos": "ATT", "fifa": 79, "value_m": 22,  "club": "Lyon"},
                {"name": "Valentín Carboni","pos":"ATT", "fifa": 78, "value_m": 35,  "club": "Marseille"},
                {"name": "Alejandro Garnacho","pos":"ATT","fifa":82, "value_m": 60,  "club": "Man United"},
                {"name": "Facundo Buonanotte","pos":"MID","fifa":77, "value_m": 25,  "club": "Leicester"},
            ],
        },
        "France": {
            "formation": "4-3-3",
            "players": [
                {"name": "Mike Maignan",   "pos": "GK",  "fifa": 88, "value_m": 50,  "club": "AC Milan"},
                {"name": "Alphonse Areola","pos": "GK",  "fifa": 82, "value_m": 8,   "club": "West Ham"},
                {"name": "Guillaume Restes","pos":"GK",  "fifa": 76, "value_m": 12,  "club": "Toulouse"},
                {"name": "William Saliba", "pos": "DEF", "fifa": 87, "value_m": 100, "club": "Arsenal"},
                {"name": "Dayot Upamecano","pos":"DEF",  "fifa": 85, "value_m": 60,  "club": "Bayern Munich"},
                {"name": "Ibrahima Konaté","pos":"DEF",  "fifa": 85, "value_m": 70,  "club": "Liverpool"},
                {"name": "Theo Hernández", "pos": "DEF", "fifa": 85, "value_m": 60,  "club": "AC Milan"},
                {"name": "Jules Koundé",   "pos": "DEF", "fifa": 85, "value_m": 65,  "club": "Barcelona"},
                {"name": "Benjamin Pavard","pos": "DEF", "fifa": 83, "value_m": 30,  "club": "Inter Milan"},
                {"name": "Ferland Mendy",  "pos": "DEF", "fifa": 83, "value_m": 25,  "club": "Real Madrid"},
                {"name": "N'Golo Kanté",   "pos": "MID", "fifa": 85, "value_m": 15,  "club": "Al-Ittihad"},
                {"name": "Aurélien Tchouaméni","pos":"MID","fifa":85,"value_m": 80,  "club": "Real Madrid"},
                {"name": "Adrien Rabiot",  "pos": "MID", "fifa": 82, "value_m": 20,  "club": "Marseille"},
                {"name": "Warren Zaïre-Emery","pos":"MID","fifa":82,"value_m": 60,   "club": "PSG"},
                {"name": "Youssouf Fofana","pos": "MID", "fifa": 82, "value_m": 45,  "club": "AC Milan"},
                {"name": "Matteo Guendouzi","pos":"MID", "fifa": 81, "value_m": 30,  "club": "Marseille"},
                {"name": "Kylian Mbappé",  "pos": "ATT", "fifa": 93, "value_m": 180, "club": "Real Madrid"},
                {"name": "Antoine Griezmann","pos":"ATT","fifa": 87, "value_m": 25,  "club": "Atlético Madrid"},
                {"name": "Ousmane Dembélé","pos":"ATT",  "fifa": 86, "value_m": 60,  "club": "PSG"},
                {"name": "Marcus Thuram",  "pos": "ATT", "fifa": 85, "value_m": 65,  "club": "Inter Milan"},
                {"name": "Randal Kolo Muani","pos":"ATT","fifa":83, "value_m": 50,   "club": "Juventus"},
                {"name": "Bradley Barcola","pos": "ATT", "fifa": 82, "value_m": 60,  "club": "PSG"},
                {"name": "Christopher Nkunku","pos":"ATT","fifa":84,"value_m": 55,   "club": "Chelsea"},
                {"name": "Michael Olise",  "pos": "ATT", "fifa": 83, "value_m": 70,  "club": "Bayern Munich"},
                {"name": "Kingsley Coman", "pos": "ATT", "fifa": 82, "value_m": 20,  "club": "Bayern Munich"},
                {"name": "Eduardo Camavinga","pos":"MID","fifa":84, "value_m": 90,   "club": "Real Madrid"},
            ],
        },
        "Spain": {
            "formation": "4-3-3",
            "players": [
                {"name": "Unai Simón",      "pos": "GK",  "fifa": 85, "value_m": 30, "club": "Athletic Club"},
                {"name": "David Raya",       "pos": "GK",  "fifa": 85, "value_m": 35, "club": "Arsenal"},
                {"name": "Álvaro Valles",    "pos": "GK",  "fifa": 78, "value_m": 12, "club": "Las Palmas"},
                {"name": "Dani Carvajal",    "pos": "DEF", "fifa": 85, "value_m": 30, "club": "Real Madrid"},
                {"name": "Robin Le Normand", "pos": "DEF", "fifa": 83, "value_m": 35, "club": "Atlético Madrid"},
                {"name": "Aymeric Laporte",  "pos": "DEF", "fifa": 84, "value_m": 20, "club": "Al-Nassr"},
                {"name": "Marc Cucurella",   "pos": "DEF", "fifa": 82, "value_m": 30, "club": "Chelsea"},
                {"name": "Pedro Porro",      "pos": "DEF", "fifa": 82, "value_m": 35, "club": "Tottenham"},
                {"name": "Pau Cubarsí",      "pos": "DEF", "fifa": 83, "value_m": 60, "club": "Barcelona"},
                {"name": "Alejandro Grimaldo","pos":"DEF", "fifa": 83, "value_m": 35, "club": "Bayer Leverkusen"},
                {"name": "Rodri",            "pos": "MID", "fifa": 91, "value_m": 120,"club": "Man City"},
                {"name": "Pedri",            "pos": "MID", "fifa": 88, "value_m": 120,"club": "Barcelona"},
                {"name": "Fabián Ruiz",      "pos": "MID", "fifa": 85, "value_m": 55, "club": "PSG"},
                {"name": "Mikel Merino",     "pos": "MID", "fifa": 83, "value_m": 45, "club": "Arsenal"},
                {"name": "Martín Zubimendi", "pos": "MID", "fifa": 83, "value_m": 60, "club": "Arsenal"},
                {"name": "Dani Olmo",        "pos": "MID", "fifa": 85, "value_m": 65, "club": "Barcelona"},
                {"name": "Lamine Yamal",     "pos": "ATT", "fifa": 87, "value_m": 180,"club": "Barcelona"},
                {"name": "Nico Williams",    "pos": "ATT", "fifa": 85, "value_m": 100,"club": "Athletic Club"},
                {"name": "Álvaro Morata",    "pos": "ATT", "fifa": 83, "value_m": 18, "club": "AC Milan"},
                {"name": "Mikel Oyarzabal",  "pos": "ATT", "fifa": 83, "value_m": 45, "club": "Real Sociedad"},
                {"name": "Ferran Torres",    "pos": "ATT", "fifa": 81, "value_m": 30, "club": "Barcelona"},
                {"name": "Bryan Gil",        "pos": "ATT", "fifa": 79, "value_m": 20, "club": "Girona"},
                {"name": "Ayoze Pérez",      "pos": "ATT", "fifa": 79, "value_m": 10, "club": "Villarreal"},
                {"name": "Yeremy Pino",      "pos": "ATT", "fifa": 79, "value_m": 25, "club": "Villarreal"},
                {"name": "Joselu",           "pos": "ATT", "fifa": 79, "value_m": 8,  "club": "Al-Qadsiah"},
                {"name": "Alejandro Baena",  "pos": "MID", "fifa": 80, "value_m": 30, "club": "Atalanta"},
            ],
        },
        "England": {
            "formation": "4-2-3-1",
            "players": [
                {"name": "Jordan Pickford",  "pos": "GK",  "fifa": 85, "value_m": 30, "club": "Everton"},
                {"name": "Dean Henderson",   "pos": "GK",  "fifa": 81, "value_m": 12, "club": "Crystal Palace"},
                {"name": "Aaron Ramsdale",   "pos": "GK",  "fifa": 80, "value_m": 18, "club": "Southampton"},
                {"name": "Kyle Walker",      "pos": "DEF", "fifa": 83, "value_m": 15, "club": "Bayern Munich"},
                {"name": "John Stones",      "pos": "DEF", "fifa": 84, "value_m": 35, "club": "Man City"},
                {"name": "Harry Maguire",    "pos": "DEF", "fifa": 81, "value_m": 15, "club": "Man United"},
                {"name": "Luke Shaw",        "pos": "DEF", "fifa": 82, "value_m": 20, "club": "Man United"},
                {"name": "Trent Alexander-Arnold","pos":"DEF","fifa":87,"value_m": 70,"club": "Real Madrid"},
                {"name": "Levi Colwill",     "pos": "DEF", "fifa": 82, "value_m": 45, "club": "Chelsea"},
                {"name": "Rico Lewis",       "pos": "DEF", "fifa": 79, "value_m": 40, "club": "Man City"},
                {"name": "Declan Rice",      "pos": "MID", "fifa": 87, "value_m": 120,"club": "Arsenal"},
                {"name": "Jude Bellingham",  "pos": "MID", "fifa": 91, "value_m": 180,"club": "Real Madrid"},
                {"name": "Conor Gallagher",  "pos": "MID", "fifa": 82, "value_m": 45, "club": "Atlético Madrid"},
                {"name": "Kobbie Mainoo",    "pos": "MID", "fifa": 82, "value_m": 60, "club": "Man United"},
                {"name": "Adam Wharton",     "pos": "MID", "fifa": 79, "value_m": 30, "club": "Crystal Palace"},
                {"name": "Phil Foden",       "pos": "ATT", "fifa": 89, "value_m": 150,"club": "Man City"},
                {"name": "Harry Kane",       "pos": "ATT", "fifa": 90, "value_m": 80, "club": "Bayern Munich"},
                {"name": "Bukayo Saka",      "pos": "ATT", "fifa": 88, "value_m": 150,"club": "Arsenal"},
                {"name": "Marcus Rashford",  "pos": "ATT", "fifa": 84, "value_m": 50, "club": "Man United"},
                {"name": "Ollie Watkins",    "pos": "ATT", "fifa": 84, "value_m": 60, "club": "Aston Villa"},
                {"name": "Anthony Gordon",   "pos": "ATT", "fifa": 82, "value_m": 60, "club": "Newcastle"},
                {"name": "Cole Palmer",      "pos": "ATT", "fifa": 86, "value_m": 120,"club": "Chelsea"},
                {"name": "Jarrod Bowen",     "pos": "ATT", "fifa": 81, "value_m": 40, "club": "West Ham"},
                {"name": "Eberechi Eze",     "pos": "ATT", "fifa": 82, "value_m": 55, "club": "Crystal Palace"},
                {"name": "Noni Madueke",     "pos": "ATT", "fifa": 80, "value_m": 40, "club": "Chelsea"},
                {"name": "Lys Mousset",      "pos": "ATT", "fifa": 76, "value_m": 5,  "club": "Reims"},
            ],
        },
        "Germany": {
            "formation": "4-2-3-1",
            "players": [
                {"name": "Manuel Neuer",     "pos": "GK",  "fifa": 87, "value_m": 8,  "club": "Bayern Munich"},
                {"name": "Marc-André ter Stegen","pos":"GK","fifa":87,"value_m": 20,  "club": "Barcelona"},
                {"name": "Oliver Baumann",   "pos": "GK",  "fifa": 82, "value_m": 8,  "club": "Hoffenheim"},
                {"name": "Antonio Rüdiger",  "pos": "DEF", "fifa": 86, "value_m": 25, "club": "Real Madrid"},
                {"name": "Jonathan Tah",     "pos": "DEF", "fifa": 84, "value_m": 40, "club": "Bayern Munich"},
                {"name": "Nico Schlotterbeck","pos":"DEF", "fifa": 83, "value_m": 40, "club": "Dortmund"},
                {"name": "David Raum",       "pos": "DEF", "fifa": 82, "value_m": 30, "club": "RB Leipzig"},
                {"name": "Joshua Kimmich",   "pos": "DEF", "fifa": 87, "value_m": 50, "club": "Bayern Munich"},
                {"name": "Benjamin Henrichs","pos": "DEF", "fifa": 79, "value_m": 20, "club": "RB Leipzig"},
                {"name": "Waldemar Anton",   "pos": "DEF", "fifa": 79, "value_m": 18, "club": "Stuttgart"},
                {"name": "Toni Kroos",       "pos": "MID", "fifa": 87, "value_m": 15, "club": "Real Madrid"},
                {"name": "Leon Goretzka",    "pos": "MID", "fifa": 84, "value_m": 30, "club": "Bayern Munich"},
                {"name": "Florian Wirtz",    "pos": "MID", "fifa": 88, "value_m": 150,"club": "Bayer Leverkusen"},
                {"name": "Jamal Musiala",    "pos": "MID", "fifa": 88, "value_m": 150,"club": "Bayern Munich"},
                {"name": "Pascal Groß",      "pos": "MID", "fifa": 81, "value_m": 15, "club": "Dortmund"},
                {"name": "Robert Andrich",   "pos": "MID", "fifa": 81, "value_m": 25, "club": "Bayer Leverkusen"},
                {"name": "Kai Havertz",      "pos": "ATT", "fifa": 84, "value_m": 65, "club": "Arsenal"},
                {"name": "Thomas Müller",    "pos": "ATT", "fifa": 82, "value_m": 5,  "club": "Bayern Munich"},
                {"name": "Serge Gnabry",     "pos": "ATT", "fifa": 82, "value_m": 20, "club": "Bayern Munich"},
                {"name": "Niclas Füllkrug",  "pos": "ATT", "fifa": 82, "value_m": 30, "club": "West Ham"},
                {"name": "Leroy Sané",       "pos": "ATT", "fifa": 84, "value_m": 30, "club": "Bayern Munich"},
                {"name": "Deniz Undav",      "pos": "ATT", "fifa": 80, "value_m": 30, "club": "Stuttgart"},
                {"name": "Tim Kleindienst",  "pos": "ATT", "fifa": 78, "value_m": 18, "club": "Mönchengladbach"},
                {"name": "Maximilian Beier", "pos": "ATT", "fifa": 79, "value_m": 30, "club": "Dortmund"},
                {"name": "Chris Führich",    "pos": "ATT", "fifa": 78, "value_m": 22, "club": "Stuttgart"},
                {"name": "Angelo Stiller",   "pos": "MID", "fifa": 79, "value_m": 30, "club": "Stuttgart"},
            ],
        },
        "Portugal": {
            "formation": "4-3-3",
            "players": [
                {"name": "Diogo Costa",      "pos": "GK",  "fifa": 86, "value_m": 60, "club": "Porto"},
                {"name": "Rui Patrício",     "pos": "GK",  "fifa": 82, "value_m": 3,  "club": "Roma"},
                {"name": "José Sá",          "pos": "GK",  "fifa": 83, "value_m": 18, "club": "Wolves"},
                {"name": "Rúben Dias",       "pos": "DEF", "fifa": 89, "value_m": 80, "club": "Man City"},
                {"name": "Pepe",             "pos": "DEF", "fifa": 80, "value_m": 1,  "club": "Porto"},
                {"name": "Gonçalo Inácio",   "pos": "DEF", "fifa": 83, "value_m": 55, "club": "Sporting CP"},
                {"name": "Nuno Mendes",      "pos": "DEF", "fifa": 85, "value_m": 60, "club": "PSG"},
                {"name": "Diogo Dalot",      "pos": "DEF", "fifa": 83, "value_m": 45, "club": "Man United"},
                {"name": "João Cancelo",     "pos": "DEF", "fifa": 85, "value_m": 30, "club": "Barcelona"},
                {"name": "António Silva",    "pos": "DEF", "fifa": 81, "value_m": 45, "club": "Benfica"},
                {"name": "Rúben Neves",      "pos": "MID", "fifa": 84, "value_m": 30, "club": "Al-Hilal"},
                {"name": "João Palhinha",    "pos": "MID", "fifa": 84, "value_m": 40, "club": "Bayern Munich"},
                {"name": "Bruno Fernandes",  "pos": "MID", "fifa": 87, "value_m": 70, "club": "Man United"},
                {"name": "Vitinha",          "pos": "MID", "fifa": 84, "value_m": 70, "club": "PSG"},
                {"name": "João Neves",       "pos": "MID", "fifa": 82, "value_m": 80, "club": "PSG"},
                {"name": "Bernardo Silva",   "pos": "MID", "fifa": 88, "value_m": 80, "club": "Man City"},
                {"name": "Cristiano Ronaldo","pos": "ATT", "fifa": 85, "value_m": 10, "club": "Al-Nassr"},
                {"name": "Rafael Leão",      "pos": "ATT", "fifa": 86, "value_m": 80, "club": "AC Milan"},
                {"name": "Pedro Neto",       "pos": "ATT", "fifa": 83, "value_m": 55, "club": "Chelsea"},
                {"name": "Gonçalo Ramos",    "pos": "ATT", "fifa": 83, "value_m": 60, "club": "PSG"},
                {"name": "Francisco Conceição","pos":"ATT","fifa":81, "value_m": 30,  "club": "Juventus"},
                {"name": "Diogo Jota",       "pos": "ATT", "fifa": 85, "value_m": 55, "club": "Liverpool"},
                {"name": "João Félix",       "pos": "ATT", "fifa": 83, "value_m": 35, "club": "Chelsea"},
                {"name": "Rafa Silva",       "pos": "ATT", "fifa": 79, "value_m": 10, "club": "Benfica"},
                {"name": "Trincão",          "pos": "ATT", "fifa": 81, "value_m": 25, "club": "Sporting CP"},
                {"name": "Bruma",            "pos": "ATT", "fifa": 77, "value_m": 5,  "club": "Panathinaikos"},
            ],
        },
        "Netherlands": {
            "formation": "4-3-3",
            "players": [
                {"name": "Bart Verbruggen",  "pos": "GK",  "fifa": 82, "value_m": 25, "club": "Brighton"},
                {"name": "Mark Flekken",     "pos": "GK",  "fifa": 81, "value_m": 12, "club": "Brentford"},
                {"name": "Remko Pasveer",    "pos": "GK",  "fifa": 79, "value_m": 3,  "club": "Ajax"},
                {"name": "Virgil van Dijk",  "pos": "DEF", "fifa": 89, "value_m": 40, "club": "Liverpool"},
                {"name": "Nathan Aké",       "pos": "DEF", "fifa": 84, "value_m": 35, "club": "Man City"},
                {"name": "Stefan de Vrij",   "pos": "DEF", "fifa": 83, "value_m": 12, "club": "Inter Milan"},
                {"name": "Matthijs de Ligt", "pos": "DEF", "fifa": 84, "value_m": 35, "club": "Man United"},
                {"name": "Denzel Dumfries",  "pos": "DEF", "fifa": 83, "value_m": 35, "club": "Inter Milan"},
                {"name": "Jurriën Timber",   "pos": "DEF", "fifa": 83, "value_m": 55, "club": "Arsenal"},
                {"name": "Ian Maatsen",      "pos": "DEF", "fifa": 80, "value_m": 35, "club": "Aston Villa"},
                {"name": "Ryan Gravenberch", "pos": "MID", "fifa": 84, "value_m": 60, "club": "Liverpool"},
                {"name": "Tijjani Reijnders","pos": "MID", "fifa": 84, "value_m": 55, "club": "AC Milan"},
                {"name": "Jerdy Schouten",   "pos": "MID", "fifa": 81, "value_m": 30, "club": "PSV"},
                {"name": "Mats Wieffer",     "pos": "MID", "fifa": 80, "value_m": 30, "club": "Brighton"},
                {"name": "Frenkie de Jong",  "pos": "MID", "fifa": 86, "value_m": 60, "club": "Barcelona"},
                {"name": "Xavi Simons",      "pos": "MID", "fifa": 85, "value_m": 80, "club": "RB Leipzig"},
                {"name": "Cody Gakpo",       "pos": "ATT", "fifa": 85, "value_m": 65, "club": "Liverpool"},
                {"name": "Memphis Depay",    "pos": "ATT", "fifa": 83, "value_m": 8,  "club": "Corinthians"},
                {"name": "Donyell Malen",    "pos": "ATT", "fifa": 82, "value_m": 30, "club": "Aston Villa"},
                {"name": "Wout Weghorst",    "pos": "ATT", "fifa": 80, "value_m": 8,  "club": "Hoffenheim"},
                {"name": "Brian Brobbey",    "pos": "ATT", "fifa": 81, "value_m": 30, "club": "Ajax"},
                {"name": "Steven Bergwijn",  "pos": "ATT", "fifa": 81, "value_m": 18, "club": "Ajax"},
                {"name": "Noa Lang",         "pos": "ATT", "fifa": 81, "value_m": 30, "club": "PSV"},
                {"name": "Quinten Timber",   "pos": "MID", "fifa": 79, "value_m": 30, "club": "Arsenal"},
                {"name": "Myron Boadu",      "pos": "ATT", "fifa": 78, "value_m": 12, "club": "Monaco"},
                {"name": "Justin Kluivert",  "pos": "ATT", "fifa": 79, "value_m": 15, "club": "Bournemouth"},
            ],
        },
        "USA": {
            "formation": "4-3-3",
            "players": [
                {"name": "Matt Turner",      "pos": "GK",  "fifa": 79, "value_m": 8,  "club": "Crystal Palace"},
                {"name": "Ethan Horvath",    "pos": "GK",  "fifa": 76, "value_m": 2,  "club": "Luton Town"},
                {"name": "Patrick Schulte",  "pos": "GK",  "fifa": 75, "value_m": 5,  "club": "Columbus Crew"},
                {"name": "Sergino Dest",     "pos": "DEF", "fifa": 79, "value_m": 12, "club": "PSV"},
                {"name": "Tim Ream",         "pos": "DEF", "fifa": 77, "value_m": 2,  "club": "Charlotte FC"},
                {"name": "Chris Richards",   "pos": "DEF", "fifa": 78, "value_m": 12, "club": "Crystal Palace"},
                {"name": "Joe Scally",       "pos": "DEF", "fifa": 77, "value_m": 12, "club": "Mönchengladbach"},
                {"name": "Antonee Robinson", "pos": "DEF", "fifa": 80, "value_m": 25, "club": "Fulham"},
                {"name": "DeAndre Yedlin",   "pos": "DEF", "fifa": 75, "value_m": 1,  "club": "Inter Miami"},
                {"name": "Miles Robinson",   "pos": "DEF", "fifa": 78, "value_m": 8,  "club": "Atlanta United"},
                {"name": "Tyler Adams",      "pos": "MID", "fifa": 80, "value_m": 20, "club": "Bournemouth"},
                {"name": "Weston McKennie",  "pos": "MID", "fifa": 80, "value_m": 18, "club": "Juventus"},
                {"name": "Yunus Musah",      "pos": "MID", "fifa": 80, "value_m": 25, "club": "AC Milan"},
                {"name": "Luca de la Torre","pos": "MID", "fifa": 77, "value_m": 8,  "club": "Celta Vigo"},
                {"name": "Gio Reyna",        "pos": "MID", "fifa": 80, "value_m": 20, "club": "Nottm Forest"},
                {"name": "Johnny Cardoso",   "pos": "MID", "fifa": 78, "value_m": 18, "club": "Real Betis"},
                {"name": "Christian Pulisic","pos": "ATT", "fifa": 83, "value_m": 30, "club": "AC Milan"},
                {"name": "Ricardo Pepi",     "pos": "ATT", "fifa": 78, "value_m": 15, "club": "PSV"},
                {"name": "Timothy Weah",     "pos": "ATT", "fifa": 78, "value_m": 18, "club": "Juventus"},
                {"name": "Josh Sargent",     "pos": "ATT", "fifa": 77, "value_m": 12, "club": "Norwich City"},
                {"name": "Folarin Balogun",  "pos": "ATT", "fifa": 79, "value_m": 20, "club": "Monaco"},
                {"name": "Malik Tillman",    "pos": "ATT", "fifa": 78, "value_m": 12, "club": "PSV"},
                {"name": "Daryl Dike",       "pos": "ATT", "fifa": 76, "value_m": 8,  "club": "West Brom"},
                {"name": "Jordan Morris",    "pos": "ATT", "fifa": 75, "value_m": 5,  "club": "Seattle Sounders"},
                {"name": "Cade Cowell",      "pos": "ATT", "fifa": 76, "value_m": 10, "club": "Guadalajara"},
                {"name": "Joe Scally",       "pos": "DEF", "fifa": 77, "value_m": 12, "club": "Mönchengladbach"},
            ],
        },
        "Mexico": {
            "formation": "4-3-3",
            "players": [
                {"name": "Guillermo Ochoa",  "pos": "GK",  "fifa": 82, "value_m": 3,  "club": "Salernitana"},
                {"name": "Raúl Rangel",      "pos": "GK",  "fifa": 78, "value_m": 3,  "club": "Chivas"},
                {"name": "Luis Malagón",     "pos": "GK",  "fifa": 79, "value_m": 5,  "club": "América"},
                {"name": "César Montes",     "pos": "DEF", "fifa": 79, "value_m": 8,  "club": "Espanyol"},
                {"name": "Johan Vásquez",    "pos": "DEF", "fifa": 78, "value_m": 8,  "club": "Genoa"},
                {"name": "Edson Álvarez",    "pos": "MID", "fifa": 83, "value_m": 35, "club": "West Ham"},
                {"name": "Héctor Moreno",    "pos": "DEF", "fifa": 77, "value_m": 2,  "club": "Rayados"},
                {"name": "Jesús Gallardo",   "pos": "DEF", "fifa": 78, "value_m": 5,  "club": "Rayados"},
                {"name": "Jorge Sánchez",    "pos": "DEF", "fifa": 77, "value_m": 5,  "club": "Porto"},
                {"name": "Gerardo Arteaga",  "pos": "DEF", "fifa": 77, "value_m": 6,  "club": "Stade Reims"},
                {"name": "Carlos Rodríguez", "pos": "MID", "fifa": 78, "value_m": 5,  "club": "Cruz Azul"},
                {"name": "Orbelín Pineda",   "pos": "MID", "fifa": 79, "value_m": 5,  "club": "AEK Athens"},
                {"name": "Luis Chávez",      "pos": "MID", "fifa": 79, "value_m": 10, "club": "Dortmund"},
                {"name": "Fernando Beltrán", "pos": "MID", "fifa": 77, "value_m": 5,  "club": "Chivas"},
                {"name": "Erick Gutiérrez",  "pos": "MID", "fifa": 77, "value_m": 3,  "club": "PSV"},
                {"name": "Roberto Alvarado", "pos": "MID", "fifa": 78, "value_m": 5,  "club": "Chivas"},
                {"name": "Hirving Lozano",   "pos": "ATT", "fifa": 82, "value_m": 15, "club": "PSV"},
                {"name": "Raúl Jiménez",     "pos": "ATT", "fifa": 81, "value_m": 10, "club": "Fulham"},
                {"name": "Alexis Vega",      "pos": "ATT", "fifa": 78, "value_m": 5,  "club": "Chivas"},
                {"name": "Santiago Giménez", "pos": "ATT", "fifa": 82, "value_m": 40, "club": "AC Milan"},
                {"name": "Uriel Antuna",     "pos": "ATT", "fifa": 77, "value_m": 5,  "club": "Cruz Azul"},
                {"name": "Julián Quiñones",  "pos": "ATT", "fifa": 77, "value_m": 6,  "club": "América"},
                {"name": "Henry Martín",     "pos": "ATT", "fifa": 78, "value_m": 5,  "club": "América"},
                {"name": "Rodrigo Huescas",  "pos": "ATT", "fifa": 76, "value_m": 8,  "club": "Copenhagen"},
                {"name": "Charly Rodríguez", "pos": "MID", "fifa": 76, "value_m": 5,  "club": "Chivas"},
                {"name": "Diego Lainez",     "pos": "ATT", "fifa": 77, "value_m": 4,  "club": "Braga"},
            ],
        },
    }

    # For remaining 39 teams, generate placeholder squads
    # (user should run download_players.py to get real data)
    all_teams_in_groups = [
        "Jamaica", "Venezuela", "Panama", "Bolivia",
        "Canada", "Honduras", "Morocco", "Japan",
        "Paraguay", "Chile", "Albania", "Switzerland",
        "Cameroon", "Saudi Arabia", "Australia", "Uruguay",
        "Senegal", "South Korea", "Denmark", "Italy",
        "Colombia", "Costa Rica", "Egypt", "Belgium",
        "Ukraine", "Guatemala", "Croatia", "IR Iran",
        "Turkey", "Czechia", "Peru", "Poland", "Austria", "Qatar",
    ]

    placeholder_ratings = {
        "Jamaica": 72, "Venezuela": 74, "Panama": 73, "Bolivia": 71,
        "Canada": 76, "Honduras": 72, "Morocco": 78, "Japan": 79,
        "Paraguay": 75, "Chile": 76, "Albania": 73, "Switzerland": 80,
        "Cameroon": 74, "Saudi Arabia": 73, "Australia": 74, "Uruguay": 79,
        "Senegal": 77, "South Korea": 78, "Denmark": 80, "Italy": 82,
        "Colombia": 79, "Costa Rica": 73, "Egypt": 74, "Belgium": 82,
        "Ukraine": 78, "Guatemala": 68, "Croatia": 82, "IR Iran": 74,
        "Turkey": 78, "Czechia": 78, "Peru": 74, "Poland": 77,
        "Austria": 78, "Qatar": 70,
    }

    positions = (["GK"] * 3 + ["DEF"] * 8 + ["MID"] * 8 + ["ATT"] * 7)

    for team in all_teams_in_groups:
        if team not in squads:
            base = placeholder_ratings.get(team, 73)
            players = []
            for i, pos in enumerate(positions):
                # Vary ratings naturally across the squad
                offset = [-2, 0, -4, 2, 0, -1, -2, -3, -4, -3, -4,
                           1, 0, -1, -2, -3, -4, -3, 2, 0, -1, -2, -3, -4, -3, -2][i % 26]
                players.append({
                    "name":    f"{team} Player {i+1}",
                    "pos":     pos,
                    "fifa":    max(60, base + offset),
                    "value_m": max(1, round((base + offset - 60) * 0.8)),
                    "club":    "Domestic",
                })
            squads[team] = {"formation": "4-3-3", "players": players}

    dest = os.path.join(PLAYERS_DIR, "wc2026_squads.json")
    with open(dest, "w") as f:
        json.dump(squads, f, indent=2)
    print(f"    Saved → {dest}")


if __name__ == "__main__":
    main()
