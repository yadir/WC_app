"""
flags.py — map tournament teams to country flags (served from flagcdn.com).

flagcdn gives clean, consistently-sized PNG flags via a simple URL pattern:
    https://flagcdn.com/w80/<code>.png
Home-nation subdivisions use ISO 3166-2 style codes (gb-eng, gb-sct).
"""

# Canonical team name (from config.py) -> flagcdn code.
TEAM_ISO = {
    "Mexico": "mx", "South Africa": "za", "South Korea": "kr", "Czechia": "cz",
    "Canada": "ca", "Switzerland": "ch", "Qatar": "qa",
    "Bosnia and Herzegovina": "ba",
    "Brazil": "br", "Morocco": "ma", "Haiti": "ht", "Scotland": "gb-sct",
    "United States": "us", "Paraguay": "py", "Australia": "au", "Turkey": "tr",
    "Germany": "de", "Curacao": "cw", "Ivory Coast": "ci", "Ecuador": "ec",
    "Netherlands": "nl", "Japan": "jp", "Tunisia": "tn", "Sweden": "se",
    "Belgium": "be", "Egypt": "eg", "Iran": "ir", "New Zealand": "nz",
    "Spain": "es", "Cape Verde": "cv", "Saudi Arabia": "sa", "Uruguay": "uy",
    "France": "fr", "Senegal": "sn", "Norway": "no", "Iraq": "iq",
    "Argentina": "ar", "Algeria": "dz", "Austria": "at", "Jordan": "jo",
    "Portugal": "pt", "Uzbekistan": "uz", "Colombia": "co", "DR Congo": "cd",
    "England": "gb-eng", "Croatia": "hr", "Ghana": "gh", "Panama": "pa",
}

# Spellings used in the fixture list (schedule.py) that differ from canonical.
FIXTURE_ALIASES = {
    "USA": "United States",
    "Turkiye": "Turkey",
    "Bosnia & Herz.": "Bosnia and Herzegovina",
    "Bosnia": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
}


def canonical(name):
    name = name.strip()
    return FIXTURE_ALIASES.get(name, name)


def flag_url(team, width=80):
    """Return a flagcdn URL for a team, or None if we don't have a code."""
    code = TEAM_ISO.get(canonical(team))
    if not code:
        return None
    return f"https://flagcdn.com/w{width}/{code}.png"
