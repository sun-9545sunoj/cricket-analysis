import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CRICSHEET_BASE_URL = "https://cricsheet.org/downloads"

# Men's international formats ingested from Cricsheet, keyed by match_type
ARCHIVES = {
    "T20": "t20s_male_json.zip",
    "ODI": "odis_male_json.zip",
}

# Phase boundaries are 0-indexed over numbers: (first over, last over exclusive, name)
FORMATS = {
    "T20": {
        "overs": 20,
        "phases": [(0, 6, "Powerplay"), (6, 15, "Middle"), (15, 20, "Death")],
    },
    "ODI": {
        "overs": 50,
        "phases": [(0, 10, "Powerplay"), (10, 40, "Middle"), (40, 50, "Death")],
    },
}

DEFAULT_TEAM_A = "India"
DEFAULT_TEAM_B = "New Zealand"

# Dismissals that are not credited to the bowler
NON_BOWLER_DISMISSALS = ("run out", "retired hurt", "retired out", "obstructing the field", "retired not out")


def data_dir():
    """Data directory, overridable with CRICKET_DATA_DIR (used by tests)."""
    return os.environ.get("CRICKET_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))


def db_path():
    return os.path.join(data_dir(), "cricket.duckdb")


def raw_dir():
    return os.path.join(data_dir(), "raw")


def model_path(name):
    return os.path.join(data_dir(), "models", name)


def total_balls(fmt):
    return FORMATS[fmt]["overs"] * 6


def phase_of(fmt, over):
    for start, end, name in FORMATS[fmt]["phases"]:
        if start <= over < end:
            return name
    return FORMATS[fmt]["phases"][-1][2]


def phase_case_sql(over_col="d.over", format_col="m.format"):
    """SQL CASE expression mapping an over number to its phase name for each format."""
    whens = []
    for fmt, cfg in FORMATS.items():
        for start, end, name in cfg["phases"]:
            whens.append(f"WHEN {format_col} = '{fmt}' AND {over_col} >= {start} AND {over_col} < {end} THEN '{name}'")
    return "CASE " + " ".join(whens) + " ELSE 'Death' END"


def overs_to_balls(overs):
    """Cricket overs notation (13.2 = 13 overs and 2 balls) to legal balls."""
    whole = int(overs)
    part = int(round((overs - whole) * 10))
    if part > 5:
        raise ValueError(f"Invalid overs value {overs}: the ball part must be 0-5")
    return whole * 6 + part
