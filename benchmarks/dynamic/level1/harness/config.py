"""Central configuration for the DMQ benchmark harness.

All paths are absolute; the harness may be launched from any cwd, but DMView
itself is always executed with cwd=DMVIEW_ROOT.
"""
from pathlib import Path

# --- Environment ---------------------------------------------------------
# DMView 3.4 install root (PSS/e 36.2 dynamic-model review tool) - the harness
# always runs DMView with cwd=DMVIEW_ROOT.
DMVIEW_ROOT = Path(r"D:\DMView3.4(final)")
PY311 = Path(r"C:\Users\xiegr\AppData\Local\Programs\Python\Python311\python.exe")

# PowerAgentBench layout: this benchmark lives under benchmarks/dynamic/level1
# and reads its pristine case from cases/solar_wecc/psse.
HARNESS = Path(__file__).resolve().parent
LEVEL_DIR = HARNESS.parent                                  # .../dynamic/level1
PAB_ROOT = LEVEL_DIR.parents[2]                             # PowerAgentBench root
RUNS_DIR = LEVEL_DIR / "runs"
RESULTS_DIR = LEVEL_DIR / "results"           # aggregated artifacts + calibration
PRISTINE_DIR = PAB_ROOT / "cases" / "solar_wecc" / "psse"   # never modified
ENV_FILE = PAB_ROOT / ".env"

RESULTS_GLOBAL = DMVIEW_ROOT / "RESULTs"      # DMView's shared output folder
RESULTS_LOCK = RESULTS_GLOBAL / ".harness_lock"

INI_NAME = "AgentBench_DMQ"                   # -> D:\DMView3.4(final)\AgentBench_DMQ.ini
INI_PATH = DMVIEW_ROOT / f"{INI_NAME}.ini"

CRITERIA_FILE = HARNESS / "criteria.json"

# --- Benchmark definition (Solar.ini 8-test suite) ------------------------
# name -> (type, data) exactly as written into the generated INI [Tests] section
TESTS = [
    ("Test01_FS",       "['FS', '10']"),
    ("Test02_VOLTDOWN", "['VOLT','DATAs\\\\ERCOT_VOLT-STEP-DOWN.xlsx']"),
    ("Test03_VOLTUP",   "['VOLT','DATAs\\\\ERCOT_VOLT-STEP-UP.xlsx']"),
    ("Test04_FRQDOWN",  "['FREQ','DATAs\\\\ERCOT_FRQ-STEP-DOWN.xlsx']"),
    ("Test05_FRQUP",    "['FREQ','DATAs\\\\ERCOT_FRQ-STEP-UP.xlsx']"),
    ("Test06_HVRT",     "['VOLT','DATAs\\\\ERCOT_Legacy_HVRT.XLSX']"),
    ("Test07_LVRT",     "['VOLT','DATAs\\\\ERCOT_Legacy_LVRT.XLSX']"),
    ("Test08_SCR2",     "['SCR2', '5->3->1.5->1.2, 5, 1']"),
]
TEST_NAMES = [t[0] for t in TESTS]
SCR2_LEVELS = [5.0, 3.0, 1.5, 1.2]
SCR2_DWELL_S = 5.0

# --- Action space ---------------------------------------------------------
GAIN_NAMES = ("Kqp", "Kqi", "Kvp", "Kvi")
GAIN_MIN = 0.0          # exclusive
GAIN_MAX = 100.0        # inclusive
CORRUPTED_GAINS = {"Kqp": 10.0, "Kqi": 50.0, "Kvp": 10.0, "Kvi": 50.0}
KNOWN_GOOD_GAINS = {"Kqp": 1.0, "Kqi": 5.0, "Kvp": 1.0, "Kvi": 5.0}  # paper Sonnet row

# --- Agent / run parameters ------------------------------------------------
ITERATION_BUDGET = 5     # agent-triggered run_test_suite calls (baseline excluded)
MAX_TURNS = 40           # hard cap on assistant turns per run
SUITE_TIMEOUT_S = 1800   # wall-clock limit for one full DMView suite
MAX_TOKENS = 8000
RUNS_PER_MODEL = 10

MODELS = {
    "opus":   "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku":  "claude-haiku-4-5",
}
# models that support adaptive thinking (omit the thinking param for the rest)
ADAPTIVE_THINKING_MODELS = {"claude-opus-4-8", "claude-sonnet-4-6"}

# USD per million tokens (input, output)
PRICING = {
    "claude-opus-4-8":   (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5":  (1.00, 5.00),
}


def est_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICING[model_id]
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout
