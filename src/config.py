"""Central configuration: paths, constants and default hyper-parameters.

Every other module imports from here so that a path or a hyper-parameter is
defined exactly once. Paths are built with ``pathlib`` so they work on Windows
and Linux alike.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATASET_PATH = RAW_DATA_DIR / "student-mat.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = RESULTS_DIR / "reports"
FIGURES_DIR = RESULTS_DIR / "figures"

DOCS_DIR = PROJECT_ROOT / "docs"

# --------------------------------------------------------------------------- #
# Dataset facts (verified by src/data/load.py)
# --------------------------------------------------------------------------- #
DATASET_NAME = "UCI Student Performance (Mathematics)"
DATASET_URL = "https://archive.ics.uci.edu/dataset/320/student+performance"
EXPECTED_ROWS = 395
EXPECTED_COLUMNS = 33
CSV_SEPARATOR = ";"

# Grade columns. G3 is the final grade (0-20) and is ONLY ever used to build
# the target. It must never appear as an input feature.
GRADE_COLUMNS = ["G1", "G2", "G3"]
TARGET_SOURCE_COLUMN = "G3"
PASS_THRESHOLD = 10  # Pass if G3 >= 10

# Two separate experiments:
#   "full"  -> uses G1 and G2 as inputs (prediction after mid-year grades exist)
#   "early" -> drops G1 and G2 (prediction at the start of the year)
EXPERIMENT_VARIANTS = {
    "full": {"drop_columns": ["G3"], "description": "Uses first- and second-period grades G1, G2"},
    "early": {"drop_columns": ["G1", "G2", "G3"], "description": "Early prediction without G1, G2"},
}

# Grade bands for the OPTIONAL experimental five-class model (G3 scale 0-20).
# Documented thresholds, loosely following the Portuguese 20-point scale.
GRADE_BANDS = [
    ("A", 16, 20),
    ("B", 14, 15),
    ("C", 12, 13),
    ("D", 10, 11),
    ("F", 0, 9),
]

# --------------------------------------------------------------------------- #
# Split and reproducibility
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.20
RANDOM_SEED = 42
EXPERIMENT_SEEDS = [42, 43, 44, 45, 46]

# --------------------------------------------------------------------------- #
# Federated simulation
# --------------------------------------------------------------------------- #
NUM_CLIENTS = 3
CLIENT_NAMES = ["Institution A", "Institution B", "Institution C"]
# Target share of PASS students per client in the Non-IID (label-skew) split.
NON_IID_PASS_FRACTIONS = [0.90, 0.67, 0.43]

# --------------------------------------------------------------------------- #
# Model and training hyper-parameters
# --------------------------------------------------------------------------- #
HIDDEN_LAYERS = [32, 16]
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
LOCAL_EPOCHS = 5
NUM_ROUNDS = 20
CENTRALIZED_EPOCHS = NUM_ROUNDS * LOCAL_EPOCHS  # same total passes over the data
LOCAL_VALIDATION_FRACTION = 0.20  # each client holds out part of its data for validation
