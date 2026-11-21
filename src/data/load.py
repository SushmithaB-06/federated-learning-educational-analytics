"""Load and verify the raw UCI student-mat.csv file.

Run directly to print a verification report:

    python -m src.data.load
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import pandas as pd

from src import config

# Human-readable descriptions of the 33 original attributes (from the UCI
# dataset documentation). Used by the EDA notebook, dashboard and README.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "school": "Student's school (GP = Gabriel Pereira, MS = Mousinho da Silveira)",
    "sex": "Student's sex (F / M)",
    "age": "Age in years (15-22)",
    "address": "Home address type (U = urban, R = rural)",
    "famsize": "Family size (LE3 = 3 or fewer, GT3 = more than 3)",
    "Pstatus": "Parents' cohabitation status (T = together, A = apart)",
    "Medu": "Mother's education (0 none ... 4 higher education)",
    "Fedu": "Father's education (0 none ... 4 higher education)",
    "Mjob": "Mother's job (teacher, health, services, at_home, other)",
    "Fjob": "Father's job (teacher, health, services, at_home, other)",
    "reason": "Reason for choosing the school (home, reputation, course, other)",
    "guardian": "Student's guardian (mother, father, other)",
    "traveltime": "Home-to-school travel time (1 = <15 min ... 4 = >1 hour)",
    "studytime": "Weekly study time (1 = <2 h ... 4 = >10 h)",
    "failures": "Number of past class failures (0-3, 3 means 3 or more)",
    "schoolsup": "Extra educational support from school (yes / no)",
    "famsup": "Family educational support (yes / no)",
    "paid": "Extra paid classes in the subject (yes / no)",
    "activities": "Extra-curricular activities (yes / no)",
    "nursery": "Attended nursery school (yes / no)",
    "higher": "Wants to take higher education (yes / no)",
    "internet": "Internet access at home (yes / no)",
    "romantic": "In a romantic relationship (yes / no)",
    "famrel": "Quality of family relationships (1 very bad ... 5 excellent)",
    "freetime": "Free time after school (1 very low ... 5 very high)",
    "goout": "Going out with friends (1 very low ... 5 very high)",
    "Dalc": "Workday alcohol consumption (1 very low ... 5 very high)",
    "Walc": "Weekend alcohol consumption (1 very low ... 5 very high)",
    "health": "Current health status (1 very bad ... 5 very good)",
    "absences": "Number of school absences (0-93)",
    "G1": "First-period grade (0-20)",
    "G2": "Second-period grade (0-20)",
    "G3": "Final grade (0-20) - used ONLY to build the target",
}

# Columns that are categorical text in the raw file.
CATEGORICAL_COLUMNS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
    "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic",
]

# Everything that is not categorical and not a grade is numeric (integer scales/counts).
NUMERIC_COLUMNS = [
    c for c in FEATURE_DESCRIPTIONS if c not in CATEGORICAL_COLUMNS and c not in config.GRADE_COLUMNS
]


class DatasetError(RuntimeError):
    """Raised when the raw dataset is missing or does not look as expected."""


@dataclass
class DatasetSummary:
    """Facts about the raw file, computed (never hard-coded)."""

    rows: int
    columns: int
    missing_values: int
    pass_count: int
    fail_count: int

    @property
    def pass_rate(self) -> float:
        return self.pass_count / self.rows if self.rows else 0.0


def load_raw_dataset(path=config.RAW_DATASET_PATH) -> pd.DataFrame:
    """Read the semicolon-separated CSV and validate its basic shape.

    Raises:
        DatasetError: if the file is missing or the columns are not the 33 expected ones.
    """
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at {path}.\n"
            f"Download student-mat.csv from {config.DATASET_URL} and place it there."
        )

    df = pd.read_csv(path, sep=config.CSV_SEPARATOR)

    expected = list(FEATURE_DESCRIPTIONS.keys())
    if list(df.columns) != expected:
        raise DatasetError(
            "Unexpected columns in dataset.\n"
            f"Expected: {expected}\nFound:    {list(df.columns)}"
        )
    if df.shape[1] != config.EXPECTED_COLUMNS:
        raise DatasetError(f"Expected {config.EXPECTED_COLUMNS} columns, found {df.shape[1]}")
    return df


def make_binary_target(df: pd.DataFrame) -> pd.Series:
    """Pass = 1 if G3 >= PASS_THRESHOLD else Fail = 0."""
    return (df[config.TARGET_SOURCE_COLUMN] >= config.PASS_THRESHOLD).astype(int).rename("pass")


def make_grade_band_target(df: pd.DataFrame) -> pd.Series:
    """Map G3 to the documented five letter bands A-F (optional experiment)."""
    def band(g3: int) -> str:
        for label, low, high in config.GRADE_BANDS:
            if low <= g3 <= high:
                return label
        raise ValueError(f"G3 value {g3} outside 0-20")

    return df[config.TARGET_SOURCE_COLUMN].apply(band).rename("grade_band")


def summarise(df: pd.DataFrame) -> DatasetSummary:
    target = make_binary_target(df)
    return DatasetSummary(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        missing_values=int(df.isna().sum().sum()),
        pass_count=int(target.sum()),
        fail_count=int((target == 0).sum()),
    )


def _print_report() -> int:
    try:
        df = load_raw_dataset()
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1

    s = summarise(df)
    print(f"Dataset        : {config.DATASET_NAME}")
    print(f"File           : {config.RAW_DATASET_PATH}")
    print(f"Rows x Columns : {s.rows} x {s.columns}  (expected {config.EXPECTED_ROWS} x {config.EXPECTED_COLUMNS})")
    print(f"Missing values : {s.missing_values}")
    print(f"Categorical    : {len(CATEGORICAL_COLUMNS)} columns")
    print(f"Numeric        : {len(NUMERIC_COLUMNS)} columns (+ grades G1, G2, G3)")
    print(f"Target         : pass = (G3 >= {config.PASS_THRESHOLD})")
    print(f"Pass / Fail    : {s.pass_count} / {s.fail_count}  (pass rate {s.pass_rate:.1%})")
    print("G3 range       :", int(df.G3.min()), "to", int(df.G3.max()))
    bands = make_grade_band_target(df).value_counts().reindex([b[0] for b in config.GRADE_BANDS])
    print("Grade bands    :", ", ".join(f"{k}={v}" for k, v in bands.items()))
    ok = s.rows == config.EXPECTED_ROWS and s.missing_values == 0
    print("Verification   :", "OK" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_print_report())
