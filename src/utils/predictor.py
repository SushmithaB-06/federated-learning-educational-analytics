"""Schema-driven student prediction used by the dashboard.

The predictor loads, for a chosen variant ("full" or "early"):

* the fitted preprocessor (exactly the one fit on the training rows),
* the feature schema (which raw columns, allowed categories, numeric ranges),
* one of the trained models (centralized / FedAvg IID / FedAvg Non-IID).

A prediction therefore goes through *precisely* the same transformation as
training data did. Inputs are validated against the schema before anything is
transformed, so a typo or an out-of-range value gives a clear message instead
of a silent wrong answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.data.preprocess import load_preprocessor, load_schema
from src.models.io import load_model
from src.models.train import predict_proba

MODEL_CHOICES = {
    "fedavg_iid": "FedAvg global model (IID clients)",
    "fedavg_non_iid": "FedAvg global model (Non-IID clients)",
    "centralized": "Centralized baseline",
}


class ValidationError(ValueError):
    """One or more input fields are missing or invalid."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass
class Prediction:
    label: str                 # "Pass" or "Fail"
    probability_pass: float
    probability_fail: float
    model_key: str
    model_label: str
    variant: str
    confidence_note: str


class StudentPredictor:
    def __init__(self, variant: str = "full", model_key: str = "fedavg_iid"):
        if variant not in config.EXPERIMENT_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}")
        if model_key not in MODEL_CHOICES:
            raise ValueError(f"Unknown model {model_key!r}; choose from {list(MODEL_CHOICES)}")
        self.variant = variant
        self.model_key = model_key
        self.schema = load_schema(variant)
        self.preprocessor = load_preprocessor(variant)
        self.model, self.model_metadata = load_model(config.MODELS_DIR / variant / f"{model_key}.pt")
        if self.model.input_dim != self.schema["n_features_after_encoding"]:
            raise RuntimeError("Model and preprocessor disagree on feature count; re-run the pipeline")

    # ------------------------------------------------------------------ #
    @property
    def fields(self) -> dict:
        return self.schema["fields"]

    @property
    def input_columns(self) -> list[str]:
        return self.schema["input_columns"]

    def default_student(self) -> dict:
        """A valid example row (medians / first category), handy for the UI and tests."""
        out = {}
        for col, spec in self.fields.items():
            out[col] = spec["options"][0] if spec["type"] == "categorical" else spec["default"]
        return out

    def validate(self, student: dict) -> dict:
        """Return a cleaned copy or raise ValidationError listing every problem."""
        problems, clean = [], {}
        for col in self.input_columns:
            spec = self.fields[col]
            if col not in student or student[col] is None or student[col] == "":
                problems.append(f"'{col}' is required")
                continue
            value = student[col]
            if spec["type"] == "categorical":
                if str(value) not in spec["options"]:
                    problems.append(f"'{col}' must be one of {spec['options']}, got {value!r}")
                else:
                    clean[col] = str(value)
            else:
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    problems.append(f"'{col}' must be a number, got {value!r}")
                    continue
                if not num.is_integer():
                    problems.append(f"'{col}' must be a whole number, got {value!r}")
                elif not spec["min"] <= num <= spec["max"]:
                    problems.append(f"'{col}' must be between {spec['min']} and {spec['max']}, got {int(num)}")
                else:
                    clean[col] = int(num)
        extra = set(student) - set(self.input_columns)
        if "G3" in extra:
            problems.append("'G3' is the target and must not be supplied as an input")
        if problems:
            raise ValidationError(problems)
        return clean

    def transform(self, student: dict) -> np.ndarray:
        clean = self.validate(student)
        frame = pd.DataFrame([clean])[self.input_columns]
        return self.preprocessor.transform(frame).astype(np.float32)

    def predict(self, student: dict) -> Prediction:
        p_pass = float(predict_proba(self.model, self.transform(student))[0])
        margin = abs(p_pass - 0.5)
        note = ("high confidence" if margin > 0.35 else "moderate confidence" if margin > 0.15
                else "low confidence - close to the decision boundary")
        return Prediction(
            label="Pass" if p_pass >= 0.5 else "Fail",
            probability_pass=p_pass,
            probability_fail=1 - p_pass,
            model_key=self.model_key,
            model_label=MODEL_CHOICES[self.model_key],
            variant=self.variant,
            confidence_note=note,
        )

    def sensitivity(self, student: dict, top_k: int = 8) -> pd.DataFrame:
        """What-if analysis: change one numeric field at a time to its min / max.

        This is not a formal explanation method; it shows how much the predicted
        pass probability moves for this particular student when a single input
        changes, holding everything else fixed.
        """
        base = self.predict(student).probability_pass
        rows = []
        for col in self.input_columns:
            spec = self.fields[col]
            if spec["type"] != "numeric":
                continue
            lo, hi = dict(student, **{col: spec["min"]}), dict(student, **{col: spec["max"]})
            p_lo = self.predict(lo).probability_pass
            p_hi = self.predict(hi).probability_pass
            rows.append({"feature": col, "current": student[col], "p_pass_at_min": p_lo, "p_pass_at_max": p_hi,
                         "swing": p_hi - p_lo, "abs_swing": abs(p_hi - p_lo)})
        df = pd.DataFrame(rows).sort_values("abs_swing", ascending=False).head(top_k)
        df.attrs["baseline_p_pass"] = base
        return df.reset_index(drop=True)
