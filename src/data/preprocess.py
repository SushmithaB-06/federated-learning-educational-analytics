"""Split, encode and scale the dataset without leakage, and save the artifacts.

Pipeline (same for both experiment variants "full" and "early"):

1. Build the target ``pass = (G3 >= 10)`` and drop G3 from the inputs.
   For the "early" variant also drop G1 and G2.
2. Stratified 80/20 train/test split (seed 42) **before** any fitting.
3. Fit a ``ColumnTransformer`` (one-hot for categoricals, standard scaling for
   numerics) on the *training rows only*, then transform train and test.
4. Save: processed arrays (``data/processed``), fitted preprocessor + feature
   schema (``results/models/<variant>/``).

The schema file is what the prediction form in the dashboard is built from, so
inference always sees exactly the columns, categories and ranges used in training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data import load

PREPROCESSOR_FILENAME = "preprocessor.joblib"
SCHEMA_FILENAME = "schema.json"
PROCESSED_FILENAME = "processed.npz"


@dataclass
class SplitData:
    """Processed train/test arrays plus the raw G3-derived labels for the test set."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    band_train: np.ndarray  # letter bands, used only by the optional 5-class model
    band_test: np.ndarray
    feature_names: list[str]


def variant_dir(variant: str) -> Path:
    return config.MODELS_DIR / variant


def input_columns(variant: str) -> list[str]:
    """Raw columns the model is allowed to see for a given variant."""
    drop = set(config.EXPERIMENT_VARIANTS[variant]["drop_columns"])
    assert "G3" in drop, "G3 must always be dropped from the inputs"
    return [c for c in load.FEATURE_DESCRIPTIONS if c not in drop]


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [c for c in columns if c in load.CATEGORICAL_COLUMNS]
    numeric = [c for c in columns if c not in load.CATEGORICAL_COLUMNS]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", StandardScaler(), numeric),
        ],
        verbose_feature_names_out=False,
    )


def split_raw(df: pd.DataFrame, seed: int = config.RANDOM_SEED):
    """Stratified split of the raw frame. Returns (train_df, test_df)."""
    y = load.make_binary_target(df)
    return train_test_split(df, test_size=config.TEST_SIZE, stratify=y, random_state=seed)


def build_schema(train_df: pd.DataFrame, columns: list[str], feature_names: list[str], variant: str) -> dict:
    """Describe every raw input so the dashboard form can validate user input."""
    fields = {}
    for col in columns:
        entry = {"description": load.FEATURE_DESCRIPTIONS[col]}
        if col in load.CATEGORICAL_COLUMNS:
            entry["type"] = "categorical"
            entry["options"] = sorted(train_df[col].unique().tolist())
        else:
            entry["type"] = "numeric"
            entry["min"] = int(train_df[col].min())
            entry["max"] = int(train_df[col].max())
            entry["default"] = int(round(train_df[col].median()))
        fields[col] = entry
    return {
        "variant": variant,
        "description": config.EXPERIMENT_VARIANTS[variant]["description"],
        "target": f"pass = (G3 >= {config.PASS_THRESHOLD})",
        "input_columns": columns,
        "n_input_columns": len(columns),
        "n_features_after_encoding": len(feature_names),
        "feature_names": feature_names,
        "fields": fields,
    }


def prepare_variant(df: pd.DataFrame, variant: str, seed: int = config.RANDOM_SEED) -> tuple[SplitData, dict]:
    """Run the full leakage-safe pipeline for one variant and save its artifacts."""
    columns = input_columns(variant)
    train_df, test_df = split_raw(df, seed)

    pre = build_preprocessor(columns)
    X_train = pre.fit_transform(train_df[columns]).astype(np.float32)  # fit on TRAIN only
    X_test = pre.transform(test_df[columns]).astype(np.float32)
    feature_names = pre.get_feature_names_out().tolist()

    data = SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=load.make_binary_target(train_df).to_numpy(),
        y_test=load.make_binary_target(test_df).to_numpy(),
        # Fixed-width unicode (not Python objects) so np.load works with allow_pickle=False.
        band_train=load.make_grade_band_target(train_df).to_numpy().astype("U1"),
        band_test=load.make_grade_band_target(test_df).to_numpy().astype("U1"),
        feature_names=feature_names,
    )
    schema = build_schema(train_df, columns, feature_names, variant)

    out = variant_dir(variant)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, out / PREPROCESSOR_FILENAME)
    (out / SCHEMA_FILENAME).write_text(json.dumps(schema, indent=2))

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.PROCESSED_DATA_DIR / f"{variant}_{PROCESSED_FILENAME}",
        **{k: v for k, v in asdict(data).items() if k != "feature_names"},
        feature_names=np.array(feature_names, dtype="U"),
    )
    return data, schema


def load_processed(variant: str) -> SplitData:
    """Reload arrays saved by :func:`prepare_variant`."""
    path = config.PROCESSED_DATA_DIR / f"{variant}_{PROCESSED_FILENAME}"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python main.py prepare` first.")
    z = np.load(path, allow_pickle=False)
    return SplitData(
        X_train=z["X_train"], X_test=z["X_test"],
        y_train=z["y_train"], y_test=z["y_test"],
        band_train=z["band_train"], band_test=z["band_test"],
        feature_names=z["feature_names"].tolist(),
    )


def load_preprocessor(variant: str) -> ColumnTransformer:
    path = variant_dir(variant) / PREPROCESSOR_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python main.py prepare` first.")
    return joblib.load(path)


def load_schema(variant: str) -> dict:
    path = variant_dir(variant) / SCHEMA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python main.py prepare` first.")
    return json.loads(path.read_text())
