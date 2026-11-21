"""Leakage and consistency tests for the preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load, preprocess


@pytest.fixture(scope="module")
def df():
    return load.load_raw_dataset()


@pytest.mark.parametrize("variant", list(config.EXPERIMENT_VARIANTS))
def test_g3_never_an_input(variant):
    cols = preprocess.input_columns(variant)
    assert "G3" not in cols


def test_early_variant_drops_period_grades():
    cols = preprocess.input_columns("early")
    assert "G1" not in cols and "G2" not in cols
    assert "G1" in preprocess.input_columns("full")


def test_split_is_stratified_and_disjoint(df):
    train_df, test_df = preprocess.split_raw(df)
    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df.index).isdisjoint(test_df.index)
    full_rate = load.make_binary_target(df).mean()
    assert abs(load.make_binary_target(test_df).mean() - full_rate) < 0.02


def test_split_is_deterministic(df):
    a, _ = preprocess.split_raw(df)
    b, _ = preprocess.split_raw(df)
    assert list(a.index) == list(b.index)


def test_scaler_fit_on_train_only(df, tmp_path, monkeypatch):
    # Redirect artifact folders into tmp so the test never touches real results.
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    data, schema = preprocess.prepare_variant(df, "full")

    train_df, _ = preprocess.split_raw(df)
    pre = preprocess.load_preprocessor("full")
    scaler = pre.named_transformers_["num"]
    numeric_cols = [c for c in schema["input_columns"] if c not in load.CATEGORICAL_COLUMNS]
    # The scaler's learned means must equal the TRAIN means, not the full-data means.
    np.testing.assert_allclose(scaler.mean_, train_df[numeric_cols].mean().to_numpy(), rtol=1e-6)
    assert not np.allclose(scaler.mean_, df[numeric_cols].mean().to_numpy())


def test_feature_count_and_schema_consistency(df, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    data, schema = preprocess.prepare_variant(df, "full")
    assert data.X_train.shape[1] == schema["n_features_after_encoding"] == len(schema["feature_names"])
    assert data.X_train.shape[1] == data.X_test.shape[1]
    # Every raw column appears in the schema with a type the form can render.
    for col in schema["input_columns"]:
        assert schema["fields"][col]["type"] in {"categorical", "numeric"}

    # Reloading gives identical arrays.
    reloaded = preprocess.load_processed("full")
    np.testing.assert_array_equal(reloaded.X_test, data.X_test)
    np.testing.assert_array_equal(reloaded.y_test, data.y_test)


def test_transform_matches_training_for_single_row(df, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
    data, schema = preprocess.prepare_variant(df, "full")
    _, test_df = preprocess.split_raw(df)
    pre = preprocess.load_preprocessor("full")
    one = pre.transform(test_df[schema["input_columns"]].iloc[[0]]).astype(np.float32)
    np.testing.assert_allclose(one[0], data.X_test[0], atol=1e-6)
