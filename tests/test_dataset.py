"""Tests for raw dataset loading and target construction."""
import pandas as pd
import pytest

from src import config
from src.data import load


@pytest.fixture(scope="module")
def df():
    return load.load_raw_dataset()


def test_shape_matches_uci_documentation(df):
    assert df.shape == (config.EXPECTED_ROWS, config.EXPECTED_COLUMNS)


def test_no_missing_values(df):
    assert df.isna().sum().sum() == 0


def test_binary_target_threshold(df):
    y = load.make_binary_target(df)
    assert set(y.unique()) == {0, 1}
    assert (y == (df.G3 >= 10).astype(int)).all()
    # Total is always 395; pass + fail must add up.
    assert y.sum() + (y == 0).sum() == len(df)


def test_grade_bands_cover_whole_scale(df):
    bands = load.make_grade_band_target(df)
    assert bands.isna().sum() == 0
    # F band must coincide with the binary Fail class.
    assert ((bands == "F") == (df.G3 < 10)).all()


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(load.DatasetError, match="Download"):
        load.load_raw_dataset(tmp_path / "nope.csv")


def test_wrong_columns_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad, sep=";", index=False)
    with pytest.raises(load.DatasetError, match="Unexpected columns"):
        load.load_raw_dataset(bad)
