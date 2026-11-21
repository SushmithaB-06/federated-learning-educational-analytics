"""Tests for IID and Non-IID client partitioning."""
import numpy as np
import pytest

from src import config
from src.clients import partition

rng = np.random.default_rng(0)
# Synthetic labels with the real training-set class counts (212 Pass / 104 Fail), shuffled.
Y = rng.permutation(np.array([1] * 212 + [0] * 104))


@pytest.mark.parametrize("scheme", ["iid", "non_iid"])
def test_every_record_used_exactly_once(scheme):
    parts = partition.make_partitions(Y, scheme)
    all_idx = np.sort(np.concatenate(parts))
    assert np.array_equal(all_idx, np.arange(len(Y)))
    assert len(parts) == config.NUM_CLIENTS


@pytest.mark.parametrize("scheme", ["iid", "non_iid"])
def test_deterministic_for_same_seed(scheme):
    a = partition.make_partitions(Y, scheme, seed=7)
    b = partition.make_partitions(Y, scheme, seed=7)
    c = partition.make_partitions(Y, scheme, seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_iid_clients_have_similar_pass_rates():
    rates = [Y[p].mean() for p in partition.partition_iid(Y)]
    assert max(rates) - min(rates) < 0.05


def test_non_iid_clients_have_skewed_pass_rates_in_requested_order():
    rates = [Y[p].mean() for p in partition.partition_non_iid(Y)]
    assert rates == sorted(rates, reverse=True)
    assert rates[0] - rates[-1] > 0.3
    for r, target in zip(rates, config.NON_IID_PASS_FRACTIONS):
        assert abs(r - target) < 0.03


def test_non_iid_rejects_impossible_fractions():
    with pytest.raises(ValueError, match="pass fractions"):
        partition.partition_non_iid(Y, pass_fractions=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        partition.partition_non_iid(Y, pass_fractions=[0.9, 0.5])  # wrong number of clients


def test_describe_table_sums_to_total():
    parts = partition.partition_iid(Y)
    tbl = partition.describe_partitions(Y, parts)
    assert tbl["students"].sum() == len(Y)
    assert (tbl["pass"] + tbl["fail"] == tbl["students"]).all()
