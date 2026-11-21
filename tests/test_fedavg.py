"""Tests for the FedAvg aggregation rule."""
import pytest
import torch

from src.strategies.fedavg import fedavg


def _w(value, shape=(2, 2)):
    return {"layer.weight": torch.full(shape, float(value)), "layer.bias": torch.full((shape[0],), float(value))}


def test_weighted_average_matches_formula():
    # n=100 with w=0.8 and n=300 with w=0.4  ->  0.25*0.8 + 0.75*0.4 = 0.5
    out = fedavg([_w(0.8), _w(0.4)], [100, 300])
    assert torch.allclose(out["layer.weight"], torch.full((2, 2), 0.5))
    assert torch.allclose(out["layer.bias"], torch.full((2,), 0.5))


def test_equal_sizes_is_plain_mean():
    out = fedavg([_w(1.0), _w(3.0), _w(5.0)], [10, 10, 10])
    assert torch.allclose(out["layer.weight"], torch.full((2, 2), 3.0))


def test_single_client_returns_its_own_weights():
    out = fedavg([_w(0.123)], [42])
    assert torch.allclose(out["layer.weight"], torch.full((2, 2), 0.123))


def test_larger_client_dominates():
    out = fedavg([_w(0.0), _w(1.0)], [1, 999])
    assert out["layer.weight"].mean().item() > 0.99


def test_output_does_not_alias_inputs():
    a, b = _w(1.0), _w(2.0)
    out = fedavg([a, b], [1, 1])
    out["layer.weight"] += 100
    assert a["layer.weight"].max().item() == 1.0


def test_rejects_mismatched_architectures():
    with pytest.raises(ValueError):
        fedavg([_w(1.0, (2, 2)), _w(1.0, (3, 3))], [1, 1])


def test_rejects_bad_sizes():
    with pytest.raises(ValueError):
        fedavg([_w(1.0), _w(1.0)], [1])
    with pytest.raises(ValueError):
        fedavg([_w(1.0), _w(1.0)], [0, 5])
    with pytest.raises(ValueError):
        fedavg([], [])
