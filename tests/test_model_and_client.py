"""Tests for the MLP, local training, model I/O, the client boundary and the server loop."""
import inspect

import numpy as np
import pytest
import torch

from src import config
from src.clients.client import ClientUpdate, FederatedClient
from src.models import io as model_io
from src.models.mlp import StudentMLP, weights_num_bytes
from src.models.train import predict_label, predict_proba, train_epochs
from src.server import server as server_module
from src.server.server import FederatedServer

rng = np.random.default_rng(1)
X = rng.normal(size=(120, 58)).astype(np.float32)
Y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)  # learnable synthetic rule


def test_input_output_dimensions():
    model = StudentMLP(58)
    out = model(torch.zeros(7, 58))
    assert out.shape == (7, 1)
    assert model.num_parameters() == 58 * 32 + 32 + 32 * 16 + 16 + 16 * 1 + 1  # 2433


def test_architecture_summary_params_add_up():
    model = StudentMLP(56)
    assert sum(r["params"] for r in model.architecture_summary()) == model.num_parameters()


def test_predict_proba_in_unit_interval():
    p = predict_proba(StudentMLP(58), X)
    assert p.shape == (120,) and p.min() >= 0 and p.max() <= 1
    assert set(np.unique(predict_label(StudentMLP(58), X))) <= {0, 1}


def test_training_reduces_loss_and_is_deterministic():
    m1 = StudentMLP(58)
    torch.manual_seed(0); m1 = StudentMLP(58)
    losses = train_epochs(m1, X, Y, epochs=15, seed=3)
    assert losses[-1] < losses[0]
    torch.manual_seed(0); m2 = StudentMLP(58)
    losses2 = train_epochs(m2, X, Y, epochs=15, seed=3)
    assert losses == losses2


def test_get_set_weights_roundtrip():
    a, b = StudentMLP(58), StudentMLP(58)
    b.set_weights(a.get_weights())
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    assert weights_num_bytes(a.get_weights()) == a.num_parameters() * 4


def test_save_and_load_model(tmp_path):
    model = StudentMLP(58)
    train_epochs(model, X, Y, epochs=2)
    path = tmp_path / "m.pt"
    model_io.save_model(model, path, {"note": "test"})
    loaded, meta = model_io.load_model(path)
    assert meta["note"] == "test"
    np.testing.assert_allclose(predict_proba(loaded, X), predict_proba(model, X), atol=1e-6)
    assert path.with_suffix(".json").exists()


def test_missing_model_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="experiments"):
        model_io.load_model(tmp_path / "missing.pt")


def test_client_update_contains_only_weights_and_scalars():
    client = FederatedClient("A", X, Y, seed=0)
    update = client.train(StudentMLP(58).get_weights(), 58, local_epochs=1, round_seed=1)
    assert isinstance(update, ClientUpdate)
    assert set(update.__dict__) == {"client_name", "weights", "num_samples", "train_loss", "val_loss"}
    assert update.num_samples == client.num_train_samples
    assert client.num_train_samples + client.num_val_samples == len(Y)
    # No tensor in the update has as many rows as the client's dataset: it is weights, not data.
    assert all(v.ndim <= 2 and v.shape[0] != update.num_samples for v in update.weights.values())


def test_server_source_never_touches_client_raw_data():
    src = inspect.getsource(server_module)
    assert "._X" not in src.replace("self._X_test", "") and "._y" not in src.replace("self._y_test", "")


def test_server_round_history_and_fedavg_consistency():
    parts = np.array_split(np.arange(120), 3)
    clients = [FederatedClient(f"C{i}", X[p], Y[p], seed=i) for i, p in enumerate(parts)]
    server = FederatedServer(clients, 58, X, Y, seed=0)
    hist = server.run(num_rounds=3, local_epochs=1)
    assert [r.round for r in hist.rounds] == [1, 2, 3]
    r = hist.rounds[0]
    assert set(r.client_train_loss) == {"C0", "C1", "C2"}
    assert r.bytes_uplink == r.bytes_downlink == 3 * 2433 * 4
    assert 0 <= r.test_accuracy <= 1
