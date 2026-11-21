"""The central aggregation server for the federated simulation.

One communication round:

1. Server sends the current global weights to every client.
2. Each client trains locally for ``local_epochs`` on its own records.
3. Each client returns a ``ClientUpdate`` (weights + sample count + losses).
4. Server aggregates with FedAvg (sample-weighted mean).
5. The aggregated weights become the new global model.

The server records per-round history for convergence plots. Test-set accuracy is
logged for monitoring only; no decision (early stopping, model choice) is made
from it, so the final test metrics remain an honest hold-out estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src import config
from src.clients.client import FederatedClient
from src.models.mlp import StudentMLP, weights_num_bytes
from src.models.train import evaluate_loss, predict_label
from src.strategies.fedavg import fedavg


@dataclass
class RoundRecord:
    round: int
    train_loss: float            # sample-weighted mean of clients' last-epoch training loss
    val_loss: float              # sample-weighted mean of global model's loss on client validation slices
    test_loss: float             # monitoring only
    test_accuracy: float         # monitoring only
    client_train_loss: dict[str, float]
    client_val_loss: dict[str, float]
    bytes_uplink: int            # total bytes sent client -> server this round
    bytes_downlink: int          # total bytes sent server -> clients this round


@dataclass
class FederatedHistory:
    rounds: list[RoundRecord] = field(default_factory=list)

    def to_records(self) -> list[dict]:
        return [r.__dict__ for r in self.rounds]


class FederatedServer:
    def __init__(self, clients: list[FederatedClient], input_dim: int, X_test: np.ndarray, y_test: np.ndarray,
                 seed: int = config.RANDOM_SEED):
        self.clients = clients
        self.input_dim = input_dim
        self._X_test, self._y_test = X_test, y_test
        torch.manual_seed(seed)
        self.global_model = StudentMLP(input_dim)   # random initial global model
        self.seed = seed
        self.history = FederatedHistory()

    def run(self, num_rounds: int = config.NUM_ROUNDS, local_epochs: int = config.LOCAL_EPOCHS,
            verbose: bool = False) -> FederatedHistory:
        for rnd in range(1, num_rounds + 1):
            global_weights = self.global_model.get_weights()
            msg_bytes = weights_num_bytes(global_weights)

            # Steps 1-3: broadcast, local training, collect updates.
            updates = [c.train(global_weights, self.input_dim, local_epochs, round_seed=self.seed * 1000 + rnd)
                       for c in self.clients]

            # Step 4-5: weighted aggregation becomes the new global model.
            new_weights = fedavg([u.weights for u in updates], [u.num_samples for u in updates])
            self.global_model.set_weights(new_weights)

            # Monitoring.
            n_total = sum(u.num_samples for u in updates)
            train_loss = sum(u.train_loss * u.num_samples for u in updates) / n_total
            client_val = {c.name: c.evaluate_global(new_weights, self.input_dim) for c in self.clients}
            n_val = sum(c.num_val_samples for c in self.clients)
            val_loss = sum(client_val[c.name] * c.num_val_samples for c in self.clients) / n_val
            test_loss = evaluate_loss(self.global_model, self._X_test, self._y_test)
            test_acc = float((predict_label(self.global_model, self._X_test) == self._y_test).mean())

            self.history.rounds.append(RoundRecord(
                round=rnd, train_loss=train_loss, val_loss=val_loss, test_loss=test_loss, test_accuracy=test_acc,
                client_train_loss={u.client_name: u.train_loss for u in updates},
                client_val_loss=client_val,
                bytes_uplink=msg_bytes * len(updates), bytes_downlink=msg_bytes * len(self.clients)))
            if verbose:
                print(f"round {rnd:2d} | train {train_loss:.4f} | val {val_loss:.4f} | test acc {test_acc:.3f}")
        return self.history
