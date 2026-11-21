"""A simulated institution that trains locally and shares only model weights.

The ``FederatedClient`` keeps its student records in private attributes. The only
thing it ever returns to the outside world is a ``ClientUpdate`` containing model
parameters, a sample count and two scalar losses. The server module never
touches ``_X`` or ``_y`` (a test enforces this).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src import config
from src.models.mlp import StudentMLP
from src.models.train import evaluate_loss, train_epochs


@dataclass
class ClientUpdate:
    """What a client sends back to the server after local training."""

    client_name: str
    weights: dict[str, torch.Tensor]
    num_samples: int          # n_k, used by FedAvg weighting
    train_loss: float         # mean loss over the last local epoch
    val_loss: float           # loss on the client's own held-out validation slice


class FederatedClient:
    def __init__(self, name: str, X: np.ndarray, y: np.ndarray, seed: int = config.RANDOM_SEED,
                 val_fraction: float = config.LOCAL_VALIDATION_FRACTION):
        self.name = name
        self.seed = seed
        # Each client holds back a small validation slice so we can monitor the
        # global model on *client-side* data without ever using the test set.
        stratify = y if min(np.bincount(y)) >= 2 else None
        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
            X, y, test_size=val_fraction, stratify=stratify, random_state=seed)

    # --- facts the server is allowed to know ------------------------------- #
    @property
    def num_train_samples(self) -> int:
        return int(len(self._y_train))

    @property
    def num_val_samples(self) -> int:
        return int(len(self._y_val))

    def class_counts(self) -> dict[str, int]:
        return {"pass": int(self._y_train.sum()), "fail": int((self._y_train == 0).sum())}

    # --- local work -------------------------------------------------------- #
    def train(self, global_weights: dict[str, torch.Tensor], input_dim: int, local_epochs: int,
              round_seed: int) -> ClientUpdate:
        """Start from the global model, train on local data, return new weights."""
        model = StudentMLP(input_dim)
        model.set_weights(global_weights)
        losses = train_epochs(model, self._X_train, self._y_train, epochs=local_epochs, seed=round_seed)
        return ClientUpdate(
            client_name=self.name,
            weights=model.get_weights(),
            num_samples=self.num_train_samples,
            train_loss=float(losses[-1]),
            val_loss=evaluate_loss(model, self._X_val, self._y_val),
        )

    def evaluate_global(self, global_weights: dict[str, torch.Tensor], input_dim: int) -> float:
        """Validation loss of the *global* model on this client's validation slice."""
        model = StudentMLP(input_dim)
        model.set_weights(global_weights)
        return evaluate_loss(model, self._X_val, self._y_val)
