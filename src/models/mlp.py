"""The shared neural network used by every client and by the centralized baseline.

Architecture (for the "full" variant, 58 inputs):

    58 inputs -> Linear(58, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1)

The single output is a *logit*; ``sigmoid(logit)`` is the probability of Pass.
Using ``BCEWithLogitsLoss`` (sigmoid + binary cross-entropy in one numerically
stable step) is the standard choice for a binary classifier.

The same class, with the same ``input_dim`` and ``hidden`` sizes, is created
at every client. Identical architecture is what makes weight averaging valid.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src import config


class StudentMLP(nn.Module):
    """Small multi-layer perceptron for Pass/Fail prediction."""

    def __init__(self, input_dim: int, hidden: list[int] | None = None, output_dim: int = 1):
        super().__init__()
        hidden = list(hidden or config.HIDDEN_LAYERS)
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)

    # ------------------------------------------------------------------ #
    # Weight helpers used by the federated simulation
    # ------------------------------------------------------------------ #
    def get_weights(self) -> dict[str, torch.Tensor]:
        """Copy of all parameters (what a client sends to the server)."""
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        self.load_state_dict({k: v.clone() for k, v in weights.items()})

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def architecture_summary(self) -> list[dict]:
        rows, prev = [], self.input_dim
        for i, h in enumerate(self.hidden, 1):
            rows.append({"layer": f"Hidden {i}", "in": prev, "out": h, "activation": "ReLU", "params": prev * h + h})
            prev = h
        rows.append({"layer": "Output", "in": prev, "out": self.output_dim, "activation": "sigmoid (at inference)",
                     "params": prev * self.output_dim + self.output_dim})
        return rows


def weights_num_bytes(weights: dict[str, torch.Tensor]) -> int:
    """Size of one weight message in bytes (float32), used for communication cost."""
    return int(sum(v.numel() * v.element_size() for v in weights.values()))
