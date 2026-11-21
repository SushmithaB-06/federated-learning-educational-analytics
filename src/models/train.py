"""Plain supervised training helpers shared by clients and the centralized baseline."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src import config
from src.models.mlp import StudentMLP
from src.utils.seed import seed_everything


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def train_epochs(
    model: StudentMLP,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float = config.LEARNING_RATE,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.RANDOM_SEED,
) -> list[float]:
    """Train in place with Adam + BCEWithLogits. Returns mean training loss per epoch."""
    seed_everything(seed)
    Xt, yt = to_tensor(X), to_tensor(y).unsqueeze(1)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=generator)
        total = 0.0
        for start in range(0, len(Xt), batch_size):
            idx = perm[start:start + batch_size]
            optimiser.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        losses.append(total / len(Xt))
    return losses


@torch.no_grad()
def evaluate_loss(model: StudentMLP, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    logits = model(to_tensor(X))
    return float(nn.BCEWithLogitsLoss()(logits, to_tensor(y).unsqueeze(1)))


@torch.no_grad()
def predict_proba(model: StudentMLP, X: np.ndarray) -> np.ndarray:
    """Probability of Pass for each row."""
    model.eval()
    return torch.sigmoid(model(to_tensor(X))).squeeze(1).numpy()


def predict_label(model: StudentMLP, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (predict_proba(model, X) >= threshold).astype(int)
