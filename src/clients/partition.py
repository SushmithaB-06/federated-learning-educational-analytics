"""Split the TRAINING set into simulated institutions (clients).

Only *indices* are produced here, so the same partition can be applied to the
"full" and "early" feature matrices (both share the same train/test split).

Two schemes:

* **IID** - every client gets a stratified, random slice of the training data,
  so all clients have roughly the same Pass/Fail ratio.
* **Non-IID (label skew)** - clients receive deliberately different Pass
  rates (default 90 % / 67 % / 43 %). This models *statistical heterogeneity*:
  in reality a top-ranked school and a struggling school do not have the same
  distribution of outcomes, and FedAvg is known to suffer when client data
  differ like this.

Both schemes are deterministic for a given seed and are verified to use every
training record exactly once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def _check_partition(parts: list[np.ndarray], n_total: int) -> None:
    """Every index used exactly once: no overlap, no loss."""
    all_idx = np.concatenate(parts)
    if len(all_idx) != n_total or len(np.unique(all_idx)) != n_total:
        raise ValueError("Partition must cover every record exactly once")


def partition_iid(y: np.ndarray, num_clients: int = config.NUM_CLIENTS, seed: int = config.RANDOM_SEED) -> list[np.ndarray]:
    """Stratified random split: each client has about the same class ratio."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(num_clients)]
    for label in (0, 1):
        idx = rng.permutation(np.where(y == label)[0])
        for k, chunk in enumerate(np.array_split(idx, num_clients)):
            parts[k].append(chunk)
    parts = [rng.permutation(np.concatenate(p)) for p in parts]
    _check_partition(parts, len(y))
    return parts


def partition_non_iid(
    y: np.ndarray,
    num_clients: int = config.NUM_CLIENTS,
    seed: int = config.RANDOM_SEED,
    pass_fractions: list[float] = config.NON_IID_PASS_FRACTIONS,
) -> list[np.ndarray]:
    """Label-skew split: equal client sizes, deliberately different Pass rates.

    The requested fractions are targets; rounding is corrected so that every
    Pass and every Fail record is assigned exactly once.
    """
    if len(pass_fractions) != num_clients:
        raise ValueError("Need one pass fraction per client")
    rng = np.random.default_rng(seed)
    pass_idx = rng.permutation(np.where(y == 1)[0])
    fail_idx = rng.permutation(np.where(y == 0)[0])

    sizes = [len(c) for c in np.array_split(np.arange(len(y)), num_clients)]
    pass_counts = [int(round(f * n)) for f, n in zip(pass_fractions, sizes)]
    # The fractions must be consistent with how many Pass records actually exist:
    # we only tolerate a rounding residual of at most one record per client.
    diff = len(pass_idx) - sum(pass_counts)
    if abs(diff) > num_clients:
        raise ValueError(
            f"Requested pass fractions imply {sum(pass_counts)} Pass records but the data has "
            f"{len(pass_idx)}; choose fractions whose weighted mean is about {len(pass_idx) / len(y):.2f}")
    # Spread the small residual one record at a time so every Pass record is used exactly once.
    k = 0
    while diff != 0:
        step = 1 if diff > 0 else -1
        pass_counts[k % num_clients] += step
        diff -= step
        k += 1
    fail_counts = [n - p for n, p in zip(sizes, pass_counts)]
    if min(fail_counts) < 0 or sum(fail_counts) != len(fail_idx):
        raise ValueError("Requested pass fractions are not achievable with this data")

    parts, p0, f0 = [], 0, 0
    for p, f in zip(pass_counts, fail_counts):
        chunk = np.concatenate([pass_idx[p0:p0 + p], fail_idx[f0:f0 + f]])
        parts.append(rng.permutation(chunk))
        p0 += p
        f0 += f
    _check_partition(parts, len(y))
    return parts


def make_partitions(y: np.ndarray, scheme: str, seed: int = config.RANDOM_SEED) -> list[np.ndarray]:
    if scheme == "iid":
        return partition_iid(y, seed=seed)
    if scheme == "non_iid":
        return partition_non_iid(y, seed=seed)
    raise ValueError(f"Unknown scheme {scheme!r}; use 'iid' or 'non_iid'")


def describe_partitions(y: np.ndarray, parts: list[np.ndarray], names: list[str] = config.CLIENT_NAMES) -> pd.DataFrame:
    """Per-client sample counts and class balance, as a table."""
    rows = []
    for name, idx in zip(names, parts):
        yk = y[idx]
        rows.append({
            "client": name,
            "students": int(len(idx)),
            "pass": int(yk.sum()),
            "fail": int((yk == 0).sum()),
            "pass_rate": round(float(yk.mean()), 4),
        })
    return pd.DataFrame(rows)
