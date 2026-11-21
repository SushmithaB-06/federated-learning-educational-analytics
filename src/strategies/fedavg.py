"""Federated Averaging (McMahan et al., 2017).

    w_global = sum_k (n_k / N) * w_k

where ``w_k`` are the parameters returned by client *k*, ``n_k`` the number of
training samples that client used, and ``N = sum_k n_k``.

Worked example with two clients and a single weight:
    client A: n=100, w=0.8      client B: n=300, w=0.4
    N = 400 -> w_global = (100/400)*0.8 + (300/400)*0.4 = 0.2 + 0.3 = 0.5

Clients with more data pull the average more strongly. The function is pure
(no side effects) so it is easy to unit-test.
"""
from __future__ import annotations

import torch


def fedavg(client_weights: list[dict[str, torch.Tensor]], client_sizes: list[int]) -> dict[str, torch.Tensor]:
    """Sample-weighted average of client state dicts."""
    if not client_weights:
        raise ValueError("No client updates to aggregate")
    if len(client_weights) != len(client_sizes):
        raise ValueError("Need one sample count per client update")
    if any(n <= 0 for n in client_sizes):
        raise ValueError("Client sample counts must be positive")

    total = float(sum(client_sizes))
    keys = list(client_weights[0].keys())
    for w in client_weights[1:]:
        if list(w.keys()) != keys or any(w[k].shape != client_weights[0][k].shape for k in keys):
            raise ValueError("All clients must share the same model architecture")

    aggregated: dict[str, torch.Tensor] = {}
    for key in keys:
        acc = torch.zeros_like(client_weights[0][key], dtype=torch.float32)
        for w, n in zip(client_weights, client_sizes):
            acc += (n / total) * w[key].to(torch.float32)
        aggregated[key] = acc
    return aggregated
