
import numpy as np


def create_iid_clients(X, y, seed=42):
    """Create 3 clients with nearly balanced Pass/Fail distributions."""

    if len(y) != 316 or np.sum(y == 1) != 212 or np.sum(y == 0) != 104:
        raise ValueError("Expected 316 records: 212 Pass and 104 Fail.")

    rng = np.random.default_rng(seed)

    pass_indices = rng.permutation(np.where(y == 1)[0])
    fail_indices = rng.permutation(np.where(y == 0)[0])

    pass_counts = [71, 67, 74]
    fail_counts = [35, 38, 31]

    clients = {}
    pass_start = 0
    fail_start = 0

    for i in range(3):
        p = pass_indices[pass_start:pass_start + pass_counts[i]]
        f = fail_indices[fail_start:fail_start + fail_counts[i]]

        indices = rng.permutation(np.concatenate([p, f]))

        clients[f"Client {i+1}"] = {
            "X": X[indices],
            "y": y[indices]
        }

        pass_start += pass_counts[i]
        fail_start += fail_counts[i]

    return clients


def create_non_iid_clients(X, y, seed=42):
    """Create 3 clients with deliberately different Pass/Fail distributions."""

    if len(y) != 316 or np.sum(y == 1) != 212 or np.sum(y == 0) != 104:
        raise ValueError("Expected 316 records: 212 Pass and 104 Fail.")

    rng = np.random.default_rng(seed)

    pass_indices = rng.permutation(np.where(y == 1)[0])
    fail_indices = rng.permutation(np.where(y == 0)[0])

    pass_counts = [90, 70, 52]
    fail_counts = [16, 35, 53]

    clients = {}
    pass_start = 0
    fail_start = 0

    for i in range(3):
        p = pass_indices[pass_start:pass_start + pass_counts[i]]
        f = fail_indices[fail_start:fail_start + fail_counts[i]]

        indices = rng.permutation(np.concatenate([p, f]))

        clients[f"Client {i+1}"] = {
            "X": X[indices],
            "y": y[indices]
        }

        pass_start += pass_counts[i]
        fail_start += fail_counts[i]

    return clients


def display_client_distribution(clients, title):
    print(f"\n{title}")

    for client_name, data in clients.items():
        y_client = data["y"]

        total = len(y_client)
        passed = int(np.sum(y_client == 1))
        failed = int(np.sum(y_client == 0))

        print(
            f"{client_name}: Total={total}, "
            f"Pass={passed}, Fail={failed}"
        )
