import os
import numpy as np
import torch

from src.strategies.federated_training import federated_training


# --------------------------------------------------
# 1. LOAD PROCESSED DATA
# --------------------------------------------------

data = np.load("data/processed/student_data.npz")

X_train = data["X_train"]
y_train = data["y_train"]


# --------------------------------------------------
# 2. LOAD CLIENT PARTITIONS
# --------------------------------------------------

# We will use the Non-IID partition for this run.
# The partitioning logic will be connected in the next step.

from src.clients.data_partition import create_clients

clients = create_clients(X_train, y_train)


# --------------------------------------------------
# 3. TRAIN FEDERATED MODEL
# --------------------------------------------------

global_model, convergence_history = federated_training(
    clients,
    num_rounds=10,
    local_epochs=5
)


# --------------------------------------------------
# 4. SAVE GLOBAL MODEL
# --------------------------------------------------

os.makedirs("results/models", exist_ok=True)

torch.save(
    global_model.state_dict(),
    "results/models/federated_global_model.pth"
)

np.save(
    "results/models/federated_convergence.npy",
    np.array(convergence_history)
)

print("\nFederated training completed!")
print("Global model saved successfully!")