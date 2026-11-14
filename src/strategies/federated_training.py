
import copy

from src.models.neural_network import StudentPerformanceModel
from src.models.train import train_model
from src.strategies.fedavg import fedavg_aggregate


def federated_training(clients, num_rounds=10, local_epochs=5):

    # Initialize the global model
    global_model = StudentPerformanceModel(input_size=58)

    # Store convergence information for every round
    convergence_history = []

    for round_num in range(num_rounds):

        print(f"\n========== Federated Round {round_num + 1} ==========")

        client_models = []
        client_sizes = []
        client_final_losses = []

        # Train each client locally
        for client_id, client_data in clients.items():

            print(f"Training Client {client_id}...")

            # Copy the current global model
            local_model = copy.deepcopy(global_model)

            # Train the local model and collect loss history
            trained_model, loss_history = train_model(
                client_data["X"],
                client_data["y"],
                model=local_model,
                epochs=local_epochs,
                learning_rate=0.001,
                return_history=True
            )

            client_models.append(trained_model)
            client_sizes.append(len(client_data["y"]))
            client_final_losses.append(loss_history[-1])

        # Aggregate client models using FedAvg
        global_weights = fedavg_aggregate(
            client_models,
            client_sizes
        )

        # Update the global model
        global_model.load_state_dict(global_weights)

        # Calculate sample-weighted average local loss
        total_samples = sum(client_sizes)

        round_loss = sum(
            loss * size
            for loss, size in zip(client_final_losses, client_sizes)
        ) / total_samples

        convergence_history.append(round_loss)

        print(f"Round {round_num + 1} average local loss: {round_loss:.4f}")
        print(f"Federated Round {round_num + 1} completed!")

    # Return both the trained model and convergence history
    return global_model, convergence_history
