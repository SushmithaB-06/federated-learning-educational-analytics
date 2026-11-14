import copy
import torch


def fedavg_aggregate(client_models, client_sizes):

    global_weights = copy.deepcopy(
        client_models[0].state_dict()
    )

    total_samples = sum(client_sizes)

    for key in global_weights.keys():

        global_weights[key] = torch.zeros_like(
            global_weights[key]
        )

        for model, size in zip(client_models, client_sizes):

            client_weights = model.state_dict()[key]

            global_weights[key] += (
                client_weights * (size / total_samples)
            )

    return global_weights