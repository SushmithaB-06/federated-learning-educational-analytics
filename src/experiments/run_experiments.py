"""Stage 3 of the pipeline: run every experiment and save raw results.

    python -m src.experiments.run_experiments            # all variants, all seeds
    python -m src.experiments.run_experiments --quick    # seed 42 only (smoke test)

For each feature variant ("full", "early") and each seed:

* **centralized** - one model trained on the pooled training set
  (NUM_ROUNDS * LOCAL_EPOCHS epochs, so the same number of passes over the data).
* **fedavg_iid** - 3 clients with IID partitions, NUM_ROUNDS rounds of FedAvg.
* **fedavg_non_iid** - 3 clients with label-skewed partitions, same budget.

All three are evaluated on the *same* fixed test set. The partition itself uses
the fixed RANDOM_SEED so that every seed trains on identical client datasets and
the seed only changes model initialisation and mini-batch order.

Outputs
-------
results/reports/runs/<variant>_<method>_seed<seed>.json   raw per-run metrics + history
results/reports/experiment_summary.json                    mean +- std per method
results/reports/experiment_config.json                     every hyper-parameter used
results/models/<variant>/<method>.pt                       model trained with seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import config
from src.clients.client import FederatedClient
from src.clients.partition import make_partitions
from src.data.preprocess import load_processed
from src.evaluation.metrics import aggregate_over_seeds, always_pass_baseline, compute_metrics
from src.models.io import save_model
from src.models.mlp import StudentMLP, weights_num_bytes
from src.models.train import evaluate_loss, predict_label, predict_proba, train_epochs
from src.server.server import FederatedServer

METHODS = ["centralized", "fedavg_iid", "fedavg_non_iid"]
METHOD_LABELS = {"centralized": "Centralized", "fedavg_iid": "FedAvg (IID)", "fedavg_non_iid": "FedAvg (Non-IID)"}
RUNS_DIR = config.REPORTS_DIR / "runs"


def run_centralized(data, seed: int) -> tuple[StudentMLP, dict]:
    """Train on the pooled training set; hold out 20% of it for validation-loss monitoring."""
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        data.X_train, data.y_train, test_size=config.LOCAL_VALIDATION_FRACTION,
        stratify=data.y_train, random_state=seed)
    torch.manual_seed(seed)
    model = StudentMLP(data.X_train.shape[1])
    history = []
    for epoch in range(1, config.CENTRALIZED_EPOCHS + 1):
        loss = train_epochs(model, X_tr, y_tr, epochs=1, seed=seed * 1000 + epoch)[0]
        if epoch % config.LOCAL_EPOCHS == 0:  # log at the same cadence as one federated round
            history.append({
                "round": epoch // config.LOCAL_EPOCHS, "epoch": epoch,
                "train_loss": loss, "val_loss": evaluate_loss(model, X_val, y_val),
                "test_loss": evaluate_loss(model, data.X_test, data.y_test),
                "test_accuracy": float((predict_label(model, data.X_test) == data.y_test).mean()),
            })
    return model, {"history": history, "train_samples": int(len(y_tr)), "communication_bytes": 0}


def run_federated(data, scheme: str, seed: int) -> tuple[StudentMLP, dict]:
    parts = make_partitions(data.y_train, scheme, seed=config.RANDOM_SEED)
    clients = [FederatedClient(name, data.X_train[idx], data.y_train[idx], seed=seed)
               for name, idx in zip(config.CLIENT_NAMES, parts)]
    server = FederatedServer(clients, data.X_train.shape[1], data.X_test, data.y_test, seed=seed)
    history = server.run(config.NUM_ROUNDS, config.LOCAL_EPOCHS)
    records = history.to_records()
    return server.global_model, {
        "history": records,
        "train_samples": int(sum(c.num_train_samples for c in clients)),
        "clients": [{"name": c.name, "train_samples": c.num_train_samples, "val_samples": c.num_val_samples,
                     **c.class_counts()} for c in clients],
        "communication_bytes": int(sum(r["bytes_uplink"] + r["bytes_downlink"] for r in records)),
        "bytes_per_model_message": weights_num_bytes(server.global_model.get_weights()),
    }


def evaluate(model: StudentMLP, data) -> dict:
    proba = predict_proba(model, data.X_test)
    return compute_metrics(data.y_test, (proba >= 0.5).astype(int), proba)


def run_one(variant: str, method: str, seed: int, data) -> tuple[StudentMLP, dict]:
    t0 = time.perf_counter()
    if method == "centralized":
        model, extra = run_centralized(data, seed)
    else:
        model, extra = run_federated(data, method.replace("fedavg_", ""), seed)
    result = {
        "variant": variant, "method": method, "label": METHOD_LABELS[method], "seed": seed,
        "test_metrics": evaluate(model, data),
        "seconds": round(time.perf_counter() - t0, 2),
        **extra,
    }
    return model, result


def main(quick: bool = False) -> None:
    seeds = [config.RANDOM_SEED] if quick else config.EXPERIMENT_SEEDS
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"seeds": seeds, "variants": {}}

    for variant in config.EXPERIMENT_VARIANTS:
        data = load_processed(variant)
        baseline = always_pass_baseline(data.y_test)
        summary["variants"][variant] = {
            "description": config.EXPERIMENT_VARIANTS[variant]["description"],
            "n_features": int(data.X_train.shape[1]),
            "always_pass_baseline": {k: baseline[k] for k in ("accuracy", "f1", "fail_recall")},
            "methods": {},
        }
        print(f"\n===== variant: {variant} ({data.X_train.shape[1]} features) =====")
        print(f"always-Pass baseline accuracy: {baseline['accuracy']:.3f}")

        for method in METHODS:
            per_seed = []
            for seed in seeds:
                model, result = run_one(variant, method, seed, data)
                (RUNS_DIR / f"{variant}_{method}_seed{seed}.json").write_text(json.dumps(result, indent=1))
                per_seed.append(result)
                m = result["test_metrics"]
                print(f"{METHOD_LABELS[method]:18} seed {seed} | acc {m['accuracy']:.3f} | f1 {m['f1']:.3f} | "
                      f"fail recall {m['fail_recall']:.3f} | {result['seconds']}s")
                if seed == config.RANDOM_SEED:
                    save_model(model, config.MODELS_DIR / variant / f"{method}.pt", {
                        "variant": variant, "method": method, "label": METHOD_LABELS[method], "seed": seed,
                        "n_features": int(data.X_train.shape[1]), "test_metrics": m,
                        "trained_on": "results of python -m src.experiments.run_experiments",
                    })
            agg = aggregate_over_seeds([r["test_metrics"] for r in per_seed])
            agg["communication_bytes"] = per_seed[0]["communication_bytes"]
            agg["final_round"] = per_seed[0]["history"][-1]
            summary["variants"][variant]["methods"][method] = {"label": METHOD_LABELS[method], **agg}

    summary["config"] = {
        k: getattr(config, k) for k in (
            "NUM_CLIENTS", "CLIENT_NAMES", "NON_IID_PASS_FRACTIONS", "HIDDEN_LAYERS", "LEARNING_RATE",
            "BATCH_SIZE", "LOCAL_EPOCHS", "NUM_ROUNDS", "CENTRALIZED_EPOCHS", "LOCAL_VALIDATION_FRACTION",
            "TEST_SIZE", "RANDOM_SEED", "EXPERIMENT_SEEDS", "PASS_THRESHOLD")
    }
    summary["config"].update({"optimizer": "Adam", "loss": "BCEWithLogitsLoss", "framework": f"torch {torch.__version__}"})
    (config.REPORTS_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    (config.REPORTS_DIR / "experiment_config.json").write_text(json.dumps(summary["config"], indent=2))

    print("\n===== mean +- std over seeds =====")
    for variant, v in summary["variants"].items():
        for method, agg in v["methods"].items():
            print(f"[{variant:5}] {agg['label']:18} acc {agg['accuracy']['mean']:.3f}+-{agg['accuracy']['std']:.3f} | "
                  f"f1 {agg['f1']['mean']:.3f} | fail recall {agg['fail_recall']['mean']:.3f}")
    print(f"\nSaved {config.REPORTS_DIR / 'experiment_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="single seed smoke run")
    main(quick=parser.parse_args().quick)
