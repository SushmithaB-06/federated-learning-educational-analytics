"""Stage 4 of the pipeline: turn the raw run files into tables and figures.

    python -m src.evaluation.make_figures

Reads results/reports/runs/*.json and experiment_summary.json, writes:

results/reports/metrics_table.csv / .md     headline metrics, mean +- std
results/figures/metrics_<variant>.png       bar chart with error bars vs always-Pass baseline
results/figures/convergence_<variant>.png   train / validation loss per round (mean, +-1 std)
results/figures/test_accuracy_<variant>.png test accuracy per round (monitoring only)
results/figures/confusion_<variant>.png     confusion matrices of the seed-42 models
results/figures/client_val_loss_<variant>.png   per-client validation loss, IID vs Non-IID
results/figures/communication_cost.png      cumulative bytes exchanged per round
"""
from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.evaluation.palette import CLIENT_COLORS, METHOD_COLORS as COLORS, NEUTRAL, SEQUENTIAL
from src.experiments.run_experiments import METHOD_LABELS, METHODS, RUNS_DIR
from matplotlib.colors import LinearSegmentedColormap

BLUES = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def load_runs() -> dict[str, dict[str, list[dict]]]:
    runs: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    files = sorted(RUNS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No run files in {RUNS_DIR}. Run `python main.py experiments` first.")
    for f in files:
        r = json.loads(f.read_text())
        runs[r["variant"]][r["method"]].append(r)
    return runs


def metrics_table(summary: dict) -> pd.DataFrame:
    rows = []
    for variant, v in summary["variants"].items():
        b = v["always_pass_baseline"]
        rows.append({"variant": variant, "method": "Always-Pass baseline", "accuracy": f"{b['accuracy']:.3f}",
                     "precision": "-", "recall": "-", "f1": f"{b['f1']:.3f}", "fail_recall": f"{b['fail_recall']:.3f}",
                     "roc_auc": "-", "comm_MB": "-"})
        for method, agg in v["methods"].items():
            fmt = lambda k: f"{agg[k]['mean']:.3f} +- {agg[k]['std']:.3f}"
            rows.append({"variant": variant, "method": agg["label"], "accuracy": fmt("accuracy"),
                         "precision": fmt("precision"), "recall": fmt("recall"), "f1": fmt("f1"),
                         "fail_recall": fmt("fail_recall"), "roc_auc": fmt("roc_auc"),
                         "comm_MB": f"{agg['communication_bytes'] / 1e6:.2f}"})
    return pd.DataFrame(rows)


def plot_metrics(summary: dict, variant: str) -> None:
    v = summary["variants"][variant]
    keys = ["accuracy", "precision", "recall", "f1", "fail_recall"]
    labels = ["Accuracy", "Precision\n(Pass)", "Recall\n(Pass)", "F1\n(Pass)", "Recall\n(Fail)"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.26
    x = np.arange(len(keys))
    for i, method in enumerate(METHODS):
        agg = v["methods"][method]
        means = [agg[k]["mean"] for k in keys]
        stds = [agg[k]["std"] for k in keys]
        bars = ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, color=COLORS[method], label=agg["label"])
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.2f}", ha="center", fontsize=7.5)
    ax.axhline(v["always_pass_baseline"]["accuracy"], color=NEUTRAL, ls=":", lw=1.2,
               label=f"Always-Pass accuracy ({v['always_pass_baseline']['accuracy']:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score on held-out test set (79 students)")
    ax.set_title(f"Test metrics, {variant} variant - mean +- std over {summary['seeds'].__len__()} seeds")
    ax.legend(loc="upper center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"metrics_{variant}.png", bbox_inches="tight")
    plt.close(fig)


def _stack(runs: list[dict], key: str) -> np.ndarray:
    return np.array([[h[key] for h in r["history"]] for r in runs])


def plot_convergence(runs: dict, variant: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, key, title in zip(axes, ["train_loss", "val_loss"],
                              ["Training loss (clients' local data)", "Validation loss (clients' held-out slices)"]):
        for method in METHODS:
            arr = _stack(runs[variant][method], key)
            rounds = np.arange(1, arr.shape[1] + 1)
            mean, std = arr.mean(0), arr.std(0)
            ax.plot(rounds, mean, color=COLORS[method], label=METHOD_LABELS[method], lw=2)
            ax.fill_between(rounds, mean - std, mean + std, color=COLORS[method], alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("communication round (centralized: 5 epochs = 1 round)")
        ax.set_ylabel("binary cross-entropy")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Convergence, {variant} variant (mean +- 1 std over seeds)", y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"convergence_{variant}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method in METHODS:
        arr = _stack(runs[variant][method], "test_accuracy")
        rounds = np.arange(1, arr.shape[1] + 1)
        ax.plot(rounds, arr.mean(0), color=COLORS[method], label=METHOD_LABELS[method], lw=2)
        ax.fill_between(rounds, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), color=COLORS[method], alpha=0.15)
    ax.set_xlabel("communication round")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Test accuracy per round, {variant} (monitoring only, no model selection)")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"test_accuracy_{variant}.png")
    plt.close(fig)


def plot_confusions(runs: dict, variant: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, method in zip(axes, METHODS):
        run = next(r for r in runs[variant][method] if r["seed"] == config.RANDOM_SEED)
        cm = np.array(run["test_metrics"]["confusion_matrix"])
        ax.imshow(cm, cmap=BLUES, vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred Fail", "pred Pass"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true Fail", "true Pass"])
        m = run["test_metrics"]
        ax.set_title(f"{METHOD_LABELS[method]}\nacc {m['accuracy']:.3f}  F1 {m['f1']:.3f}", fontsize=10)
    fig.suptitle(f"Confusion matrices on the test set, {variant} variant, seed {config.RANDOM_SEED}", y=1.03)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"confusion_{variant}.png", bbox_inches="tight")
    plt.close(fig)


def plot_client_val_loss(runs: dict, variant: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, method in zip(axes, ["fedavg_iid", "fedavg_non_iid"]):
        run = next(r for r in runs[variant][method] if r["seed"] == config.RANDOM_SEED)
        rounds = [h["round"] for h in run["history"]]
        for color, client in zip(CLIENT_COLORS, run["clients"]):
            name = client["name"]
            rate = client["pass"] / (client["pass"] + client["fail"])
            ax.plot(rounds, [h["client_val_loss"][name] for h in run["history"]], color=color, lw=2,
                    label=f"{name} ({rate:.0%} pass)")
        ax.set_title(f"{METHOD_LABELS[method]}: global model's loss on each client")
        ax.set_xlabel("communication round")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("validation loss")
    fig.suptitle(f"Client-level validation loss, {variant} variant, seed {config.RANDOM_SEED}", y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"client_val_loss_{variant}.png", bbox_inches="tight")
    plt.close(fig)


def plot_communication(runs: dict) -> None:
    run = runs["full"]["fedavg_iid"][0]
    per_round = np.array([h["bytes_uplink"] + h["bytes_downlink"] for h in run["history"]]) / 1e6
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(range(1, len(per_round) + 1), per_round, color=COLORS["fedavg_iid"], label="per round")
    ax.plot(range(1, len(per_round) + 1), per_round.cumsum(), color=COLORS["fedavg_non_iid"], marker="o", ms=3,
            label="cumulative")
    ax.set_xlabel("communication round")
    ax.set_ylabel("MB exchanged (3 clients, float32 weights)")
    ax.set_title(f"Communication cost: {run['bytes_per_model_message'] / 1e3:.1f} KB per model message, "
                 f"{per_round.sum():.2f} MB total")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "communication_cost.png")
    plt.close(fig)


def main() -> None:
    summary_path = config.REPORTS_DIR / "experiment_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found. Run `python main.py experiments` first.")
    summary = json.loads(summary_path.read_text())
    runs = load_runs()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    table = metrics_table(summary)
    table.to_csv(config.REPORTS_DIR / "metrics_table.csv", index=False)
    (config.REPORTS_DIR / "metrics_table.md").write_text(table.to_markdown(index=False))
    print(table.to_string(index=False))

    for variant in summary["variants"]:
        plot_metrics(summary, variant)
        plot_convergence(runs, variant)
        plot_confusions(runs, variant)
        plot_client_val_loss(runs, variant)
    plot_communication(runs)
    print(f"\nFigures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
