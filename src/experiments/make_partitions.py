"""Stage 2 of the pipeline: build and document the IID and Non-IID client partitions.

    python -m src.experiments.make_partitions

Partitions are index lists over the TRAINING set and are identical for both
feature variants, because both variants use the same train/test split.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.clients.partition import describe_partitions, make_partitions
from src.data.preprocess import load_processed
from src.evaluation.palette import FAIL_COLOR, PASS_COLOR


def plot_distributions(tables: dict[str, "pd.DataFrame"], path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (scheme, tbl) in zip(axes, tables.items()):
        x = np.arange(len(tbl))
        ax.bar(x, tbl["pass"], color=PASS_COLOR, label="Pass", edgecolor="white", linewidth=2)
        ax.bar(x, tbl["fail"], bottom=tbl["pass"], color=FAIL_COLOR, label="Fail", edgecolor="white", linewidth=2)
        for i, r in tbl.iterrows():
            ax.text(i, r["students"] + 2, f"{r['pass_rate']:.0%} pass", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(tbl["client"])
        ax.set_title({"iid": "IID: similar class balance", "non_iid": "Non-IID: label skew"}[scheme])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("training students")
    axes[0].set_ylim(0, max(t["students"].max() for t in tables.values()) * 1.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    y_train = load_processed("full").y_train
    out = {"seed": config.RANDOM_SEED, "num_clients": config.NUM_CLIENTS, "train_rows": int(len(y_train)), "schemes": {}}
    tables = {}
    for scheme in ("iid", "non_iid"):
        parts = make_partitions(y_train, scheme)
        tbl = describe_partitions(y_train, parts)
        tables[scheme] = tbl
        out["schemes"][scheme] = {
            "clients": tbl.to_dict(orient="records"),
            "indices": [p.tolist() for p in parts],
        }
        print(f"\n{scheme.upper()}")
        print(tbl.to_string(index=False))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "partitions.json").write_text(json.dumps(out, indent=2))
    plot_distributions(tables, config.FIGURES_DIR / "client_distributions.png")
    print(f"\nSaved partitions.json and client_distributions.png")


if __name__ == "__main__":
    main()
