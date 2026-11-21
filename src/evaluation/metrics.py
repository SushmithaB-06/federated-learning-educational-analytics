"""Classification metrics computed on the held-out test set.

"Pass" (label 1) is treated as the positive class for precision/recall/F1, which
is the scikit-learn default for binary labels. Because catching students who will
*fail* is what an institution actually cares about, Fail recall is also reported.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_NAMES = ["Fail", "Pass"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """All headline metrics plus confusion matrix and per-class report, JSON-friendly."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fail_recall": float(report["Fail"]["recall"]),
        "fail_precision": float(report["Fail"]["precision"]),
        "balanced_accuracy": float((report["Fail"]["recall"] + report["Pass"]["recall"]) / 2),
        "confusion_matrix": cm.tolist(),  # rows = true [Fail, Pass], cols = predicted [Fail, Pass]
        "per_class": {k: report[k] for k in CLASS_NAMES},
        "n_test": int(len(y_true)),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return out


def always_pass_baseline(y_true: np.ndarray) -> dict:
    """Trivial model that predicts Pass for everyone - the bar any real model must beat."""
    return compute_metrics(y_true, np.ones_like(y_true))


HEADLINE_KEYS = ["accuracy", "precision", "recall", "f1", "fail_recall", "balanced_accuracy", "roc_auc"]


def aggregate_over_seeds(metric_dicts: list[dict]) -> dict:
    """Mean and standard deviation of each headline metric across repeated runs."""
    agg = {"n_runs": len(metric_dicts)}
    for key in HEADLINE_KEYS:
        vals = [m[key] for m in metric_dicts if key in m]
        if vals:
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": [float(v) for v in vals]}
    cms = np.array([m["confusion_matrix"] for m in metric_dicts])
    agg["confusion_matrix_mean"] = cms.mean(axis=0).round(2).tolist()
    return agg
