"""Tests for evaluation metrics."""
import numpy as np

from src.evaluation import metrics


def test_perfect_predictions():
    y = np.array([0, 1, 1, 0, 1])
    m = metrics.compute_metrics(y, y, y.astype(float))
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 3]]
    assert m["roc_auc"] == 1.0


def test_known_confusion_matrix_values():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    m = metrics.compute_metrics(y_true, y_pred)
    # TN=2 FP=2 FN=1 TP=5
    assert m["confusion_matrix"] == [[2, 2], [1, 5]]
    assert abs(m["accuracy"] - 0.7) < 1e-9
    assert abs(m["precision"] - 5 / 7) < 1e-9
    assert abs(m["recall"] - 5 / 6) < 1e-9
    assert abs(m["fail_recall"] - 0.5) < 1e-9


def test_always_pass_baseline_has_zero_fail_recall():
    y = np.array([1] * 53 + [0] * 26)
    b = metrics.always_pass_baseline(y)
    assert abs(b["accuracy"] - 53 / 79) < 1e-9
    assert b["fail_recall"] == 0.0


def test_aggregate_mean_std():
    runs = [{"accuracy": 0.8, "f1": 0.9, "confusion_matrix": [[1, 1], [1, 1]]},
            {"accuracy": 0.6, "f1": 0.7, "confusion_matrix": [[3, 1], [1, 3]]}]
    agg = metrics.aggregate_over_seeds(runs)
    assert agg["n_runs"] == 2
    assert abs(agg["accuracy"]["mean"] - 0.7) < 1e-9
    assert abs(agg["accuracy"]["std"] - 0.1) < 1e-9
    assert agg["confusion_matrix_mean"] == [[2.0, 1.0], [1.0, 2.0]]
    assert "roc_auc" not in agg  # absent in inputs, so absent in output
