"""Stage 1 of the pipeline: verify the dataset and build both preprocessing variants.

    python -m src.experiments.prepare_data
"""
from __future__ import annotations

import json

import numpy as np

from src import config
from src.data import load, preprocess


def main() -> None:
    df = load.load_raw_dataset()
    summary = load.summarise(df)
    report = {
        "dataset": config.DATASET_NAME,
        "source": config.DATASET_URL,
        "rows": summary.rows,
        "columns": summary.columns,
        "missing_values": summary.missing_values,
        "target": f"pass = (G3 >= {config.PASS_THRESHOLD})",
        "pass_count": summary.pass_count,
        "fail_count": summary.fail_count,
        "pass_rate": round(summary.pass_rate, 4),
        "test_size": config.TEST_SIZE,
        "split_seed": config.RANDOM_SEED,
        "variants": {},
    }

    for variant in config.EXPERIMENT_VARIANTS:
        data, schema = preprocess.prepare_variant(df, variant)
        n_train, n_test = len(data.y_train), len(data.y_test)
        report["variants"][variant] = {
            "description": schema["description"],
            "input_columns": schema["n_input_columns"],
            "features_after_encoding": schema["n_features_after_encoding"],
            "train_rows": n_train,
            "test_rows": n_test,
            "train_pass": int(data.y_train.sum()),
            "train_fail": int((data.y_train == 0).sum()),
            "test_pass": int(data.y_test.sum()),
            "test_fail": int((data.y_test == 0).sum()),
            "always_pass_baseline_accuracy": round(float(data.y_test.mean()), 4),
        }
        print(
            f"[{variant:5}] {schema['n_input_columns']} raw inputs -> {schema['n_features_after_encoding']} features | "
            f"train {n_train} ({int(data.y_train.sum())} pass) | test {n_test} ({int(data.y_test.sum())} pass)"
        )
        # Sanity: scaler was fit on train, so scaled training numerics have ~0 mean.
        num_start = schema["n_features_after_encoding"] - len(
            [c for c in schema["input_columns"] if c not in load.CATEGORICAL_COLUMNS]
        )
        assert abs(float(np.mean(data.X_train[:, num_start:]))) < 1e-5

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "dataset_summary.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
