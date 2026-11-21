"""Single entry point for the whole pipeline.

Usage (from the project root, with the venv active):

    python main.py verify        # check the raw dataset
    python main.py prepare       # split, preprocess, save artifacts
    python main.py partition     # build IID / Non-IID client splits
    python main.py experiments   # centralized + federated runs (all seeds)
    python main.py figures       # evaluation tables and plots
    python main.py all           # everything above, in order
    python main.py dashboard     # launch Streamlit

Each stage is also runnable on its own as ``python -m src.experiments.<stage>``.
"""
from __future__ import annotations

import subprocess
import sys

STAGES = {
    "verify": "src.data.load",
    "prepare": "src.experiments.prepare_data",
    "partition": "src.experiments.make_partitions",
    "experiments": "src.experiments.run_experiments",
    "figures": "src.evaluation.make_figures",
}
PIPELINE_ORDER = ["verify", "prepare", "partition", "experiments", "figures"]


def run_module(module: str) -> None:
    print(f"\n=== {module} ===")
    subprocess.run([sys.executable, "-m", module], check=True)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {*STAGES, "all", "dashboard"}:
        print(__doc__)
        return 2
    stage = argv[1]
    if stage == "dashboard":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"], check=True)
        return 0
    stages = PIPELINE_ORDER if stage == "all" else [stage]
    for s in stages:
        run_module(STAGES[s])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
