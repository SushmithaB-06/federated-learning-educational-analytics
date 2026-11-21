"""Save and load trained models together with the metadata needed to rebuild them."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.models.mlp import StudentMLP


def save_model(model: StudentMLP, path: Path, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "hidden": model.hidden,
        "output_dim": model.output_dim,
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    path.with_suffix(".json").write_text(json.dumps(
        {k: v for k, v in payload.items() if k != "state_dict"}, indent=2))


def load_model(path: Path) -> tuple[StudentMLP, dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file {path} not found. Run `python main.py experiments` first.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = StudentMLP(payload["input_dim"], payload["hidden"], payload["output_dim"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("metadata", {})
