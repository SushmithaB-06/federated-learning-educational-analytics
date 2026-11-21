"""Shared helpers for every dashboard page: paths, cached loaders, styling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.evaluation import palette  # noqa: E402

METHODS = ["centralized", "fedavg_iid", "fedavg_non_iid"]
METHOD_LABELS = {"centralized": "Centralized", "fedavg_iid": "FedAvg (IID)", "fedavg_non_iid": "FedAvg (Non-IID)"}
VARIANT_LABELS = {"full": "Full model (uses G1, G2)", "early": "Early model (no G1, G2) - experimental"}

PIPELINE_HINT = (
    "Run the pipeline first from the project root:\n\n"
    "```powershell\n.\\.venv\\Scripts\\Activate.ps1\npython main.py all\n```"
)


# --------------------------------------------------------------------------- #
# Artifact loading (cached, with friendly errors)
# --------------------------------------------------------------------------- #
def require(*paths: Path) -> None:
    """Stop the page with a clear message if any required artifact is missing."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        st.error("Some result files are missing, so this page cannot be shown yet.")
        for p in missing:
            st.code(str(p.relative_to(PROJECT_ROOT)), language=None)
        st.markdown(PIPELINE_HINT)
        st.stop()


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def dataset_summary() -> dict:
    p = config.REPORTS_DIR / "dataset_summary.json"
    require(p)
    return load_json(str(p))


def experiment_summary() -> dict:
    p = config.REPORTS_DIR / "experiment_summary.json"
    require(p)
    return load_json(str(p))


def partitions() -> dict:
    p = config.REPORTS_DIR / "partitions.json"
    require(p)
    return load_json(str(p))


@st.cache_data(show_spinner=False)
def load_runs(variant: str) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {m: [] for m in METHODS}
    for f in sorted((config.REPORTS_DIR / "runs").glob(f"{variant}_*.json")):
        r = json.loads(f.read_text())
        runs[r["method"]].append(r)
    return runs


def runs(variant: str) -> dict[str, list[dict]]:
    require(config.REPORTS_DIR / "runs")
    out = load_runs(variant)
    if not any(out.values()):
        st.error(f"No experiment runs found for variant '{variant}'.")
        st.markdown(PIPELINE_HINT)
        st.stop()
    return out


@st.cache_data(show_spinner=False)
def raw_dataframe() -> pd.DataFrame:
    from src.data import load
    return load.load_raw_dataset()


def raw_df() -> pd.DataFrame:
    require(config.RAW_DATASET_PATH)
    return raw_dataframe()


@st.cache_resource(show_spinner="Loading model...")
def predictor(variant: str, model_key: str):
    from src.utils.predictor import StudentPredictor
    return StudentPredictor(variant, model_key)


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def style(fig: go.Figure, height: int = 380, legend_bottom: bool = True) -> go.Figure:
    """Apply the shared look: quiet grid, system font, legend below the plot."""
    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=palette.INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, bgcolor="rgba(0,0,0,0)") if legend_bottom else {},
    )
    fig.update_xaxes(showgrid=False, linecolor=palette.GRID, zeroline=False)
    fig.update_yaxes(gridcolor=palette.GRID, zeroline=False)
    return fig


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def mean_std(agg: dict, key: str, digits: int = 3) -> str:
    return f"{agg[key]['mean']:.{digits}f} ± {agg[key]['std']:.{digits}f}"


def callout(kind: str, text: str) -> None:
    {"info": st.info, "warning": st.warning, "success": st.success, "error": st.error}[kind](text)
