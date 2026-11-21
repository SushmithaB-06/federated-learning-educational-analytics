"""Page 5 - training and convergence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import common
from src import config
from src.evaluation import palette


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _band_lines(runs: dict, key: str, title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    for method in common.METHODS:
        arr = np.array([[h[key] for h in r["history"]] for r in runs[method]])
        rounds = np.arange(1, arr.shape[1] + 1)
        mean, std = arr.mean(0), arr.std(0)
        col = palette.METHOD_COLORS[method]
        fig.add_scatter(x=np.r_[rounds, rounds[::-1]], y=np.r_[mean + std, (mean - std)[::-1]], fill="toself",
                        fillcolor=_hex_to_rgba(col, 0.12), line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                        showlegend=False)
        fig.add_scatter(x=rounds, y=mean, mode="lines", name=common.METHOD_LABELS[method], line=dict(color=col, width=2.5),
                        hovertemplate=f"{common.METHOD_LABELS[method]}<br>round %{{x}}<br>{ytitle}: %{{y:.4f}}<extra></extra>")
    fig.update_layout(title=title, xaxis_title="communication round (centralized: 5 epochs = 1 round)", yaxis_title=ytitle,
                      hovermode="x unified")
    return common.style(fig, 380)


def render() -> None:
    common.page_header("Training & Convergence", "What the server observed round by round")
    variant = st.radio("Feature variant", list(common.VARIANT_LABELS), format_func=common.VARIANT_LABELS.get, horizontal=True)
    runs = common.runs(variant)
    summ = common.experiment_summary()
    cfg = summ["config"]

    st.subheader("Training configuration")
    c = st.columns(6)
    c[0].metric("Clients", cfg["NUM_CLIENTS"])
    c[1].metric("Rounds", cfg["NUM_ROUNDS"])
    c[2].metric("Local epochs / round", cfg["LOCAL_EPOCHS"])
    c[3].metric("Batch size", cfg["BATCH_SIZE"])
    c[4].metric("Learning rate", cfg["LEARNING_RATE"])
    c[5].metric("Seeds", len(summ["seeds"]))
    st.caption(f"Model: input → {' → '.join(map(str, cfg['HIDDEN_LAYERS']))} → 1 (ReLU hidden layers), loss {cfg['loss']}, "
               f"optimizer {cfg['optimizer']}. The centralized baseline trains for {cfg['CENTRALIZED_EPOCHS']} epochs, the "
               f"same number of passes over the data as {cfg['NUM_ROUNDS']} rounds × {cfg['LOCAL_EPOCHS']} local epochs.")

    st.divider()
    st.subheader("Loss curves (mean ± 1 std over seeds)")
    a, b = st.columns(2)
    a.plotly_chart(_band_lines(runs, "train_loss", "Training loss on clients' local data", "binary cross-entropy"),
                   use_container_width=True)
    b.plotly_chart(_band_lines(runs, "val_loss", "Validation loss on clients' held-out slices", "binary cross-entropy"),
                   use_container_width=True)

    # Data-driven commentary.
    def _curve(method, key):
        return np.array([[h[key] for h in r["history"]] for r in runs[method]]).mean(0)

    msgs = []
    for method in common.METHODS:
        v = _curve(method, "val_loss")
        best = int(v.argmin()) + 1
        if v[-1] > v.min() * 1.15:
            msgs.append(f"**{common.METHOD_LABELS[method]}**: validation loss is lowest at round {best} "
                        f"({v.min():.3f}) and rises to {v[-1]:.3f} by round {len(v)}, a sign of over-fitting with no early stopping.")
        else:
            msgs.append(f"**{common.METHOD_LABELS[method]}**: validation loss is lowest at round {best} ({v.min():.3f}) "
                        f"and stays near it ({v[-1]:.3f} at round {len(v)}).")
    st.markdown("\n".join(f"- {m}" for m in msgs))
    common.callout("info", "Training loss and validation loss are **not** test performance. Test metrics are on the Model Evaluation page.")

    st.divider()
    st.subheader("Test accuracy per round (monitoring only)")
    st.plotly_chart(_band_lines(runs, "test_accuracy", "Global model accuracy on the fixed test set after each round", "accuracy"),
                    use_container_width=True)
    st.caption("Shown for transparency. No decision (model selection, early stopping) was made from this curve; the model "
               "after the final round is the one evaluated.")

    st.divider()
    st.subheader(f"Client-level view (seed {config.RANDOM_SEED})")
    a, b = st.columns(2)
    for col, method in zip((a, b), ["fedavg_iid", "fedavg_non_iid"]):
        run = next(r for r in runs[method] if r["seed"] == config.RANDOM_SEED)
        fig = go.Figure()
        for color, client in zip(palette.CLIENT_COLORS, run["clients"]):
            name = client["name"]
            rate = client["pass"] / (client["pass"] + client["fail"])
            fig.add_scatter(x=[h["round"] for h in run["history"]], y=[h["client_val_loss"][name] for h in run["history"]],
                            mode="lines", name=f"{name} ({rate:.0%} pass)", line=dict(color=color, width=2.5),
                            hovertemplate=f"{name}<br>round %{{x}}<br>val loss %{{y:.4f}}<extra></extra>")
        fig.update_layout(title=f"{common.METHOD_LABELS[method]}: global model's validation loss on each client",
                          xaxis_title="communication round", yaxis_title="validation loss", hovermode="x unified")
        col.plotly_chart(common.style(fig, 360), use_container_width=True)
    st.caption("Under Non-IID data the global model fits the majority-Pass institutions better than the institution "
               "with many failing students: a single averaged model is a compromise between differing populations.")

    st.divider()
    st.subheader("Round-by-round table")
    seed = st.selectbox("Seed", summ["seeds"])
    method = st.selectbox("Method", common.METHODS, format_func=common.METHOD_LABELS.get)
    run = next(r for r in runs[method] if r["seed"] == seed)
    hist = pd.DataFrame(run["history"])
    keep = [c for c in ["round", "epoch", "train_loss", "val_loss", "test_loss", "test_accuracy", "bytes_uplink", "bytes_downlink"]
            if c in hist.columns]
    st.dataframe(hist[keep].round(4), hide_index=True, use_container_width=True, height=300)

    st.divider()
    st.subheader("Communication cost")
    fl_run = runs["fedavg_iid"][0]
    per_msg = fl_run["bytes_per_model_message"]
    total = fl_run["communication_bytes"]
    c1, c2, c3 = st.columns(3)
    c1.metric("One model message", f"{per_msg / 1e3:.1f} KB", help="2,433 float32 parameters × 4 bytes")
    c2.metric("Per round (3 up + 3 down)", f"{6 * per_msg / 1e3:.1f} KB")
    c3.metric(f"Total over {cfg['NUM_ROUNDS']} rounds", f"{total / 1e6:.2f} MB")
    st.caption("For comparison, the raw training CSV of all 316 students is about 45 KB, so for a dataset this small FL "
               "sends *more* bytes than centralisation would. FL's benefit here is governance, not bandwidth.")
