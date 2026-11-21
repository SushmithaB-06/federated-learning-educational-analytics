"""Page 6 - model evaluation on the held-out test set."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import common
from src import config
from src.evaluation import palette

KEYS = [("accuracy", "Accuracy"), ("precision", "Precision (Pass)"), ("recall", "Recall (Pass)"),
        ("f1", "F1 (Pass)"), ("fail_recall", "Recall (Fail)"), ("roc_auc", "ROC AUC")]


def _bar_chart(methods: dict, baseline: float, n_seeds: int) -> go.Figure:
    fig = go.Figure()
    for m in common.METHODS:
        agg = methods[m]
        fig.add_bar(name=agg["label"], x=[k[1] for k in KEYS], y=[agg[k[0]]["mean"] for k in KEYS],
                    error_y=dict(type="data", array=[agg[k[0]]["std"] for k in KEYS], thickness=1.2, width=4),
                    marker_color=palette.METHOD_COLORS[m], text=[f"{agg[k[0]]['mean']:.2f}" for k in KEYS],
                    textposition="outside", hovertemplate="%{x}<br>%{y:.3f}<extra>" + agg["label"] + "</extra>")
    fig.add_hline(y=baseline, line_dash="dot", line_color=palette.NEUTRAL,
                  annotation_text=f"always-Pass accuracy {baseline:.3f}", annotation_position="top left")
    fig.update_layout(barmode="group", title=f"Test-set metrics, mean ± std over {n_seeds} seeds", yaxis_range=[0, 1.15],
                      yaxis_title="score", bargap=0.25, bargroupgap=0.08)
    fig.update_traces(marker_line_color="white", marker_line_width=1.5)
    return common.style(fig, 420)


def _confusion(cm: list[list[int]], title: str) -> go.Figure:
    z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["predicted Fail", "predicted Pass"], y=["true Fail", "true Pass"],
        colorscale=[[0, palette.SEQUENTIAL[0]], [1, palette.SEQUENTIAL[-1]]], showscale=False,
        text=[[str(v) for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=22),
        hovertemplate="%{y} / %{x}: %{z} students<extra></extra>", xgap=3, ygap=3))
    fig.update_layout(title=title, yaxis_autorange="reversed")
    return common.style(fig, 330, legend_bottom=False)


def render() -> None:
    common.page_header("Model Evaluation", "All three models, same 79 held-out students, never seen during training")
    variant = st.radio("Feature variant", list(common.VARIANT_LABELS), format_func=common.VARIANT_LABELS.get, horizontal=True)
    summ = common.experiment_summary()
    v = summ["variants"][variant]
    methods = v["methods"]
    baseline = v["always_pass_baseline"]["accuracy"]
    n_seeds = len(summ["seeds"])
    runs = common.runs(variant)

    if variant == "early":
        common.callout("error",
                       f"**The early model is not usable.** Its accuracy ({methods['fedavg_iid']['accuracy']['mean']:.3f} for FedAvg IID) "
                       f"is at or below the always-Pass baseline ({baseline:.3f}) and it finds only about "
                       f"{methods['fedavg_iid']['fail_recall']['mean']:.0%} of failing students. Without G1 and G2 the remaining "
                       "attributes carry too little signal in this small dataset. It is kept as an honest negative result.")

    st.subheader("Headline comparison")
    rows = [{"Method": methods[m]["label"], **{lab: common.mean_std(methods[m], k) for k, lab in KEYS}} for m in common.METHODS]
    rows.append({"Method": "Always-Pass baseline", "Accuracy": f"{baseline:.3f}", "Precision (Pass)": f"{baseline:.3f}",
                 "Recall (Pass)": "1.000", "F1 (Pass)": f"{v['always_pass_baseline']['f1']:.3f}", "Recall (Fail)": "0.000", "ROC AUC": "-"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.plotly_chart(_bar_chart(methods, baseline, n_seeds), use_container_width=True)

    # Honest, data-driven interpretation.
    accs = {m: methods[m]["accuracy"] for m in common.METHODS}
    best = max(accs, key=lambda m: accs[m]["mean"])
    worst = min(accs, key=lambda m: accs[m]["mean"])
    spread = accs[best]["mean"] - accs[worst]["mean"]
    max_std = max(a["std"] for a in accs.values())
    st.subheader("Interpretation")
    bullets = [
        f"Highest mean accuracy: **{methods[best]['label']}** ({accs[best]['mean']:.3f}); lowest: {methods[worst]['label']} "
        f"({accs[worst]['mean']:.3f}). The gap is {100 * spread:.1f} points, "
        + ("**smaller than** the seed-to-seed standard deviation" if spread < max_std else "larger than the seed-to-seed standard deviation")
        + f" (up to {max_std:.3f}), so " + ("the three approaches are statistically indistinguishable on this test set."
                                            if spread < max_std else "the difference may be real but the test set is small."),
        f"All models beat the always-Pass baseline by {100 * (accs[worst]['mean'] - baseline):+.1f} to {100 * (accs[best]['mean'] - baseline):+.1f} points"
        if accs[worst]["mean"] > baseline else
        f"Not every model beats the always-Pass baseline ({baseline:.3f}); this variant should not be used for decisions.",
        f"Recall of the Fail class, the students an institution most wants to find, is "
        + ", ".join(f"{methods[m]['label']} {methods[m]['fail_recall']['mean']:.2f}" for m in common.METHODS) + ".",
        f"Non-IID FedAvg vs IID FedAvg: accuracy {methods['fedavg_non_iid']['accuracy']['mean']:.3f} vs "
        f"{methods['fedavg_iid']['accuracy']['mean']:.3f}; precision {methods['fedavg_non_iid']['precision']['mean']:.3f} vs "
        f"{methods['fedavg_iid']['precision']['mean']:.3f}. Label skew did not break training here, but the client-level "
        "curves on the Training page show the global model fits the low-pass-rate institution worst.",
        f"With only {methods['centralized']['confusion_matrix_mean'][0][0] + methods['centralized']['confusion_matrix_mean'][0][1]:.0f} failing and "
        f"{methods['centralized']['confusion_matrix_mean'][1][0] + methods['centralized']['confusion_matrix_mean'][1][1]:.0f} passing students in the test set, "
        "one student is about 1.3 accuracy points. Treat all differences below ~3 points as noise.",
    ]
    st.markdown("\n".join(f"- {b}" for b in bullets))

    st.divider()
    st.subheader(f"Confusion matrices (seed {config.RANDOM_SEED} models)")
    cols = st.columns(3)
    for col, m in zip(cols, common.METHODS):
        run = next(r for r in runs[m] if r["seed"] == config.RANDOM_SEED)
        tm = run["test_metrics"]
        col.plotly_chart(_confusion(tm["confusion_matrix"], f"{common.METHOD_LABELS[m]} · acc {tm['accuracy']:.3f}"),
                         use_container_width=True)
    st.caption("Rows are the true class, columns the prediction. Top-right cells are failing students the model missed.")

    st.divider()
    st.subheader("Per-class report")
    m_sel = st.selectbox("Model", common.METHODS, format_func=common.METHOD_LABELS.get)
    seed_sel = st.selectbox("Seed", summ["seeds"])
    run = next(r for r in runs[m_sel] if r["seed"] == seed_sel)
    rep = run["test_metrics"]["per_class"]
    st.dataframe(pd.DataFrame(rep).T.rename(columns={"f1-score": "f1", "support": "students"}).round(3), use_container_width=True)

    st.subheader("Per-seed results")
    per_seed = pd.DataFrame([{"method": common.METHOD_LABELS[m], "seed": r["seed"], **{k: round(r["test_metrics"][k], 3) for k, _ in KEYS}}
                             for m in common.METHODS for r in runs[m]])
    st.dataframe(per_seed, hide_index=True, use_container_width=True, height=300)
