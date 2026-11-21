"""Page 1 - project overview."""
from __future__ import annotations

import streamlit as st

from dashboard import common
from dashboard.views.architecture import architecture_svg
from src import config


def render() -> None:
    common.page_header(
        "Federated Learning for Educational Analytics",
        "Predicting student Pass/Fail across simulated institutions without pooling raw student records",
    )

    ds = common.dataset_summary()
    summ = common.experiment_summary()
    full = summ["variants"]["full"]["methods"]
    base = summ["variants"]["full"]["always_pass_baseline"]["accuracy"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students in dataset", ds["rows"])
    c2.metric("Simulated institutions", summ["config"]["NUM_CLIENTS"])
    c3.metric("Communication rounds", summ["config"]["NUM_ROUNDS"])
    c4.metric("FedAvg (IID) test accuracy", common.pct(full["fedavg_iid"]["accuracy"]["mean"]),
              f"{100 * (full['fedavg_iid']['accuracy']['mean'] - base):+.1f} pts vs always-Pass",
              help=f"Mean over {len(summ['seeds'])} seeds, std {full['fedavg_iid']['accuracy']['std']:.3f}")
    c5.metric("Centralized test accuracy", common.pct(full["centralized"]["accuracy"]["mean"]),
              f"{100 * (full['centralized']['accuracy']['mean'] - base):+.1f} pts vs always-Pass",
              help=f"Same data, model and training budget, trained on pooled data")

    st.divider()
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Problem statement")
        st.markdown(
            "Educational institutions hold rich data about their students, and models trained on more data from "
            "more institutions are usually better. But student records are personal and governed by privacy law "
            "and institutional policy, so pooling them in one central database is often impossible.\n\n"
            "**Federated learning (FL)** offers a way out: every institution trains the same model *locally*, and "
            "only the model's numbers (weights) travel to a central server, which averages them into a shared "
            "global model."
        )
        st.subheader("Objectives")
        st.markdown(
            f"1. Build a reproducible FL simulation with {config.NUM_CLIENTS} clients and a FedAvg server.\n"
            "2. Compare federated training with a centralized baseline on the same held-out test set.\n"
            "3. Study the effect of **IID vs Non-IID** (statistically different) client data.\n"
            "4. Serve a validated Pass/Fail predictor and an interactive analytics dashboard.\n"
            "5. Report results and limitations honestly, including the privacy properties FedAvg does *not* have."
        )
    with right:
        st.subheader("How the system works")
        st.markdown(architecture_svg(compact=True), unsafe_allow_html=True)
        st.caption("Raw student rows stay inside each institution; only model weights cross the dashed line.")

    st.divider()
    st.subheader("What this project is, and is not")
    a, b = st.columns(2)
    with a:
        common.callout("success",
                       "**It is** a controlled simulation: the public UCI Student Performance dataset is split into "
                       "three partitions that stand in for three institutions. Everything is reproducible with fixed seeds.")
    with b:
        common.callout("warning",
                       "**It is not** a deployment with real colleges, and plain FedAvg does **not** give formal privacy "
                       "guarantees. See the Privacy & Limitations page.")

    st.subheader("Headline result (full model, mean over seeds)")
    rows = []
    for m in common.METHODS:
        agg = full[m]
        rows.append({"Method": agg["label"], "Accuracy": common.mean_std(agg, "accuracy"),
                     "F1 (Pass)": common.mean_std(agg, "f1"), "Recall (Fail)": common.mean_std(agg, "fail_recall")})
    rows.append({"Method": "Always-Pass baseline", "Accuracy": f"{base:.3f}", "F1 (Pass)": "-", "Recall (Fail)": "0.000"})
    st.table(rows)
    st.caption("Differences between the three trained models are within one standard deviation across seeds: "
               "on this dataset federated training matches centralized training without pooling the data.")
