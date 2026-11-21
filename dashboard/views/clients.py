"""Page 4 - client data distribution (IID vs Non-IID)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import common
from src.evaluation import palette


def _stacked(tbl: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name="Pass", x=tbl["client"], y=tbl["pass"], marker_color=palette.PASS_COLOR,
                text=tbl["pass"], textposition="inside", hovertemplate="%{x}<br>Pass: %{y}<extra></extra>")
    fig.add_bar(name="Fail", x=tbl["client"], y=tbl["fail"], marker_color=palette.FAIL_COLOR,
                text=tbl["fail"], textposition="inside", hovertemplate="%{x}<br>Fail: %{y}<extra></extra>")
    for _, r in tbl.iterrows():
        fig.add_annotation(x=r["client"], y=r["students"], text=f"{r['pass_rate']:.0%} pass", showarrow=False,
                           yshift=14, font=dict(size=12, color=palette.INK_SECONDARY))
    fig.update_layout(barmode="stack", title=title, yaxis_title="training students", bargap=0.35,
                      yaxis_range=[0, tbl["students"].max() * 1.2])
    fig.update_traces(marker_line_color="white", marker_line_width=2)
    return common.style(fig, 380)


def render() -> None:
    common.page_header("Client Data Distribution", "How the 316 training students are divided among three simulated institutions")
    parts = common.partitions()
    schemes = parts["schemes"]

    st.markdown(
        f"The **test set (79 students) is never given to any client**. Only the {parts['train_rows']} training students "
        f"are partitioned, deterministically with seed {parts['seed']}, and each student goes to exactly one institution. "
        "Both partitions are used with the same model, the same number of rounds and the same test set, so any "
        "difference in results comes from the data distribution alone.")

    tabs = st.tabs(["Side by side", "IID", "Non-IID"])
    tbl_iid = pd.DataFrame(schemes["iid"]["clients"])
    tbl_non = pd.DataFrame(schemes["non_iid"]["clients"])

    with tabs[0]:
        a, b = st.columns(2)
        a.plotly_chart(_stacked(tbl_iid, "IID: similar class balance everywhere"), use_container_width=True)
        b.plotly_chart(_stacked(tbl_non, "Non-IID: deliberately skewed pass rates"), use_container_width=True)
    with tabs[1]:
        st.plotly_chart(_stacked(tbl_iid, "IID partition"), use_container_width=True)
        st.dataframe(tbl_iid, hide_index=True, use_container_width=True)
    with tabs[2]:
        st.plotly_chart(_stacked(tbl_non, "Non-IID partition"), use_container_width=True)
        st.dataframe(tbl_non, hide_index=True, use_container_width=True)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("IID - independent and identically distributed")
        st.markdown(
            "Pass and Fail students are shuffled and dealt out evenly, so every institution sees roughly the same "
            f"{tbl_iid['pass_rate'].mean():.0%} pass rate. This is the easy case for FedAvg: each local model learns "
            "approximately the same thing, so averaging them loses little.")
    with right:
        st.subheader("Non-IID - statistical heterogeneity")
        st.markdown(
            f"Institution A gets {tbl_non.loc[0, 'pass_rate']:.0%} passing students, Institution C only "
            f"{tbl_non.loc[2, 'pass_rate']:.0%}. This **label skew** imitates reality: a selective school and a "
            "struggling school have different outcome distributions. Each local model now drifts toward its own "
            "majority class, and the averaged model has to compromise. The literature (McMahan et al., 2017; "
            "Zhao et al., 2018) reports that this slows convergence and can reduce accuracy; the Training and "
            "Evaluation pages show what happened here.")

    st.subheader("Verification")
    st.markdown(
        "- Every training index appears in exactly one client (checked in code and in `tests/test_partition.py`).\n"
        "- Partitions are identical on every run thanks to a fixed random seed.\n"
        "- Each client additionally holds back 20% of its own students as a private validation slice used only "
        "to monitor loss; the global model never trains on it.")
