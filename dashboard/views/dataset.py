"""Page 2 - dataset analytics."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import common
from src import config
from src.data import load
from src.evaluation import palette


def render() -> None:
    common.page_header("Dataset Analytics", f"{config.DATASET_NAME} · source: {config.DATASET_URL}")
    ds = common.dataset_summary()
    df = common.raw_df()
    y = load.make_binary_target(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", ds["rows"])
    c2.metric("Original columns", ds["columns"])
    c3.metric("Missing values", ds["missing_values"])
    c4.metric("Pass (G3 ≥ 10)", ds["pass_count"], common.pct(ds["pass_rate"]))
    c5.metric("Fail (G3 < 10)", ds["fail_count"], common.pct(1 - ds["pass_rate"]))

    st.markdown(
        "The data describe students of two Portuguese secondary schools in a mathematics course (2005-2006). "
        "Grades are on a 0-20 scale; 10 is the pass mark. **G3 is only used to make the label and is never an input.**"
    )

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(
            x=["Pass", "Fail"], y=[ds["pass_count"], ds["fail_count"]],
            marker_color=[palette.PASS_COLOR, palette.FAIL_COLOR], text=[ds["pass_count"], ds["fail_count"]],
            textposition="outside", hovertemplate="%{x}: %{y} students<extra></extra>"))
        fig.update_layout(title="Class distribution", yaxis_title="students", showlegend=False)
        st.plotly_chart(common.style(fig, 340), use_container_width=True)
        st.caption(f"A model that always predicts Pass is {common.pct(ds['pass_rate'])} accurate. "
                   "Every real model must beat that, so we also report F1 and recall of the Fail class.")
    with right:
        counts = df.G3.value_counts().reindex(range(0, 21), fill_value=0)
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[palette.FAIL_COLOR if g < 10 else palette.PASS_COLOR for g in counts.index],
            hovertemplate="G3 = %{x}: %{y} students<extra></extra>"))
        fig.add_vline(x=9.5, line_dash="dot", line_color=palette.NEUTRAL, annotation_text="pass mark")
        fig.update_layout(title="Final grade G3 (0-20)", xaxis_title="G3", yaxis_title="students", bargap=0.15)
        st.plotly_chart(common.style(fig, 340), use_container_width=True)
        st.caption(f"{int((df.G3 == 0).sum())} students have G3 = 0 (most likely did not sit the final exam).")

    st.divider()
    st.subheader("Explore a feature")
    cols = list(load.FEATURE_DESCRIPTIONS)
    feature = st.selectbox("Feature", [c for c in cols if c != "G3"], index=cols.index("failures"),
                           format_func=lambda c: f"{c} - {load.FEATURE_DESCRIPTIONS[c]}")
    rate = df.groupby(feature)["G3"].apply(lambda g: (g >= 10).mean())
    n = df[feature].value_counts().reindex(rate.index)
    fig = go.Figure(go.Bar(
        x=rate.index.astype(str), y=rate.values, marker_color=palette.METHOD_COLORS["centralized"],
        text=[f"{r:.0%} (n={c})" for r, c in zip(rate.values, n.values)], textposition="outside",
        hovertemplate=f"{feature} = %{{x}}<br>pass rate %{{y:.1%}}<extra></extra>"))
    fig.add_hline(y=y.mean(), line_dash="dot", line_color=palette.NEUTRAL, annotation_text="overall pass rate")
    fig.update_layout(title=f"Pass rate by {feature}", yaxis_title="pass rate", yaxis_range=[0, 1.12],
                      xaxis_title=feature)
    st.plotly_chart(common.style(fig, 360), use_container_width=True)

    st.divider()
    st.subheader("Which numeric features relate to the final grade?")
    num_cols = load.NUMERIC_COLUMNS + ["G1", "G2"]
    corr = df[num_cols + ["G3"]].corr()["G3"].drop("G3").sort_values()
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker_color=[palette.PASS_COLOR if v > 0 else palette.FAIL_COLOR for v in corr.values],
        text=[f"{v:+.2f}" for v in corr.values], textposition="outside",
        hovertemplate="%{y}: r = %{x:.3f}<extra></extra>"))
    fig.update_layout(title="Pearson correlation with G3", xaxis_title="correlation", xaxis_range=[-0.6, 1.05])
    st.plotly_chart(common.style(fig, 480), use_container_width=True)
    common.callout(
        "info",
        f"G1 and G2 correlate with G3 at r = {df.G1.corr(df.G3):.2f} and {df.G2.corr(df.G3):.2f}. A model that uses "
        "them predicts the final result *after* mid-year grades exist. The separate **early** model drops them to "
        "test what is predictable at the start of the year.")

    st.divider()
    st.subheader("Attribute dictionary")
    table = pd.DataFrame({
        "attribute": cols,
        "type": ["categorical" if c in load.CATEGORICAL_COLUMNS else ("grade" if c in config.GRADE_COLUMNS else "numeric")
                 for c in cols],
        "distinct values": [df[c].nunique() for c in cols],
        "description": [load.FEATURE_DESCRIPTIONS[c] for c in cols],
    })
    st.dataframe(table, use_container_width=True, hide_index=True, height=420)

    st.subheader("Limitations of the data")
    st.markdown(
        "- Small: 395 students, 79 in the test set. One misclassified student moves accuracy by about 1.3 points.\n"
        "- Two schools in one country and one school year, so results do not generalise widely.\n"
        "- Many attributes are self-reported survey answers.\n"
        "- The institutions in this project are **partitions of this single dataset**, not real organisations.")
