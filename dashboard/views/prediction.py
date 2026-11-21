"""Page 7 - interactive student prediction, built from the training schema."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard import common
from src import config
from src.evaluation import palette
from src.utils.predictor import MODEL_CHOICES, ValidationError

GROUPS = {
    "Student & school": ["school", "sex", "age", "address", "famsize", "Pstatus", "reason", "guardian", "traveltime"],
    "Family": ["Medu", "Fedu", "Mjob", "Fjob", "famrel", "famsup", "paid", "internet"],
    "Study & support": ["studytime", "failures", "schoolsup", "activities", "nursery", "higher", "absences"],
    "Lifestyle & health": ["romantic", "freetime", "goout", "Dalc", "Walc", "health"],
    "Earlier grades": ["G1", "G2"],
}


def _field(col: str, spec: dict, key_prefix: str):
    label = f"{col}"
    help_text = spec["description"]
    if spec["type"] == "categorical":
        return st.selectbox(label, spec["options"], help=help_text, key=f"{key_prefix}_{col}")
    return st.number_input(label, min_value=int(spec["min"]), max_value=int(spec["max"]), value=int(spec["default"]),
                           step=1, help=f"{help_text} (allowed {spec['min']}-{spec['max']})", key=f"{key_prefix}_{col}")


def render() -> None:
    common.page_header("Student Prediction", "Enter a student's details and get a Pass/Fail estimate from a trained global model")
    common.require(config.MODELS_DIR / "full" / "fedavg_iid.pt", config.MODELS_DIR / "full" / "schema.json")

    c1, c2 = st.columns(2)
    variant = c1.selectbox("Feature variant", list(common.VARIANT_LABELS), format_func=common.VARIANT_LABELS.get)
    model_key = c2.selectbox("Model", list(MODEL_CHOICES), format_func=MODEL_CHOICES.get)
    pred = common.predictor(variant, model_key)
    meta = pred.model_metadata.get("test_metrics", {})

    if variant == "early":
        common.callout("error", "The early model (no G1/G2) performed **at or below the always-Pass baseline** in our "
                                "experiments. Its predictions are shown for completeness only and must not be used for decisions.")
    else:
        common.callout("info", "This model uses first- and second-period grades (G1, G2). It estimates the **final** result "
                               "once mid-year grades are known; it is not an early-warning model for the start of the year.")
    if meta:
        st.caption(f"Selected model's held-out test accuracy: {meta['accuracy']:.3f}, F1 {meta['f1']:.3f}, "
                   f"Fail recall {meta['fail_recall']:.3f} (seed {pred.model_metadata.get('seed')}).")

    st.markdown("The form below is generated from the saved training schema, so every input the model expects is "
                "present, with the same categories and ranges seen in training. G3 (the target) is deliberately absent.")

    student = {}
    with st.form("student_form"):
        for group, cols in GROUPS.items():
            cols = [c for c in cols if c in pred.fields]
            if not cols:
                continue
            st.markdown(f"**{group}**")
            grid = st.columns(min(4, len(cols)))
            for i, col in enumerate(cols):
                with grid[i % len(grid)]:
                    student[col] = _field(col, pred.fields[col], f"{variant}")
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if not submitted:
        st.caption("Fill in the form and press Predict. Default values are the training-set medians.")
        return

    try:
        result = pred.predict(student)
    except ValidationError as exc:
        st.error("Please fix the following inputs:")
        for p in exc.problems:
            st.markdown(f"- {p}")
        return

    st.divider()
    a, b = st.columns([1, 1.4])
    with a:
        color = palette.PASS_COLOR if result.label == "Pass" else palette.FAIL_COLOR
        st.markdown(
            f"<div style='padding:18px 22px;border-radius:12px;border:2px solid {color};'>"
            f"<div style='font-size:13px;color:{palette.INK_SECONDARY}'>Predicted outcome</div>"
            f"<div style='font-size:42px;font-weight:700;color:{color}'>{result.label}</div>"
            f"<div style='font-size:13px;color:{palette.INK_SECONDARY}'>{result.confidence_note}</div></div>",
            unsafe_allow_html=True)
        st.metric("Probability of Pass", common.pct(result.probability_pass))
        st.metric("Probability of Fail", common.pct(result.probability_fail))
        st.caption(f"Model: {result.model_label}")
    with b:
        fig = go.Figure(go.Bar(
            x=[result.probability_fail, result.probability_pass], y=["Fail", "Pass"], orientation="h",
            marker_color=[palette.FAIL_COLOR, palette.PASS_COLOR],
            text=[common.pct(result.probability_fail), common.pct(result.probability_pass)], textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.add_vline(x=0.5, line_dash="dot", line_color=palette.NEUTRAL, annotation_text="decision threshold 0.5")
        fig.update_layout(title="Class probabilities", xaxis_range=[0, 1.15], xaxis_tickformat=".0%")
        st.plotly_chart(common.style(fig, 260, legend_bottom=False), use_container_width=True)

    common.callout("warning",
                   "This is a statistical estimate from a model trained on 316 Portuguese students in 2005-06. It is not a "
                   "judgement about any real person and must not be the sole basis for an academic decision.")

    st.subheader("What-if: which inputs move this prediction most?")
    sens = pred.sensitivity(student, top_k=8)
    fig = go.Figure()
    fig.add_bar(x=sens["feature"], y=sens["p_pass_at_min"], name="at minimum value", marker_color=palette.SEQUENTIAL[2],
                hovertemplate="%{x} = min → P(pass) %{y:.1%}<extra></extra>")
    fig.add_bar(x=sens["feature"], y=sens["p_pass_at_max"], name="at maximum value", marker_color=palette.SEQUENTIAL[5],
                hovertemplate="%{x} = max → P(pass) %{y:.1%}<extra></extra>")
    fig.add_hline(y=result.probability_pass, line_dash="dot", line_color=palette.NEUTRAL,
                  annotation_text=f"current P(pass) {result.probability_pass:.0%}")
    fig.update_layout(barmode="group", title="Pass probability if one input is set to its minimum / maximum, all else fixed",
                      yaxis_tickformat=".0%", yaxis_range=[0, 1.1])
    st.plotly_chart(common.style(fig, 380), use_container_width=True)
    st.caption("A one-at-a-time sensitivity check for this specific student. It shows how the model reacts, not causal "
               "effects, and it is not a formal explanation method.")
