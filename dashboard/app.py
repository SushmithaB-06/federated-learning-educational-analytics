"""Federated Learning for Educational Analytics - Streamlit dashboard.

Run from the project root:

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.views import (  # noqa: E402
    architecture, clients, dataset, evaluation, home, prediction, privacy, training,
)

st.set_page_config(
    page_title="Federated Learning for Educational Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Every view exposes a function called ``render``; Streamlit would infer the same URL for all of them,
# so each page gets an explicit, unique url_path.
PAGES = [
    st.Page(home.render, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(dataset.render, title="Dataset Analytics", icon="📊", url_path="dataset"),
    st.Page(architecture.render, title="Federated Architecture", icon="🏛️", url_path="architecture"),
    st.Page(clients.render, title="Client Data Distribution", icon="🏫", url_path="clients"),
    st.Page(training.render, title="Training & Convergence", icon="📈", url_path="training"),
    st.Page(evaluation.render, title="Model Evaluation", icon="🎯", url_path="evaluation"),
    st.Page(prediction.render, title="Student Prediction", icon="🔮", url_path="prediction"),
    st.Page(privacy.render, title="Privacy & Limitations", icon="🔒", url_path="privacy"),
]

with st.sidebar:
    st.markdown("### 🎓 FL for Educational Analytics")
    st.caption("Simulated federated learning on the UCI Student Performance dataset")

nav = st.navigation(PAGES)

with st.sidebar:
    st.divider()
    st.caption("All numbers shown are read from files produced by `python main.py all`. Nothing is hard-coded.")

nav.run()
