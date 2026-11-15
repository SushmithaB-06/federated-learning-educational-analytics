
from pathlib import Path
import textwrap

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduFL | Educational Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSS_PATH = PROJECT_ROOT / "dashboard" / "assets" / "style.css"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-mat.csv"

CLIENT_DATA_PATH = (
    PROJECT_ROOT / "results" / "reports"
    / "client_distribution_comparison.csv"
)


if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True
        )


# ============================================================
# 3. PROJECT CONFIGURATION AND RECORDED RESULTS
# ============================================================

TOTAL_STUDENTS = 395
TRAINING_SAMPLES = 316
TESTING_SAMPLES = 79
INPUT_FEATURES = 58

SIMULATED_CLIENTS = 3
FEDERATED_ROUNDS = 10
LOCAL_EPOCHS = 5

# Recorded model evaluation results
MODEL_RESULTS = {
    "Centralized": {
        "Accuracy": 84.81,
        "Precision": 91.84,
        "Recall": 84.91,
        "F1-score": 88.24
    },
    "IID FedAvg": {
        "Accuracy": 87.34,
        "Precision": 93.88,
        "Recall": 86.79,
        "F1-score": 90.20
    },
    "Non-IID FedAvg": {
        "Accuracy": 87.34,
        "Precision": 93.88,
        "Recall": 86.79,
        "F1-score": 90.20
    }
}

# Confusion matrix format:
# Rows = Actual class (Fail, Pass)
# Columns = Predicted class (Fail, Pass)
CONFUSION_MATRICES = {
    "Centralized": [
        [22, 4],
        [8, 45]
    ],
    "IID FedAvg": [
        [23, 3],
        [7, 46]
    ],
    "Non-IID FedAvg": [
        [23, 3],
        [7, 46]
    ]
}


# ============================================================
# 4. SIDEBAR
# ============================================================

pages = [
    "Overview",
    "Dataset Analytics",
    "Client Analytics",
    "Model Performance",
    "Federated Training",
    "Student Prediction"
]

with st.sidebar:

    st.markdown("# 🎓 EduFL")
    st.caption("FEDERATED LEARNING PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        pages,
        label_visibility="visible"
    )

    st.divider()

    st.markdown("**PROJECT STATUS**")
    st.markdown("🟢 Initial experiments completed")

    st.caption("PyTorch • FedAvg • Streamlit")

    st.divider()

    st.markdown("**DEVELOPMENT STATUS**")
    st.caption("Overview: Available")
    st.caption("Dataset Analytics: Available")
    st.caption("Client Analytics: Available")
    st.caption("Model Performance: Available")
    st.caption("Federated Training: Available")
    st.caption("Student Prediction: Upcoming")


# ============================================================
# 5. COMMON HEADER
# ============================================================

header_html = """
<div class="main-header">
    <div>
        <p class="eyebrow">EDUCATIONAL INTELLIGENCE PLATFORM</p>
        <h1>Federated Learning Analytics</h1>
        <p class="subtitle">
            Privacy-aware student performance prediction
            across simulated educational institutions.
        </p>
    </div>
    <div class="status-pill">
        ● PROTOTYPE • ACTIVE DEVELOPMENT
    </div>
</div>
"""

st.markdown(
    textwrap.dedent(header_html),
    unsafe_allow_html=True
)


# ============================================================
# 6. OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown("## Project Overview")

    st.caption(
        "An overview of the dataset, federated experiments "
        "and current model performance."
    )

    st.markdown("### Key Performance Indicators")

    metric_data = [
        ("TOTAL STUDENTS", "395", "Dataset records"),
        ("FEDERATED ACCURACY", "87.34%", "Recorded FedAvg result"),
        ("FEDERATED F1-SCORE", "90.20%", "Recorded FedAvg result"),
        ("CENTRALIZED ACCURACY", "84.81%", "Baseline model"),
        ("SIMULATED CLIENTS", "03", "Educational groups"),
        ("FEDERATED ROUNDS", "10", "5 local epochs per round")
    ]

    for row_index in range(2):

        columns = st.columns(3)

        for column_index in range(3):

            label, value, detail = metric_data[
                row_index * 3 + column_index
            ]

            metric_html = f"""
            <div class="metric-card">
                <p class="metric-label">{label}</p>
                <h2>{value}</h2>
                <p class="metric-detail">{detail}</p>
            </div>
            """

            with columns[column_index]:
                st.markdown(
                    textwrap.dedent(metric_html),
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.markdown("### Experiment Summary")

    left, right = st.columns([1.2, 1])

    with left:

        st.markdown("#### Model Accuracy Comparison")

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Centralized", "IID FedAvg", "Non-IID FedAvg"],
                y=[84.81, 87.34, 87.34],
                text=["84.81%", "87.34%", "87.34%"],
                textposition="outside",
                marker_color=["#64748b", "#8b5cf6", "#06b6d4"]
            )
        )

        fig.update_layout(
            height=350,
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 105]),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dce5f5")
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "IID and Non-IID models produced identical "
            "predictions on the latest evaluated test set."
        )

    with right:

        st.markdown("#### Experiment Configuration")

        config = pd.DataFrame({
            "Configuration": [
                "Dataset",
                "Training samples",
                "Testing samples",
                "Input features",
                "Simulated clients",
                "Aggregation",
                "Framework",
                "Learning task"
            ],
            "Value": [
                "Student Performance",
                "316",
                "79",
                "58",
                "3",
                "FedAvg",
                "PyTorch",
                "Pass / Fail"
            ]
        })

        st.dataframe(
            config,
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### Dataset Allocation")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Training records", TRAINING_SAMPLES)
        st.progress(TRAINING_SAMPLES / TOTAL_STUDENTS)

    with c2:
        st.metric("Testing records", TESTING_SAMPLES)
        st.progress(TESTING_SAMPLES / TOTAL_STUDENTS)

    st.caption("80:20 stratified train-test split.")

    st.markdown("---")
    st.markdown("### Platform Development")

    modules = [
        ("📊", "Dataset Analytics",
         "Explore student records and grade distributions.", "AVAILABLE"),
        ("🌐", "Client Analytics",
         "Explore IID and Non-IID client distributions.", "AVAILABLE"),
        ("🧠", "Model Performance",
         "Compare centralized and federated models.", "AVAILABLE"),
        ("🔄", "Federated Training",
         "Visualize local training and aggregation.", "AVAILABLE"),
        ("🎯", "Pass / Fail Prediction",
         "Predict student pass or fail status.", "PLANNED"),
        ("🎓", "Grade Prediction",
         "Predict grades A, B, C, D and F.", "PLANNED")
    ]

    module_columns = st.columns(3)

    for index, (icon, title, description, status) in enumerate(modules):

        module_html = f"""
        <div class="module-card">
            <div style="font-size:28px">{icon}</div>
            <h4>{title}</h4>
            <p>{description}</p>
            <span style="color:#22d3ee;font-size:11px;font-weight:700">
                {status}
            </span>
        </div>
        """

        with module_columns[index % 3]:
            st.markdown(
                textwrap.dedent(module_html),
                unsafe_allow_html=True
            )


# ============================================================
# 7. DATASET ANALYTICS
# ============================================================

elif page == "Dataset Analytics":

    st.markdown("## Dataset Analytics")

    st.caption(
        "Explore student records, academic performance "
        "and final grade distribution."
    )

    if not DATA_PATH.exists():

        st.error(f"Dataset not found: {DATA_PATH}")

    else:

        try:

            df = pd.read_csv(DATA_PATH, sep=";")

            if "G3" not in df.columns:

                st.error("Required G3 column not found.")

            else:

                df["Pass_Status"] = df["G3"].apply(
                    lambda x: "Pass" if x >= 10 else "Fail"
                )

                grade_labels = ["F", "D", "C", "B", "A"]

                df["Grade"] = pd.cut(
                    df["G3"],
                    bins=[-1, 9, 11, 13, 15, 20],
                    labels=grade_labels
                )

                st.markdown("### Dataset Overview")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Total Students", len(df))
                c2.metric("Original Features", len(df.columns) - 2)
                c3.metric(
                    "Passing Students",
                    int((df["Pass_Status"] == "Pass").sum())
                )
                c4.metric(
                    "Failing Students",
                    int((df["Pass_Status"] == "Fail").sum())
                )

                st.markdown("---")
                st.markdown("### Academic Performance Distribution")

                left, right = st.columns(2)

                with left:

                    pass_counts = (
                        df["Pass_Status"]
                        .value_counts()
                        .reindex(["Pass", "Fail"], fill_value=0)
                    )

                    fig = go.Figure(
                        go.Pie(
                            labels=pass_counts.index,
                            values=pass_counts.values,
                            hole=0.55,
                            marker_colors=["#10b981", "#ef4444"],
                            textinfo="label+percent"
                        )
                    )

                    fig.update_layout(
                        title="Pass vs Fail",
                        height=350,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#dce5f5")
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with right:

                    grade_counts = (
                        df["Grade"]
                        .value_counts()
                        .reindex(grade_labels, fill_value=0)
                    )

                    fig = go.Figure(
                        go.Bar(
                            x=grade_counts.index.astype(str),
                            y=grade_counts.values,
                            text=grade_counts.values,
                            textposition="outside",
                            marker_color=[
                                "#ef4444", "#f59e0b", "#8b5cf6",
                                "#06b6d4", "#10b981"
                            ]
                        )
                    )

                    fig.update_layout(
                        title="Five-Grade Distribution",
                        xaxis_title="Grade",
                        yaxis_title="Students",
                        height=350,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#dce5f5")
                    )

                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Final Marks Distribution")

                fig = go.Figure(
                    go.Histogram(
                        x=df["G3"],
                        nbinsx=21,
                        marker_color="#8b5cf6"
                    )
                )

                fig.update_layout(
                    xaxis_title="Final Marks (G3)",
                    yaxis_title="Number of Students",
                    height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#dce5f5")
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Academic Statistics")

                a, b, c = st.columns(3)

                a.metric("Average Marks", f"{df['G3'].mean():.2f}/20")
                b.metric("Highest Marks", int(df["G3"].max()))
                c.metric("Lowest Marks", int(df["G3"].min()))

                st.markdown("### Student Dataset Preview")

                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Dataset Quality")

                q1, q2 = st.columns(2)

                q1.metric(
                    "Missing Values",
                    int(df.isnull().sum().sum())
                )
                q2.metric(
                    "Duplicate Records",
                    int(df.duplicated().sum())
                )

        except Exception as error:

            st.error(f"Dataset analysis failed: {error}")


# ============================================================
# 8. CLIENT ANALYTICS
# ============================================================

elif page == "Client Analytics":

    st.markdown("## Client Analytics")

    st.caption(
        "Explore how student records are distributed "
        "across simulated educational institutions."
    )

    if not CLIENT_DATA_PATH.exists():

        st.error(
            "Client distribution file not found. "
            "Please generate client_distribution_comparison.csv "
            "from the partitioning notebook."
        )

    else:

        try:

            client_df = pd.read_csv(CLIENT_DATA_PATH)

            required_columns = {
                "Client", "Distribution", "Pass", "Fail",
                "Total", "Pass_Percentage", "Fail_Percentage"
            }

            if not required_columns.issubset(client_df.columns):

                st.error("The client CSV is missing required columns.")

            else:

                st.markdown("### Client Overview")

                iid_total = int(
                    client_df.loc[
                        client_df["Distribution"] == "IID", "Total"
                    ].sum()
                )

                non_iid_total = int(
                    client_df.loc[
                        client_df["Distribution"] == "Non-IID", "Total"
                    ].sum()
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Simulated Clients", 3)
                c2.metric("Students per Partition", "316")
                c3.metric("IID Records", iid_total)
                c4.metric("Non-IID Records", non_iid_total)

                st.markdown("---")

                selected_distribution = st.selectbox(
                    "Select data distribution",
                    ["IID", "Non-IID"]
                )

                selected_df = client_df[
                    client_df["Distribution"] == selected_distribution
                ].copy()

                st.markdown(
                    f"### {selected_distribution} Client Distribution"
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=selected_df["Client"],
                        y=selected_df["Pass"],
                        name="Pass",
                        marker_color="#10b981"
                    )
                )

                fig.add_trace(
                    go.Bar(
                        x=selected_df["Client"],
                        y=selected_df["Fail"],
                        name="Fail",
                        marker_color="#ef4444"
                    )
                )

                fig.update_layout(
                    barmode="stack",
                    height=400,
                    xaxis_title="Simulated Educational Client",
                    yaxis_title="Number of Students",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#dce5f5"),
                    legend_title="Student Outcome"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Client-wise Pass Percentage")

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=selected_df["Client"],
                        y=selected_df["Pass_Percentage"],
                        text=[
                            f"{value:.2f}%"
                            for value in selected_df["Pass_Percentage"]
                        ],
                        textposition="outside",
                        marker_color="#8b5cf6"
                    )
                )

                fig.update_layout(
                    height=350,
                    yaxis_title="Passing Students (%)",
                    yaxis=dict(range=[0, 105]),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#dce5f5")
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.markdown("### IID vs Non-IID Comparison")

                comparison_fig = go.Figure()

                for distribution, color in [
                    ("IID", "#8b5cf6"),
                    ("Non-IID", "#06b6d4")
                ]:

                    subset = client_df[
                        client_df["Distribution"] == distribution
                    ]

                    comparison_fig.add_trace(
                        go.Bar(
                            x=subset["Client"],
                            y=subset["Pass_Percentage"],
                            name=distribution,
                            marker_color=color
                        )
                    )

                comparison_fig.update_layout(
                    barmode="group",
                    height=400,
                    xaxis_title="Simulated Client",
                    yaxis_title="Pass Percentage (%)",
                    yaxis=dict(range=[0, 105]),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#dce5f5")
                )

                st.plotly_chart(
                    comparison_fig,
                    use_container_width=True
                )

                st.markdown("### Detailed Client Statistics")

                st.dataframe(
                    client_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("---")
                st.markdown("### Understanding the Distribution")

                st.info(
                    "IID: Each simulated client has a broadly similar "
                    "Pass/Fail distribution. Non-IID: The proportion of "
                    "Pass and Fail students varies across clients. "
                    "This simulates heterogeneous educational data."
                )

                st.caption(
                    "These are simulated client partitions of the same "
                    "316-student training dataset, not actual separate colleges."
                )

        except Exception as error:

            st.error(f"Client analytics failed: {error}")


# ============================================================
# 9. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown("## Model Performance Analysis")

    st.caption(
        "Compare centralized machine learning with "
        "federated learning using the recorded test results."
    )

    st.markdown("### Evaluation Configuration")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Testing Samples", TESTING_SAMPLES)
    c2.metric("Input Features", INPUT_FEATURES)
    c3.metric("Federated Clients", SIMULATED_CLIENTS)
    c4.metric("Federated Rounds", FEDERATED_ROUNDS)

    st.markdown("---")

    # --------------------------------------------------------
    # PERFORMANCE METRICS TABLE
    # --------------------------------------------------------

    st.markdown("### Model Evaluation Metrics")

    results_df = pd.DataFrame(MODEL_RESULTS).T
    results_df.index.name = "Model"

    st.dataframe(
        results_df.style.format("{:.2f}%"),
        use_container_width=True
    )

    # --------------------------------------------------------
    # ACCURACY COMPARISON
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### 1. Accuracy Comparison")

    fig = go.Figure()

    colors = ["#64748b", "#8b5cf6", "#06b6d4"]

    for index, model_name in enumerate(MODEL_RESULTS):

        fig.add_trace(
            go.Bar(
                x=[model_name],
                y=[MODEL_RESULTS[model_name]["Accuracy"]],
                name=model_name,
                marker_color=colors[index],
                text=[
                    f'{MODEL_RESULTS[model_name]["Accuracy"]:.2f}%'
                ],
                textposition="outside"
            )
        )

    fig.update_layout(
        height=400,
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 105]),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5f5")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Accuracy represents the percentage of all test predictions "
        "that were correct."
    )

    # --------------------------------------------------------
    # PRECISION, RECALL AND F1 COMPARISON
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### 2. Precision, Recall and F1-score")

    metric_fig = go.Figure()

    metric_colors = {
        "Precision": "#8b5cf6",
        "Recall": "#06b6d4",
        "F1-score": "#10b981"
    }

    for metric_name, color in metric_colors.items():

        metric_fig.add_trace(
            go.Bar(
                name=metric_name,
                x=list(MODEL_RESULTS.keys()),
                y=[
                    MODEL_RESULTS[model][metric_name]
                    for model in MODEL_RESULTS
                ],
                marker_color=color
            )
        )

    metric_fig.update_layout(
        barmode="group",
        height=420,
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 105]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5f5"),
        legend_title="Evaluation Metric"
    )

    st.plotly_chart(metric_fig, use_container_width=True)

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### 3. Confusion Matrix Analysis")

    st.caption(
        "Rows represent actual student outcomes. "
        "Columns represent model-predicted outcomes."
    )

    selected_model = st.selectbox(
        "Choose a model to inspect",
        list(CONFUSION_MATRICES.keys())
    )

    matrix = CONFUSION_MATRICES[selected_model]

    matrix_fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted Fail", "Predicted Pass"],
            y=["Actual Fail", "Actual Pass"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
            hovertemplate=(
                "Actual: %{y}<br>"
                "Predicted: %{x}<br>"
                "Students: %{z}<extra></extra>"
            )
        )
    )

    matrix_fig.update_layout(
        title=f"{selected_model} Confusion Matrix",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5f5")
    )

    st.plotly_chart(matrix_fig, use_container_width=True)

    tn, fp = matrix[0]
    fn, tp = matrix[1]

    a, b, c, d = st.columns(4)

    a.metric("True Negatives", tn)
    b.metric("False Positives", fp)
    c.metric("False Negatives", fn)
    d.metric("True Positives", tp)

    st.markdown("#### Understanding the Confusion Matrix")

    st.write(
        f"""
        - **True Negative (TN = {tn}):** Students who actually failed
          and were correctly predicted as Fail.
        - **False Positive (FP = {fp}):** Students who actually failed
          but were incorrectly predicted as Pass.
        - **False Negative (FN = {fn}):** Students who actually passed
          but were incorrectly predicted as Fail.
        - **True Positive (TP = {tp}):** Students who actually passed
          and were correctly predicted as Pass.
        """
    )

    # --------------------------------------------------------
    # METRIC EXPLANATIONS
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### 4. Understanding Evaluation Metrics")

    explanations = [
        (
            "Accuracy",
            "How many total predictions were correct?",
            "Correct predictions / Total predictions"
        ),
        (
            "Precision",
            "Out of all students predicted as Pass, how many actually passed?",
            "True Positives / (True Positives + False Positives)"
        ),
        (
            "Recall",
            "Out of all students who actually passed, how many were identified?",
            "True Positives / (True Positives + False Negatives)"
        ),
        (
            "F1-score",
            "A combined measure of Precision and Recall.",
            "2 × (Precision × Recall) / (Precision + Recall)"
        )
    ]

    for title, description, formula in explanations:

        with st.expander(title):

            st.write(description)
            st.code(formula)

    # --------------------------------------------------------
    # PERFORMANCE INTERPRETATION
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("### 5. Experiment Interpretation")

    st.info(
        """
        In the recorded evaluation, the federated models achieved
        87.34% accuracy, compared with 84.81% for the centralized model.

        The federated models also recorded an F1-score of 90.20%,
        compared with 88.24% for the centralized model.

        The latest IID and Non-IID evaluations produced identical
        predictions on the same test set.
        """
    )

    st.warning(
        "These results are from a particular experimental split and "
        "training run. They do not establish that federated learning "
        "will always outperform centralized learning. Further repeated "
        "experiments are needed to assess stability."
    )

    st.caption(
        "Confusion matrix class order: Fail = 0, Pass = 1. "
        "All matrices use the same 79-student test set."
    )



# ============================================================
# 10. FEDERATED TRAINING
# ============================================================

elif page == "Federated Training":

    st.markdown("## Federated Training")

    st.caption(
        "Explore how simulated clients train local neural networks "
        "and how the central server combines their model parameters."
    )

    iid_losses = [
        0.6320100, 0.5442927, 0.4699556, 0.4142982, 0.3570195,
        0.2735714, 0.2351688, 0.1915158, 0.1577655, 0.1443359
    ]

    non_iid_losses = [
        0.5918464, 0.5120522, 0.4417597, 0.3516884, 0.2965044,
        0.2434958, 0.1991381, 0.1601837, 0.1386247, 0.1264606
    ]

    st.markdown("### Training Configuration")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Simulated Clients", SIMULATED_CLIENTS)
    c2.metric("Communication Rounds", FEDERATED_ROUNDS)
    c3.metric("Local Epochs / Round", LOCAL_EPOCHS)
    c4.metric("Training Students", TRAINING_SAMPLES)

    st.markdown("---")
    st.markdown("### Federated Learning Workflow")

    workflow = [
        (
            "1. Initialize Global Model",
            "The server initializes the global neural network."
        ),
        (
            "2. Distribute Model",
            "The current global model is sent to each participating client."
        ),
        (
            "3. Local Training",
            "Each client trains a copy of the model using its local student records."
        ),
        (
            "4. Send Model Updates",
            "Clients send model parameters to the server; raw student records remain local."
        ),
        (
            "5. FedAvg Aggregation",
            "The server combines client parameters using a sample-count-weighted average."
        ),
        (
            "6. Repeat Communication",
            "The updated global model is distributed for the next communication round."
        )
    ]

    for start in range(0, len(workflow), 3):

        cols = st.columns(3)

        for offset, col in enumerate(cols):

            idx = start + offset

            if idx < len(workflow):

                title, description = workflow[idx]

                with col:

                    st.markdown(
                        f"""
                        <div class="module-card" style="min-height:150px">
                            <h4>{title}</h4>
                            <p>{description}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.markdown("---")
    st.markdown("### Federated Averaging (FedAvg)")

    st.write(
        "The server calculates a weighted average of the locally trained "
        "model parameters. A client with more training records receives "
        "a proportionally larger weight."
    )

    st.latex(
        r"w_{global} = \sum_{k=1}^{K} \frac{n_k}{N} w_k"
    )

    st.markdown(
        """
        **Formula terms**

        - **K:** Number of participating clients.
        - **nₖ:** Number of training samples at client k.
        - **N:** Total number of training samples across clients.
        - **wₖ:** Model parameters received from client k.
        - **w_global:** Aggregated global model parameters.
        """
    )

    client_sizes = pd.DataFrame({
        "Simulated Client": ["Client 1", "Client 2", "Client 3"],
        "Training Records": [106, 105, 105]
    })

    st.dataframe(
        client_sizes,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### Training Convergence: IID vs Non-IID")

    st.caption(
        "These recorded values are the sample-weighted average of "
        "clients' final local training losses in each communication round."
    )

    rounds = list(range(1, FEDERATED_ROUNDS + 1))

    convergence_fig = go.Figure()

    convergence_fig.add_trace(
        go.Scatter(
            x=rounds,
            y=iid_losses,
            mode="lines+markers",
            name="IID",
            line=dict(color="#8b5cf6", width=3)
        )
    )

    convergence_fig.add_trace(
        go.Scatter(
            x=rounds,
            y=non_iid_losses,
            mode="lines+markers",
            name="Non-IID",
            line=dict(color="#06b6d4", width=3)
        )
    )

    convergence_fig.update_layout(
        title="Weighted Average Local Training Loss by Round",
        xaxis_title="Communication Round",
        yaxis_title="Training Loss",
        xaxis=dict(dtick=1),
        height=430,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5f5")
    )

    st.plotly_chart(
        convergence_fig,
        use_container_width=True
    )

    st.markdown("### Round-wise Loss Values")

    convergence_df = pd.DataFrame({
        "Communication Round": rounds,
        "IID Loss": iid_losses,
        "Non-IID Loss": non_iid_losses
    })

    st.dataframe(
        convergence_df.style.format({
            "IID Loss": "{:.6f}",
            "Non-IID Loss": "{:.6f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### Interpreting the Experiment")

    st.write(
        f"After {FEDERATED_ROUNDS} rounds, the recorded IID loss is "
        f"**{iid_losses[-1]:.4f}**, and the recorded Non-IID loss is "
        f"**{non_iid_losses[-1]:.4f}**. Both curves decrease over the "
        "recorded rounds, indicating that the local training losses "
        "declined during this experiment."
    )

    st.warning(
        "These curves show local training loss, not global test loss. "
        "A decreasing training loss does not by itself establish better "
        "generalization or higher test accuracy."
    )

    st.info(
        "IID and Non-IID clients are simulated partitions of the same "
        "316-record UCI Student Performance training split. They do not "
        "represent actual independent colleges or institutions."
    )


# ============================================================
# 11. STUDENT PREDICTION
# ============================================================

elif page == "Student Prediction":

    st.markdown("## Student Prediction")

    st.info(
        "Student prediction is the next development module. "
        "We will connect the trained global model and matching "
        "preprocessing pipeline before enabling predictions."
    )

    st.caption(
        "The current trained classifier predicts Pass / Fail. "
        "Five-grade prediction (A, B, C, D, F) is not yet implemented."
    )
