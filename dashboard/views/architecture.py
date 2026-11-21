"""Page 3 - federated architecture and the FedAvg workflow."""
from __future__ import annotations

import streamlit as st

from dashboard import common
from src import config
from src.evaluation import palette


def architecture_svg(compact: bool = False) -> str:
    """Inline SVG diagram: three institutions, local models, server, global model."""
    w, h = 760, 300 if compact else 360
    colors = palette.CLIENT_COLORS
    names = config.CLIENT_NAMES
    boxes = []
    for i, (name, col) in enumerate(zip(names, colors)):
        x = 30 + i * 250
        y = h - 120
        boxes.append(f"""
        <g>
          <rect x="{x}" y="{y}" width="200" height="90" rx="10" fill="#fcfcfb" stroke="{col}" stroke-width="2"/>
          <text x="{x + 100}" y="{y + 24}" text-anchor="middle" font-size="14" font-weight="600" fill="{palette.INK}">{name}</text>
          <rect x="{x + 14}" y="{y + 38}" width="80" height="38" rx="6" fill="{col}" opacity="0.15"/>
          <text x="{x + 54}" y="{y + 54}" text-anchor="middle" font-size="10" fill="{palette.INK}">raw student</text>
          <text x="{x + 54}" y="{y + 67}" text-anchor="middle" font-size="10" fill="{palette.INK}">records (private)</text>
          <rect x="{x + 106}" y="{y + 38}" width="80" height="38" rx="6" fill="{col}"/>
          <text x="{x + 146}" y="{y + 61}" text-anchor="middle" font-size="11" fill="white" font-weight="600">local model</text>
          <!-- uplink: weights to server -->
          <line x1="{x + 146}" y1="{y}" x2="{w / 2 - 60 + i * 60}" y2="118" stroke="{col}" stroke-width="2" marker-end="url(#arrow)"/>
          <!-- downlink: global model back -->
          <line x1="{w / 2 - 40 + i * 60}" y1="124" x2="{x + 60}" y2="{y}" stroke="{palette.NEUTRAL}" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#arrowg)"/>
        </g>""")
    svg = f"""
    <svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, Segoe UI, sans-serif">
      <defs>
        <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{palette.INK_SECONDARY}"/></marker>
        <marker id="arrowg" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{palette.NEUTRAL}"/></marker>
      </defs>
      <rect x="{w / 2 - 150}" y="30" width="300" height="90" rx="12" fill="#f1f1ee" stroke="{palette.METHOD_COLORS['centralized']}" stroke-width="2"/>
      <text x="{w / 2}" y="56" text-anchor="middle" font-size="15" font-weight="700" fill="{palette.INK}">Aggregation server</text>
      <text x="{w / 2}" y="78" text-anchor="middle" font-size="12" fill="{palette.INK_SECONDARY}">FedAvg:  w = Σ (n_k / N) · w_k</text>
      <rect x="{w / 2 - 60}" y="88" width="120" height="24" rx="6" fill="{palette.METHOD_COLORS['centralized']}"/>
      <text x="{w / 2}" y="104" text-anchor="middle" font-size="11" fill="white" font-weight="600">global model</text>
      <line x1="20" y1="{h - 135}" x2="{w - 20}" y2="{h - 135}" stroke="{palette.NEUTRAL}" stroke-dasharray="6 5"/>
      <text x="24" y="{h - 141}" font-size="10" fill="{palette.INK_SECONDARY}">only model weights + sample counts cross this line  ↑   (solid = client → server, dashed = global model → clients)</text>
      {''.join(boxes)}
    </svg>"""
    # Strip indentation so Markdown renderers never mistake the markup for a code block.
    return "\n".join(line.strip() for line in svg.splitlines() if line.strip())


def render() -> None:
    common.page_header("Federated Architecture", "Who holds what, and what travels over the network")
    st.html(architecture_svg())

    st.subheader("One communication round, step by step")
    steps = [
        ("1. Broadcast", "The server sends the current global model weights to every institution."),
        ("2. Local training", f"Each institution trains that model on **its own** students for {config.LOCAL_EPOCHS} epochs "
                              f"(Adam, learning rate {config.LEARNING_RATE}, batch size {config.BATCH_SIZE})."),
        ("3. Upload", "Each institution returns its updated weights, how many students it trained on (n_k), and two scalar loss values."),
        ("4. Aggregate", "The server computes the sample-weighted average of the weights (FedAvg)."),
        ("5. Update", "The average becomes the new global model."),
        ("6. Repeat", f"Steps 1-5 run for {config.NUM_ROUNDS} rounds. The final global model is evaluated on the fixed test set."),
    ]
    cols = st.columns(3)
    for i, (title, body) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"**{title}**  \n{body}")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Raw data vs model updates")
        st.table([
            {"": "Student records (33 columns per student)", "Stays at institution": "✅", "Sent to server": "❌"},
            {"": "Model weights (2,433 numbers)", "Stays at institution": "copy", "Sent to server": "✅"},
            {"": "Number of training students n_k", "Stays at institution": "-", "Sent to server": "✅"},
            {"": "Training / validation loss (2 numbers)", "Stays at institution": "-", "Sent to server": "✅"},
        ])
        st.caption("Weights are the parameters of the neural network, not a transformed copy of the data. "
                   "They can still leak information in principle; see Privacy & Limitations.")
    with right:
        st.subheader("Federated Averaging (FedAvg)")
        st.latex(r"w_{\text{global}} = \sum_{k=1}^{K} \frac{n_k}{N}\, w_k, \qquad N = \sum_{k=1}^{K} n_k")
        st.markdown(
            "**Worked example** with two institutions and one weight:\n\n"
            "- Institution A: 100 students, weight 0.8\n"
            "- Institution B: 300 students, weight 0.4\n\n"
            "N = 400, so the global weight is 100/400 × 0.8 + 300/400 × 0.4 = 0.2 + 0.3 = **0.5**. "
            "Institutions with more students pull the average harder. The same formula is applied to every one of "
            "the model's 2,433 parameters.")

    st.divider()
    st.subheader("Why a custom simulation instead of the Flower framework?")
    st.markdown(
        "Flower (flwr) is an excellent production FL framework, but its simulation engine depends on Ray, which is "
        "heavy and fragile on Windows, and it hides the aggregation step behind a strategy class. For a project whose "
        "goal is to *understand and defend* FedAvg, a transparent 30-line implementation in `src/strategies/fedavg.py` "
        "plus an explicit round loop in `src/server/server.py` is easier to test, explain and reproduce. "
        "The algorithm is identical; what differs is only the transport layer, which here is an in-process function call.")
