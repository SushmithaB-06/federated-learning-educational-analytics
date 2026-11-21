"""Page 8 - privacy properties, honest limitations and future work."""
from __future__ import annotations

import streamlit as st
import torch

from dashboard import common
from src import config


def render() -> None:
    common.page_header("Privacy & Limitations", "What federated learning does for privacy, what it does not, and what this project leaves open")

    st.subheader("Why raw data stays local")
    st.markdown(
        "In the simulation each `FederatedClient` object keeps its student arrays in private attributes, and the server "
        "code never reads them (a unit test scans the server source to enforce this). The only object that leaves a "
        "client is a `ClientUpdate`: model weights, a sample count and two loss values.")

    model_path = config.MODELS_DIR / "full" / "fedavg_iid.pt"
    if model_path.exists():
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        rows = [{"tensor": k, "shape": "×".join(map(str, v.shape)), "values": v.numel()} for k, v in payload["state_dict"].items()]
        total = sum(r["values"] for r in rows)
        a, b = st.columns([1.2, 1])
        with a:
            st.markdown("**Exactly what one upload contains**")
            st.dataframe(rows, hide_index=True, use_container_width=True)
        with b:
            st.metric("Numbers per upload", f"{total:,}")
            st.metric("Bytes per upload", f"{total * 4:,}")
            st.markdown("plus `num_samples` (1 integer), `train_loss` and `val_loss` (2 floats). No student row, no feature value, "
                        "no identifier.")

    st.divider()
    st.subheader("What FedAvg does **not** guarantee")
    common.callout("warning", "Standard FedAvg is a *data-minimisation* technique, not a formal privacy guarantee.")
    st.markdown(
        "- **Model updates can leak information.** Gradient-inversion and membership-inference attacks have reconstructed "
        "training examples or confirmed a record's presence from shared updates, especially with small batches and few clients "
        "(Zhu et al., 2019; Shokri et al., 2017).\n"
        "- **The server is trusted here.** It sees each institution's weights individually. Secure aggregation (Bonawitz et al., "
        "2017) would let it see only the sum.\n"
        "- **No differential privacy.** DP-FedAvg (McMahan et al., 2018) adds calibrated noise so that any single student's "
        "influence is bounded. This project does not implement it, so it makes **no formal privacy claim**.\n"
        "- **Honest participants are assumed.** Poisoning or free-riding clients are out of scope.")

    st.divider()
    st.subheader("Limitations of this study")
    st.markdown(
        "- **Simulation, not deployment.** Three partitions of one public dataset stand in for institutions; no real "
        "organisation's data or systems are involved, and no network transport is implemented.\n"
        "- **Tiny data.** 395 students, 79 in the test set. Differences of a few accuracy points are within noise, which is why "
        "results are reported as mean ± std over 5 seeds and compared to the always-Pass baseline.\n"
        "- **The useful model depends on G1/G2.** Without them (early variant) prediction is no better than the baseline on "
        "this dataset.\n"
        "- **No early stopping.** Validation loss rises in later rounds for the centralized model and slightly for FL; the fixed "
        "20-round budget was kept for a fair comparison.\n"
        "- **Label skew only.** Non-IID here means different Pass/Fail ratios with equal client sizes; feature skew and "
        "quantity skew were not studied.\n"
        "- **One dataset, one country, one year.** Results should not be generalised.")

    st.subheader("Future enhancements")
    st.markdown(
        "1. **Differential privacy**: clip client updates and add Gaussian noise (DP-FedAvg), then report the privacy-utility trade-off.\n"
        "2. **Secure aggregation** so the server only ever sees the weighted sum.\n"
        "3. **Early stopping on client validation loss** and tuning of local epochs.\n"
        "4. **More heterogeneity types** (feature skew, unequal client sizes, more clients) and FedProx / SCAFFOLD for comparison.\n"
        "5. **Second dataset** (the Portuguese-language course file, 649 students) to test generalisation.\n"
        "6. **Real transport** with a framework such as Flower once the method is validated.")

    st.divider()
    st.subheader("References mentioned on this page")
    st.markdown(
        "- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep "
        "networks from decentralized data. *AISTATS*.\n"
        "- Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018). Federated learning with non-IID data. *arXiv:1806.00582*.\n"
        "- Bonawitz, K. et al. (2017). Practical secure aggregation for privacy-preserving machine learning. *ACM CCS*.\n"
        "- McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning differentially private recurrent language models. *ICLR*.\n"
        "- Zhu, L., Liu, Z., & Han, S. (2019). Deep leakage from gradients. *NeurIPS*.\n"
        "- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership inference attacks against machine learning models. *IEEE S&P*.\n"
        "- Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. *Proceedings of 5th FUBUTEC*.")
    st.caption("Verify each reference against the original source before citing it in the report.")
