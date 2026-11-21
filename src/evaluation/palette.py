"""One colour system shared by the matplotlib figures and the Streamlit dashboard.

Categorical hues come from a validated palette (adjacent-pair colour-vision-deficiency
separation and normal-vision separation both pass). Bars and stacks are always
direct-labelled as well, so colour is never the only carrier of meaning.
"""

METHOD_COLORS = {
    "centralized": "#2a78d6",     # blue
    "fedavg_iid": "#1baf7a",      # aqua
    "fedavg_non_iid": "#eb6834",  # orange
}
CLIENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # Institution A, B, C
PASS_COLOR = "#008300"
FAIL_COLOR = "#e34948"
NEUTRAL = "#898781"        # muted ink / baselines
GRID = "#e1e0d9"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
