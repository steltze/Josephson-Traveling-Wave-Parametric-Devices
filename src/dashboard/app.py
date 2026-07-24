"""
Interactive S-parameter viewer.

This script is the Streamlit process launched by `dashboard.Dashboard.run`.
It never runs a simulation itself — it only receives S-matrices (already
computed by your own script) through a `--data <pickle path>` argument, and
lets you pick which S_ij from which run to plot in dB, without re-running
anything.

Don't run this file directly; use `Dashboard` from your own script instead:

    from dashboard import Dashboard
    Dashboard([S1, S2], freqs=cfg.freqs, labels=["eps=0.05", "eps=0.10"], ks_state=cfg.ks_state).run()
"""

from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Fixed-order categorical palette (CVD-validated), assigned in this order
# (never cycled/recycled by rank) so multi-series S-parameter plots stay
# colorblind-safe.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

st.set_page_config(page_title="TWPA/TWPC S-parameters", layout="wide")
st.title("S-parameter viewer")

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=False)
args, _ = parser.parse_known_args(sys.argv[1:])

if not args.data:
    st.warning(
        "No data loaded. Launch this from your own script instead of running "
        "`streamlit run` directly:\n\n"
        "```python\n"
        "from dashboard import Dashboard\n"
        "Dashboard([S1, S2], freqs=cfg.freqs, labels=[\"run1\", \"run2\"], "
        "ks_state=cfg.ks_state).run()\n"
        "```"
    )
    st.stop()

with open(args.data, "rb") as f:
    runs = pickle.load(f)

st.caption(f"Loaded {len(runs)} run(s): " + ", ".join(r["label"] for r in runs))

ctrl_col, plot_col = st.columns([1, 3])

with ctrl_col:
    st.subheader("S-parameters")

    option_to_run_pair = {}
    for run_idx, run in enumerate(runs):
        N = run["S"].shape[1]
        port_labels = run["port_labels"]
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                s_ij = f"{port_labels[j - 1]} → {port_labels[i - 1]}"
                label = f"{run['label']}: {s_ij}" if len(runs) > 1 else s_ij
                option_to_run_pair[label] = (run_idx, i, j)

    options = list(option_to_run_pair)
    default = options[:1]
    selected = st.multiselect("Select S-parameters to plot", options=options, default=default)

    auto_ylim = st.checkbox("Auto y-axis (dB)", value=True)
    ylim = None
    if not auto_ylim:
        ymin = st.number_input("y min (dB)", value=-60.0)
        ymax = st.number_input("y max (dB)", value=5.0)
        ylim = (ymin, ymax)

with plot_col:
    if selected:
        fig = go.Figure()

        for color_idx, label in enumerate(selected):
            run_idx, i, j = option_to_run_pair[label]
            run = runs[run_idx]
            port_labels = run["port_labels"]

            s_ij = run["S"][:, i - 1, j - 1]
            db = 20.0 * np.log10(np.abs(s_ij))

            name = f"{port_labels[j - 1]} → {port_labels[i - 1]}"
            if len(runs) > 1:
                name = f"{run['label']}: {name}"

            fig.add_trace(
                go.Scatter(
                    x=run["freqs"],
                    y=db,
                    mode="lines",
                    name=name,
                    line=dict(color=PALETTE[color_idx % len(PALETTE)], width=2),
                )
            )

        fig.update_layout(
            xaxis_title="Frequency (GHz)",
            yaxis_title="Magnitude (dB)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=40),
        )
        if ylim is not None:
            fig.update_yaxes(range=list(ylim))

        # Scroll-wheel zoom, drag-to-zoom-box, and a double-click reset are
        # all built into the plotly modebar/canvas by default.
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
    else:
        st.info("Select at least one S-parameter to plot.")
