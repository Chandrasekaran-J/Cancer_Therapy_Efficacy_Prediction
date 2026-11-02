"""Compare therapy-specific biomarker tables with explicit projection semantics."""

import argparse
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

X_REAL = [-6, -5, -4, -3, -2, -1, 0, 0.25, 0.5, 0.75, 1.0]
Y_REAL = [-30, -7, -5, -3, -1, 0, 0.5, 1.0, 1.5, 2.0]
X_DISPLAY = list(range(-5, 6))
Y_DISPLAY = list(range(-4, 6))


def load(path):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    required = {"rxn_id", "log2_fold_change", "rxn_name"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    frame = frame[["rxn_id", "log2_fold_change", "rxn_name"]].copy()
    frame["log2_fold_change"] = pd.to_numeric(frame["log2_fold_change"], errors="raise")
    if frame["rxn_id"].duplicated().any():
        raise ValueError("Reaction identifiers must be unique within each therapy table")
    if not np.isfinite(frame["log2_fold_change"]).all():
        raise ValueError("Fold changes must be finite")
    return frame


def to_display(value, real, display):
    """Piecewise-linear coordinates with extrapolation, avoiding edge clipping."""
    if value < real[0]:
        return display[0] + (value - real[0]) * (display[1] - display[0]) / (real[1] - real[0])
    if value > real[-1]:
        return display[-1] + (value - real[-1]) * (display[-1] - display[-2]) / (real[-1] - real[-2])
    return float(np.interp(value, real, display))


def build_figure(chemo, radio):
    shared = chemo.merge(radio, on="rxn_id", suffixes=("_chemo", "_radio"), validate="one_to_one")
    chemo_only = chemo[~chemo.rxn_id.isin(shared.rxn_id)]
    radio_only = radio[~radio.rxn_id.isin(shared.rxn_id)]
    tx = lambda value: to_display(value, X_REAL, X_DISPLAY)
    ty = lambda value: to_display(value, Y_REAL, Y_DISPLAY)
    fig = go.Figure()
    common_hover = [
        f"<b>{escape(str(r.rxn_id))}</b><br>{escape(str(r.rxn_name_chemo))}<br>"
        f"Chemotherapy log₂FC: {r.log2_fold_change_chemo:+.3f}<br>"
        f"Radiotherapy log₂FC: {r.log2_fold_change_radio:+.3f}"
        for r in shared.itertuples()
    ]
    fig.add_trace(go.Scatter(
        x=[tx(v) for v in shared.log2_fold_change_chemo],
        y=[ty(v) for v in shared.log2_fold_change_radio],
        mode="markers", name=f"Shared (n={len(shared)})",
        marker=dict(color="mediumpurple", size=10, symbol="circle"),
        customdata=common_hover, hovertemplate="%{customdata}<extra></extra>"))
    for table, therapy, colour, symbol, horizontal in (
            (chemo_only, "Chemotherapy", "#1abcb0", "triangle-up", True),
            (radio_only, "Radiotherapy", "#f5a623", "square", False)):
        opposite = "Radiotherapy" if horizontal else "Chemotherapy"
        hover = [
            f"<b>{escape(str(r.rxn_id))}</b><br>{escape(str(r.rxn_name))}<br>"
            f"{therapy} log₂FC: {r.log2_fold_change:+.3f}<br>"
            f"{opposite}: not selected in the supplied table.<br>"
            "The other coordinate is an axis projection, not a measured zero effect."
            for r in table.itertuples()
        ]
        fig.add_trace(go.Scatter(
            x=[tx(v) for v in table.log2_fold_change] if horizontal else [tx(0)] * len(table),
            y=[ty(0)] * len(table) if horizontal else [ty(v) for v in table.log2_fold_change],
            mode="markers", name=f"{therapy}-only, projected (n={len(table)})",
            marker=dict(color=colour, size=9, symbol=symbol),
            customdata=hover, hovertemplate="%{customdata}<extra></extra>"))
    fig.add_hline(y=ty(0), line_dash="dash", line_color="grey", line_width=1)
    fig.add_vline(x=tx(0), line_dash="dash", line_color="grey", line_width=1)
    fig.update_layout(
        title=dict(text="Reaction effects across chemotherapy and radiotherapy", font_size=18),
        xaxis=dict(title="Chemotherapy log₂FC (sensitive / resistant)",
                   tickvals=X_DISPLAY, ticktext=[str(v) for v in X_REAL]),
        yaxis=dict(title="Radiotherapy log₂FC (sensitive / resistant)",
                   tickvals=Y_DISPLAY, ticktext=[str(v) for v in Y_REAL]),
        annotations=[dict(x=0, y=-0.18, xref="paper", yref="paper", xanchor="left",
                          showarrow=False, align="left", font_size=12,
                          text="Both axes use piecewise-linear tick spacing.<br>"
                               "Therapy-only candidates are projected to the other zero axis; "
                               "that coordinate is not an estimated effect.")],
        legend=dict(orientation="h", y=1.10, x=0),
        font=dict(size=13), template="plotly_white",
        width=1050, height=760, margin=dict(l=90, r=35, t=120, b=130))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemo", default="./Chemotherapy/biomarker_rxns_mapped.csv")
    parser.add_argument("--radio", default="./Radiation_Therapy/biomarker_rxns_mapped.csv")
    parser.add_argument("--output", default="biomarker_scatter.html")
    args = parser.parse_args()
    fig = build_figure(load(args.chemo), load(args.radio))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True, full_html=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
