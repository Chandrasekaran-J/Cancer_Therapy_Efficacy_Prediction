# Stage 6 — Visualization

Generates the cross-therapy log₂FC scatter plot comparing chemo and radiation biomarkers.

- Purple circles: reactions significant in **both** therapies (plotted at actual chemo/radio log2FC)
- Teal triangles: chemo-only biomarkers (pinned to y = 0)
- Amber squares: radio-only biomarkers (pinned to x = 0)

Output is a self-contained interactive `biomarker_scatter.html` (Plotly). The y-axis uses a non-uniform scale that maps the extreme outlier at $-30$ to a position adjacent to $-7$ for readability, while preserving uniform pixel spacing between all ticks.
