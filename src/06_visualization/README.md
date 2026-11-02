# Stage 6 — Cross-therapy visualization

`biomarker_scatter.py` compares selected chemotherapy and radiotherapy reaction effects and writes an interactive HTML file with Plotly embedded for offline use.

- Purple circles: shared candidates, at their computed chemotherapy and radiotherapy summary coordinates.
- Teal triangles: chemotherapy-only candidates, projected to the radiotherapy zero axis.
- Orange squares: radiotherapy-only candidates, projected to the chemotherapy zero axis.

The projected coordinate for a therapy-only candidate is a display convention, not a computed effect of zero in the other therapy. Both axes use piecewise-linear spacing between the displayed tick labels. Hover text retains the original effect values and explicitly identifies projections.

```bash
python src/06_visualization/biomarker_scatter.py \
  --chemo results/biomarkers/Chemotherapy/biomarker_rxns_mapped.csv \
  --radio results/biomarkers/Radiation_Therapy/biomarker_rxns_mapped.csv \
  --output results/biomarker_scatter.html
```
