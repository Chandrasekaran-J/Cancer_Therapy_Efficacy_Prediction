# Stage 4 — Differential Biomarker Discovery

Identifies metabolic reactions with significantly different flux/presence between Sensitive and Resistant cohorts, per therapy.

**Tests used (adaptive by cohort size):**
- Large cohorts ($N > 500$): **Mann-Whitney U** + Fisher's exact (FDR < 0.05)
- Small cohorts ($N \leq 500$): **Permutation test** (10,000 shuffles, vectorised) + Fisher's exact (FDR < 0.20)

**Biomarker classification:**
- `flux_only` — magnitude test significant only
- `presence_only` — Fisher's exact only
- `both` — both tests significant
- `none` — neither

**Outputs:** per-therapy CSV files with rxn_id, log2FC, p-values, FDR, biomarker type, gene/metabolite annotations.

Volcano colours require both a flux-test adjusted p-value below the cohort cutoff and `abs(log2FC) > 0.5`. Grey points are below one or both display cutoffs, including some reactions selected by the two-test union. The displayed bar-chart subset follows test-category and adjusted-p ordering, not an AUC ranking. Its single-reaction AUC is oriented as `max(AUC, 1-AUC)`. Plots are exported as 300-dpi PNG and vector PDF.
