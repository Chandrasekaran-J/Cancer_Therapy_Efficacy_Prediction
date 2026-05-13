# Stage 4 — Differential Biomarker Discovery

Identifies metabolic reactions with significantly different flux/presence between Sensitive and Resistant cohorts, per therapy.

**Tests used (adaptive by cohort size):**
- Large cohorts ($N > 500$): **Mann-Whitney U** + Fisher's exact (FDR < 0.05)
- Small cohorts ($N \leq 500$): **Permutation test** (10,000 shuffles, vectorised) + Fisher's exact (FDR < 0.20)

**Biomarker classification:**
- `flux_only` — magnitude test significant only
- `presence_only` — Fisher's exact only
- `both` — both tests significant (strongest)
- `none` — neither

**Outputs:** per-therapy CSV files with rxn_id, log2FC, p-values, FDR, biomarker type, gene/metabolite annotations.
