# Predicting Cancer Therapy Response Through Patient-Specific Genome-Scale Metabolic Models

**CS6024 — Computational Systems Biology, IIT Madras (2026)**
**Author:** Chandrasekaran J · BS22B013 · Department of Biotechnology

A multi-cohort framework for biomarker discovery and ML-based therapy recommendation from TCGA transcriptomic data. We integrate 11,505 TCGA RNA-seq profiles into the Recon3D human metabolic network using the GIMME algorithm, perform differential flux analysis to identify therapy-stratified biomarker reactions, and train LightGBM classifiers reaching out-of-fold AUCs of **0.785** (chemotherapy) and **0.818** (radiation therapy).

> 📄 **Full project report:** [`docs/CS6024_project_report.pdf`](docs/CS6024_project_report.pdf)

---

## Key Results

| Metric | Chemotherapy | Radiation Therapy |
|---|---|---|
| Cohort size | 1,719 patients | 1,595 patients |
| Significant biomarker reactions (FDR < 0.05) | 47 | 95 |
| Unique encoding genes | 53 | 81 |
| Best fold AUC | 0.826 | 0.845 |
| **Out-of-Fold AUC** | **0.785** | **0.818** |

- **10,603** patient-specific GSMMs successfully constructed (92.2% of 11,505 input profiles)
- **12.5× larger** cohort than Lewis & Kemp (2021), the closest comparable prior work
- All 37 shared biomarkers are **directionally concordant** between therapies, indicating a general metabolic resistance phenotype

---

## Repository Structure

```
CS6024_Project/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── LICENSE
├── .gitignore
│
├── docs/
│   └── CS6024_project_report.pdf          # Final 10-page report
│
├── src/
│   ├── 01_data_prep/
│   │   └── gene_mapping.py                # HGNC ↔ Entrez ↔ Recon3D gene-ID mapping
│   │
│   ├── 02_fba/                            # FBA pipeline (SGE array jobs)
│   │   ├── precompute_rxn_list_gurobi.py  # One-time: canonical reaction ordering
│   │   ├── step1_fba_chunk_gurobi.py      # Per-chunk FBA worker
│   │   └── step1_merge_gurobi.py          # Merge chunks → fba_matrix.h5
│   │
│   ├── 03_qc/                             # Model QC + FVA + gap-filling
│   │   ├── step2_qc_chunk_gurobi.py       # Per-chunk QC worker
│   │   ├── step2_merge_report_gurobi.py   # Merge QC reports
│   │   └── step2b_fva_gapfill_gurobi.py   # Loopless FVA + MILP gap-fill
│   │
│   ├── 04_biomarkers/
│   │   └── biomarkers_parallel.py         # Differential flux analysis (per therapy)
│   │
│   ├── 05_ml/
│   │   └── ml_pipeline.py                 # LightGBM + Optuna + SHAP
│   │
│   └── 06_visualization/
│       └── biomarker_scatter.py           # Cross-therapy log2FC scatter plot
│
└── results/                               # Pipeline outputs (created at runtime)
```

---

## Pipeline Overview

The analysis pipeline runs in six stages. Each stage produces inputs for the next; the entire pipeline was executed on the IIT Madras Actinium HPC cluster (SGE scheduler).

```
TCGA RNA-seq (11,505 patients)
        │
        ▼
[1] gene_mapping.py  ───────────► gene_map.pkl
        │
        ▼  (GIMME — external step, COBRApy)
Patient-specific GIMME models (10,603 .pkl files)
        │
        ▼
[2] FBA pipeline (parallel SGE array)
    ├── precompute_rxn_list_gurobi.py   → rxn_list.pkl
    ├── step1_fba_chunk_gurobi.py       → chunks/*.npz
    └── step1_merge_gurobi.py           → fba_matrix.h5  (11505 × 10600)
        │
        ▼
[3] QC + FVA + gap-fill (parallel SGE array)
    ├── step2_qc_chunk_gurobi.py        → qc_chunk_*.csv
    ├── step2_merge_report_gurobi.py    → qc_report.csv
    └── step2b_fva_gapfill_gurobi.py    → gap-filled models
        │
        ▼
[4] biomarkers_parallel.py  ──────► biomarkers/*.csv  (per therapy)
        │
        ▼
[5] ml_pipeline.py  ──────────────► models/*.pkl, shap_*.png, OOF predictions
        │
        ▼
[6] biomarker_scatter.py  ────────► biomarker_scatter.html (interactive)
```

---

## Quick Start

### Requirements

- Python 3.10+
- Gurobi Optimizer (WLS or full licence)
- HPC environment with SGE scheduler (recommended for full pipeline)
- ~150 GB disk space for full FBA matrix + intermediate files

```bash
pip install -r requirements.txt
```

### Running Individual Stages

```bash
# 1. Gene mapping (one-time)
python src/01_data_prep/gene_mapping.py

# 2. FBA pipeline
python src/02_fba/precompute_rxn_list_gurobi.py
qsub submit_fba.sh                      # SGE array job: step1_fba_chunk
python src/02_fba/step1_merge_gurobi.py

# 3. QC and (optional) gap-filling
qsub submit_qc.sh                       # SGE array job: step2_qc_chunk
python src/03_qc/step2_merge_report_gurobi.py
qsub submit_fva.sh --phase fva          # Optional: FVA + gap-fill
qsub submit_fva.sh --phase gapfill
python src/03_qc/step2b_fva_gapfill_gurobi.py --phase report

# 4. Differential biomarker analysis
python src/04_biomarkers/biomarkers_parallel.py --n-jobs 60

# 5. ML pipeline
python src/05_ml/ml_pipeline.py

# 6. Cross-therapy scatter plot
python src/06_visualization/biomarker_scatter.py
```

### Per-script Options

Each script has its own `--help`:

```bash
python src/04_biomarkers/biomarkers_parallel.py --help
python src/05_ml/ml_pipeline.py --help
```

For example, run biomarker analysis on a single therapy:

```bash
python src/04_biomarkers/biomarkers_parallel.py --therapy Chemotherapy
```

Run just one step of the ML pipeline:

```bash
python src/05_ml/ml_pipeline.py --step train
python src/05_ml/ml_pipeline.py --step predict
```

---

## Data Sources

This pipeline operates on three external data sources (not included in the repo):

| Source | Description | Access |
|---|---|---|
| TCGA RNA-seq | STAR-augmented gene counts (TPM, unstranded), 11,505 files | [GDC Data Portal](https://portal.gdc.cancer.gov/) |
| TCGA Clinical | Patient demographics + treatment outcomes (RECIST-labelled) | GDC Data Portal |
| Recon3D | Human metabolic network reference (SBML + MATLAB) | [VMH Database](https://www.vmh.life/) |

Patient-level data should be placed under a separate `data/` directory (gitignored). See individual scripts for expected paths.

---

## Methods Summary

- **GSMM construction:** GIMME algorithm (Becker & Palsson 2008) on Recon3D (10,600 reactions, 2,248 genes) with cutoff $\ln(10) \approx 2.303$ and 85% of $v_{\max}$ biomass constraint
- **FBA solver:** GLPK for GIMME phase, Gurobi for downstream FBA (WLS licence) — both guarantee LP optimality
- **Differential flux tests:**
  - Mann-Whitney U for cohorts with $N > 500$
  - Vectorised permutation test (10,000 shuffles) for $N \leq 500$
  - Fisher's exact for presence/absence in parallel
  - Benjamini-Hochberg FDR correction at $q < 0.05$
- **ML classifier:** LightGBM with Optuna hyperparameter tuning (TPE, 50 trials, 3-fold inner CV) and outer 5-fold stratified CV; SHAP TreeSHAP for interpretability
- **Class imbalance:** `scale_pos_weight = N_Resistant / N_Sensitive`

Full methodological details are in [`docs/CS6024_project_report.pdf`](docs/CS6024_project_report.pdf).

---

## Citation

If you use any part of this work, please cite the underlying methods:

```
Lewis, J. E. & Kemp, M. L. (2021) Integration of machine learning and
  genome-scale metabolic modeling identifies multi-omics biomarkers for
  radiation resistance. Nature Communications 12: 2700.

Brunk, E. et al. (2018) Recon3D enables a three-dimensional view of gene
  variation in human metabolism. Nature Biotechnology 36: 272–281.

Becker, S. A. & Palsson, B. Ø. (2008) Context-specific metabolic networks
  are consistent with experiments. PLoS Comput. Biol. 4: e1000082.
```

---

## Acknowledgements

The author thanks the course instructor for early guidance on the project direction — in particular, the timely correction that whole-genome shotgun sequencing data describes microbial communities rather than human tumour tissue, which prompted a revision of the original dual-pipeline proposal in favour of the present transcriptomics-based GSMM framework. Thanks also to the TCGA Research Network and the NCI Genomic Data Commons for open access to pan-cancer multi-omics data, and to the developers of COBRApy, LightGBM, Optuna, and SHAP for their open-source contributions.

---

## License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.
