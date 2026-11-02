# Predicting Cancer Therapy Response Through Patient-Specific Genome-Scale Metabolic Models

**Authors:** Chandrasekaran J and Karthik Rajesh

A research workflow connecting TCGA transcriptomics, genome-scale metabolic modelling, reaction-level biomarker discovery and machine learning for cancer therapy response. The study integrates 11,505 RNA-seq profiles with Recon3D using GIMME and examines outcomes across six treatment categories.

## Study results

| Quantity | Chemotherapy | Radiotherapy |
|---|---:|---:|
| Labelled patient–therapy observations | 1,719 | 1,595 |
| Selected reaction features | 47 | 95 |
| Genes mapped from selected reactions | 53 | 81 |
| Pooled out-of-fold AUROC | 0.785 | 0.818 |
| Balanced accuracy | 0.681 | 0.745 |
| F1 score | 0.783 | 0.847 |

The two reaction sets share 37 candidates with concordant effect directions. These are the study's original results; the presentation and inference fixes in this repository do not constitute a new training run.

## Repository contents

| Directory | Contents |
|---|---|
| `src/01_data_prep/` | Description of the upstream gene-mapping inputs |
| `src/02_fba/` | Canonical reaction ordering, parallel FBA workers and HDF5 merging |
| `src/03_qc/` | Model diagnostics and optional FVA/gap-filling utilities |
| `src/04_biomarkers/` | Treatment-stratified differential tests, annotations and plots |
| `src/05_ml/` | LightGBM training, prediction, shared feature encoding and SHAP plotting |
| `src/06_visualization/` | Interactive cross-therapy comparison |
| `tests/` | Regression checks for feature encoding and figure semantics |
| `submit_chunk.sh` | SGE array-job template for the FBA and QC workers |
| `results/` | Location for generated outputs |

The executable workflow starts from prepared GIMME model files and sample/outcome manifests. The upstream expression-preprocessing and GIMME-extraction scripts are not included. Model files, flux matrices, trained estimators and patient-level data are supplied separately by the researcher.

## Installation

Use Python 3.10 or later and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

FBA and the optional model-diagnostic utilities require an installed Gurobi Optimizer and a valid licence. Cluster paths and resource requests must be adapted to the local environment. Scripts with argument parsers document their inputs through `--help`.

## Running the workflow

Prepare the model pickles, the canonical reaction list, the transcriptomics manifest and the treatment-label table first. Review the path constants in the FBA scripts for the original HPC layout.

```bash
# Run once to build the canonical reaction ordering.
python src/02_fba/precompute_rxn_list_gurobi.py

# Activate your Python environment before submitting jobs.
# Create the log directory before the scheduler opens its output files.
mkdir -p logs
qsub -t 1-58 submit_chunk.sh fba \
  --manifest data/transcriptomics_manifest.tsv \
  --pkl-dir data/models --out-dir results/fba_chunks --chunk-size 200

# Run after the array completes.
python src/02_fba/step1_merge_gurobi.py \
  --results-dir results/fba_chunks --out results/fba_matrix.h5 \
  --summary-csv results/fba_summary.csv

qsub -t 1-58 submit_chunk.sh qc \
  --manifest data/transcriptomics_manifest.tsv \
  --pkl-dir data/models --out-dir results/qc_chunks --chunk-size 200
python src/03_qc/step2_merge_report_gurobi.py \
  --qc-dir results/qc_chunks --out-csv results/qc_report.csv \
  --out-txt results/qc_report.txt

python src/04_biomarkers/biomarkers_parallel.py \
  --h5 results/fba_matrix.h5 --manifest data/patient_therapy_labels.tsv \
  --out-dir results/biomarkers --n-jobs 8

python src/05_ml/ml_pipeline.py --step train \
  --h5 results/fba_matrix.h5 --manifest data/patient_therapy_labels.tsv \
  --clinical data/clinical.tsv --biomarker-dir results/biomarkers \
  --out-dir results/ml

python src/06_visualization/biomarker_scatter.py \
  --chemo results/biomarkers/Chemotherapy/biomarker_rxns_mapped.csv \
  --radio results/biomarkers/Radiation_Therapy/biomarker_rxns_mapped.csv \
  --output results/biomarker_scatter.html
```

Set the array range to `ceil(number_of_manifest_rows / chunk_size)`. The workers read `SGE_TASK_ID` directly. FVA and gap filling are optional utilities, not prerequisites for reproducing the study's reported single-solution FBA analysis.

To redraw a SHAP bar chart from an existing importance table without retraining:

```bash
python src/05_ml/plot_shap_importance.py \
  --csv results/ml/Chemotherapy_shap_importance.csv \
  --therapy Chemotherapy --out-dir results/ml
```

The plotting command reclassifies reaction features from their names, preserves the supplied SHAP values, and writes PNG and vector PDF figures. Both `rxn_` and the legacy `flux_` prefix are recognized.

## Analysis conventions

- **Model construction:** Recon3D, GIMME expression threshold `ln(10)`, biomass requirement of 85% of the reference optimum, and pruning threshold `1e-6`.
- **Outcome labels:** complete response, partial response and no measurable disease form the sensitive group; stable disease, persistent disease, progressive disease and no response form the resistant group. Unknown or uninformative outcomes are excluded.
- **Differential analysis:** two-sided Mann–Whitney tests for cohorts larger than 500 observations; 10,000-shuffle two-sided permutation tests otherwise; and two-sided Fisher tests using the implemented positive-flux threshold `v > 1e-9`. Benjamini–Hochberg adjustment is applied separately to each test family. Selection takes the union at 0.05 for the large cohorts and 0.20 for the smaller cohorts.
- **Effect summary:** `log2(clip((mean_sensitive + 1e-9) / (mean_resistant + 1e-9), 1e-9, 1e9))`, using signed flux means. Ratios near zero or involving negative flux require interpretation in the reaction's model direction.
- **Classification:** binary reaction activity `abs(v) > 1e-9` plus 12 clinical covariates. Sensitive is the positive class. The study uses full-cohort reaction selection and three-fold Optuna tuning followed by five-fold stratified classification with fixed selected features and hyperparameters. The scores describe this workflow; independent validation of the complete selection procedure remains future work.
- **SHAP:** feature attributions are calculated for the final refitted model. Blue bars identify reaction features and orange bars identify clinical covariates.

See the stage READMEs for figure conventions and input details.

## Data and software

Expression and clinical data originate from [TCGA through the GDC Data Portal](https://portal.gdc.cancer.gov/). The reference reconstruction is [Recon3D](https://doi.org/10.1038/nbt.4072). Data and model outputs are excluded from version control.

## Checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
bash -n submit_chunk.sh
```

## Reference

Chandrasekaran J and Karthik Rajesh. *Predicting Cancer Therapy Response Through Patient-Specific Genome-Scale Metabolic Models*. Manuscript in preparation.

Software is distributed under the [MIT licence](LICENSE).
