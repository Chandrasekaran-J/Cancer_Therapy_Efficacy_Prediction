# Stage 5 — Machine Learning

Trains two separate LightGBM binary classifiers — one per therapy — on biomarker reaction presence/absence features combined with 13 clinical covariates.

**Setup:**
- Hyperparameter tuning: Optuna with TPE sampler, 50 trials, 3-fold inner CV
- Honest evaluation: outer 5-fold stratified CV
- Class imbalance: `scale_pos_weight = N_Resistant / N_Sensitive`
- Interpretability: SHAP TreeSHAP on the final retrained model

**Results:**
- Chemotherapy OOF AUC: **0.785**
- Radiation Therapy OOF AUC: **0.818**

Predictions on unlabelled TCGA patients are produced in the `--step predict` phase.
