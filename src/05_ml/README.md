# Stage 5 — Machine learning

Two LightGBM classifiers combine reaction activity with 12 clinical covariates to predict chemotherapy and radiotherapy outcome labels. Sensitive is encoded as 1 and resistant as 0.

The original workflow selects reactions across each full treatment cohort, tunes hyperparameters with 50 Optuna TPE trials and three stratified folds, and then evaluates fixed features and hyperparameters with five stratified folds. TPE and the five-fold split use seed 42; the tuning splits use seed 0. Class weighting uses `N_resistant / N_sensitive`. Binary predictions use a 0.5 threshold; F1 is the binary positive-class score. The pooled out-of-fold AUROCs are 0.785 and 0.818, respectively.

`feature_schema.py` supplies the same binary `abs(flux) > 1e-9` encoding to training and prediction. It recognizes the current `rxn_` prefix and legacy `flux_` prefix. A required reaction missing from an inference matrix raises an error instead of silently becoming zero.

`plot_shap_importance.py` can redraw existing SHAP importance CSV files without patient data or retraining. It assigns steel blue to reaction features and dark orange to clinical features, preserves numeric values, and exports both PNG and vector PDF. The final-refit SHAP values describe the fitted classifier.

```bash
python src/05_ml/plot_shap_importance.py \
  --csv results/ml/Chemotherapy_shap_importance.csv \
  --therapy Chemotherapy --out-dir results/ml
```

Use `ml_pipeline.py --help` for the training and prediction inputs, including `--biomarker-dir`. The historical `therapy_recommendations.csv` filename is retained for compatibility; its contents are model classifications for research use.
