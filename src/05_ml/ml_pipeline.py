"""
ml_pipeline.py
Complete ML pipeline for therapy response prediction.

Builds two separate LightGBM models:
  1. Chemotherapy sensitivity model  (47 biomarker fluxes + clinical)
  2. Radiation Therapy sensitivity model (95 biomarker fluxes + clinical)

Then predicts on all unlabeled patients and recommends therapy.

Usage:
    python ml_pipeline.py               # full pipeline
    python ml_pipeline.py --step build  # only build feature matrices
    python ml_pipeline.py --step train  # only train models
    python ml_pipeline.py --step predict # only predict unlabeled
"""

import os
import argparse
import logging
import warnings
import pickle

from feature_schema import is_reaction_feature, reaction_activity, reaction_feature_row
from plot_shap_importance import plot_shap_importance

import numpy as np
import pandas as pd
import h5py
import lightgbm as lgb
import shap

from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    f1_score, classification_report
)
from sklearn.preprocessing import LabelEncoder

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
H5_PATH       = "/data/chandrasekaran/algo/Biomarkers/fba_matrix.h5"
MANIFEST_PATH = "/data/chandrasekaran/algo/Biomarkers/Final_biomarkers/patient_therapy_labels.tsv"
CLINICAL_PATH = "/data/chandrasekaran/algo/Biomarkers/clinical.tsv"
BIOMARKER_DIR = "/data/chandrasekaran/algo/Biomarkers/Final_biomarkers"
OUT_DIR       = "/data/chandrasekaran/algo/Biomarkers/ML"

CHEMO_BIOMARKERS_CSV = os.path.join(BIOMARKER_DIR, "Chemotherapy/top_biomarkers.csv")
RAD_BIOMARKERS_CSV   = os.path.join(BIOMARKER_DIR, "Radiation_Therapy/top_biomarkers.csv")

# Clinical features to use
CLINICAL_FEATURES = [
    "age_at_diagnosis_years",          # numeric
    "demographic.gender",              # categorical
    "demographic.race",                # categorical
    "demographic.ethnicity",           # categorical
    "cases.disease_type",              # categorical
    "cases.primary_site",              # categorical
    "diagnoses.ajcc_pathologic_stage", # categorical
    "diagnoses.tissue_or_organ_of_origin", # categorical
    "diagnoses.primary_diagnosis",     # categorical
    "diagnoses.tumor_grade",           # categorical
    "diagnoses.prior_malignancy",      # categorical
    "diagnoses.prior_treatment",       # categorical
]

N_FOLDS       = 5
OPTUNA_TRIALS = 50   # hyperparameter search iterations
N_JOBS_CV      = 30   # parallel jobs during CV (per LightGBM model) — 60/2 therapies
THRESHOLD     = 0.5  # for therapy recommendation


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load and clean clinical data
# ══════════════════════════════════════════════════════════════════════════════

def load_clinical(clinical_path):
    log.info("Loading clinical data...")
    df = pd.read_csv(clinical_path, sep="\t", dtype=str, low_memory=False)

    # Replace TCGA's '--' missing marker with NaN
    df = df.replace("'--", np.nan).replace("'--'", np.nan)

    # Age: days → years
    df["age_at_diagnosis_years"] = pd.to_numeric(
        df["diagnoses.age_at_diagnosis"], errors="coerce"
    ) / 365.25

    # Deduplicate: keep one row per case_id
    # Safe because clinical/demographic columns are same across treatment rows
    df = df.drop_duplicates(subset=["cases.case_id"], keep="first")
    df = df.set_index("cases.case_id")

    # Fill categorical missing with "Unknown"
    cat_cols = [c for c in CLINICAL_FEATURES if c != "age_at_diagnosis_years"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").str.strip().str.lower()

    log.info(f"  Clinical: {len(df)} unique patients")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Load flux matrix
# ══════════════════════════════════════════════════════════════════════════════

def load_flux(h5_path):
    log.info("Loading flux matrix...")
    with h5py.File(h5_path, "r") as hf:
        fluxes  = hf["fluxes"][:]
        uuids   = np.array([u.decode() if isinstance(u, bytes) else u
                            for u in hf["uuids"][:]])
        rxn_ids = np.array([r.decode() if isinstance(r, bytes) else r
                            for r in hf["rxn_ids"][:]])
    log.info(f"  Flux: {fluxes.shape[0]} patients x {fluxes.shape[1]} reactions")
    return fluxes, uuids, rxn_ids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Build feature matrix for one therapy
# ══════════════════════════════════════════════════════════════════════════════

def build_features(therapy, manifest, clinical, fluxes, uuids, rxn_ids, biomarker_csv):
    log.info(f"\nBuilding feature matrix: {therapy}")

    # Load biomarker reaction IDs for this therapy
    bio_df      = pd.read_csv(biomarker_csv)
    biomarker_rxns = bio_df["rxn_id"].tolist()
    rxn_to_idx  = {r: i for i, r in enumerate(rxn_ids)}
    valid_rxns  = [r for r in biomarker_rxns if r in rxn_to_idx]
    rxn_indices = [rxn_to_idx[r] for r in valid_rxns]
    log.info(f"  Biomarker reactions: {len(valid_rxns)}")

    # Filter manifest to this therapy + labeled patients
    sub = manifest[
        (manifest["therapy_group"] == therapy) &
        (manifest["sensitivity"].isin(["Sensitive", "Resistant"]))
    ].copy()
    log.info(f"  Labeled patients: {len(sub)}")

    # UUID → HDF5 row index
    uuid_to_h5idx = {u: i for i, u in enumerate(uuids)}

    rows = []
    for _, row in sub.iterrows():
        uid      = row["uuid"]
        case_uid = row["case_uuid"]
        label    = 1 if row["sensitivity"] == "Sensitive" else 0

        if uid not in uuid_to_h5idx:
            continue

        h5_idx = uuid_to_h5idx[uid]

        # Flux → binary presence/absence (1 if active, 0 if silent)
        flux_vals   = fluxes[h5_idx, rxn_indices]
        binary_vals = reaction_activity(flux_vals)

        # Clinical features
        clin = {}
        if case_uid in clinical.index:
            clin_row = clinical.loc[case_uid]
            for feat in CLINICAL_FEATURES:
                clin[feat] = clin_row[feat] if feat in clin_row.index else np.nan
        else:
            for feat in CLINICAL_FEATURES:
                clin[feat] = np.nan if feat == "age_at_diagnosis_years" else "Unknown"

        entry = {"uuid": uid, "case_uuid": case_uid, "sensitivity": label}
        for rxn, val in zip(valid_rxns, binary_vals):
            entry[f"rxn_{rxn}"] = val  # rxn_ prefix (binary, not raw flux)
        entry.update(clin)
        rows.append(entry)

    df = pd.DataFrame(rows)
    log.info(f"  Feature matrix: {len(df)} patients x {len(df.columns)} columns")

    # Encode categoricals as LightGBM category dtype
    cat_cols = [c for c in CLINICAL_FEATURES if c != "age_at_diagnosis_years"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype("category")

    return df, valid_rxns


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Train LightGBM with stratified K-fold CV + SHAP
# ══════════════════════════════════════════════════════════════════════════════

def train_model(therapy, feature_df, valid_rxns, out_dir):
    log.info(f"\nTraining model: {therapy}")
    os.makedirs(out_dir, exist_ok=True)

    # Feature/label split
    meta_cols    = ["uuid", "case_uuid", "sensitivity"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    cat_cols     = [c for c in CLINICAL_FEATURES if c != "age_at_diagnosis_years"
                    and c in feature_cols]

    X = feature_df[feature_cols]
    y = feature_df["sensitivity"].values

    n_s = y.sum()
    n_r = (y == 0).sum()
    log.info(f"  Sensitive={n_s}, Resistant={n_r}, Features={len(feature_cols)}")

    # Three-fold hyperparameter search, preceding five-fold evaluation.
    # This reproduces the reported workflow; it is not fully nested CV.
    log.info("  Running Optuna hyperparameter search...")

    def objective(trial):
        p = {
            "objective":        "binary",
            "metric":           "auc",
            "boosting_type":    "gbdt",
            "verbosity":        -1,
            "n_jobs":           -1,
            "random_state":     42,
            "scale_pos_weight": n_r / n_s,
            # Hyperparameters to tune
            "num_leaves":        trial.suggest_int("num_leaves", 16, 128),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        aucs = []
        for tr_idx, va_idx in inner_skf.split(X, y):
            m = lgb.LGBMClassifier(**p)
            m.fit(X.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X.iloc[va_idx], y[va_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(-1)],
                  categorical_feature=cat_cols)
            aucs.append(roc_auc_score(y[va_idx], m.predict_proba(X.iloc[va_idx])[:, 1]))
        return np.mean(aucs)

    # Parallel Optuna hyperparameter search
    log.info(f"  Optuna: {OPTUNA_TRIALS} trials × 3-fold CV, parallel workers=10...")
    
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    
    # Optuna's n_jobs enables parallel trial evaluation
    # (Bayesian sampler still sees completed results sequentially for optimization)
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=10, show_progress_bar=False)

    best_params = {
        "objective":        "binary",
        "metric":           "auc",
        "boosting_type":    "gbdt",
        "verbosity":        -1,
        "n_jobs":           -1,
        "random_state":     42,
        "scale_pos_weight": n_r / n_s,
        **study.best_params,
    }
    log.info(f"  Best params (trial {study.best_trial.number}, AUC={study.best_value:.4f}):")
    for k, v in study.best_params.items():
        log.info(f"    {k}: {v}")

    # Save best params
    pd.DataFrame([study.best_params]).to_csv(
        os.path.join(out_dir, f"{therapy.replace(' ','_')}_best_params.csv"), index=False
    )

    # Five-fold evaluation with the selected feature set and fixed best params.
    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    cv_aucs = []
    cv_bacs = []
    cv_f1s  = []
    oof_preds = np.zeros(len(y))
    fold_models = []
    all_shap_vals = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)],
            categorical_feature=cat_cols,
        )

        preds   = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds

        auc = roc_auc_score(y_val, preds)
        bac = balanced_accuracy_score(y_val, (preds >= THRESHOLD).astype(int))
        f1  = f1_score(y_val, (preds >= THRESHOLD).astype(int))

        cv_aucs.append(auc)
        cv_bacs.append(bac)
        cv_f1s.append(f1)
        fold_models.append(model)

        log.info(f"  Fold {fold+1}: AUC={auc:.4f}  BAC={bac:.4f}  F1={f1:.4f}")

        # SHAP values for this fold
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # positive class
        all_shap_vals.append(shap_vals)

    # Overall OOF metrics
    oof_auc = roc_auc_score(y, oof_preds)
    oof_bac = balanced_accuracy_score(y, (oof_preds >= THRESHOLD).astype(int))
    oof_f1  = f1_score(y, (oof_preds >= THRESHOLD).astype(int))

    log.info(f"\n  CV Summary ({N_FOLDS}-fold):")
    log.info(f"    AUC : {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    log.info(f"    BAC : {np.mean(cv_bacs):.4f} ± {np.std(cv_bacs):.4f}")
    log.info(f"    F1  : {np.mean(cv_f1s):.4f}  ± {np.std(cv_f1s):.4f}")
    log.info(f"  OOF AUC={oof_auc:.4f}  BAC={oof_bac:.4f}  F1={oof_f1:.4f}")

    # Save CV metrics
    metrics = pd.DataFrame({
        "fold":  list(range(1, N_FOLDS+1)) + ["OOF"],
        "auc":   cv_aucs + [oof_auc],
        "bac":   cv_bacs + [oof_bac],
        "f1":    cv_f1s  + [oof_f1],
    })
    metrics.to_csv(os.path.join(out_dir, f"{therapy.replace(' ','_')}_cv_metrics.csv"), index=False)

    # Train final model on ALL data
    log.info("  Training final model on all data...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X, y, categorical_feature=cat_cols)

    # Save model
    model_path = os.path.join(out_dir, f"{therapy.replace(' ','_')}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "feature_cols": feature_cols,
                     "cat_cols": cat_cols, "therapy": therapy}, f)
    log.info(f"  Model saved: {model_path}")

    # ── SHAP Analysis ──────────────────────────────────────────────────────────
    log.info("  Computing SHAP on full data...")
    explainer_final = shap.TreeExplainer(final_model)
    shap_vals_final = explainer_final.shap_values(X)
    if isinstance(shap_vals_final, list):
        shap_vals_final = shap_vals_final[1]

    # Mean |SHAP| per feature
    mean_shap = np.abs(shap_vals_final).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":    feature_cols,
        "mean_abs_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    # Classify rxn_ features correctly, then save CSV, PNG and vector PDF.
    shap_df = plot_shap_importance(shap_df, therapy, out_dir)

    # SHAP beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals_final, X, max_display=30, show=False)
    plt.title(f"SHAP Summary — {therapy}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{therapy.replace(' ','_')}_shap_beeswarm.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, f"{therapy.replace(' ','_')}_shap_beeswarm.pdf"), bbox_inches="tight")
    plt.close()

    log.info(f"  Top 5 features by SHAP:")
    for _, row in shap_df.head(5).iterrows():
        log.info(f"    {row['rank']}. {row['feature']:<40} SHAP={row['mean_abs_shap']:.5f} ({row['feature_type']})")

    return final_model, feature_cols, cat_cols, shap_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Predict on unlabeled patients + recommend therapy
# ══════════════════════════════════════════════════════════════════════════════

def predict_unlabeled(manifest, clinical, fluxes, uuids, rxn_ids,
                      chemo_model_info, rad_model_info, out_dir):
    log.info("\nPredicting on unlabeled patients...")

    # Patients without sensitivity label
    unlabeled = manifest[
        ~manifest["sensitivity"].isin(["Sensitive", "Resistant"])
    ].drop_duplicates(subset=["uuid"]).copy()
    log.info(f"  Unlabeled patients: {len(unlabeled)}")

    uuid_to_h5idx = {u: i for i, u in enumerate(uuids)}
    rxn_to_idx    = {r: i for i, r in enumerate(rxn_ids)}

    def build_patient_features(uid, case_uid, feature_cols, cat_cols):
        if uid not in uuid_to_h5idx:
            return None
        h5_idx    = uuid_to_h5idx[uid]
        flux_cols = [c for c in feature_cols if is_reaction_feature(c)]
        clin_cols = [c for c in feature_cols if not is_reaction_feature(c)]
        row = reaction_feature_row(fluxes[h5_idx], flux_cols, rxn_to_idx)

        if case_uid in clinical.index:
            clin_row = clinical.loc[case_uid]
            for col in clin_cols:
                row[col] = clin_row[col] if col in clin_row.index else (
                    np.nan if col == "age_at_diagnosis_years" else "Unknown"
                )
        else:
            for col in clin_cols:
                row[col] = np.nan if col == "age_at_diagnosis_years" else "Unknown"

        return row

    chemo_model, chemo_fcols, chemo_cat = (
        chemo_model_info["model"],
        chemo_model_info["feature_cols"],
        chemo_model_info["cat_cols"],
    )
    rad_model, rad_fcols, rad_cat = (
        rad_model_info["model"],
        rad_model_info["feature_cols"],
        rad_model_info["cat_cols"],
    )

    results = []
    for _, row in unlabeled.iterrows():
        uid      = row["uuid"]
        case_uid = row["case_uuid"]

        chemo_row = build_patient_features(uid, case_uid, chemo_fcols, chemo_cat)
        rad_row   = build_patient_features(uid, case_uid, rad_fcols,   rad_cat)

        if chemo_row is None or rad_row is None:
            continue

        chemo_X = pd.DataFrame([chemo_row])[chemo_fcols]
        rad_X   = pd.DataFrame([rad_row])[rad_fcols]

        for col in chemo_cat:
            if col in chemo_X.columns:
                chemo_X[col] = chemo_X[col].astype("category")
        for col in rad_cat:
            if col in rad_X.columns:
                rad_X[col] = rad_X[col].astype("category")

        p_chemo = float(chemo_model.predict_proba(chemo_X)[0, 1])
        p_rad   = float(rad_model.predict_proba(rad_X)[0, 1])

        # Recommendation logic
        chemo_sens = p_chemo >= THRESHOLD
        rad_sens   = p_rad   >= THRESHOLD

        if chemo_sens and rad_sens:
            recommendation = "Both_Therapies"
            confidence     = max(p_chemo, p_rad)
        elif chemo_sens:
            recommendation = "Chemotherapy"
            confidence     = p_chemo
        elif rad_sens:
            recommendation = "Radiation_Therapy"
            confidence     = p_rad
        else:
            recommendation = "Resistant_to_Both"
            confidence     = 1 - min(p_chemo, p_rad)

        results.append({
            "uuid":                uid,
            "case_uuid":           case_uid,
            "p_chemo_sensitive":   round(p_chemo, 4),
            "p_radiation_sensitive": round(p_rad, 4),
            "recommendation":      recommendation,
            "confidence":          round(confidence, 4),
        })

    results_df = pd.DataFrame(results)

    # Summary
    rec_counts = results_df["recommendation"].value_counts()
    log.info(f"\n  Therapy Recommendations ({len(results_df)} patients):")
    for rec, cnt in rec_counts.items():
        log.info(f"    {rec:<25}: {cnt} ({cnt/len(results_df)*100:.1f}%)")

    out_path = os.path.join(out_dir, "therapy_recommendations.csv")
    results_df.to_csv(out_path, index=False)
    log.info(f"\n  Recommendations saved: {out_path}")

    # Plot recommendation distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    color_map = {
        "Chemotherapy":      "steelblue",
        "Radiation_Therapy": "darkorange",
        "Both_Therapies":    "purple",
        "Resistant_to_Both": "grey",
    }
    bars   = rec_counts.index.tolist()
    counts = rec_counts.values.tolist()
    colors = [color_map.get(b, "green") for b in bars]
    ax.bar(bars, counts, color=colors, alpha=0.8)
    ax.set_ylabel("Number of Patients", fontsize=12)
    ax.set_title("Therapy Recommendations — Unlabeled Patients", fontsize=13)
    ax.set_xticklabels(bars, rotation=20, ha="right")
    for i, v in enumerate(counts):
        ax.text(i, v + 10, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "therapy_recommendations_plot.png"), dpi=150)
    plt.close()

    # Scatter: p_chemo vs p_radiation
    fig, ax = plt.subplots(figsize=(8, 7))
    c_map = {
        "Chemotherapy": "steelblue", "Radiation_Therapy": "darkorange",
        "Both_Therapies": "purple",  "Resistant_to_Both": "grey",
    }
    for rec, grp in results_df.groupby("recommendation"):
        ax.scatter(grp["p_chemo_sensitive"], grp["p_radiation_sensitive"],
                   c=c_map.get(rec, "green"), s=8, alpha=0.5, label=rec)
    ax.axvline(THRESHOLD, color="black", linestyle="--", lw=0.8)
    ax.axhline(THRESHOLD, color="black", linestyle="--", lw=0.8)
    ax.set_xlabel("P(Chemo Sensitive)", fontsize=12)
    ax.set_ylabel("P(Radiation Sensitive)", fontsize=12)
    ax.set_title("Therapy Sensitivity Probability Space", fontsize=13)
    ax.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "therapy_probability_scatter.png"), dpi=150)
    plt.close()

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all",
                        choices=["all", "build", "train", "predict"])
    parser.add_argument("--h5",       default=H5_PATH)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--clinical", default=CLINICAL_PATH)
    parser.add_argument("--biomarker-dir", default=BIOMARKER_DIR,
                        help="Directory containing per-therapy top_biomarkers.csv files")
    parser.add_argument("--out-dir",  default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load data (always needed) ──────────────────────────────────────────────
    clinical = load_clinical(args.clinical)
    fluxes, uuids, rxn_ids = load_flux(args.h5)

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    manifest.columns = manifest.columns.str.strip()
    manifest["uuid"] = manifest["file_name"].str.replace(
        r"\.rna_seq\.augmented_star_gene_counts\.tsv$", "", regex=True
    )
    for col in ["therapy_group", "sensitivity", "case_uuid", "uuid"]:
        manifest[col] = manifest[col].str.strip()

    # ── STEP BUILD ────────────────────────────────────────────────────────────
    if args.step in ["all", "build"]:
        log.info("\n" + "="*60 + "\nSTEP 1: Building feature matrices\n" + "="*60)

        chemo_df, chemo_rxns = build_features(
            "Chemotherapy", manifest, clinical, fluxes, uuids, rxn_ids,
            os.path.join(args.biomarker_dir, "Chemotherapy", "top_biomarkers.csv")
        )
        chemo_df.to_csv(os.path.join(args.out_dir, "chemo_features.csv"), index=False)
        log.info(f"  Saved: chemo_features.csv ({len(chemo_df)} patients)")

        rad_df, rad_rxns = build_features(
            "Radiation Therapy", manifest, clinical, fluxes, uuids, rxn_ids,
            os.path.join(args.biomarker_dir, "Radiation_Therapy", "top_biomarkers.csv")
        )
        rad_df.to_csv(os.path.join(args.out_dir, "radiation_features.csv"), index=False)
        log.info(f"  Saved: radiation_features.csv ({len(rad_df)} patients)")

    # ── STEP TRAIN ────────────────────────────────────────────────────────────
    if args.step in ["all", "train"]:
        log.info("\n" + "="*60 + "\nSTEP 2: Training models\n" + "="*60)

        chemo_df = pd.read_csv(os.path.join(args.out_dir, "chemo_features.csv"))
        rad_df   = pd.read_csv(os.path.join(args.out_dir, "radiation_features.csv"))

        # Re-encode categoricals
        cat_cols = [c for c in CLINICAL_FEATURES if c != "age_at_diagnosis_years"]
        for col in cat_cols:
            if col in chemo_df.columns:
                chemo_df[col] = chemo_df[col].fillna("Unknown").astype("category")
            if col in rad_df.columns:
                rad_df[col]   = rad_df[col].fillna("Unknown").astype("category")

        chemo_model, chemo_fcols, chemo_cat, chemo_shap = train_model(
            "Chemotherapy", chemo_df,
            [r for r in chemo_df.columns if r.startswith("rxn_")],
            args.out_dir
        )

        rad_model, rad_fcols, rad_cat, rad_shap = train_model(
            "Radiation Therapy", rad_df,
            [r for r in rad_df.columns if r.startswith("rxn_")],
            args.out_dir
        )

    # ── STEP PREDICT ──────────────────────────────────────────────────────────
    if args.step in ["all", "predict"]:
        log.info("\n" + "="*60 + "\nSTEP 3: Predicting on unlabeled patients\n" + "="*60)

        # Load saved models
        with open(os.path.join(args.out_dir, "Chemotherapy_model.pkl"), "rb") as f:
            chemo_info = pickle.load(f)
        with open(os.path.join(args.out_dir, "Radiation_Therapy_model.pkl"), "rb") as f:
            rad_info = pickle.load(f)

        predict_unlabeled(
            manifest, clinical, fluxes, uuids, rxn_ids,
            chemo_info, rad_info, args.out_dir
        )

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
