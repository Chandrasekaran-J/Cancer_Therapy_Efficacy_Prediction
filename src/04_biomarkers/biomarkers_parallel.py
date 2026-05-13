"""
biomarkers_parallel.py
Differential flux biomarker analysis — per therapy group, parallelized.

Tests used (adaptive by cohort size):
  Large cohorts (N > 500): Mann-Whitney U  + Fisher's exact  | FDR < 0.05
  Small cohorts (N ≤ 500): Permutation test + Fisher's exact | FDR < 0.20

Significant = passes FDR cutoff in EITHER magnitude OR presence/absence test
Biomarker type:
  "flux_only"     → magnitude test significant, Fisher's not
  "presence_only" → Fisher's significant, magnitude not
  "both"          → both significant (strongest)
  "none"          → neither

Usage:
    python biomarkers_parallel.py                          # all therapies
    python biomarkers_parallel.py --therapy Chemotherapy   # single therapy
    python biomarkers_parallel.py --therapy "Hormone Therapy"
    python biomarkers_parallel.py --n-jobs 60
"""

import os
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import h5py
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = "/data/chandrasekaran/algo/Biomarkers"
H5_PATH  = os.path.join(BASE_DIR, "fba_matrix.h5")
MANIFEST = "/data/chandrasekaran/algo/Biomarkers/Final_biomarkers/patient_therapy_labels.tsv"
OUT_DIR  = "/data/chandrasekaran/algo/Biomarkers/Final_biomarkers"

THERAPY_GROUPS = [
    "Chemotherapy",
    "Radiation Therapy",
    "Hormone Therapy",
    "Surgery",
    "Immunotherapy",
    "Targeted Therapy",
]

# Small therapies — use permutation test + relaxed FDR
SMALL_THERAPIES = {
    "Hormone Therapy",
    "Surgery",
    "Immunotherapy",
    "Targeted Therapy",
}

LARGE_N_THRESHOLD  = 500
N_PERMUTATIONS     = 10000
PRESENCE_THRESHOLD = 1e-9
MIN_PER_CLASS      = 5
FDR_CUTOFF_LARGE   = 0.05   # Chemo, Radiation
FDR_CUTOFF_SMALL   = 0.20   # Hormone, Surgery, Immunotherapy, Targeted


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_flux_matrix(h5_path):
    log.info(f"Loading flux matrix from {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        fluxes  = hf["fluxes"][:]
        uuids   = hf["uuids"][:].astype(str)
        rxn_ids = hf["rxn_ids"][:].astype(str)
    uuids   = np.array([u.decode() if isinstance(u, bytes) else u for u in uuids])
    rxn_ids = np.array([r.decode() if isinstance(r, bytes) else r for r in rxn_ids])
    log.info(f"Flux matrix: {fluxes.shape[0]} patients x {fluxes.shape[1]} reactions")
    return fluxes, uuids, rxn_ids


def load_manifest(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = df.columns.str.strip()
    for col in ["file_name", "therapy_group", "sensitivity"]:
        df[col] = df[col].str.strip()
    df["uuid"] = df["file_name"].str.replace(
        r"\.rna_seq\.augmented_star_gene_counts\.tsv$", "", regex=True
    )
    log.info(f"Manifest: {len(df)} rows | "
             f"S={(df['sensitivity']=='Sensitive').sum()} "
             f"R={(df['sensitivity']=='Resistant').sum()}")
    return df


def get_therapy_subset(fluxes, uuids, manifest, therapy):
    sub = manifest[
        (manifest["therapy_group"] == therapy) &
        (manifest["sensitivity"].isin(["Sensitive", "Resistant"]))
    ].copy()

    uuid_to_idx = {u: i for i, u in enumerate(uuids)}
    rows, labels, valid_uuids = [], [], []
    for _, row in sub.iterrows():
        uid = row["uuid"]
        if uid in uuid_to_idx:
            rows.append(uuid_to_idx[uid])
            labels.append(1 if row["sensitivity"] == "Sensitive" else 0)
            valid_uuids.append(uid)

    if not rows:
        return None, None, None

    X = np.nan_to_num(fluxes[rows, :], nan=0.0)
    y = np.array(labels, dtype=np.int32)
    return X, y, np.array(valid_uuids)


# ══════════════════════════════════════════════════════════════════════════════
# Per-reaction processing — called in parallel
# ══════════════════════════════════════════════════════════════════════════════

def _process_reaction(i, X_s, X_r, X, y, rxn_ids, use_permutation, n_perm, rng_seed):
    """Process one reaction. Returns (record_dict, silent_rxn_or_None)."""
    s      = X_s[:, i]
    r      = X_r[:, i]
    mean_s = float(s.mean())
    mean_r = float(r.mean())
    eps    = 1e-9
    rxn_id = rxn_ids[i]

    # Silent: zero flux in ALL patients in BOTH groups
    if s.max() == 0 and r.max() == 0:
        return {
            "rxn_id":              rxn_id,
            "mean_flux_sensitive": 0.0,
            "mean_flux_resistant": 0.0,
            "log2_fold_change":    0.0,
            "pvalue_magnitude":    1.0,
            "pvalue_fisher":       1.0,
            "auc":                 0.5,
            "active_sensitive":    0,
            "active_resistant":    0,
            "test_used":           "none_silent",
        }, rxn_id

    # Magnitude test
    if use_permutation:
        rng      = np.random.default_rng(rng_seed + i)
        obs_diff = abs(mean_s - mean_r)
        combined = X[:, i]
        n_s      = len(s)
        # Vectorized permutation — all at once, no Python loop
        idx_matrix  = np.array([rng.permutation(len(combined)) for _ in range(n_perm)])
        perm_matrix = combined[idx_matrix]
        perm_diffs  = np.abs(
            perm_matrix[:, :n_s].mean(axis=1) - perm_matrix[:, n_s:].mean(axis=1)
        )
        pval_mag  = float((perm_diffs >= obs_diff).sum() + 1) / (n_perm + 1)
        test_used = "permutation"
    else:
        try:
            _, pval_mag = mannwhitneyu(s, r, alternative="two-sided")
        except Exception:
            pval_mag = 1.0
        test_used = "mann_whitney"

    # Fisher's exact test (presence / absence)
    s_present = int((s > PRESENCE_THRESHOLD).sum())
    s_absent  = len(s) - s_present
    r_present = int((r > PRESENCE_THRESHOLD).sum())
    r_absent  = len(r) - r_present

    try:
        _, pval_fisher = fisher_exact(
            [[s_present, s_absent],
             [r_present, r_absent]],
            alternative="two-sided"
        )
    except Exception:
        pval_fisher = 1.0

    # AUC
    try:
        auc = roc_auc_score(y, X[:, i])
        auc = max(auc, 1 - auc)
    except Exception:
        auc = 0.5

    # Fold change
    fc = np.clip((mean_s + eps) / (mean_r + eps), 1e-9, 1e9)

    return {
        "rxn_id":              rxn_id,
        "mean_flux_sensitive": mean_s,
        "mean_flux_resistant": mean_r,
        "log2_fold_change":    float(np.log2(fc)),
        "pvalue_magnitude":    float(pval_mag),
        "pvalue_fisher":       float(pval_fisher),
        "auc":                 float(auc),
        "active_sensitive":    s_present,
        "active_resistant":    r_present,
        "test_used":           test_used,
    }, None


# ══════════════════════════════════════════════════════════════════════════════
# Differential flux — parallel processing + FDR
# ══════════════════════════════════════════════════════════════════════════════

def differential_flux(X, y, rxn_ids, n_jobs=4, fdr_cutoff=FDR_CUTOFF_LARGE):
    n_total  = len(y)
    use_perm = n_total <= LARGE_N_THRESHOLD

    sens_idx = np.where(y == 1)[0]
    res_idx  = np.where(y == 0)[0]
    X_s = X[sens_idx]
    X_r = X[res_idx]

    test_name = "Permutation (vectorized)" if use_perm else "Mann-Whitney U"
    log.info(f"  Magnitude test: {test_name} | FDR cutoff: {fdr_cutoff} | "
             f"n_jobs={n_jobs} | reactions={X.shape[1]}")

    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_reaction)(
            i, X_s, X_r, X, y, rxn_ids,
            use_perm, N_PERMUTATIONS, rng_seed=42
        )
        for i in range(X.shape[1])
    )

    records     = [r[0] for r in results]
    silent_rxns = [r[1] for r in results if r[1] is not None]

    df = pd.DataFrame(records)

    # FDR correction separately for each test
    _, padj_mag,    _, _ = multipletests(df["pvalue_magnitude"], method="fdr_bh")
    _, padj_fisher, _, _ = multipletests(df["pvalue_fisher"],    method="fdr_bh")
    df["pvalue_magnitude_adj"] = padj_mag
    df["pvalue_fisher_adj"]    = padj_fisher

    # Classify using therapy-specific FDR cutoff
    sig_mag    = df["pvalue_magnitude_adj"] < fdr_cutoff
    sig_fisher = df["pvalue_fisher_adj"]    < fdr_cutoff

    def classify(m, f):
        if m and f:     return "both"
        if m and not f: return "flux_only"
        if f and not m: return "presence_only"
        return "none"

    df["biomarker_type"] = [classify(m, f) for m, f in zip(sig_mag, sig_fisher)]
    df["significant"]    = sig_mag | sig_fisher

    df["direction"] = df["log2_fold_change"].apply(
        lambda x: "Higher_in_Sensitive" if x > 0 else "Higher_in_Resistant"
    )

    priority = {"both": 0, "flux_only": 1, "presence_only": 2, "none": 3}
    df["_sort"] = df["biomarker_type"].map(priority)
    df = (df.sort_values(["_sort", "pvalue_magnitude_adj"])
            .drop("_sort", axis=1)
            .reset_index(drop=True))
    df["rank"] = df.index + 1

    df = df[[
        "rank", "rxn_id",
        "mean_flux_sensitive", "mean_flux_resistant",
        "log2_fold_change", "direction",
        "pvalue_magnitude", "pvalue_magnitude_adj",
        "pvalue_fisher",    "pvalue_fisher_adj",
        "auc",
        "active_sensitive", "active_resistant",
        "test_used", "biomarker_type", "significant",
    ]]

    return df, silent_rxns


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_volcano(df, therapy, out_dir, fdr_cutoff):
    fig, ax = plt.subplots(figsize=(10, 7))

    sig_mag = df["pvalue_magnitude_adj"] < fdr_cutoff
    up      = sig_mag & (df["log2_fold_change"] >  0.5)
    down    = sig_mag & (df["log2_fold_change"] < -0.5)
    ns      = ~(up | down)
    y_vals  = -np.log10(df["pvalue_magnitude_adj"].clip(lower=1e-300))

    ax.scatter(df.loc[ns,   "log2_fold_change"], y_vals[ns],   c="grey", s=8,  alpha=0.4, label="NS")
    ax.scatter(df.loc[up,   "log2_fold_change"], y_vals[up],   c="red",  s=12, alpha=0.7, label=f"Sensitive (n={up.sum()})")
    ax.scatter(df.loc[down, "log2_fold_change"], y_vals[down], c="blue", s=12, alpha=0.7, label=f"Resistant (n={down.sum()})")

    ax.axhline(-np.log10(fdr_cutoff), color="black", linestyle="--", lw=0.8, alpha=0.6)
    ax.axvline( 0.5, color="red",  linestyle="--", lw=0.8, alpha=0.5)
    ax.axvline(-0.5, color="blue", linestyle="--", lw=0.8, alpha=0.5)

    for _, row in df[df["significant"]].head(10).iterrows():
        ax.annotate(row["rxn_id"],
                    (row["log2_fold_change"],
                     -np.log10(max(row["pvalue_magnitude_adj"], 1e-300))),
                    fontsize=6, alpha=0.85)

    ax.set_xlabel("log2(Fold Change: Sensitive / Resistant)", fontsize=12)
    ax.set_ylabel("-log10(Adjusted p-value, magnitude test)", fontsize=12)
    ax.set_title(f"Differential Flux — {therapy} (FDR<{fdr_cutoff})", fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "volcano_plot.png"), dpi=150)
    plt.close()


def plot_top_biomarkers(df, therapy, out_dir, fdr_cutoff, n=30):
    sig = df[df["significant"]].head(n)
    if sig.empty:
        log.warning(f"[{therapy}] No significant reactions to plot.")
        return

    color_map = {"both": "purple", "flux_only": "steelblue", "presence_only": "darkorange"}
    colors = [color_map.get(t, "grey") for t in sig["biomarker_type"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sig) * 0.35)))
    ax.barh(range(len(sig)), sig["auc"], color=colors, alpha=0.80)
    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(sig["rxn_id"], fontsize=8)
    ax.set_xlabel("AUC (reaction discriminability)", fontsize=12)
    ax.set_title(f"Significant Biomarkers — {therapy} (FDR<{fdr_cutoff})", fontsize=13)
    ax.axvline(0.5, color="black", linestyle="--", lw=0.8)
    ax.invert_yaxis()

    patches = [
        mpatches.Patch(color="purple",     alpha=0.8, label="Both tests"),
        mpatches.Patch(color="steelblue",  alpha=0.8, label="Flux magnitude only"),
        mpatches.Patch(color="darkorange", alpha=0.8, label="Presence/absence only"),
    ]
    ax.legend(handles=patches, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_biomarkers.png"), dpi=150)
    plt.close()


def plot_biomarker_type_breakdown(df, therapy, out_dir):
    counts = df[df["significant"]]["biomarker_type"].value_counts()
    if counts.empty:
        return
    color_map = {"both": "purple", "flux_only": "steelblue", "presence_only": "darkorange"}
    colors = [color_map.get(k, "grey") for k in counts.index]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index, colors=colors,
           autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Biomarker Type Breakdown — {therapy}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "biomarker_type_breakdown.png"), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Per-Therapy Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_therapy(therapy, fluxes, uuids, rxn_ids, manifest, out_base, n_jobs_rxn=4):
    log.info(f"\n{'='*60}\nTHERAPY: {therapy}\n{'='*60}")

    therapy_dir = os.path.join(out_base, therapy.replace(" ", "_"))
    os.makedirs(therapy_dir, exist_ok=True)

    X, y, uuids_sub = get_therapy_subset(fluxes, uuids, manifest, therapy)
    if X is None:
        log.warning(f"[{therapy}] No matching patients in flux matrix.")
        return

    n_s = int(y.sum())
    n_r = int((y == 0).sum())
    log.info(f"[{therapy}] Patients: {len(y)} (Sensitive={n_s}, Resistant={n_r})")

    if n_s < MIN_PER_CLASS or n_r < MIN_PER_CLASS:
        log.warning(f"[{therapy}] Too few samples (min {MIN_PER_CLASS}/class). Skipping.")
        return

    # Adaptive FDR cutoff
    fdr_cutoff = FDR_CUTOFF_SMALL if therapy in SMALL_THERAPIES else FDR_CUTOFF_LARGE
    regime     = "large (Mann-Whitney)" if len(y) > LARGE_N_THRESHOLD else "small (Permutation)"
    log.info(f"[{therapy}] Test regime: {regime} | FDR cutoff: {fdr_cutoff}")

    diff_df, silent_rxns = differential_flux(
        X, y, rxn_ids, n_jobs=n_jobs_rxn, fdr_cutoff=fdr_cutoff
    )

    n_silent       = len(silent_rxns)
    n_sig_total    = int(diff_df["significant"].sum())
    n_sig_both     = int((diff_df["biomarker_type"] == "both").sum())
    n_sig_flux     = int((diff_df["biomarker_type"] == "flux_only").sum())
    n_sig_presence = int((diff_df["biomarker_type"] == "presence_only").sum())

    log.info(f"[{therapy}] Significant: {n_sig_total} total | "
             f"both={n_sig_both} flux_only={n_sig_flux} presence_only={n_sig_presence}")

    diff_df.to_csv(os.path.join(therapy_dir, "differential_flux.csv"), index=False)

    sig_df = diff_df[diff_df["significant"]]
    sig_df.to_csv(os.path.join(therapy_dir, "top_biomarkers.csv"), index=False)

    with open(os.path.join(therapy_dir, "silent_reactions.txt"), "w") as f:
        f.write(f"# Silent reactions in {therapy}\n")
        f.write(f"# Total: {n_silent}\n")
        f.write(f"# Zero flux in ALL patients (both Sensitive and Resistant)\n\n")
        for rxn in silent_rxns:
            f.write(f"{rxn}\n")

    plot_volcano(diff_df, therapy, therapy_dir, fdr_cutoff)
    plot_top_biomarkers(diff_df, therapy, therapy_dir, fdr_cutoff)
    plot_biomarker_type_breakdown(diff_df, therapy, therapy_dir)

    lines = [
        f"{'='*60}",
        f"Therapy          : {therapy}",
        f"Patients         : {len(y)} (Sensitive={n_s}, Resistant={n_r})",
        f"Test regime      : {regime}",
        f"FDR cutoff       : {fdr_cutoff}",
        f"Reactions tested : {len(diff_df)}",
        f"Silent reactions : {n_silent}",
        f"Active reactions : {len(diff_df) - n_silent}",
        f"",
        f"Significant (FDR < {fdr_cutoff}):",
        f"  Total          : {n_sig_total}",
        f"  Both tests     : {n_sig_both}  (strongest biomarkers)",
        f"  Flux only      : {n_sig_flux}  (magnitude differs)",
        f"  Presence only  : {n_sig_presence} (ON in one group, OFF in other)",
        f"",
        f"Top 10 significant biomarkers:",
        f"{'Rank':<5} {'Reaction':<28} {'Type':<15} {'log2FC':>7} {'FDR_mag':>10} {'FDR_fis':>10} {'AUC':>6}",
        f"{'-'*85}",
    ]
    for _, row in sig_df.head(10).iterrows():
        lines.append(
            f"{int(row['rank']):<5} {row['rxn_id']:<28} "
            f"{row['biomarker_type']:<15} "
            f"{row['log2_fold_change']:>7.3f} "
            f"{row['pvalue_magnitude_adj']:>10.2e} "
            f"{row['pvalue_fisher_adj']:>10.2e} "
            f"{row['auc']:>6.3f}"
        )
    lines.append(f"{'='*60}")

    summary = "\n".join(lines)
    print(f"\n{summary}\n")
    with open(os.path.join(therapy_dir, "summary.txt"), "w") as f:
        f.write(summary)

    log.info(f"[{therapy}] Done -> {therapy_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--therapy",  default="all",
                        help="Therapy group or 'all' or 'small' (Hormone/Surgery/Immuno/Targeted)")
    parser.add_argument("--h5",       default=H5_PATH)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--out-dir",  default=OUT_DIR)
    parser.add_argument("--n-jobs",   type=int, default=60)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fluxes, uuids, rxn_ids = load_flux_matrix(args.h5)
    manifest = load_manifest(args.manifest)

    # --therapy small → run only the 4 small cohorts
    if args.therapy == "small":
        therapies = list(SMALL_THERAPIES)
    elif args.therapy == "all":
        therapies = THERAPY_GROUPS
    else:
        therapies = [args.therapy]

    if len(therapies) > 1:
        n_jobs_per_therapy = max(1, args.n_jobs // len(therapies))
        log.info(f"Running {len(therapies)} therapies in parallel | "
                 f"{n_jobs_per_therapy} cores/therapy")
        Parallel(n_jobs=len(therapies), backend="loky")(
            delayed(run_therapy)(
                therapy, fluxes, uuids, rxn_ids, manifest, args.out_dir,
                n_jobs_rxn=n_jobs_per_therapy
            )
            for therapy in therapies
        )
    else:
        log.info(f"Single therapy: {args.n_jobs} cores for reactions")
        run_therapy(
            therapies[0], fluxes, uuids, rxn_ids, manifest, args.out_dir,
            n_jobs_rxn=args.n_jobs
        )

    log.info("\nAll done.")


if __name__ == "__main__":
    main()