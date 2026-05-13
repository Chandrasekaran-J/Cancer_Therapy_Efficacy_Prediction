"""
step2_merge_report.py
Run after all step2_qc_chunk.py tasks finish.

Combines all qc_chunk_*.csv files into:
    qc_full_report.csv        — per-patient QC table
    qc_summary_stats.txt      — human-readable summary

Usage:
    python step2_merge_report.py [--qc-dir qc_results]
"""

import os
import argparse
import glob
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = "/data/chandrasekaran/algo/Biomarkers"
QC_DIR   = os.path.join(BASE_DIR, "qc_results")
OUT_CSV  = os.path.join(BASE_DIR, "qc_full_report.csv")
OUT_TXT  = os.path.join(BASE_DIR, "qc_summary_stats.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc-dir",     default=QC_DIR)
    parser.add_argument("--out-csv",    default=OUT_CSV)
    parser.add_argument("--out-txt",    default=OUT_TXT)
    args = parser.parse_args()

    chunk_files = sorted(glob.glob(os.path.join(args.qc_dir, "qc_chunk_*.csv")))
    log.info(f"Found {len(chunk_files)} QC chunk files")
    if not chunk_files:
        raise FileNotFoundError(f"No qc_chunk_*.csv in {args.qc_dir}")

    df = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
    df.to_csv(args.out_csv, index=False)
    log.info(f"Full QC report: {len(df)} patients → {args.out_csv}")

    # ── Summary stats ───────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("GSMM Quality Control Summary")
    lines.append("=" * 60)
    lines.append(f"Total models assessed  : {len(df)}")

    for label in ["Sensitive", "Resistant", "Unknown"]:
        sub = df[df["label"] == label]
        lines.append(f"\n  Label = {label}  ({len(sub)} patients)")

        if len(sub) == 0:
            continue

        # FBA status
        status_counts = sub["fba_status"].value_counts()
        lines.append(f"    FBA optimal        : {status_counts.get('optimal', 0)}")
        lines.append(f"    FBA infeasible     : {status_counts.get('infeasible', 0)}")
        n_err = sum(v for k, v in status_counts.items() if k not in ("optimal","infeasible"))
        lines.append(f"    FBA error/other    : {n_err}")

        # Objective value
        opt = sub[sub["fba_status"] == "optimal"]
        if len(opt):
            lines.append(f"    Objective (opt)    : mean={opt['fba_objective'].mean():.4f}  "
                         f"std={opt['fba_objective'].std():.4f}  "
                         f"min={opt['fba_objective'].min():.4f}")

        # Model size
        lines.append(f"    Reactions/model    : mean={sub['n_reactions'].mean():.0f}  "
                     f"std={sub['n_reactions'].std():.0f}")
        lines.append(f"    Genes/model        : mean={sub['n_genes'].mean():.0f}  "
                     f"std={sub['n_genes'].std():.0f}")
        lines.append(f"    Dead-end mets      : mean={sub['n_dead_end_mets'].mean():.1f}")

        # Gap filling
        n_gap = sub["requires_gap_filling"].sum()
        lines.append(f"    Flagged for gaps   : {n_gap} ({100*n_gap/len(sub):.1f}%)")

    # ── Models that PASS QC (for downstream analysis) ─────────────────────
    pass_mask = (
        (df["fba_status"] == "optimal") &
        (df["fba_objective"] > 1e-4) &
        (df["label"].isin(["Sensitive", "Resistant"]))
    )
    df_pass = df[pass_mask]
    lines.append("\n" + "-" * 60)
    lines.append(f"Models passing QC for biomarker analysis: {len(df_pass)}")
    lines.append(f"  Sensitive : {(df_pass['label']=='Sensitive').sum()}")
    lines.append(f"  Resistant : {(df_pass['label']=='Resistant').sum()}")

    # ── Save list of passing UUIDs ─────────────────────────────────────────
    pass_uuid_path = os.path.join(os.path.dirname(args.out_csv), "qc_passing_uuids.txt")
    df_pass["uuid"].to_csv(pass_uuid_path, index=False, header=False)
    lines.append(f"\nPassing UUIDs saved to: {pass_uuid_path}")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    with open(args.out_txt, "w") as f:
        f.write(summary_text)

    print(summary_text)
    log.info(f"Summary stats → {args.out_txt}")


if __name__ == "__main__":
    main()
