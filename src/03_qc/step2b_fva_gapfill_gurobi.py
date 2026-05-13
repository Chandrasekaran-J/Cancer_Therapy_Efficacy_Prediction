"""
step2b_fva_gapfill_gurobi.py
Single SGE array worker that runs:
    Phase 1 (--phase fva)     : Loopless FVA → count blocked reactions
    Phase 2 (--phase gapfill) : MILP gap filling → save gap-filled PKL models
    Phase 3 (--phase report)  : Merge all results → before/after QC reports
                                (run manually ONCE after all array tasks finish)

Usage:
    # Phase 1 — submit array job:
    qsub submit_fva_gapfill.sh --phase fva

    # After Phase 1 finishes, Phase 2 — submit array job:
    qsub submit_fva_gapfill.sh --phase gapfill

    # After Phase 2 finishes, Phase 3 — run manually ONCE:
    python step2b_fva_gapfill_gurobi.py --phase report
"""

import os, sys
os.environ["OPTLANG_DEFAULT_INTERFACE"] = "optlang.gurobi_interface"

import argparse
import copy
import glob
import logging
import pickle
import warnings

import numpy as np
import pandas as pd
import cobra
from cobra.flux_analysis import flux_variability_analysis, gapfill

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = "/data/chandrasekaran/algo/Biomarkers"
PKL_DIR         = "/data/chandrasekaran/algo/GIMME_output"
GAPFILL_PKL_DIR = os.path.join(BASE_DIR, "gapfill_models")
FVA_DIR         = os.path.join(BASE_DIR, "fva_results")
GF_DIR          = os.path.join(BASE_DIR, "gapfill_results")
QC_CSV          = os.path.join(BASE_DIR, "qc_full_report.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(phase)s|TASK%(task)s] %(levelname)s %(message)s",
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_task_id() -> int:
    return int(os.environ.get("SGE_TASK_ID", 1))


def make_log(task_id: int, phase: str):
    return logging.LoggerAdapter(logging.getLogger(), {"task": task_id, "phase": phase.upper()})


def load_manifest(path: str) -> dict:
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, dtype=str)
    elif path.endswith((".tsv", ".txt")):
        df = pd.read_csv(path, sep="\t", dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    if "file_uuid" not in df.columns:
        raise ValueError(f"Manifest missing 'file_uuid'. Got: {df.columns.tolist()}")
    if "sensitivity" not in df.columns:
        raise ValueError(f"Manifest missing 'sensitivity'. Got: {df.columns.tolist()}")
    return dict(zip(df["file_uuid"].str.strip(), df["sensitivity"].str.strip()))


def get_chunk(pkl_dir: str, task_id: int, chunk_size: int):
    all_pkls = sorted([f for f in os.listdir(pkl_dir) if f.endswith(".pkl")])
    start = (task_id - 1) * chunk_size
    end   = min(task_id * chunk_size, len(all_pkls))
    return all_pkls[start:end], len(all_pkls)


def count_dead_ends(model: cobra.Model) -> int:
    return sum(1 for m in model.metabolites if len(m.reactions) == 1)


def find_recon3d(hint: str):
    """Try multiple locations for Recon3D universal model."""
    candidates = [
        hint,
        "/data/chandrasekaran/algo/Recon3D_universal.pkl",
        "/data/chandrasekaran/algo/Recon3D.pkl",
        "/data/chandrasekaran/algo/Recon3D.xml",
        "/data/chandrasekaran/algo/Recon3D/Recon3D.xml",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Cannot find Recon3D model. Tried: {candidates}\n"
        "Set --recon3d to the correct path."
    )


def load_universal(path: str, log) -> cobra.Model:
    pkl_cache = path if path.endswith(".pkl") else path + ".cache.pkl"
    if path.endswith(".pkl") and os.path.exists(path):
        log.info(f"Loading Recon3D pkl from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    log.info(f"Loading Recon3D XML from {path} (may take a minute...)")
    model = cobra.io.read_sbml_model(path)
    log.info(f"Caching Recon3D to {pkl_cache}")
    with open(pkl_cache, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Loopless FVA
# ══════════════════════════════════════════════════════════════════════════════

def phase_fva(args):
    task_id = get_task_id()
    log = make_log(task_id, "FVA")
    os.makedirs(FVA_DIR, exist_ok=True)

    out_path = os.path.join(FVA_DIR, f"fva_chunk_{task_id:04d}.npz")
    if os.path.exists(out_path):
        log.info(f"Already done → {out_path}. Skipping.")
        return

    # Load pre-computed FBA objectives from QC CSV
    log.info(f"Loading FBA objectives from {QC_CSV}")
    qc_df = pd.read_csv(QC_CSV, dtype=str)
    uuid_to_obj = dict(zip(
        qc_df["uuid"].str.strip(),
        pd.to_numeric(qc_df["fba_objective"], errors="coerce")
    ))

    chunk, n_total = get_chunk(args.pkl_dir, task_id, args.chunk_size)
    log.info(f"Chunk: {len(chunk)} files (total PKLs={n_total})")

    uuids_out      = []
    n_blocked_out  = []
    n_variable_out = []

    for idx, pkl_fname in enumerate(chunk):
        uuid     = pkl_fname.split(".")[0]
        pkl_path = os.path.join(args.pkl_dir, pkl_fname)
        uuids_out.append(uuid)

        fba_obj = uuid_to_obj.get(uuid)
        if not fba_obj or np.isnan(fba_obj) or fba_obj <= 0:
            log.warning(f"  [{idx+1}] {uuid[:8]} no valid FBA obj, skipping")
            n_blocked_out.append(-1)
            n_variable_out.append(-1)
            continue

        try:
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)

            log.info(f"  [{idx+1}/{len(chunk)}] {uuid[:8]} — loopless FVA (fraction={args.fraction_optimum})")
            fva = flux_variability_analysis(
                model,
                reaction_list=model.reactions,
                fraction_of_optimum=args.fraction_optimum,
                loopless=True,
                processes=1,
            )

            tol = 1e-9
            blocked   = ((fva["minimum"].abs() < tol) & (fva["maximum"].abs() < tol)).sum()
            variable  = ((fva["maximum"] - fva["minimum"]).abs() > tol).sum()

            n_blocked_out.append(int(blocked))
            n_variable_out.append(int(variable))
            log.info(f"  [{idx+1}/{len(chunk)}] {uuid[:8]} blocked={blocked} variable={variable}")

        except Exception as e:
            log.error(f"  FAILED {uuid}: {e}")
            n_blocked_out.append(-1)
            n_variable_out.append(-1)

    np.savez_compressed(
        out_path,
        uuids          = np.array(uuids_out,      dtype=object),
        n_blocked_rxns = np.array(n_blocked_out,  dtype=np.int32),
        n_variable_rxns= np.array(n_variable_out, dtype=np.int32),
    )
    log.info(f"Saved FVA chunk → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Gap Filling
# ══════════════════════════════════════════════════════════════════════════════

def phase_gapfill(args):
    task_id = get_task_id()
    log = make_log(task_id, "GAPFILL")
    os.makedirs(GF_DIR, exist_ok=True)
    os.makedirs(GAPFILL_PKL_DIR, exist_ok=True)

    out_path = os.path.join(GF_DIR, f"gapfill_chunk_{task_id:04d}.npz")
    if os.path.exists(out_path):
        log.info(f"Already done → {out_path}. Skipping.")
        return

    # Load Recon3D universal model
    recon3d_path = find_recon3d(args.recon3d)
    universal = load_universal(recon3d_path, log)
    log.info(f"Universal model: {len(universal.reactions)} reactions")

    # Load pre-computed FBA objectives — NO re-running FBA
    log.info(f"Loading pre-computed FBA objectives from {QC_CSV}")
    qc_df = pd.read_csv(QC_CSV, dtype=str)
    uuid_to_obj = dict(zip(
        qc_df["uuid"].str.strip(),
        pd.to_numeric(qc_df["fba_objective"], errors="coerce")
    ))

    chunk, n_total = get_chunk(args.pkl_dir, task_id, args.chunk_size)
    log.info(f"Chunk: {len(chunk)} files (total PKLs={n_total})")

    uuids_out      = []
    gf_status_out  = []
    n_added_out    = []
    obj_before_out = []
    obj_after_out  = []

    for idx, pkl_fname in enumerate(chunk):
        uuid     = pkl_fname.split(".")[0]
        pkl_path = os.path.join(args.pkl_dir, pkl_fname)
        uuids_out.append(uuid)

        gf_pkl = os.path.join(GAPFILL_PKL_DIR, pkl_fname)

        # Use pre-computed FBA objective — no re-solving
        obj_before = float(uuid_to_obj.get(uuid) or 0.0)
        obj_before_out.append(obj_before)

        try:
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)

            log.info(f"  [{idx+1}/{len(chunk)}] {uuid[:8]} obj_before={obj_before:.2f} — gap filling...")

            # MILP gap fill (Shlomi et al. 2010, state-of-the-art)
            solutions = gapfill(
                model,
                universal,
                demand_reactions=False,
                exchange_reactions=False,
                iterations=1,
            )

            if solutions and solutions[0]:
                rxns_to_add = solutions[0]
                gf_model = copy.deepcopy(model)
                for rxn in rxns_to_add:
                    gf_model.add_reactions([rxn.copy()])
                n_added = len(rxns_to_add)
                status  = "filled"
            else:
                gf_model = copy.deepcopy(model)
                n_added  = 0
                status   = "no_gaps"

            # FBA after
            sol_after = gf_model.optimize()
            obj_after = float(sol_after.objective_value) if sol_after.status == "optimal" else 0.0

            gf_status_out.append(status)
            n_added_out.append(n_added)
            obj_after_out.append(obj_after)

            # Save gap-filled model
            with open(gf_pkl, "wb") as f:
                pickle.dump(gf_model, f, protocol=pickle.HIGHEST_PROTOCOL)

            log.info(
                f"  [{idx+1}/{len(chunk)}] {uuid[:8]} "
                f"status={status} added={n_added} obj_after={obj_after:.2f}"
            )

        except Exception as e:
            log.error(f"  FAILED {uuid}: {e}")
            gf_status_out.append("error")
            n_added_out.append(-1)
            obj_after_out.append(0.0)
            # Copy original as fallback
            try:
                import shutil
                shutil.copy2(pkl_path, gf_pkl)
            except:
                pass

    np.savez_compressed(
        out_path,
        uuids          = np.array(uuids_out,      dtype=object),
        gapfill_status = np.array(gf_status_out,  dtype=object),
        n_rxns_added   = np.array(n_added_out,    dtype=np.int32),
        fba_obj_before = np.array(obj_before_out, dtype=np.float32),
        fba_obj_after  = np.array(obj_after_out,  dtype=np.float32),
    )
    log.info(f"Saved gapfill chunk → {out_path}")
    n_filled = sum(1 for s in gf_status_out if s == "filled")
    log.info(f"Chunk summary: filled={n_filled} no_gaps={gf_status_out.count('no_gaps')} errors={gf_status_out.count('error')}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Report (run ONCE manually after all array tasks finish)
# ══════════════════════════════════════════════════════════════════════════════

def phase_report(args):
    log = make_log(0, "REPORT")

    # ── Load original QC report ───────────────────────────────────────────────
    log.info(f"Loading original QC report: {QC_CSV}")
    qc_df = pd.read_csv(QC_CSV, dtype=str)

    # ── Merge FVA results ─────────────────────────────────────────────────────
    fva_files = sorted(glob.glob(os.path.join(FVA_DIR, "fva_chunk_*.npz")))
    log.info(f"Found {len(fva_files)} FVA chunk files")

    uuid_to_blocked  = {}
    uuid_to_variable = {}
    for cf in fva_files:
        data = np.load(cf, allow_pickle=True)
        for u, nb, nv in zip(data["uuids"], data["n_blocked_rxns"], data["n_variable_rxns"]):
            uuid_to_blocked[str(u)]  = int(nb)
            uuid_to_variable[str(u)] = int(nv)

    # ── Merge gapfill results ─────────────────────────────────────────────────
    gf_files = sorted(glob.glob(os.path.join(GF_DIR, "gapfill_chunk_*.npz")))
    log.info(f"Found {len(gf_files)} gapfill chunk files")

    uuid_to_gf = {}
    for cf in gf_files:
        data = np.load(cf, allow_pickle=True)
        for u, gs, na, ob, oa in zip(
            data["uuids"], data["gapfill_status"],
            data["n_rxns_added"], data["fba_obj_before"], data["fba_obj_after"]
        ):
            uuid_to_gf[str(u)] = {
                "gapfill_status": str(gs),
                "n_rxns_added":   int(na),
                "fba_obj_before": float(ob),
                "fba_obj_after":  float(oa),
            }

    # ── Compute post-gapfill model metrics from PKL ───────────────────────────
    log.info("Computing post-gapfill model metrics from PKL files...")
    uuid_to_post_qc = {}
    for pkl_fname in sorted(os.listdir(GAPFILL_PKL_DIR)):
        if not pkl_fname.endswith(".pkl"):
            continue
        uuid = pkl_fname.split(".")[0]
        pkl_path = os.path.join(GAPFILL_PKL_DIR, pkl_fname)
        try:
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            sol = model.optimize()
            uuid_to_post_qc[uuid] = {
                "n_reactions_gf":    len(model.reactions),
                "n_genes_gf":        len(model.genes),
                "n_metabolites_gf":  len(model.metabolites),
                "n_dead_end_mets_gf": count_dead_ends(model),
                "fba_status_gf":     sol.status,
                "fba_objective_gf":  float(sol.objective_value) if sol.status == "optimal" else 0.0,
            }
        except Exception as e:
            log.error(f"Failed to load {uuid[:8]}: {e}")
            uuid_to_post_qc[uuid] = {
                "n_reactions_gf": None, "n_genes_gf": None,
                "n_metabolites_gf": None, "n_dead_end_mets_gf": None,
                "fba_status_gf": "error", "fba_objective_gf": 0.0,
            }

    # ── Build output dataframe ────────────────────────────────────────────────
    out = qc_df.copy()

    # Add FVA columns
    out["n_blocked_rxns"]  = out["uuid"].map(uuid_to_blocked)
    out["n_variable_rxns"] = out["uuid"].map(uuid_to_variable)

    # Add gapfill columns
    for col in ["gapfill_status", "n_rxns_added", "fba_obj_before", "fba_obj_after"]:
        out[col] = out["uuid"].map(lambda u, c=col: uuid_to_gf.get(u, {}).get(c))

    # Add post-gapfill QC columns
    for col in ["n_reactions_gf", "n_genes_gf", "n_metabolites_gf",
                "n_dead_end_mets_gf", "fba_status_gf", "fba_objective_gf"]:
        out[col] = out["uuid"].map(lambda u, c=col: uuid_to_post_qc.get(u, {}).get(c))

    # Write reports
    out_csv = os.path.join(BASE_DIR, "qc_full_report_gapfill.csv")
    out.to_csv(out_csv, index=False)
    log.info(f"✓ Report written: {out_csv}")

    # ── Passing UUIDs (optimal after gapfill) ────────────────────────────────
    passing = out[out["fba_status_gf"] == "optimal"]
    passing_path = os.path.join(BASE_DIR, "qc_passing_uuids_gapfill.txt")
    with open(passing_path, "w") as f:
        f.write("\n".join(passing["uuid"].tolist()) + "\n")
    log.info(f"✓ Passing UUIDs: {passing_path} ({len(passing)} models)")

    # ── Summary stats ─────────────────────────────────────────────────────────
    N = len(out)

    def safe_num(series):
        return pd.to_numeric(series, errors="coerce")

    blocked         = safe_num(out["n_blocked_rxns"])
    dead_before     = safe_num(out["n_dead_end_mets"])
    dead_after      = safe_num(out["n_dead_end_mets_gf"])
    rxns_added      = safe_num(out["n_rxns_added"])
    obj_before      = safe_num(out["fba_obj_before"])
    obj_after       = safe_num(out["fba_obj_after"])
    n_orig_optimal  = (out["fba_status"] == "optimal").sum()
    n_gf_optimal    = (out["fba_status_gf"] == "optimal").sum()
    n_filled        = (out["gapfill_status"] == "filled").sum()

    summary = f"""
{'='*60}
GSMM Quality Control — Post Gap Filling Summary
{'='*60}
Total models               : {N}

FBA Status:
  Optimal BEFORE gapfill   : {n_orig_optimal} / {N} ({100*n_orig_optimal/N:.1f}%)
  Optimal AFTER  gapfill   : {n_gf_optimal}  / {N} ({100*n_gf_optimal/N:.1f}%)

Loopless FVA (blocked reactions):
  Mean per model            : {blocked.mean():.1f}
  Std                       : {blocked.std():.1f}
  Min / Max                 : {blocked.min():.0f} / {blocked.max():.0f}
  Models with 0 blocked     : {(blocked == 0).sum()}
  Models with >100 blocked  : {(blocked > 100).sum()}
  Models with >500 blocked  : {(blocked > 500).sum()}

Gap Filling:
  Models with gaps filled   : {n_filled}
  Models with no gaps       : {(out['gapfill_status'] == 'no_gaps').sum()}
  Models with errors        : {(out['gapfill_status'] == 'error').sum()}
  Reactions added (mean)    : {rxns_added[rxns_added >= 0].mean():.1f}
  Reactions added (max)     : {rxns_added.max():.0f}

Dead-end metabolites:
  Before gapfill (mean)     : {dead_before.mean():.1f}
  After  gapfill (mean)     : {dead_after.mean():.1f}
  Reduction (mean)          : {(dead_before - dead_after).mean():.1f}

FBA Objective (biomass):
  Before gapfill (mean±std) : {obj_before.mean():.4f} ± {obj_before.std():.4f}
  After  gapfill (mean±std) : {obj_after.mean():.4f} ± {obj_after.std():.4f}

Models passing QC           : {len(passing)} / {N}
{'='*60}
"""
    print(summary)
    summary_path = os.path.join(BASE_DIR, "qc_summary_gapfill.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    log.info(f"✓ Summary written: {summary_path}")

    # ── Before/After comparison ───────────────────────────────────────────────
    compare = f"""
{'='*60}
BEFORE vs AFTER Gap Filling Comparison
{'='*60}
{'Metric':<35} {'Before':>12} {'After':>12}
{'-'*60}
{'FBA Optimal (%)':<35} {100*n_orig_optimal/N:>11.1f}% {100*n_gf_optimal/N:>11.1f}%
{'Dead-end metabolites (mean)':<35} {dead_before.mean():>12.1f} {dead_after.mean():>12.1f}
{'FBA Objective mean':<35} {obj_before.mean():>12.4f} {obj_after.mean():>12.4f}
{'Models passing QC':<35} {'0':>12} {len(passing):>12}
{'='*60}
"""
    print(compare)
    compare_path = os.path.join(BASE_DIR, "qc_comparison.txt")
    with open(compare_path, "w") as f:
        f.write(compare)
    log.info(f"✓ Comparison written: {compare_path}")
    log.info("Report generation complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FVA + Gap Fill pipeline")
    parser.add_argument("--phase",            required=True, choices=["fva", "gapfill", "report"])
    parser.add_argument("--manifest",         default="/data/chandrasekaran/algo/transcriptomics_manifest.tsv")
    parser.add_argument("--chunk-size",       type=int,   default=200)
    parser.add_argument("--pkl-dir",          default=PKL_DIR)
    parser.add_argument("--fraction-optimum", type=float, default=0.95)
    parser.add_argument("--recon3d",          default="/data/chandrasekaran/algo/Recon3D.xml", help="Path to Recon3D universal model (.pkl or .xml)")
    args = parser.parse_args()

    if args.phase == "fva":
        phase_fva(args)
    elif args.phase == "gapfill":
        phase_gapfill(args)
    elif args.phase == "report":
        phase_report(args)


if __name__ == "__main__":
    main()
