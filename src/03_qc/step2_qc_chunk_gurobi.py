"""
step2_qc_chunk.py
SGE array worker — quality-control checks on a chunk of GIMME patient models.

Per-model checks:
  1. FBA feasibility + objective value (already done in step1, here for cross-check)
  2. Blocked reactions  — reactions unable to carry flux in any condition
     (lightweight: checks FBA flux only; full FVA optional with --full-fva)
  3. Dead-end metabolites — only produced OR only consumed within the model
  4. Zero-flux reactions in FBA solution (proxy for potentially blocked)
  5. Model completeness stats: n_rxns, n_genes, n_metabolites

Output per task:
    qc_results/qc_chunk_{task_id:04d}.csv

After all tasks: python step2_merge_report.py

Usage:
    python step2_qc_chunk.py --manifest <path> [--chunk-size 200] [--full-fva]
"""

import os, sys
os.environ["OPTLANG_DEFAULT_INTERFACE"] = "optlang.gurobi_interface"

import argparse
import logging
import pickle
import warnings

import numpy as np
import pandas as pd
import cobra
from cobra.flux_analysis import find_blocked_reactions

warnings.filterwarnings("ignore")

BASE_DIR  = "/data/chandrasekaran/algo/Biomarkers"
PKL_DIR   = "/data/chandrasekaran/algo/GIMME_output"
QC_DIR    = "/data/chandrasekaran/algo/Biomarkers/qc_results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TASK%(task)s] %(levelname)s %(message)s",
)


def get_task_id() -> int:
    return int(os.environ.get("SGE_TASK_ID", 1))


def load_manifest(path: str) -> dict:
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path, dtype=str)
    elif path.endswith(".tsv") or path.endswith(".txt"):
        df = pd.read_csv(path, sep="\t", dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    return dict(zip(df["file_uuid"].str.strip(), df["sensitivity"].str.strip()))


def find_dead_end_metabolites(model: cobra.Model) -> list:
    """
    Dead-end metabolites: only appear as reactant OR only as product
    across all reactions in the model.  These indicate gaps / missing reactions.
    """
    produced_by = {m.id: [] for m in model.metabolites}
    consumed_by = {m.id: [] for m in model.metabolites}
    for rxn in model.reactions:
        for met, stoich in rxn.metabolites.items():
            if stoich < 0:
                consumed_by[met.id].append(rxn.id)
            elif stoich > 0:
                produced_by[met.id].append(rxn.id)

    dead_ends = []
    for met in model.metabolites:
        is_only_produced = len(produced_by[met.id]) > 0 and len(consumed_by[met.id]) == 0
        is_only_consumed = len(consumed_by[met.id]) > 0 and len(produced_by[met.id]) == 0
        if is_only_produced or is_only_consumed:
            dead_ends.append({
                "metabolite_id"   : met.id,
                "metabolite_name" : met.name,
                "compartment"     : met.compartment,
                "only_produced"   : is_only_produced,
                "only_consumed"   : is_only_consumed,
                "producing_rxns"  : ";".join(produced_by[met.id]),
                "consuming_rxns"  : ";".join(consumed_by[met.id]),
            })
    return dead_ends


def qc_model(model: cobra.Model, full_fva: bool = False) -> dict:
    """
    Run all QC checks on a single COBRApy model.
    Returns a dict of per-model QC metrics.
    """
    record = {
        "n_reactions"          : len(model.reactions),
        "n_genes"              : len(model.genes),
        "n_metabolites"        : len(model.metabolites),
        "fba_status"           : "not_run",
        "fba_objective"        : np.nan,
        "n_zero_flux_rxns"     : np.nan,   # |flux| < 1e-6 in FBA
        "n_blocked_rxns"       : np.nan,   # blocked (full FVA) if --full-fva
        "n_dead_end_mets"      : np.nan,
        "n_structural_rxns"    : sum(1 for r in model.reactions if not r.gene_reaction_rule.strip()),
        "requires_gap_filling" : False,
        "notes"                : "",
    }

    # ── FBA ──────────────────────────────────────────────────────────────────
    try:
        sol = model.optimize()
        record["fba_status"]    = sol.status
        record["fba_objective"] = float(sol.objective_value) if sol.status == "optimal" else 0.0

        if sol.status == "optimal":
            fluxes = sol.fluxes
            n_zero = int((fluxes.abs() < 1e-6).sum())
            record["n_zero_flux_rxns"] = n_zero
        else:
            record["requires_gap_filling"] = True
            record["notes"] += "infeasible_fba;"
    except Exception as e:
        record["fba_status"] = f"error:{e}"
        record["notes"]     += f"fba_exception;"

    # ── Blocked reactions (full FVA — slow, optional) ─────────────────────
    if full_fva:
        try:
            blocked = find_blocked_reactions(model, open_exchanges=True)
            # Exclude structural reactions (exchanges/demands) — they're expected to be 0
            structural_ids = {r.id for r in model.reactions if not r.gene_reaction_rule.strip()}
            non_structural_blocked = [r for r in blocked if r not in structural_ids]
            record["n_blocked_rxns"] = len(non_structural_blocked)
            if len(non_structural_blocked) > 0:
                record["requires_gap_filling"] = True
                record["notes"] += f"blocked_rxns:{len(non_structural_blocked)};"
        except Exception as e:
            record["notes"] += f"fva_exception:{e};"

    # ── Dead-end metabolites ───────────────────────────────────────────────
    try:
        dead_ends = find_dead_end_metabolites(model)
        # Only internal metabolites (not exchange compartment) are meaningful
        internal_dead_ends = [
            d for d in dead_ends
            if not (d["metabolite_id"].endswith("[e]") or d["metabolite_id"].endswith("_e"))
        ]
        record["n_dead_end_mets"] = len(internal_dead_ends)
        if len(internal_dead_ends) > 10:   # threshold — arbitrary, tune as needed
            record["requires_gap_filling"] = True
            record["notes"] += f"dead_ends:{len(internal_dead_ends)};"
    except Exception as e:
        record["notes"] += f"dead_end_exception:{e};"

    # ── Objective too low (model probably under-constrained) ─────────────
    if record["fba_status"] == "optimal" and record["fba_objective"] < 1e-4:
        record["requires_gap_filling"] = True
        record["notes"] += "near_zero_objective;"

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",   required=True)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--pkl-dir",    default=PKL_DIR)
    parser.add_argument("--out-dir",    default=QC_DIR)
    parser.add_argument("--full-fva",   action="store_true",
                        help="Run full FVA to find blocked reactions (slow!). "
                             "Adds ~10x runtime. Recommended on a small subset first.")
    args = parser.parse_args()

    task_id = get_task_id()
    log = logging.LoggerAdapter(logging.getLogger(), {"task": task_id})

    if args.full_fva:
        log.info("WARNING: --full-fva enabled. Each model will run FVA (slow).")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"qc_chunk_{task_id:04d}.csv")
    if os.path.exists(out_csv):
        log.info(f"Already done → {out_csv}. Skipping.")
        return

    uuid_to_label = load_manifest(args.manifest)

    all_pkls = sorted([f for f in os.listdir(args.pkl_dir) if f.endswith(".pkl")])
    cs    = args.chunk_size
    start = (task_id - 1) * cs
    end   = min(task_id * cs, len(all_pkls))
    chunk = all_pkls[start:end]
    log.info(f"Chunk [{start}:{end}] — {len(chunk)} files")

    rows = []
    for i, pkl_fname in enumerate(chunk):
        uuid     = pkl_fname.split(".")[0]
        pkl_path = os.path.join(args.pkl_dir, pkl_fname)
        label    = uuid_to_label.get(uuid, "Unknown")

        try:
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
            metrics = qc_model(model, full_fva=args.full_fva)
        except Exception as e:
            log.error(f"  Load failed {uuid}: {e}")
            metrics = {k: np.nan for k in [
                "n_reactions","n_genes","n_metabolites","fba_status","fba_objective",
                "n_zero_flux_rxns","n_blocked_rxns","n_dead_end_mets",
                "n_structural_rxns","requires_gap_filling"
            ]}
            metrics["fba_status"] = f"load_error:{e}"
            metrics["notes"]      = "load_failed"
            metrics["requires_gap_filling"] = True

        metrics["uuid"]  = uuid
        metrics["label"] = label
        rows.append(metrics)

        log.info(
            f"  [{i+1}/{len(chunk)}] {uuid[:8]}... "
            f"label={label} status={metrics['fba_status']} "
            f"rxns={metrics['n_reactions']} dead_ends={metrics['n_dead_end_mets']}"
        )

    df = pd.DataFrame(rows)
    # Put uuid and label first
    cols = ["uuid","label"] + [c for c in df.columns if c not in ("uuid","label")]
    df   = df[cols]
    df.to_csv(out_csv, index=False)
    log.info(f"Saved QC results → {out_csv}")

    # Quick summary
    n_gap = df["requires_gap_filling"].sum() if "requires_gap_filling" in df else "?"
    log.info(f"Models flagged for gap filling: {n_gap}/{len(df)}")


if __name__ == "__main__":
    main()
