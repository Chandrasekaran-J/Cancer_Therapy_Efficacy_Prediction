"""
step1_fba_chunk.py
SGE array worker — runs FBA on a chunk of patient GIMME models.

Each task writes:
    fba_results/chunk_{task_id:04d}.npz
        fluxes   : float32 (chunk_size, N_rxns)  — NaN = reaction absent
        presence : bool    (chunk_size, N_rxns)  — True = reaction in model
        uuids    : (chunk_size,)  str
        labels   : (chunk_size,)  str  e.g. 'Sensitive' / 'Resistant'
        obj_vals : (chunk_size,)  float32  — FBA objective value
        statuses : (chunk_size,)  str  — 'optimal' / 'infeasible' / 'error'

After ALL tasks complete, run step1_merge.py to combine into one HDF5.

Usage (SGE sets SGE_TASK_ID automatically):
    python step1_fba_chunk.py --manifest <path.csv> [--chunk-size 200]

MUST BE FIRST — sets solver before any cobra import.
"""

import os, sys
os.environ["OPTLANG_DEFAULT_INTERFACE"] = "optlang.gurobi_interface"

import argparse
import logging
import math
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import cobra

warnings.filterwarnings("ignore")  # suppress solver verbosity

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = "/data/chandrasekaran/algo/Biomarkers"
PKL_DIR     = "/data/chandrasekaran/algo/GIMME_output"
RXN_LIST    = os.path.join(BASE_DIR, "rxn_list.pkl")
OUT_DIR     = os.path.join(BASE_DIR, "fba_results")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TASK%(task)s] %(levelname)s %(message)s",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_task_id() -> int:
    """Return 1-based SGE task ID (or 1 for local testing)."""
    return int(os.environ.get("SGE_TASK_ID", 1))


def load_manifest(path: str) -> dict:
    """
    Load the transcriptomics manifest.
    Expects columns: file_uuid  (str) and  sensitivity  (str).
    Returns dict {uuid: label}.
    """
    # Handle both .csv and .tsv; the manifest screenshot shows xlsx — support all.
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path, dtype=str)
    elif path.endswith(".tsv") or path.endswith(".txt"):
        df = pd.read_csv(path, sep="\t", dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)

    df.columns = df.columns.str.strip()

    if "file_uuid" not in df.columns:
        raise ValueError(f"Manifest missing 'file_uuid' column. Got: {df.columns.tolist()}")
    if "sensitivity" not in df.columns:
        raise ValueError(f"Manifest missing 'sensitivity' column. Got: {df.columns.tolist()}")

    uuid_to_label = dict(zip(df["file_uuid"].str.strip(), df["sensitivity"].str.strip()))
    return uuid_to_label


def run_fba_on_model(model: cobra.Model) -> tuple:
    """
    Run FBA.  Returns (status, objective_value, flux_dict).
    flux_dict: {reaction_id: flux_value}  (only reactions in this model)
    """
    try:
        sol = model.optimize()
        if sol.status == "optimal":
            return "optimal", float(sol.objective_value), sol.fluxes.to_dict()
        else:
            return sol.status, 0.0, {}
    except Exception as e:
        return f"error:{e}", 0.0, {}


def align_fluxes(flux_dict: dict, rxn_ids: list) -> tuple:
    """
    Align a per-model flux dict to the canonical Recon3D reaction order.
    Returns:
        presence : np.bool_  array   True if reaction exists in GIMME model
        fluxes   : np.float32 array  NaN for absent reactions
    """
    presence = np.zeros(len(rxn_ids), dtype=bool)
    fluxes   = np.full(len(rxn_ids), np.nan, dtype=np.float32)
    for i, rid in enumerate(rxn_ids):
        if rid in flux_dict:
            presence[i] = True
            fluxes[i]   = np.float32(flux_dict[rid])
    return presence, fluxes


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",   required=True, help="Path to transcriptomics manifest file")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--pkl-dir",    default=PKL_DIR)
    parser.add_argument("--out-dir",    default=OUT_DIR)
    args = parser.parse_args()

    task_id = get_task_id()
    log = logging.LoggerAdapter(logging.getLogger(), {"task": task_id})

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Skip if chunk already done ───────────────────────────────────────────
    out_path = os.path.join(args.out_dir, f"chunk_{task_id:04d}.npz")
    if os.path.exists(out_path):
        log.info(f"Already done → {out_path}. Skipping.")
        return

    # ── Load canonical reaction list ─────────────────────────────────────────
    log.info(f"Loading rxn_list from {RXN_LIST}")
    with open(RXN_LIST, "rb") as f:
        rxn_payload = pickle.load(f)
    rxn_ids = rxn_payload["rxn_ids"]          # list of ~10,600 reaction IDs
    n_rxns  = len(rxn_ids)
    rxn_idx = {rid: i for i, rid in enumerate(rxn_ids)}
    log.info(f"Canonical reaction list: {n_rxns} reactions")

    # ── Load manifest ────────────────────────────────────────────────────────
    log.info(f"Loading manifest from {args.manifest}")
    uuid_to_label = load_manifest(args.manifest)
    log.info(f"Manifest: {len(uuid_to_label)} entries")

    # ── Enumerate PKL files ──────────────────────────────────────────────────
    all_pkls = sorted([
        f for f in os.listdir(args.pkl_dir) if f.endswith(".pkl")
    ])
    n_total = len(all_pkls)
    log.info(f"Total PKL files found: {n_total}")

    # Chunk slice (1-based task_id)
    cs     = args.chunk_size
    start  = (task_id - 1) * cs          # 0-based inclusive
    end    = min(task_id * cs, n_total)  # 0-based exclusive
    chunk  = all_pkls[start:end]
    log.info(f"Processing chunk [{start}:{end}] — {len(chunk)} files")

    if not chunk:
        log.warning("Empty chunk — check -t range in SGE submission.")
        return

    # ── Process chunk ────────────────────────────────────────────────────────
    chunk_fluxes   = np.full((len(chunk), n_rxns), np.nan, dtype=np.float32)
    chunk_presence = np.zeros((len(chunk), n_rxns), dtype=bool)
    chunk_uuids    = []
    chunk_labels   = []
    chunk_objvals  = np.zeros(len(chunk), dtype=np.float32)
    chunk_statuses = []

    for local_idx, pkl_fname in enumerate(chunk):
        # Extract UUID from filename: <uuid>.rna_seq.augmented_star_gene_counts.pkl
        uuid = pkl_fname.split(".")[0]
        pkl_path = os.path.join(args.pkl_dir, pkl_fname)

        label = uuid_to_label.get(uuid, "Unknown")
        chunk_uuids.append(uuid)
        chunk_labels.append(label)

        try:
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)

            status, obj_val, flux_dict = run_fba_on_model(model)
            chunk_objvals[local_idx]  = obj_val
            chunk_statuses.append(status)

            if status == "optimal":
                presence, fluxes = align_fluxes(flux_dict, rxn_ids)
                chunk_presence[local_idx] = presence
                chunk_fluxes[local_idx]   = fluxes

            log.info(
                f"  [{local_idx+1}/{len(chunk)}] {uuid[:8]}... "
                f"label={label} status={status} obj={obj_val:.4f} "
                f"rxns={model.reactions.__len__()}"
            )

        except Exception as e:
            log.error(f"  FAILED {uuid}: {e}")
            chunk_statuses.append(f"load_error:{e}")

    # ── Save chunk ────────────────────────────────────────────────────────────
    np.savez_compressed(
        out_path,
        fluxes   = chunk_fluxes,
        presence = chunk_presence,
        uuids    = np.array(chunk_uuids,    dtype=object),
        labels   = np.array(chunk_labels,   dtype=object),
        obj_vals = chunk_objvals,
        statuses = np.array(chunk_statuses, dtype=object),
    )
    log.info(f"Saved chunk → {out_path}")

    n_ok   = sum(1 for s in chunk_statuses if s == "optimal")
    n_fail = len(chunk_statuses) - n_ok
    log.info(f"Chunk summary: {n_ok} optimal, {n_fail} failed/infeasible")


if __name__ == "__main__":
    main()
