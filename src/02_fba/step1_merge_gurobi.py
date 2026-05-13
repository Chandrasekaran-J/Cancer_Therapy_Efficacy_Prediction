"""
step1_merge.py
Run AFTER all step1_fba_chunk.py SGE tasks complete.

Combines all chunk_{task_id:04d}.npz files into a single HDF5:
    fba_matrix.h5
        /fluxes      float32 (N, R)  — NaN = reaction absent from GIMME model
        /presence    uint8   (N, R)  — 1 = reaction present in GIMME model
        /uuids       bytes   (N,)
        /labels      bytes   (N,)
        /obj_vals    float32 (N,)
        /statuses    bytes   (N,)
        /rxn_ids     bytes   (R,)    — canonical Recon3D order

Also writes:
    fba_summary.csv  — per-patient FBA summary

Usage:
    python step1_merge.py [--results-dir fba_results] [--out fba_matrix.h5]
"""

import os
import argparse
import glob
import logging
import pickle
import re

import h5py
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR    = "/data/chandrasekaran/algo/Biomarkers"
RXN_LIST    = os.path.join(BASE_DIR, "rxn_list.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "fba_results")
OUT_H5      = os.path.join(BASE_DIR, "fba_matrix.h5")
OUT_CSV     = os.path.join(BASE_DIR, "fba_summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--out",         default=OUT_H5)
    parser.add_argument("--summary-csv", default=OUT_CSV)
    args = parser.parse_args()

    # ── Load canonical reaction list ──────────────────────────────────────────
    with open(RXN_LIST, "rb") as f:
        rxn_payload = pickle.load(f)
    rxn_ids = rxn_payload["rxn_ids"]
    n_rxns  = len(rxn_ids)
    log.info(f"Canonical reaction list: {n_rxns} reactions")

    # ── Find all chunk files, sorted by task id ────────────────────────────────
    chunk_files = sorted(glob.glob(os.path.join(args.results_dir, "chunk_*.npz")))
    log.info(f"Found {len(chunk_files)} chunk files")
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files in {args.results_dir}")

    # ── Load all chunks into lists ─────────────────────────────────────────────
    all_fluxes   = []
    all_presence = []
    all_uuids    = []
    all_labels   = []
    all_obj_vals = []
    all_statuses = []

    for cf in chunk_files:
        log.info(f"  Loading {os.path.basename(cf)} ...")
        data = np.load(cf, allow_pickle=True)
        all_fluxes.append(data["fluxes"])      # (chunk, R)
        all_presence.append(data["presence"])  # (chunk, R)
        all_uuids.extend(data["uuids"].tolist())
        all_labels.extend(data["labels"].tolist())
        all_obj_vals.extend(data["obj_vals"].tolist())
        all_statuses.extend(data["statuses"].tolist())

    # ── Stack ─────────────────────────────────────────────────────────────────
    flux_matrix   = np.vstack(all_fluxes).astype(np.float32)   # (N, R)
    pres_matrix   = np.vstack(all_presence).astype(np.uint8)   # (N, R)
    uuids_arr     = np.array(all_uuids,    dtype=object)
    labels_arr    = np.array(all_labels,   dtype=object)
    obj_vals_arr  = np.array(all_obj_vals, dtype=np.float32)
    statuses_arr  = np.array(all_statuses, dtype=object)
    rxn_ids_arr   = np.array(rxn_ids,      dtype=object)

    N, R = flux_matrix.shape
    log.info(f"Total patients: {N}, reactions: {R}")

    # ── Validate dimensions ────────────────────────────────────────────────────
    if R != n_rxns:
        raise ValueError(f"Reaction count mismatch: got {R}, expected {n_rxns}")

    # ── Write HDF5 ────────────────────────────────────────────────────────────
    log.info(f"Writing HDF5 → {args.out}")
    with h5py.File(args.out, "w") as hf:
        hf.create_dataset("fluxes",   data=flux_matrix,  compression="gzip", compression_opts=4)
        hf.create_dataset("presence", data=pres_matrix,  compression="gzip", compression_opts=4)
        hf.create_dataset("obj_vals", data=obj_vals_arr)

        # String arrays stored as variable-length bytes
        dt_str = h5py.special_dtype(vlen=str)
        hf.create_dataset("uuids",    data=uuids_arr.astype(str),    dtype=dt_str)
        hf.create_dataset("labels",   data=labels_arr.astype(str),   dtype=dt_str)
        hf.create_dataset("statuses", data=statuses_arr.astype(str), dtype=dt_str)
        hf.create_dataset("rxn_ids",  data=rxn_ids_arr.astype(str),  dtype=dt_str)

        # Metadata attributes
        hf.attrs["n_patients"]  = N
        hf.attrs["n_reactions"] = R
        hf.attrs["notes"] = (
            "fluxes: NaN = reaction absent from GIMME model. "
            "presence: 1 = reaction in model, 0 = pruned by GIMME."
        )

    log.info(f"HDF5 written: {args.out}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    n_rxns_per_patient = pres_matrix.sum(axis=1)
    summary = pd.DataFrame({
        "uuid"        : list(uuids_arr.astype(str)),
        "label"       : list(labels_arr.astype(str)),
        "fba_status"  : list(statuses_arr.astype(str)),
        "obj_value"   : obj_vals_arr.astype(float),
        "n_rxns_in_model": n_rxns_per_patient,
    })
    summary.to_csv(args.summary_csv, index=False)
    log.info(f"Summary CSV written: {args.summary_csv}")

    # ── Quick stats ───────────────────────────────────────────────────────────
    n_optimal    = (summary["fba_status"] == "optimal").sum()
    n_sensitive  = (summary["label"] == "Sensitive").sum()
    n_resistant  = (summary["label"] == "Resistant").sum()
    n_unknown    = (summary["label"] == "Unknown").sum()
    log.info(f"FBA status  : {n_optimal}/{N} optimal")
    log.info(f"Labels      : Sensitive={n_sensitive}, Resistant={n_resistant}, Unknown={n_unknown}")
    log.info(f"Model sizes : mean={n_rxns_per_patient.mean():.0f}, "
             f"min={n_rxns_per_patient.min()}, max={n_rxns_per_patient.max()} reactions")
    log.info("Merge complete.")


if __name__ == "__main__":
    main()
