# Stage 2 — Flux Balance Analysis

Runs FBA on all 10,603 patient-specific GIMME models on the HPC cluster.

| Script | Purpose | When to run |
|---|---|---|
| `precompute_rxn_list_gurobi.py` | Build canonical 10,600-reaction ordering from Recon3D | Once, before chunks |
| `step1_fba_chunk_gurobi.py` | FBA on a chunk of patient models (SGE array worker) | Parallel array job |
| `step1_merge_gurobi.py` | Combine all chunk outputs into `fba_matrix.h5` | Once, after array job |

**Final output:** `fba_matrix.h5` — compressed HDF5 of shape `(11505, 10600)` (float32).
