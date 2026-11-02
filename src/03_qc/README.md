# Stage 3 — Quality Control + FVA + Gap-Filling

Per-model checks: FBA feasibility, objective value, dead-end metabolites, structural reactions. Optional FVA-based blocked-reaction detection and MILP gap-filling.

| Script | Purpose |
|---|---|
| `step2_qc_chunk_gurobi.py` | Per-chunk QC (SGE array worker) |
| `step2_merge_report_gurobi.py` | Merge all QC chunks into `qc_report.csv` |
| `step2b_fva_gapfill_gurobi.py` | Three-phase loopless FVA + MILP gap-fill |

`step2b` uses three phases controlled by `--phase {fva, gapfill, report}` and should be run in that order.

The study used the structural and single-solution FBA diagnostics. Complete FVA and systematic gap filling were not applied across the reported model collection; the optional utilities are included for subsequent analyses.
