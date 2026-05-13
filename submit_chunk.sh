#!/bin/bash
# ============================================================
# Example SGE submission script template
# Adapt paths, queue names, and resource requests to your cluster
# ============================================================
#
# Usage:
#   qsub -t 1-100 submit_fba.sh        # array job: 100 chunks
#   qsub -t 1-100 submit_qc.sh
#
# To run a different stage, change the python invocation at the bottom
# and adjust resources accordingly.
# ============================================================

#$ -N gsmm_pipeline
#$ -cwd
#$ -j y                                 # merge stderr into stdout
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID
#$ -q all.q                             # queue name (cluster-specific)
#$ -pe smp 2                            # 2 cores per task
#$ -l h_vmem=8G                         # 8 GB RAM per task
#$ -l h_rt=24:00:00                     # 24 h wall time

# Make sure logs directory exists
mkdir -p logs

# Activate conda environment (adjust to your setup)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gsmm_env

# Gurobi WLS licence (if applicable)
export GRB_LICENSE_FILE=$HOME/gurobi.lic

# ─── Run the chunk worker ─────────────────────────────────────
# SGE_TASK_ID is set automatically (1..N when launched with -t 1-N)
python src/02_fba/step1_fba_chunk_gurobi.py --chunk-id ${SGE_TASK_ID}

# For QC stage instead:
# python src/03_qc/step2_qc_chunk_gurobi.py --chunk-id ${SGE_TASK_ID}

# For FVA + gap-fill (uses --phase argument):
# python src/03_qc/step2b_fva_gapfill_gurobi.py --phase fva     --task-id ${SGE_TASK_ID}
# python src/03_qc/step2b_fva_gapfill_gurobi.py --phase gapfill --task-id ${SGE_TASK_ID}
