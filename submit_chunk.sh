#!/bin/bash
# Example SGE array template. Adapt queue and resource requests to your cluster.
# Create logs/ and activate your environment before qsub.
# Usage: qsub -t 1-58 submit_chunk.sh fba --manifest data/manifest.tsv ...
#        qsub -t 1-58 submit_chunk.sh qc  --manifest data/manifest.tsv ...
#$ -N gsmm_pipeline
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID
#$ -q all.q
#$ -pe smp 2
#$ -l h_vmem=8G
#$ -l h_rt=24:00:00

set -euo pipefail
stage="${1:-}"
case "$stage" in
  fba) worker="src/02_fba/step1_fba_chunk_gurobi.py" ;;
  qc)  worker="src/03_qc/step2_qc_chunk_gurobi.py" ;;
  *) echo "Usage: $0 {fba|qc} --manifest PATH [worker options]" >&2; exit 2 ;;
esac
shift
: "${SGE_TASK_ID:?Submit an array job with qsub -t, or set SGE_TASK_ID for a local run}"
# Workers read SGE_TASK_ID themselves; there is no --chunk-id argument.
exec "${GSMM_PYTHON:-python}" "$worker" "$@"
