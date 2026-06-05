#!/bin/bash
# Convenience wrapper: enumerate the sweep, render+submit the SLURM array.
#
# Usage:
#   bash scripts/submit_sweep.sh                 # submit (uses conf/sweep.yaml + conf/slurm.yaml)
#   bash scripts/submit_sweep.sh --dry-run       # render scripts only, don't sbatch
#   bash scripts/submit_sweep.sh --local         # run array in-process (no SLURM)
#
# Requires the project to be installed: `pip install -e .` inside the chemprop venv.
set -euo pipefail

cd "$(dirname "$0")/.."

PLAN="${PLAN:-conf/sweep.yaml}"
SLURM_YAML="${SLURM_YAML:-conf/slurm.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results}"
JOB_NAME="${JOB_NAME:-admet-sweep}"

mkdir -p "${RESULTS_DIR}"

python -m experiment.cli submit-sweep \
    --plan "${PLAN}" \
    --slurm "${SLURM_YAML}" \
    --results-dir "${RESULTS_DIR}" \
    --job-name "${JOB_NAME}" \
    "$@"
