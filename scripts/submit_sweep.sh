#!/bin/bash
# Enumerate sweep jobs, render SLURM scripts, and submit the array.
#
# Usage:
#   bash scripts/submit_sweep.sh                 # full sweep → results/
#   bash scripts/submit_sweep.sh --dry-run
#   bash scripts/submit_sweep.sh --llm
#
# Mini sweep:
#   PLAN=conf/sweep-mini.yaml RESULTS_DIR=results-mini JOB_NAME=admet-mini \
#     bash scripts/submit_sweep.sh --llm
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/lumi_env.sh

PLAN="${PLAN:-conf/sweep.yaml}"
SLURM_YAML="${SLURM_YAML:-conf/slurm.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results}"
JOB_NAME="${JOB_NAME:-admet-sweep}"

mkdir -p "${RESULTS_DIR}/logs"

LOCAL=0
DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        --local)   LOCAL=1 ;;
        --dry-run) DRY_RUN=1 ;;
        *)         EXTRA_ARGS+=("${arg}") ;;
    esac
done

load_aitta_token

CLI_ARGS=(
    submit-sweep
    --plan "${PLAN}"
    --slurm "${SLURM_YAML}"
    --results-dir "${RESULTS_DIR}"
    --job-name "${JOB_NAME}"
)

if [[ "${LOCAL}" -eq 1 ]]; then
    run_in_container "python -m experiment.cli ${CLI_ARGS[*]} --local ${EXTRA_ARGS[*]:-}"
    exit $?
fi

# Generate sbatch + worker inside container (sbatch is not available there).
echo "Generating ${RESULTS_DIR}/${JOB_NAME}.sbatch ..." >&2
run_in_container "python -m experiment.cli ${CLI_ARGS[*]} --dry-run ${EXTRA_ARGS[*]:-}" >&2

SBATCH_SCRIPT="${RESULTS_DIR}/${JOB_NAME}.sbatch"
WORKER_SCRIPT="${RESULTS_DIR}/${JOB_NAME}.worker.sh"
if [[ ! -f "${SBATCH_SCRIPT}" || ! -f "${WORKER_SCRIPT}" ]]; then
    echo "ERROR: missing ${SBATCH_SCRIPT} or ${WORKER_SCRIPT}" >&2
    exit 1
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "dry-run: wrote ${SBATCH_SCRIPT}" >&2
    exit 0
fi

JOB_ID=$(sbatch --parsable --partition="${PARTITION}" --export=ALL "${SBATCH_SCRIPT}")
echo "submitted ${JOB_ID} (${SBATCH_SCRIPT})" >&2
echo "${JOB_ID}"
