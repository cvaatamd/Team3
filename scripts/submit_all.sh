#!/bin/bash
# Clean start: submit sweep array → collect (--llm) via SLURM.
#
# Usage:
#   bash scripts/submit_all.sh              # mini sweep (8 jobs) → results-mini
#   bash scripts/submit_all.sh --full       # full sweep (240 jobs) → results
#   bash scripts/submit_all.sh --free-slot  # cancel blocking dev-g bash job first
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/lumi_env.sh

PLAN="conf/sweep-mini.yaml"
RESULTS_DIR="results-mini"
JOB_NAME="admet-mini"
FREE_SLOT=0

for arg in "$@"; do
    case "${arg}" in
        --full)      PLAN="conf/sweep.yaml"; RESULTS_DIR="results"; JOB_NAME="admet-sweep" ;;
        --free-slot) FREE_SLOT=1 ;;
    esac
done

[[ "${FREE_SLOT}" -eq 1 ]] && free_devg_slot

load_aitta_token
if [[ -z "${AITTA_API_TOKEN:-}" ]]; then
    echo "ERROR: set AITTA_API_TOKEN or create conf/.aitta_token" >&2
    exit 1
fi

RESULTS_ABS="${WORK_DIR}/${RESULTS_DIR}"
mkdir -p "${RESULTS_ABS}/logs"

echo "=== Step 1/2: sweep (${PLAN}, $(grep -c . "${PLAN}" 2>/dev/null || echo '?') lines) → ${RESULTS_DIR} ===" >&2
SWEEP_ID=$(PLAN="${PLAN}" RESULTS_DIR="${RESULTS_DIR}" JOB_NAME="${JOB_NAME}" \
    bash scripts/submit_sweep.sh --llm)

# afterany (not afterok): the external/both arms can legitimately fail (no
# Polaris/TDC access); we still want a report from whatever finished.
echo "=== Step 2/2: collect afterany:${SWEEP_ID} ===" >&2
COLLECT_ID=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --dependency=afterany:"${SWEEP_ID}" \
    --export=ALL,RESULTS_DIR="${RESULTS_ABS}",AITTA_API_TOKEN \
    --chdir="${WORK_DIR}" \
    --output="${RESULTS_ABS}/logs/collect-%j.out" \
    --error="${RESULTS_ABS}/logs/collect-%j.err" \
    scripts/collect_llm.sbatch)

NJOBS=$(wc -l < "${RESULTS_ABS}/manifest.jsonl")
echo "Sweep:   ${SWEEP_ID}  array 0-$((NJOBS - 1))  (${RESULTS_DIR}/${JOB_NAME}.sbatch)"
echo "Collect: ${COLLECT_ID}  (afterany:${SWEEP_ID})"
squeue -u "${USER}"
