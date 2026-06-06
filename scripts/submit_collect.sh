#!/bin/bash
# Submit collect --llm only (expects runs/ to already exist).
#
# Usage:
#   RESULTS_DIR=results-mini bash scripts/submit_collect.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/lumi_env.sh

RESULTS_DIR="${RESULTS_DIR:-results-mini}"
RESULTS_ABS="${WORK_DIR}/${RESULTS_DIR}"

load_aitta_token
if [[ -z "${AITTA_API_TOKEN:-}" ]]; then
    echo "ERROR: set AITTA_API_TOKEN or create conf/.aitta_token" >&2
    exit 1
fi

mkdir -p "${RESULTS_ABS}/logs"

JOB_ID=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --export=ALL,RESULTS_DIR="${RESULTS_ABS}",AITTA_API_TOKEN \
    --chdir="${WORK_DIR}" \
    --output="${RESULTS_ABS}/logs/collect-%j.out" \
    --error="${RESULTS_ABS}/logs/collect-%j.err" \
    scripts/collect_llm.sbatch)

echo "submitted collect ${JOB_ID} → ${RESULTS_DIR}"
