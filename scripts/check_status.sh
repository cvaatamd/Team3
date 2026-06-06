#!/bin/bash
# =============================================================================
#  check_status.sh  —  See how the sweep is going (safe to run any time).
# =============================================================================
#
#  Usage:
#     bash scripts/check_status.sh            # check the full sweep (results/)
#     bash scripts/check_status.sh --mini     # check the mini sweep (results-mini/)
#
#  It shows: your jobs in the queue, how many runs are done, and whether the
#  final report has been written yet. It changes nothing.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="results"
[[ "${1:-}" == "--mini" ]] && RESULTS_DIR="results-mini"

echo "=== Your SLURM jobs ==="
squeue -u "${USER}" -o "%.12i %.10P %.18j %.2t %.10M %R" 2>/dev/null || echo "(none)"

echo
echo "=== Progress in ${RESULTS_DIR}/ ==="
if [[ -f "${RESULTS_DIR}/manifest.jsonl" ]]; then
    TOTAL=$(wc -l < "${RESULTS_DIR}/manifest.jsonl")
else
    TOTAL="?"
fi
DONE=$(ls "${RESULTS_DIR}/runs/" 2>/dev/null | wc -l | tr -d ' ')
echo "Training runs finished: ${DONE} / ${TOTAL}"

echo
echo "=== Final outputs ==="
for f in report.md results.parquet curves.png ma_rae.png; do
    if [[ -f "${RESULTS_DIR}/${f}" ]]; then
        echo "  [ready] ${RESULTS_DIR}/${f}"
    else
        echo "  [ ... ] ${RESULTS_DIR}/${f}   (not written yet)"
    fi
done

echo
if [[ -f "${RESULTS_DIR}/report.md" ]]; then
    echo "All done. Read the report:  cat ${RESULTS_DIR}/report.md"
else
    echo "Still working. Re-run this script later to check again."
fi
