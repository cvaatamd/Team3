#!/bin/bash
# =============================================================================
#  run_rounds.sh — run the multi-round endpoint-exploration experiment.
#
#  Three independent "rounds", each a different ADMET endpoint's transfer story,
#  run as PARALLEL SLURM arrays (all 4 arms, n in {25,50,100,250}, 5 seeds):
#     round1  HLM CLint  (near-identical external)   -> results-rounds/hlm
#     round2  MBPB       (weak species-transfer ext) -> results-rounds/mbpb
#     round3  KSOL       (external needs unit conv.) -> results-rounds/ksol
#
#  Each round = sweep array (--llm, Aitta agents in workers) + a dependent
#  collect job that writes curves + an Aitta-narrated report.
#
#  Token (needed): export AITTA_API_TOKEN=... or conf/.aitta_token
#  Usage:  bash scripts/run_rounds.sh   [--dry-run]
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/lumi_env.sh

SLURM_YAML="conf/slurm-rounds.yaml"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

load_aitta_token
if [[ -z "${AITTA_API_TOKEN:-}" ]]; then
    echo "ERROR: no Aitta token. Set AITTA_API_TOKEN or create conf/.aitta_token" >&2
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: 'sbatch' not found — run on a LUMI login node." >&2
    exit 1
fi

# round_key | plan | results_subdir | job_name
ROUNDS=(
    "hlm|conf/round1-hlm.yaml|results-rounds/hlm|admet-hlm"
    "mbpb|conf/round2-mbpb.yaml|results-rounds/mbpb|admet-mbpb"
    "ksol|conf/round3-ksol.yaml|results-rounds/ksol|admet-ksol"
)

echo "============================================================"
echo " Multi-round experiment — 3 endpoint rounds (parallel arrays)"
echo " slurm config: ${SLURM_YAML}"
echo "============================================================"

for entry in "${ROUNDS[@]}"; do
    IFS='|' read -r key plan rdir jobname <<< "${entry}"
    rabs="${WORK_DIR}/${rdir}"
    mkdir -p "${rabs}/logs"

    echo
    echo ">>> Round '${key}': plan=${plan} -> ${rdir}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        PLAN="${plan}" SLURM_YAML="${SLURM_YAML}" RESULTS_DIR="${rdir}" JOB_NAME="${jobname}" \
            bash scripts/submit_sweep.sh --llm --dry-run
        echo "    (dry-run) would chain collect afterany"
        continue
    fi

    SWEEP_ID=$(PLAN="${plan}" SLURM_YAML="${SLURM_YAML}" RESULTS_DIR="${rdir}" JOB_NAME="${jobname}" \
        bash scripts/submit_sweep.sh --llm)

    COLLECT_ID=$(sbatch --parsable \
        --partition="${PARTITION}" \
        --dependency=afterany:"${SWEEP_ID}" \
        --export=ALL,RESULTS_DIR="${rabs}",AITTA_API_TOKEN \
        --chdir="${WORK_DIR}" \
        --output="${rabs}/logs/collect-%j.out" \
        --error="${rabs}/logs/collect-%j.err" \
        scripts/collect_llm.sbatch)

    NJOBS=$(wc -l < "${rabs}/manifest.jsonl")
    echo "    sweep:   ${SWEEP_ID}  (${NJOBS} jobs)"
    echo "    collect: ${COLLECT_ID}  (afterany:${SWEEP_ID}) -> ${rdir}/report.md"
done

[[ "${DRY_RUN}" -eq 1 ]] && { echo; echo "dry-run complete (nothing submitted)."; exit 0; }

echo
echo "============================================================"
echo " Submitted all 3 rounds. Monitor:  squeue -u ${USER}"
echo " Reports (when done):"
echo "   results-rounds/hlm/report.md"
echo "   results-rounds/mbpb/report.md"
echo "   results-rounds/ksol/report.md"
echo "============================================================"
squeue -u "${USER}" -o "%.10i %.12j %.2t %.6M %R" || true
