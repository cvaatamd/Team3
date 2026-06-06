#!/bin/bash
# =============================================================================
#  run_fullexp.sh — FULL 5-round ADMET transfer experiment (separate folder).
#
#  Five endpoint "rounds", each a FULL sweep (4 arms x n{25,50,100,250,500,full}
#  x 5 seeds = 120 jobs), run as PARALLEL SLURM arrays with a dependent collect:
#     hlm       HLM CLint                  (rich,  near-identical external)
#     mbpb      MBPB                       (sparse, weak species-transfer ext)
#     ksol      KSOL                       (richest, external needs unit conv.)
#     mppb      MPPB                       (mid,   species-transfer ext)
#     caco2eff  Caco-2 Permeability Efflux (mid,   analogous MDR1-MDCK ext)
#
#  Output goes to a SEPARATE folder (results-fullexp/) and never touches the
#  previous experiments (results/, results-rounds/).
#
#  DISCONNECTION-SAFE: this script only SUBMITS jobs (a few seconds) and exits.
#  SLURM then runs everything independently of your login session, and each
#  collect job is chained with --dependency=afterany, so the reports are written
#  even if you log out immediately after submitting. To be extra safe you may run
#  this launcher itself detached:  nohup bash scripts/run_fullexp.sh &>fullexp.submit.log &
#
#  Token (needed): export AITTA_API_TOKEN=...  or  conf/.aitta_token
#  Usage:  bash scripts/run_fullexp.sh   [--dry-run]
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/lumi_env.sh

SLURM_YAML="conf/slurm-fullexp.yaml"
EXP_DIR="results-fullexp"
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
    "hlm|conf/fullexp-hlm.yaml|${EXP_DIR}/hlm|fx-hlm"
    "mbpb|conf/fullexp-mbpb.yaml|${EXP_DIR}/mbpb|fx-mbpb"
    "ksol|conf/fullexp-ksol.yaml|${EXP_DIR}/ksol|fx-ksol"
    "mppb|conf/fullexp-mppb.yaml|${EXP_DIR}/mppb|fx-mppb"
    "caco2eff|conf/fullexp-caco2eff.yaml|${EXP_DIR}/caco2eff|fx-caco2eff"
)

mkdir -p "${WORK_DIR}/${EXP_DIR}"
JOBIDS_FILE="${WORK_DIR}/${EXP_DIR}/JOBIDS.txt"
[[ "${DRY_RUN}" -eq 0 ]] && : > "${JOBIDS_FILE}"

echo "============================================================"
echo " FULL experiment — 5 endpoint rounds (parallel arrays)"
echo " slurm config: ${SLURM_YAML}   output: ${EXP_DIR}/"
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
    echo "${key} sweep=${SWEEP_ID} collect=${COLLECT_ID} jobs=${NJOBS} dir=${rdir}" >> "${JOBIDS_FILE}"
done

[[ "${DRY_RUN}" -eq 1 ]] && { echo; echo "dry-run complete (nothing submitted)."; exit 0; }

echo
echo "============================================================"
echo " Submitted all 5 rounds. Job IDs saved to ${JOBIDS_FILE}"
echo " You can safely log out now — SLURM runs everything detached."
echo " Monitor:  squeue -u ${USER}        progress: bash scripts/check_status.sh"
echo " Reports (when each finishes):"
for entry in "${ROUNDS[@]}"; do IFS='|' read -r key _ rdir _ <<< "${entry}"; echo "   ${rdir}/report.md"; done
echo "============================================================"
squeue -u "${USER}" -o "%.10i %.12j %.2t %.6M %R" | head -20 || true
