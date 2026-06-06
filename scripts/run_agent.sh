#!/bin/bash
# =============================================================================
#  run_agent.sh — launch the STANDALONE Aitta-driven ADMET agent on LUMI.
#
#  This is a thin, manually-reproducible launcher around the single Python agent
#  in  src/agents/orchestrator.py . It does not depend on Cursor or any IDE.
#
#  WHAT IT DOES
#    (default)   Submit ONE SLURM job (scripts/agent.sbatch) that runs the whole
#                agent loop in a single Python process on a GPU node:
#                  Planner -> Data agent -> ML agent -> Planner (report),
#                with the three decision points routed through Aitta.
#
#  USAGE
#    bash scripts/run_agent.sh                 # standalone agent, mini plan
#    bash scripts/run_agent.sh --full          # standalone agent, full plan (serial, slow!)
#    bash scripts/run_agent.sh --no-llm        # deterministic agents (no Aitta)
#    bash scripts/run_agent.sh --check         # Aitta connectivity + plan only (no jobs)
#    bash scripts/run_agent.sh --slurm         # parallel SLURM array path (calls submit_all.sh)
#
#  OPTIONS
#    --plan FILE          sweep plan YAML (default conf/sweep-mini.yaml)
#    --results-dir DIR    output dir (default results-agent); set a unique dir per round
#    --token TOK          pass the Aitta token inline
#
#  RUNNING A FEW ROUNDS
#    Give each round its own results dir so they don't clobber each other:
#      bash scripts/run_agent.sh --results-dir results-agent/round1
#      bash scripts/run_agent.sh --results-dir results-agent/round2
#    Re-running the SAME dir is a no-op for finished jobs (results/runs/*.json are
#    reused). Aitta decisions are cached under results/llm_cache, so repeats are free.
#
#  TOKEN (needed unless --no-llm) — set ONE of:
#    export AITTA_API_TOKEN=...
#    echo '<token>' > conf/.aitta_token && chmod 600 conf/.aitta_token
#  Get a token from https://aitta-auth.csc.fi/myToken
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/lumi_env.sh

PLAN="conf/sweep-mini.yaml"
RESULTS_DIR="results-agent"
USE_LLM=1
MODE="submit"        # submit | check | slurm
PLAN_SET=0           # explicit --plan / --results-dir win over --full/--mini
RD_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)         PLAN="conf/sweep.yaml"; [[ "${RD_SET}" -eq 0 ]] && RESULTS_DIR="results-agent-full" ;;
        --mini)         PLAN="conf/sweep-mini.yaml"; [[ "${RD_SET}" -eq 0 ]] && RESULTS_DIR="results-agent" ;;
        --plan)         shift; PLAN="${1:?--plan needs a value}"; PLAN_SET=1 ;;
        --results-dir)  shift; RESULTS_DIR="${1:?--results-dir needs a value}"; RD_SET=1 ;;
        --no-llm)       USE_LLM=0 ;;
        --check)        MODE="check" ;;
        --slurm)        MODE="slurm" ;;
        --token)        shift; export AITTA_API_TOKEN="${1:?--token needs a value}" ;;
        -h|--help)      sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

load_aitta_token
if [[ "${USE_LLM}" -eq 1 && -z "${AITTA_API_TOKEN:-}" ]]; then
    echo "ERROR: no Aitta token. Set AITTA_API_TOKEN, create conf/.aitta_token," >&2
    echo "       or pass --no-llm. Get one at https://aitta-auth.csc.fi/myToken" >&2
    exit 1
fi

# --- --check: connectivity + planning dry-run (runs in container, no jobs) ----
if [[ "${MODE}" == "check" ]]; then
    LLM_FLAG=""; [[ "${USE_LLM}" -eq 0 ]] && LLM_FLAG="--no-llm"
    echo ">>> Aitta connectivity + plan check (no jobs submitted) ..."
    run_in_container "python -m agents.orchestrator --dry-run \
        --plan '${PLAN}' --results-dir '${RESULTS_DIR}' ${LLM_FLAG}"
    exit $?
fi

# --- --slurm: hand off to the proven parallel SLURM-array pipeline ------------
if [[ "${MODE}" == "slurm" ]]; then
    echo ">>> Parallel SLURM array path (sweep + dependent collect) ..."
    if [[ "${PLAN}" == "conf/sweep.yaml" ]]; then
        exec bash scripts/submit_all.sh --full
    else
        exec bash scripts/submit_all.sh
    fi
fi

# --- default: submit the single standalone agent job -------------------------
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: 'sbatch' not found — run this on a LUMI login node." >&2
    exit 1
fi

RESULTS_ABS="${WORK_DIR}/${RESULTS_DIR}"
mkdir -p "${RESULTS_ABS}/logs"

echo ">>> Submitting standalone agent: plan=${PLAN} results=${RESULTS_DIR} use_llm=${USE_LLM}"
[[ "${PLAN}" == "conf/sweep.yaml" ]] && \
    echo "    NOTE: --full runs serially on ONE GPU and can take many hours." >&2

JOB_ID=$(PLAN="${PLAN}" RESULTS_DIR="${RESULTS_DIR}" USE_LLM="${USE_LLM}" \
    sbatch --parsable \
        --partition="${PARTITION}" \
        --export=ALL,PLAN="${PLAN}",RESULTS_DIR="${RESULTS_DIR}",USE_LLM="${USE_LLM}",AITTA_API_TOKEN \
        --chdir="${WORK_DIR}" \
        --output="${RESULTS_ABS}/logs/agent-%j.out" \
        --error="${RESULTS_ABS}/logs/agent-%j.err" \
        scripts/agent.sbatch)

echo "Submitted standalone agent as SLURM job ${JOB_ID}"
echo "  logs:    ${RESULTS_ABS}/logs/agent-${JOB_ID}.out"
echo "  report:  ${RESULTS_ABS}/report.md   (when finished)"
echo "  check:   squeue -u ${USER}"
