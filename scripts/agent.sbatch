#!/bin/bash
#SBATCH --job-name=admet-agent
#SBATCH --account=project_462001520
#SBATCH --partition=small-g
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
# Log paths overridden at submit time via sbatch --output/--error.
#SBATCH --output=agent-%j.out
#SBATCH --error=agent-%j.err

# =============================================================================
#  agent.sbatch — the STANDALONE Aitta-driven ADMET agent, end to end.
#
#  Runs `python -m agents.orchestrator --mode local` inside the chemprop
#  container on a GPU node. The single Python process does the whole workflow:
#     Planner (enumerate) -> Data agent (sources, via Aitta)
#                         -> ML agent (mechanism, via Aitta) -> train/eval
#                         -> Planner (interpretation report, via Aitta)
#
#  This is independent of Cursor: just `sbatch scripts/agent.sbatch`.
#
#  Configure via environment (all optional):
#     PLAN        sweep plan YAML       (default: conf/sweep-mini.yaml)
#     RESULTS_DIR output dir            (default: results-agent)
#     USE_LLM     1=Aitta agents, 0=deterministic (default: 1)
#
#  Token (needed when USE_LLM=1) — set ONE of:
#     export AITTA_API_TOKEN=...
#     echo '<token>' > conf/.aitta_token && chmod 600 conf/.aitta_token
#  Get a token from https://aitta-auth.csc.fi/myToken
#
#  NOTE: --mode local runs the sweep serially on ONE GPU. Use the small plan
#  (conf/sweep-mini.yaml) here. For the full 240-job parallel sweep use
#  scripts/run_full_sweep.sh (SLURM array), which the agent can also drive in
#  --mode slurm from a login node.
# =============================================================================
set -euo pipefail

WORK_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3"
PLAN="${PLAN:-conf/sweep-mini.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results-agent}"
USE_LLM="${USE_LLM:-1}"
CONTAINER="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
VENV_ACTIVATE="/scratch/project_462001520/Team3/ChemProp/chemprop/bin/activate"
TOKEN_FILE="${AITTA_API_TOKEN_FILE:-${WORK_DIR}/conf/.aitta_token}"

mkdir -p "${WORK_DIR}/${RESULTS_DIR}/logs"

# Load token: env var first, then token file (sbatch does not inherit shell exports).
if [[ -z "${AITTA_API_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
    export AITTA_API_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi
if [[ "${USE_LLM}" == "1" && -z "${AITTA_API_TOKEN:-}" ]]; then
    echo "ERROR: USE_LLM=1 but no Aitta token found." >&2
    echo "  export AITTA_API_TOKEN=...   OR" >&2
    echo "  echo '<token>' > conf/.aitta_token && chmod 600 conf/.aitta_token" >&2
    echo "Get a token from https://aitta-auth.csc.fi/myToken" >&2
    exit 1
fi

# Forward env into the container.
export SINGULARITYENV_AITTA_API_TOKEN="${AITTA_API_TOKEN:-}"

LLM_FLAG=""
[[ "${USE_LLM}" != "1" ]] && LLM_FLAG="--no-llm"

echo "[agent] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[agent] plan=${PLAN} results_dir=${RESULTS_DIR} use_llm=${USE_LLM}"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

cd "${WORK_DIR}"

singularity exec \
    --bind /scratch/project_462001520:/scratch/project_462001520 \
    --bind /pfs/lustrep1:/pfs/lustrep1 \
    --pwd "${WORK_DIR}" \
    "${CONTAINER}" \
    bash -lc "source '${VENV_ACTIVATE}' && \
              python -m agents.orchestrator --mode local \
                --plan '${PLAN}' \
                --results-dir '${RESULTS_DIR}' \
                ${LLM_FLAG}"

echo "[agent] done — see ${RESULTS_DIR}/report.md, results.parquet, curves.png, ma_rae.png"
