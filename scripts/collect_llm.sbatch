#!/bin/bash
#SBATCH --job-name=admet-collect-llm
#SBATCH --account=project_462001520
#SBATCH --partition=small-g
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
# Log paths overridden at submit time via sbatch --output/--error.
#SBATCH --output=collect-%j.out
#SBATCH --error=collect-%j.err

# Collect per-job results into parquet, plots, and an LLM-written report.
#
# Set the token once (pick one):
#   export AITTA_API_TOKEN=...
#   echo '...' > conf/.aitta_token && chmod 600 conf/.aitta_token
#
# Submit:
#   bash scripts/submit_collect.sh

set -euo pipefail

WORK_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3"
RESULTS_DIR="${RESULTS_DIR:-${WORK_DIR}/results-mini}"
METRIC="${METRIC:-rae}"   # rae|mae|rmse|r2|spearman
CONTAINER="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
VENV_ACTIVATE="/scratch/project_462001520/Team3/ChemProp/chemprop/bin/activate"
TOKEN_FILE="${AITTA_API_TOKEN_FILE:-${WORK_DIR}/conf/.aitta_token}"

mkdir -p "${RESULTS_DIR}/logs"

# Load token: env var first, then token file (sbatch does not inherit your shell exports).
if [[ -z "${AITTA_API_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
    export AITTA_API_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi

if [[ -z "${AITTA_API_TOKEN:-}" ]]; then
    echo "ERROR: no Aitta token found." >&2
    echo "  export AITTA_API_TOKEN=...   OR" >&2
    echo "  echo '<token>' > conf/.aitta_token && chmod 600 conf/.aitta_token" >&2
    echo "Get a token from https://aitta-auth.csc.fi/myToken" >&2
    exit 1
fi

export ADMET_USE_LLM=1
export SINGULARITYENV_ADMET_USE_LLM=1
export SINGULARITYENV_AITTA_API_TOKEN="${AITTA_API_TOKEN}"

echo "[collect] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[collect] work_dir=${WORK_DIR}"
echo "[collect] results_dir=${RESULTS_DIR}"

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
              python -m experiment.cli collect --llm --results-dir '${RESULTS_DIR}' --metric '${METRIC}'"

echo "[collect] done — wrote results.parquet + curves/report for metric=${METRIC}"
