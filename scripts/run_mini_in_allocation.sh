#!/bin/bash
# Run the mini sweep + collect inside an existing GPU allocation.
# Use when sbatch hits AssocMaxSubmitJobLimit because an interactive bash job
# is already holding your dev-g slot.
#
# From your GPU node session (salloc/sbatch bash on dev-g):
#   cd /pfs/lustrep1/scratch/project_462001520/Team3/Team3
#   bash scripts/run_mini_in_allocation.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CONTAINER="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
VENV_ACTIVATE="/scratch/project_462001520/Team3/ChemProp/chemprop/bin/activate"
WORK_DIR="$(pwd)"
RESULTS_DIR="${WORK_DIR}/results-mini"
TOKEN_FILE="${AITTA_API_TOKEN_FILE:-conf/.aitta_token}"

if [[ -z "${AITTA_API_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
    export AITTA_API_TOKEN="$(tr -d '[:space:]' < "${TOKEN_FILE}")"
fi
export ADMET_USE_LLM=1
export SINGULARITYENV_ADMET_USE_LLM=1
export SINGULARITYENV_AITTA_API_TOKEN="${AITTA_API_TOKEN:-}"

module purge 2>/dev/null || true
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Regenerate manifest if missing.
if [[ ! -f results-mini/manifest.jsonl ]]; then
    singularity exec \
        --bind /scratch/project_462001520:/scratch/project_462001520 \
        --bind /pfs/lustrep1:/pfs/lustrep1 \
        --pwd "${WORK_DIR}" "${CONTAINER}" \
        bash -lc "source '${VENV_ACTIVATE}' && \
          python -m experiment.cli submit-sweep \
            --plan conf/sweep-mini.yaml --slurm conf/slurm.yaml \
            --results-dir results-mini --job-name admet-mini --dry-run --llm"
fi

N=$(wc -l < results-mini/manifest.jsonl)
echo "=== Running ${N} sweep jobs on $(hostname) ==="
for i in $(seq 0 $((N - 1))); do
    echo "--- job ${i}/${N} ---"
    singularity exec \
        --bind /scratch/project_462001520:/scratch/project_462001520 \
        --bind /pfs/lustrep1:/pfs/lustrep1 \
        --pwd "${WORK_DIR}" "${CONTAINER}" \
        bash -lc "source '${VENV_ACTIVATE}' && \
          python -m experiment.cli run-job \
            --manifest results-mini/manifest.jsonl \
            --index ${i} --results-dir results-mini --llm"
done

echo "=== Collect ==="
singularity exec \
    --bind /scratch/project_462001520:/scratch/project_462001520 \
    --bind /pfs/lustrep1:/pfs/lustrep1 \
    --pwd "${WORK_DIR}" "${CONTAINER}" \
    bash -lc "source '${VENV_ACTIVATE}' && \
      python -m experiment.cli collect --llm --results-dir results-mini"

echo "Done — see results-mini/report.md"
