#!/bin/bash
# Per-task worker: prepare environment, then run one job from the manifest.
# Argument: array task index (== JSONL line number).

set -euo pipefail
TASK_IDX="${1:?usage: $0 ARRAY_TASK_INDEX}"

WORK_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3"
RESULTS_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results-mini"
MANIFEST="/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results-mini/manifest.jsonl"
CONTAINER="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
VENV_ACTIVATE="/scratch/project_462001520/Team3/ChemProp/chemprop/bin/activate"

# LLM (Aitta) plumbing — only used if ADMET_USE_LLM=1 is set.
export ADMET_USE_LLM="${ADMET_USE_LLM:-1}"
TOKEN_FILE="${AITTA_API_TOKEN_FILE:-$WORK_DIR/conf/.aitta_token}"
if [[ -z "${AITTA_API_TOKEN:-}" && -f "$TOKEN_FILE" ]]; then
    export AITTA_API_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
fi
export SINGULARITYENV_ADMET_USE_LLM="$ADMET_USE_LLM"
export SINGULARITYENV_AITTA_API_TOKEN="${AITTA_API_TOKEN:-}"
LLM_FLAG=""
if [[ "$ADMET_USE_LLM" == "1" ]]; then
    LLM_FLAG="--llm"
fi

# Lumi container modules.
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

cd "$WORK_DIR"

singularity exec \
    --bind /scratch/project_462001520:/scratch/project_462001520 \
    --bind /pfs/lustrep1:/pfs/lustrep1 \
    --pwd "$WORK_DIR" \
    "$CONTAINER" \
    bash -lc "source '$VENV_ACTIVATE' && \
              python -m experiment.cli run-job \
                --manifest '$MANIFEST' \
                --index '$TASK_IDX' \
                --results-dir '$RESULTS_DIR' \
                $LLM_FLAG"
