#!/bin/bash
# Targeted re-run worker: re-execute ONE previously-failed manifest index.
# Argument: array task index (0-based position into INDICES below).
#
# These are the 21 `external`-arm jobs that wrote NaN under the old checkpoint
# race (commit 44a8d75). The race is fixed in chemprop_mt.py; --force overwrites
# the stale NaN result JSONs with good ones.
set -euo pipefail
TASK_IDX="${1:?usage: $0 ARRAY_TASK_INDEX}"

WORK_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3"
RESULTS_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results"
MANIFEST="${RESULTS_DIR}/manifest.jsonl"
CONTAINER="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
VENV_ACTIVATE="/scratch/project_462001520/Team3/ChemProp/chemprop/bin/activate"

# Manifest indices of the failed external-arm cells (see results/runs/*__external__*).
INDICES=(68 73 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 201 202 203 204)
IDX="${INDICES[$TASK_IDX]}"

# Aitta plumbing (mechanism choice is LLM-routed; decisions are cached on disk).
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

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

echo "[rerun] array task $TASK_IDX -> manifest index $IDX on host=$(hostname)"
cd "$WORK_DIR"

singularity exec \
    --bind /scratch/project_462001520:/scratch/project_462001520 \
    --bind /pfs/lustrep1:/pfs/lustrep1 \
    --pwd "$WORK_DIR" \
    "$CONTAINER" \
    bash -lc "source '$VENV_ACTIVATE' && \
              python -m experiment.cli run-job \
                --manifest '$MANIFEST' \
                --index '$IDX' \
                --results-dir '$RESULTS_DIR' \
                --force \
                $LLM_FLAG"
