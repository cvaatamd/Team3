#!/bin/bash
# Per-task worker for the prediction-regeneration sweep (Patch 1).
#
# Identical execution path to results-fullexp/<round>/fx-*.worker.sh, with ONE difference:
# it puts the PATCHED copy (further-improvements/src) first on PYTHONPATH so the run persists
# per-molecule predictions to <RESULTS_DIR>/preds/. The original src/ is left untouched.
#
# Driven entirely by environment variables (set by the sbatch / submit helper):
#   RESULTS_DIR, MANIFEST, N_JOBS, N_TASKS  (required)
#   ADMET_USE_LLM (default 1), AITTA_API_TOKEN_FILE
#
# Heavy compute (Chemprop training) runs here, on a GPU compute node — NEVER on a login node.
set -euo pipefail
TASK_IDX="${1:?usage: $0 ARRAY_TASK_INDEX}"

WORK_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3"
CODE_SRC="$WORK_DIR/further-improvements/src"          # <-- patched copy
CONTAINER="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
VENV_ACTIVATE="/scratch/project_462001520/Team3/ChemProp/chemprop/bin/activate"

: "${RESULTS_DIR:?set RESULTS_DIR}"
: "${MANIFEST:?set MANIFEST}"
: "${N_JOBS:?set N_JOBS}"
: "${N_TASKS:?set N_TASKS}"

export ADMET_USE_LLM="${ADMET_USE_LLM:-1}"
TOKEN_FILE="${AITTA_API_TOKEN_FILE:-$WORK_DIR/conf/.aitta_token}"
if [[ -z "${AITTA_API_TOKEN:-}" && -f "$TOKEN_FILE" ]]; then
    export AITTA_API_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
fi
export SINGULARITYENV_ADMET_USE_LLM="$ADMET_USE_LLM"
export SINGULARITYENV_AITTA_API_TOKEN="${AITTA_API_TOKEN:-}"
# NOTE: do NOT export SINGULARITYENV_PYTHONPATH here. The lumi-multitorch container ships its own
# PYTHONPATH (where pandas/torch live); clobbering it from outside hides those packages. Instead we
# prepend the patched copy to PYTHONPATH *inside* the container (see the bash -lc below), which both
# preserves the container packages and makes further-improvements/src win over the editable install.
LLM_FLAG=""
[[ "$ADMET_USE_LLM" == "1" ]] && LLM_FLAG="--llm"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Strided slice of the manifest so the array stays small (mirrors the fullexp workers).
RUN_INDICES=""
for (( idx=TASK_IDX; idx<N_JOBS; idx+=N_TASKS )); do
    RUN_INDICES="$RUN_INDICES $idx"
done
echo "[regen] task $TASK_IDX -> indices:$RUN_INDICES  (code=$CODE_SRC results=$RESULTS_DIR)"
export SINGULARITYENV_RUN_INDICES="$RUN_INDICES"
export SINGULARITYENV_MANIFEST="$MANIFEST"
export SINGULARITYENV_RESULTS_DIR="$RESULTS_DIR"
export SINGULARITYENV_LLM_FLAG="$LLM_FLAG"

cd "$WORK_DIR"
singularity exec \
    --bind /scratch/project_462001520:/scratch/project_462001520 \
    --bind /pfs/lustrep1:/pfs/lustrep1 \
    --pwd "$WORK_DIR" \
    "$CONTAINER" \
    bash -lc 'source "'"$VENV_ACTIVATE"'" && export PYTHONPATH="'"$CODE_SRC"'":${PYTHONPATH:-} && \
              rc=0 && \
              for IDX in $RUN_INDICES; do \
                echo "[regen] === manifest index $IDX ==="; \
                python -m experiment.cli run-job \
                  --manifest "$MANIFEST" \
                  --index "$IDX" \
                  --results-dir "$RESULTS_DIR" \
                  --force \
                  $LLM_FLAG || rc=1; \
              done; exit $rc'
