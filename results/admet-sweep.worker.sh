#!/bin/bash
# Per-task worker: prepare environment, then run one job from the manifest.
# Argument: array task index (== JSONL line number).

set -euo pipefail
TASK_IDX="${1:?usage: $0 ARRAY_TASK_INDEX}"

WORK_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3"
RESULTS_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results"
MANIFEST="/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results/manifest.jsonl"
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

# This array task handles a STRIDED slice of the manifest so the SLURM array
# stays small (<= max_array_tasks) and we never exceed LUMI's submit limit:
#   indices TASK_IDX, TASK_IDX+N_TASKS, TASK_IDX+2*N_TASKS, ... < N_JOBS
N_JOBS=240
N_TASKS=150
RUN_INDICES=""
for (( idx=TASK_IDX; idx<N_JOBS; idx+=N_TASKS )); do
    RUN_INDICES="$RUN_INDICES $idx"
done
echo "[worker] array task $TASK_IDX handles manifest indices:$RUN_INDICES"
export SINGULARITYENV_RUN_INDICES="$RUN_INDICES"

cd "$WORK_DIR"

# Launch the container ONCE and loop over our indices inside it (amortizes the
# slow container start + imports). A single job failure is logged but does not
# abort the rest; the task's exit code is non-zero if any job failed.
singularity exec \
    --bind /scratch/project_462001520:/scratch/project_462001520 \
    --bind /pfs/lustrep1:/pfs/lustrep1 \
    --pwd "$WORK_DIR" \
    "$CONTAINER" \
    bash -lc "source '$VENV_ACTIVATE' && rc=0 && \
              for IDX in \$RUN_INDICES; do \
                echo \"[worker] === manifest index \$IDX ===\"; \
                python -m experiment.cli run-job \
                  --manifest '$MANIFEST' \
                  --index \"\$IDX\" \
                  --results-dir '$RESULTS_DIR' \
                  $LLM_FLAG || rc=1; \
              done; exit \$rc"
