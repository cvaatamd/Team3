#!/bin/bash
# Submit a prediction-regeneration sweep for ONE config, using the patched copy.
#
# Steps (all light except the array itself, which runs on GPU):
#   1. build the JSONL manifest from a conf YAML  (inside the container, no GPU — fast)
#   2. render + sbatch a SLURM array that runs further-improvements/scripts/regen.worker.sh
#
# This NEVER trains on the login node: step 1 only enumerates jobs; the GPU work is the array.
#
# Usage:
#   bash further-improvements/scripts/submit_regen.sh CONF RESULTS_DIR [CONCURRENCY] [MAX_TASKS]
# Example (headline endpoints, persists preds):
#   bash further-improvements/scripts/submit_regen.sh \
#       further-improvements/conf/regen-mbpb.yaml \
#       /pfs/lustrep1/scratch/project_462001520/Team3/Team3/results-fullexp-regen/mbpb
set -euo pipefail

CONF="${1:?usage: submit_regen.sh CONF RESULTS_DIR [CONCURRENCY] [MAX_TASKS]}"
RESULTS_DIR="${2:?usage: submit_regen.sh CONF RESULTS_DIR [CONCURRENCY] [MAX_TASKS]}"
CONCURRENCY="${3:-100}"
MAX_TASKS="${4:-36}"

WORK_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3"
CODE_SRC="$WORK_DIR/further-improvements/src"
CONTAINER="/appl/local/laifs/containers/lumi-multitorch-latest.sif"
VENV_ACTIVATE="/scratch/project_462001520/Team3/ChemProp/chemprop/bin/activate"
WORKER="$WORK_DIR/further-improvements/scripts/regen.worker.sh"

mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/logs"
MANIFEST="$RESULTS_DIR/manifest.jsonl"

module purge 2>/dev/null || true
module use /appl/local/laifs/modules 2>/dev/null || true
module load lumi-aif-singularity-bindings 2>/dev/null || true

# 1) Build manifest with the PATCHED code (light; just enumerates jobs to JSONL).
singularity exec \
  --bind /scratch/project_462001520:/scratch/project_462001520 \
  --bind /pfs/lustrep1:/pfs/lustrep1 \
  --pwd "$WORK_DIR" "$CONTAINER" \
  bash -lc "source '$VENV_ACTIVATE' && export PYTHONPATH='$CODE_SRC':\${PYTHONPATH:-} && \
            python -c \"
from pathlib import Path
from experiment.runner import load_plan, write_manifest, read_manifest
plan = load_plan(Path('$CONF'))
write_manifest(plan, Path('$MANIFEST'))
print('manifest jobs:', len(read_manifest(Path('$MANIFEST'))))
\""

N_JOBS=$(grep -c . "$MANIFEST")
N_TASKS=$(( N_JOBS < MAX_TASKS ? N_JOBS : MAX_TASKS ))
ARRAY_MAX=$(( N_TASKS - 1 ))
echo "[submit_regen] $N_JOBS jobs -> array 0-$ARRAY_MAX%$CONCURRENCY  results=$RESULTS_DIR"

# 2) Render + submit the array.
SBATCH="$RESULTS_DIR/regen.sbatch"
cat > "$SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=regen-$(basename "$RESULTS_DIR")
#SBATCH --account=project_462001520
#SBATCH --partition=small-g
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --array=0-$ARRAY_MAX%$CONCURRENCY
#SBATCH --output=$RESULTS_DIR/logs/regen-%A_%a.out
#SBATCH --error=$RESULTS_DIR/logs/regen-%A_%a.err

set -euo pipefail
export RESULTS_DIR="$RESULTS_DIR"
export MANIFEST="$MANIFEST"
export N_JOBS=$N_JOBS
export N_TASKS=$N_TASKS
export ADMET_USE_LLM=${ADMET_USE_LLM:-1}
srun bash "$WORKER" "\$SLURM_ARRAY_TASK_ID"
EOF

DEP_ARG=()
if [[ -n "${SBATCH_DEP:-}" ]]; then
    DEP_ARG=(--dependency="afterok:${SBATCH_DEP}")
    echo "[submit_regen] gated on job ${SBATCH_DEP} (afterok)"
fi
JOB_ID=$(sbatch --parsable "${DEP_ARG[@]}" "$SBATCH")
echo "[submit_regen] submitted array job $JOB_ID"
echo "[submit_regen] when it finishes:"
echo "  PYTHONPATH=$CODE_SRC python further-improvements/scripts/rescore_official.py --preds-dir $RESULTS_DIR/preds"
echo "  python further-improvements/scripts/bootstrap_lift_molecule.py --preds-dir $RESULTS_DIR/preds"
