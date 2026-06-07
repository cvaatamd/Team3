#!/bin/bash
#SBATCH --job-name=regen-pf
#SBATCH --account=project_462001520
#SBATCH --partition=small-g
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --array=0-35%100
#SBATCH --output=/pfs/lustrep1/scratch/project_462001520/Team3/Team3/further-improvements/results-fewshot/pf/logs/regen-%A_%a.out
#SBATCH --error=/pfs/lustrep1/scratch/project_462001520/Team3/Team3/further-improvements/results-fewshot/pf/logs/regen-%A_%a.err

set -euo pipefail
export RESULTS_DIR="/pfs/lustrep1/scratch/project_462001520/Team3/Team3/further-improvements/results-fewshot/pf"
export MANIFEST="/pfs/lustrep1/scratch/project_462001520/Team3/Team3/further-improvements/results-fewshot/pf/manifest.jsonl"
export N_JOBS=135
export N_TASKS=36
export ADMET_USE_LLM=0
srun bash "/pfs/lustrep1/scratch/project_462001520/Team3/Team3/further-improvements/scripts/regen.worker.sh" "$SLURM_ARRAY_TASK_ID"
