#!/bin/bash
#SBATCH --job-name=admet-mini
#SBATCH --account=project_462001520
#SBATCH --partition=dev-g
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --array=0-7%4
#SBATCH --output=/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results-mini/logs/admet-mini-%A_%a.out
#SBATCH --error=/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results-mini/logs/admet-mini-%A_%a.err

set -euo pipefail
mkdir -p "/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results-mini" "/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results-mini/logs"

echo "[slurm] host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"

srun bash "/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results-mini/admet-mini.worker.sh" "$SLURM_ARRAY_TASK_ID"
