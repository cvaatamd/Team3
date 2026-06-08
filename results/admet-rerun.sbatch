#!/bin/bash
#SBATCH --job-name=admet-rerun
#SBATCH --account=project_462001520
#SBATCH --partition=small-g
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --array=0-20%16
#SBATCH --output=/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results/logs/admet-rerun-%A_%a.out
#SBATCH --error=/pfs/lustrep1/scratch/project_462001520/Team3/Team3/results/logs/admet-rerun-%A_%a.err

set -euo pipefail
echo "[slurm] host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
srun bash /pfs/lustrep1/scratch/project_462001520/Team3/Team3/results/admet-rerun.worker.sh "$SLURM_ARRAY_TASK_ID"
