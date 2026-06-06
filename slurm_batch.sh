#BATCH 
#BATCH


source chemprop/bin/activate 
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
singularity shell $SIF
bash scripts/submit_sweep.sh --dry-run --llm