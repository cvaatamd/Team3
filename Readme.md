# Team 3


## Adding chemprop to the Pytorch container 

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
singularity shell $SIF
Singularity> python -m venv chemprop --system-site-packages
Singularity> source chemprop/bin/activate
(chemprop) Singularity> pip install chemprop

## Using chemprop with the Pytorch container

export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
run $SIF bash -c 'source chemprop/bin/activate && python -c "import chemprop; print(chemprop.__version__)"'