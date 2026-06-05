<<<<<<< HEAD
# Team 3 
=======

# Team 3

## Pytorch container

This will add chemprop to a venv that will be located on disk in the working directory, not the Pytorch container itself.

Based on https://docs.lumi-supercomputer.eu/laif/software/ai-environment/#add-more-pip-packages-to-container . 

### Adding chemprop to the Pytorch container 

```
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
singularity shell $SIF
Singularity> python -m venv chemprop --system-site-packages
Singularity> source chemprop/bin/activate
(chemprop) Singularity> pip install chemprop
```

### Using chemprop with the Pytorch container

```
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
run $SIF bash -c 'source chemprop/bin/activate && python -c "import chemprop; print(chemprop.__version__)"'
```

## ChemProp container

>>>>>>> fc51b1b6158e781924807a5c8dab49660ce93e0c
1. Prepare ChemProp container in Lumi
```shell
module load CrayEnv
module load cotainr
cotainr build chemprop.sif --system=lumi-g --conda-env=chemprop_env.yml
```

2. Running the container:
```shell

#!/bin/bash
#SBATCH --job-name=Chem prop
#SBATCH --account=project_XXXX
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=small-g

srun singularity exec --bind /projappl/project_XXXX:/projappl/project_XXXX --pwd /projappl/project_XXXX  container.sif python3 /projappl/project_XXXX/wd/python.py -wd /projappl/project_462000643/WD
<<<<<<< HEAD
```
=======
```

>>>>>>> fc51b1b6158e781924807a5c8dab49660ce93e0c
