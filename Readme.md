# Team 3 
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
```