#!/bin/env bash
#SBATCH -A SNIC2022-22-370            # find your project with the "projinfo" command
#SBATCH -p alvis                      # what partition to use (usually not needed)
#SBATCH -t 0-02:00:00                 # how long time it will take to run
#SBATCH --gpus-per-node=T4:1          # choosing no. GPUs and their type
#SBATCH -J generate-cross-data            # the jobname (not needed)

# Load modules with vitual env 'venv_thesis'
# Make sure the same python version is used. Python 3.8.6. 
ml purge
ml Python/3.8.6-GCCcore-10.2.0 SciPy-bundle/2020.11-fosscuda-2020b TensorFlow/2.5.0-fosscuda-2020b 
source /mimer/NOBACKUP/groups/snic2022-22-370/venv_thesis/bin/activate

# Non-interactive
echo Hello pre pyscript test!
python3 gen_cross_data.py
echo Hello post pyscript test!