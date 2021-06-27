#!/bin/bash

#SBATCH --job-name=HSupCon
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu_veryshort
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:10:00
#SBATCH --mem=32GB


#! Mail to user if job aborts
#SBATCH --mail-type=END
#SBATCH --mail-user=yl18410@bristol.ac.uk

# Load the conda module
module load languages/anaconda3/3.7

# Load the modules required for runtime e.g
module load languages/intel/2017.01


source activate /mnt/storage/scratch/yl18410/DUQ_env_170521
echo "Activated environment"

cd Moco_Uncertainty

echo "Current working directory: $PWD"

export PYTHONPATH=/mnt/storage/home/yl18410/scratch/Moco_Uncertainty


python ./Contrastive_uncertainty/experiments/run/run_evaluate.py
