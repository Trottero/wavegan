#!/bin/bash

# SBATCH --mem=5G
# SBATCH --time=00:10:00
# SBATCH --ntasks=1
# SBATCH --gpus=1

# SBATCH --mail-user=wrlxs@hotmail.com
# SBATCH --mail-type=ALL
# SBATCH --output=output.log
# SBATCH --error=error.log
# SBATCH --job-name=tflowwavegan
# SBATCH --partition=testing

module load Miniconda3/4.7.10
conda activate ~/data/api-env

cd ~/data/wavegan

python test_tensorflow.py
