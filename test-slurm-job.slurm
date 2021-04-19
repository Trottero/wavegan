#!/bin/bash

# SBATCH --mem=5G
# SBATCH --time=00:10:00
# SBATCH --ntasks=1
# SBATCH --gpus=1

# SBATCH --mail-user=wrlxs@hotmail.com
# SBATCH --mail-type=ALL
# SBATCH --output=%x_%j.out
# SBATCH --error=%x_%j.err
# SBATCH --job-name=tflowwavegan
# SBATCH --partition=gpu-short

module load TensorFlow/1.13.1-fosscuda-2019a-Python-3.7.2
module load Miniconda3/4.7.10
echo "Loaded module"

conda init bash
source ~/.bashrc
echo "Initialized Conda"
conda activate ~/data/api-env
echo "Activated environment"

cd ~/data/wavegan

python test_tensorflow.py
