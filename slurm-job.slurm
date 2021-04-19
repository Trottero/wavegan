#!/bin/bash

#SBATCH --job-name=test_helloworld
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=benjamin@thecrave.nl
#SBATCH --mail-type="ALL"

SBATCH --mem=5G
SBATCH --time=00:10:00
SBATCH --partition=gpu-short
SBATCH --ntasks=1
SBATCH --gpus=1

SBATCH --mail-user=wrlxs@hotmail.com
SBATCH --mail-type=ALL
SBATCH --output=output.log
SBATCH --error=error.log
SBATCH --job-name=tflowwavegan
SBATCH --partition=testing

cd ~/data/wavegan

python train_wavegan.py train ./train --data_dir ./dataset/dataset/train/
