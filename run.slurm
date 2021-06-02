#!/bin/bash

#SBATCH --time=6-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=12G

#SBATCH --mail-user=jmcvink@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --job-name=tflowwavegan
#SBATCH --partition=gpu-long

module load TensorFlow/1.13.1-fosscuda-2019a-Python-3.7.2
pip install scipy==1.1.0 scipy==1.1.0 numpy==1.19.1 numba==0.48.0 librosa==0.6.2 --user


cd ~/data/API/wavegan

python train_wavegan.py train ./train-techno --data_dir ./techno-120 --data_first_slice --wavegan_dim 32 --train_batch_size 16
