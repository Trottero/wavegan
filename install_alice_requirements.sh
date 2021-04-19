cd ~/data/

# Load conda module
module load Miniconda3/4.7.10

# Recreate virtual environment
rm -rf api-env
conda create --prefix ~/data/api-env --name api-env
conda activate api-env
conda install -c anaconda tensorflow-gpu==1.12.0 -y
conda install -c anaconda scipy==1.1.0 -y
conda install -c conda-forge librosa==0.6.2 -y
conda install -c anaconda numpy==1.19.1 -y
conda install -c anaconda numba==0.48.0 -y
