#!/bin/bash

#SBATCH --partition=V4V32_SKY32M192_L 
#SBATCH --cpus-per-task=32
#SBATCH -A gol_nja224_uksr
#SBATCH --job-name=harmonize_dublin
#SBATCH --output=harmonize_log.txt
#SBATCH --time=3-00:00:00
##SBATCH --export=all module load mvapich2_ib
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module purge
module load intel/19.0.4.243
module load impi/2019.4.243

source /home/dtjo223/.bashrc
conda activate lidar
grep -c ^processor /proc/cpuinfo
nvidia-smi
python $PSCRATCH/nja224_uksr/dtjo223/lidar-harmonization/src/run.py
