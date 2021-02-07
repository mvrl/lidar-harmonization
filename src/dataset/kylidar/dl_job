#!/bin/bash

#SBATCH --partition=CAL48M192_D
#SBATCH --cpus-per-task=24
#SBATCH -A col_nja224_uksr
#SBATCH --job-name=intensity_harmonization
#SBATCH --output=lcc_run_log.txt
#SBATCH --time=01:00:00
#SBATCH --export=all module load mvapich2_ib singularity

module purge
module load intel/19.0.4.243
module load impi/2019.4.243
module load ccs/singularity
 
CONTAINER=/pscratch/nja224_uksr/lidar_container

source /home/dtjo223/.bashrc
conda activate lidar
python ~/workspace/lidar-harmonization/src/dataset/kylidar/dl_kylidar.py
