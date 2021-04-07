#!/bin/bash

#SBATCH --partition=SAN32M512_L
#SBATCH --cpus-per-task=24
#SBATCH -A col_nja224_uksr
#SBATCH --job-name=convert_laz_np
#SBATCH --output=laz_to_np_log.txt
#SBATCH --time=01:00:00
#SBATCH --export=all module load mvapich2_ib singularity

module purge
module load intel/19.0.4.243
module load impi/2019.4.243
module load ccs/singularity

CONTAINER=/psratch/nja224_uksr/lidar_container

source /home/dtjo223/.bashrc
conda activate lidar
python $PSCRATCH/nja224_uksr/dtjo223/lidar-harmonization/src/datasets/tools/laz_to_numpy.py
