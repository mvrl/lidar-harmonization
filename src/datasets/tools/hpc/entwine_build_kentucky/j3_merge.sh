#!/bin/bash
 
#SBATCH -t 1-00:00:00 				    # Time for the job to run 
#SBATCH --job-name=ENTWINE_BUILD_3   	# Name of the job

#SBATCH --cpus-per-task=8
#SBATCH --partition=HAS24M128_S # Name of the queue

#SBATCH --mail-type ALL
#SBATCH --mail-user nathan.jacobs@uky.edu
#SBATCH --account=col_nja224_uksr

#SBATCH --output=log-j3_merge-%A.out

source ./config.sh

module load ccs/anaconda/3

conda activate entwine

entwine merge ${ENTWINE_OUT} -t 8

